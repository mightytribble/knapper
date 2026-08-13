use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result};
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::schemars;
use rmcp::schemars::JsonSchema;
use rmcp::{ErrorData as McpError, ServiceExt, tool, tool_handler, tool_router};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::config::Config;
use crate::context::{self, ContextParams};
use crate::llm::{EmbedModel, RerankModel};
use crate::profile::VaultProfile;
use crate::search;
use crate::store::Store;

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HealthParams {}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Map of recently-written file paths to their mtime.
/// Used to tell the watcher "I just wrote this file, skip re-indexing it."
pub type RecentWrites = Arc<Mutex<HashMap<PathBuf, SystemTime>>>;

#[derive(Clone)]
pub struct EngraphServer {
    store: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
    vault_path: Arc<PathBuf>,
    profile: Arc<Option<VaultProfile>>,
    #[allow(dead_code)] // Required by rmcp #[tool_router] macro infrastructure
    tool_router: ToolRouter<Self>,
    /// Result reranker (None when intelligence is disabled or failed to load).
    reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>>,
    /// Tracks files recently written by MCP tools so the watcher can skip re-indexing them.
    recent_writes: RecentWrites,
    /// When true, write/edit/delete MCP tools return an error instead of executing.
    read_only: bool,
    /// Retrieval granularity settings from `config.toml`, so MCP callers get the
    /// same result shape the CLI does.
    max_chunks_per_file: usize,
    group_by: crate::config::GroupBy,
    /// Rerank-lane settings from `config.toml`.
    rerank: crate::config::RerankConfig,
    /// Ranking-stage settings from `config.toml`.
    ranking: crate::config::RankingConfig,
    lane_weights: crate::config::LaneWeights,
    /// Keyword-lane settings from `config.toml`. The BM25 weights are
    /// positional over the columns the store's index is declared with, so this
    /// has to be the config the store was built from (issue #37).
    fts: crate::config::FtsConfig,
    /// Embedding-prefix settings, for the same reason: notes written by MCP
    /// tools share a vector space with the indexed vault.
    embed: crate::prefix::EmbedComposition,
    /// The chunker settings, one step earlier than `embed` for the same reason: a
    /// note written by an MCP tool has to be cut into the rows a re-index of it
    /// would produce (issue #43), and the same now holds for where a section
    /// starts (issue #44).
    chunk_opts: crate::chunker::ChunkOptions,
}

fn read_only_err() -> McpError {
    McpError::new(
        rmcp::model::ErrorCode::INVALID_REQUEST,
        "Write operations disabled in read-only mode. Start server without --read-only to enable writes.".to_string(),
        None::<serde_json::Value>,
    )
}

fn mcp_err(e: &anyhow::Error) -> McpError {
    McpError::new(
        rmcp::model::ErrorCode::INTERNAL_ERROR,
        format!("{e:#}"),
        None::<serde_json::Value>,
    )
}

fn to_json_result<T: serde::Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let json = serde_json::to_string_pretty(value).map_err(|e| {
        McpError::new(
            rmcp::model::ErrorCode::INTERNAL_ERROR,
            e.to_string(),
            None::<serde_json::Value>,
        )
    })?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

/// Record a recently-written file path + mtime so the watcher can skip re-indexing it.
async fn record_write(recent_writes: &RecentWrites, path: &Path) {
    if let Ok(meta) = std::fs::metadata(path)
        && let Ok(mtime) = meta.modified()
    {
        recent_writes.lock().await.insert(path.to_path_buf(), mtime);
    }
}

#[tool_router(vis = "pub(crate)")]
impl EngraphServer {
    #[tool(
        name = "search",
        description = "Semantic + keyword hybrid search across the vault. Returns ranked results with file paths, scores, headings, and snippets."
    )]
    async fn search(
        &self,
        params: Parameters<crate::params::Search>,
    ) -> Result<CallToolResult, McpError> {
        let top_n = params.0.top_n.unwrap_or(10);
        let all_terms = crate::tags::merge_all_alias(params.0.tags, params.0.all);
        let scope = crate::tags::TagFilter::parse(&all_terms, &params.0.any, &params.0.none)
            .map_err(|e| mcp_err(&e))?;
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;

        // Lock the cross-encoder if it is available.
        let mut rerank_guard = match &self.reranker {
            Some(r) => Some(r.lock().await),
            None => None,
        };

        let mut config = search::SearchConfig {
            reranker: rerank_guard
                .as_mut()
                .map(|g| g.as_mut() as &mut dyn RerankModel),
            store: &store,
            rerank_candidates: 30,
            rerank: self.rerank,
            max_chunks_per_file: self.max_chunks_per_file,
            group_by: self.group_by,
            ranking: self.ranking,
            lane_weights: self.lane_weights,
            fts: self.fts,
            scope,
        };

        let output =
            search::search_with_intelligence(&params.0.query, top_n, &mut *embedder, &mut config)
                .map_err(|e| mcp_err(&e))?;

        // The answer floor (#34). The empty array is the answer, and the message
        // stops a caller from reading it as a transport failure and sending the
        // query again. The message is a second content block and does not
        // replace the JSON, because text in place of the array would break a
        // client that parses it. MCP holds both without a change to the schema,
        // and #35 owns the schema.
        if output.results.is_empty() {
            let mut result = to_json_result(&output.results)?;
            result
                .content
                .push(Content::text(crate::ranking::NO_RELEVANT_CONTENT));
            return Ok(result);
        }
        to_json_result(&output.results)
    }

    #[tool(
        name = "read",
        description = "Read a note's full content with metadata, tags, and graph edges. Accepts file path, basename, or #docid."
    )]
    async fn read(
        &self,
        params: Parameters<crate::params::Read>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let note = context::context_read(&ctx, &params.0.file, params.0.section.as_deref())
            .map_err(|e| mcp_err(&e))?;
        to_json_result(&note)
    }

    #[tool(
        name = "list",
        description = "List notes filtered by folder prefix and tag operators (all/any/none). A term is a tag path; a trailing `/` matches the tag and its descendants. Returns paths, docids, tags, and edge counts."
    )]
    async fn list(
        &self,
        params: Parameters<crate::params::List>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let all_terms = crate::tags::merge_all_alias(params.0.tags, params.0.all);
        let tags = crate::tags::TagFilter::parse(&all_terms, &params.0.any, &params.0.none)
            .map_err(|e| mcp_err(&e))?;
        let items = context::context_list(
            &ctx,
            params.0.folder.as_deref(),
            &tags,
            params.0.created_by.as_deref(),
            params.0.limit,
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&items)
    }

    #[tool(
        name = "tags",
        description = "The vault's tag vocabulary: every tag, or the subtree under one term, each with the notes carrying it. Call before filtering with list."
    )]
    async fn tags(
        &self,
        params: Parameters<crate::params::Tags>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let prefix = params.0.under.as_deref().and_then(crate::tags::parse_term);
        let rows = store.tags_under(prefix.as_ref()).map_err(|e| mcp_err(&e))?;
        to_json_result(&rows)
    }

    #[tool(
        name = "vault_map",
        description = "Vault structure overview: folders, tags, file counts, recent files. Use to orient before deeper queries."
    )]
    async fn vault_map(&self) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let map = context::vault_map(&ctx).map_err(|e| mcp_err(&e))?;
        to_json_result(&map)
    }

    #[tool(
        name = "who",
        description = "Person context bundle: their note, mentions across the vault, and graph connections."
    )]
    async fn who(
        &self,
        params: Parameters<crate::params::Who>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let person = context::context_who(&ctx, &params.0.name).map_err(|e| mcp_err(&e))?;
        to_json_result(&person)
    }

    #[tool(
        name = "project",
        description = "Project context bundle: project note, child notes, active tasks, team members, and recent daily mentions."
    )]
    async fn project(
        &self,
        params: Parameters<crate::params::Project>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let proj = context::context_project(&ctx, &params.0.name).map_err(|e| mcp_err(&e))?;
        to_json_result(&proj)
    }

    #[tool(
        name = "context",
        description = "Rich topic context with search-driven section selection and character budget trimming. Returns the most relevant note sections for a topic."
    )]
    async fn context(
        &self,
        params: Parameters<crate::params::Topic>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let ctx = ContextParams {
            store: &store,
            vault_path: &self.vault_path,
            profile: self.profile.as_ref().as_ref(),
        };
        let bundle = context::context_topic_with_search(
            &ctx,
            &params.0.query,
            params.0.budget,
            &mut *embedder,
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&bundle)
    }

    #[tool(
        name = "create",
        description = "Create a new note with automatic tag resolution, link discovery, and folder placement. Returns the created file's path, docid, and what was auto-resolved."
    )]
    async fn create(
        &self,
        params: Parameters<crate::params::Create>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        // No stdin exists on this surface, so an omitted content is an
        // error here instead of the CLI's fallback read.
        let content = params.0.content.ok_or_else(|| {
            McpError::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "content is required",
                None::<serde_json::Value>,
            )
        })?;
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let input = crate::writer::CreateNoteInput {
            content,
            filename: params.0.filename,
            type_hint: params.0.type_hint,
            tags: params.0.tags,
            folder: params.0.folder,
            created_by: "claude-code".into(),
            auto_link: params.0.auto_link,
        };
        let result = crate::writer::create_note(
            input,
            &store,
            &mut *embedder,
            self.embed,
            self.chunk_opts,
            &self.vault_path,
            self.profile.as_ref().as_ref(),
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "update",
        description = "Change an existing note. Takes a list of edits and applies them in order, in one write: one conflict check, one file write. \
             Each edit names its target. `section` is one heading. `property` is one frontmatter key. An edit that names neither targets the note's body, and an edit that names both is an error. \
             `mode` is `replace`, `append`, `prepend` or `remove`. `remove` is for a property alone. \
             `content` is a string, or a list of strings to set a list-valued property such as tags or aliases. A body edit and a section edit take a string. \
             A body edit always keeps the note's frontmatter: content that starts with its own `---` block gives the note two of them. Change the frontmatter with `property` edits in the same list. \
             Three things differ from the calls this replaces. A note changed outside engraph and not yet re-indexed fails with an mtime conflict. \
             Replacing a note's frontmatter wholesale has no spelling here — `rewrite`'s `preserve_frontmatter: false` is gone, not renamed, and write the new frontmatter with `property` edits instead. \
             A whole-note tag or alias replacement no longer stamps a `modified_by` property on the note."
    )]
    async fn update(
        &self,
        params: Parameters<crate::params::Update>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        // The whole list is read before anything is written, so a request
        // that names an impossible target is a parameter error and not a
        // half-applied write (#62).
        let edits = params.0.to_writer_edits().map_err(|e| {
            McpError::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                format!("{e:#}"),
                None::<serde_json::Value>,
            )
        })?;
        let store = self.store.lock().await;
        let input = crate::writer::UpdateInput {
            file: params.0.file,
            edits,
            modified_by: "claude-code".into(),
        };
        let result = crate::writer::update_note(&store, &self.vault_path, &input)
            .map_err(|e| mcp_err(&e))?;
        // `update_note` stores the new content hash and writes no chunks, so
        // nothing else will re-derive them: not `diff_vault`, which sees a
        // hash that already matches disk, and not the watcher, which the
        // `record_write` below tells to skip this file. Re-index here or the
        // note stays searchable only as the text it held before the edit (#62).
        let mut embedder = self.embedder.lock().await;
        // A failure here happens after the write, so a bare INTERNAL_ERROR
        // would read as "nothing happened" — say what did. Returning early
        // also skips `record_write` below, so the watcher's event on this
        // file is not suppressed and it re-indexes it on its own; that
        // recovery is deliberate, not accidental.
        crate::indexer::reindex_written_file(
            &result.path,
            &store,
            &mut *embedder,
            &self.vault_path,
            self.chunk_opts,
        )
        .with_context(|| {
            format!(
                "the file was written; its index rows were not updated for {}",
                result.path
            )
        })
        .map_err(|e| mcp_err(&e))?;
        // Record write so the watcher skips re-indexing
        let full_path = self.vault_path.join(&result.path);
        record_write(&self.recent_writes, &full_path).await;
        to_json_result(&result)
    }

    #[tool(
        name = "move_note",
        description = "Move a note to a different folder. Updates the index path."
    )]
    async fn move_note(
        &self,
        params: Parameters<crate::params::Move>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let result = crate::writer::move_note(
            &params.0.file,
            &params.0.new_folder,
            &store,
            &self.vault_path,
        )
        .map_err(|e| mcp_err(&e))?;
        to_json_result(&result)
    }

    #[tool(
        name = "archive",
        description = "Archive a note: moves it to the archive folder, removes from search index. The note is preserved on disk but invisible to search/context. `undo: true` reverses this: restores the note to its original location and re-indexes it."
    )]
    async fn archive(
        &self,
        params: Parameters<crate::params::Archive>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        // Archiving and restoring are one operation and its reverse, so they
        // are one capability with a flag rather than two names (#62).
        let result = if params.0.undo {
            let mut embedder = self.embedder.lock().await;
            crate::writer::unarchive_note(
                &params.0.file,
                &store,
                &mut *embedder,
                self.embed,
                self.chunk_opts,
                &self.vault_path,
            )
            .map_err(|e| mcp_err(&e))?
        } else {
            crate::writer::archive_note(
                &params.0.file,
                &store,
                &self.vault_path,
                self.profile.as_ref().as_ref(),
            )
            .map_err(|e| mcp_err(&e))?
        };
        to_json_result(&result)
    }

    #[tool(
        name = "health",
        description = "Vault health report: orphans, broken links, stale notes, tag hygiene, index freshness."
    )]
    async fn health(&self, _params: Parameters<HealthParams>) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let profile_ref = self.profile.as_ref().as_ref();
        let config = crate::health::HealthConfig {
            daily_folder: profile_ref.and_then(|p| p.structure.folders.daily.clone()),
            inbox_folder: profile_ref.and_then(|p| p.structure.folders.inbox.clone()),
        };
        let report =
            crate::health::generate_health_report(&store, &config).map_err(|e| mcp_err(&e))?;
        to_json_result(&report)
    }

    #[tool(
        name = "migrate",
        description = "Restructure the vault into PARA. Mode 'preview' classifies every note into Projects/Areas/Resources/Archive and returns the proposed moves with confidence scores; 'apply' performs the moves of a preview; 'undo' reverses the last migration."
    )]
    async fn migrate(
        &self,
        params: Parameters<crate::params::Migrate>,
    ) -> Result<CallToolResult, McpError> {
        // The CLI already took a mode. MCP and HTTP split it into three
        // names, which is the same capability spelled three ways (#62).
        match params.0.mode.as_str() {
            "preview" => {
                let store = self.store.lock().await;
                let profile_ref = self.profile.as_ref().as_ref();
                let preview =
                    crate::migrate::generate_preview(&store, &self.vault_path, profile_ref)
                        .map_err(|e| mcp_err(&e))?;
                to_json_result(&preview)
            }
            "apply" => {
                if self.read_only {
                    return Err(read_only_err());
                }
                let store = self.store.lock().await;
                let data_dir = crate::config::Config::data_dir().map_err(|e| mcp_err(&e))?;
                let preview = crate::migrate::resolve_preview(params.0.preview, &data_dir)
                    .map_err(|e| mcp_err(&e))?;
                let result = crate::migrate::apply_preview(&preview, &store, &self.vault_path)
                    .map_err(|e| mcp_err(&e))?;
                to_json_result(&result)
            }
            "undo" => {
                if self.read_only {
                    return Err(read_only_err());
                }
                let store = self.store.lock().await;
                let result =
                    crate::migrate::undo_last(&store, &self.vault_path).map_err(|e| mcp_err(&e))?;
                to_json_result(&result)
            }
            // The mode is the caller's own text, so a word that names no
            // operation is an invalid parameter and not an internal fault.
            other => Err(McpError::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                format!("Unknown mode: {other}. Use 'preview', 'apply' or 'undo'."),
                None::<serde_json::Value>,
            )),
        }
    }

    #[tool(
        name = "delete",
        description = "Delete a note. Soft mode (default) moves it to the archive folder. Hard mode permanently removes it from disk and index."
    )]
    async fn delete(
        &self,
        params: Parameters<crate::params::Delete>,
    ) -> Result<CallToolResult, McpError> {
        if self.read_only {
            return Err(read_only_err());
        }
        let store = self.store.lock().await;
        let mode = match params.0.mode.as_str() {
            "hard" => crate::writer::DeleteMode::Hard,
            _ => crate::writer::DeleteMode::Soft,
        };
        let archive_folder = self
            .profile
            .as_ref()
            .as_ref()
            .and_then(|p| p.structure.folders.archive.as_deref())
            .unwrap_or("04-Archive");
        crate::writer::delete_note(
            &store,
            &self.vault_path,
            &params.0.file,
            mode,
            archive_folder,
        )
        .map_err(|e| mcp_err(&e))?;
        let result = serde_json::json!({
            "deleted": params.0.file,
            "mode": params.0.mode,
        });
        to_json_result(&result)
    }

    #[tool(
        name = "reindex_file",
        description = "Re-index a single file after external edits. Reads the file from disk, re-embeds its chunks, and updates the search index. Use when a file was modified outside engraph and you need the index to reflect current content."
    )]
    async fn reindex_file(
        &self,
        params: Parameters<crate::params::ReindexFile>,
    ) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let mut embedder = self.embedder.lock().await;
        let rel_path = params.0.file;
        let full_path = self.vault_path.join(&rel_path);

        // Read file content from disk
        let content = std::fs::read_to_string(&full_path).map_err(|e| {
            McpError::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                format!("Cannot read file {rel_path}: {e}"),
                None::<serde_json::Value>,
            )
        })?;

        let content_hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            format!("{:x}", hasher.finalize())
        };

        let mut config = crate::config::Config::load().unwrap_or_default();
        // The chunker settings come from the session, not from this load: a
        // load that fails falls back to the defaults, and one file re-chunked
        // at settings the rest of the store was not built at is a set of rows
        // nothing downstream can tell apart. Carrying the captured value is
        // what `ChunkOptions` exists to give.
        config.set_chunk_options(self.chunk_opts);

        // Re-index the file (handles cleanup of old entries automatically)
        let result = crate::indexer::index_file(
            &rel_path,
            &content,
            &content_hash,
            &store,
            &mut *embedder,
            &self.vault_path,
            &config,
        )
        .map_err(|e| mcp_err(&e))?;

        // Rebuild edges for the re-indexed file
        // Outgoing only — see issue #27.
        store
            .delete_outgoing_edges_for_file(result.file_id)
            .map_err(|e| mcp_err(&e))?;
        crate::indexer::build_edges_for_file(&store, result.file_id, &content)
            .map_err(|e| mcp_err(&e))?;

        let output = serde_json::json!({
            "file": rel_path,
            "chunks": result.total_chunks,
            "docid": result.docid,
        });
        to_json_result(&output)
    }

    #[tool(
        name = "identity",
        description = "Returns compact user identity and current context. Call at session start for instant context. L0 = static identity (~50 tokens), L1 = dynamic state (~120 tokens)."
    )]
    async fn identity(&self) -> Result<CallToolResult, McpError> {
        let store = self.store.lock().await;
        let config = crate::config::Config::load().unwrap_or_default();
        let block =
            crate::identity::format_identity_block(&config, &store).map_err(|e| mcp_err(&e))?;
        Ok(CallToolResult::success(vec![Content::text(block)]))
    }

    #[tool(
        name = "init",
        description = "Run first-time setup or update identity. Use 'detect' mode to inspect the vault without changes, 'apply' mode to configure identity and index. Returns JSON."
    )]
    async fn init(
        &self,
        params: Parameters<crate::params::Init>,
    ) -> Result<CallToolResult, McpError> {
        match params.0.mode.as_deref() {
            Some("detect") => {
                let result = crate::onboarding::run_detect_json(&self.vault_path)
                    .map_err(|e| mcp_err(&e))?;
                to_json_result(&result)
            }
            Some("apply") => {
                let mut config = crate::config::Config::load().unwrap_or_default();
                let data_dir = crate::config::Config::data_dir().map_err(|e| mcp_err(&e))?;
                let flags = crate::onboarding::ApplyFlags {
                    name: params.0.name,
                    role: params.0.role,
                    purpose: params.0.purpose,
                    identity_only: false,
                    reindex_only: false,
                };
                let result = crate::onboarding::run_apply_json(
                    &self.vault_path,
                    &mut config,
                    &data_dir,
                    flags,
                )
                .map_err(|e| mcp_err(&e))?;
                to_json_result(&result)
            }
            Some(other) => Err(McpError::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                format!("Unknown mode: {other}. Use 'detect' or 'apply'."),
                None::<serde_json::Value>,
            )),
            // A server has no interactive flow, so `init` there needs a
            // mode. The CLI's no-mode form is its own prompt sequence and
            // reaches no surface but the CLI (#62).
            None => Err(McpError::new(
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "init needs mode=detect or mode=apply",
                None::<serde_json::Value>,
            )),
        }
    }
}

#[tool_handler]
impl rmcp::handler::server::ServerHandler for EngraphServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "engraph: vault intelligence for Obsidian. \
                 Read: vault_map to orient, tags for the tag vocabulary, search to find, read for content (a section parameter narrows it), who/project for context bundles, health for vault diagnostics. \
                 Write: create for new notes, update for every change to an existing one — a list of edits over the body, a section or a frontmatter property, applied in one write. \
                 Lifecycle: move_note to relocate, archive to soft-delete (`undo: true` to restore), delete for permanent removal. \
                 Index: reindex_file to refresh a single file's index after external edits. \
                 Identity: identity for user context at session start, init to run first-time onboarding (`mode: detect` or `mode: apply`). \
                 Migration: migrate with `mode: preview` to classify notes into PARA folders, `mode: apply` to execute the migration, `mode: undo` to revert.",
        )
    }
}

// ---------------------------------------------------------------------------
// HTTP server options (populated by CLI flags in Task 7)
// ---------------------------------------------------------------------------

pub struct HttpServeOpts {
    pub port: u16,
    pub host: String,
    pub no_auth: bool,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run_serve(
    data_dir: &Path,
    http_opts: Option<HttpServeOpts>,
    read_only: bool,
) -> Result<()> {
    if let Some(ref opts) = http_opts
        && opts.no_auth
        && opts.host != "127.0.0.1"
    {
        anyhow::bail!(
            "--no-auth cannot be used with --host {} (only 127.0.0.1 is allowed)",
            opts.host
        );
    }

    let db_path = data_dir.join("engraph.db");
    let models_dir = data_dir.join("models");

    let store = Store::open(&db_path)?;
    let config = Config::load()?;
    let embedder = crate::llm::LlamaEmbed::new(&models_dir, &config)?;
    store.verify_embedding_dim(embedder.dim())?;

    let vault_path_str = store.get_meta("vault_path")?.ok_or_else(|| {
        anyhow::anyhow!("No vault path in index. Run 'engraph index <path>' first.")
    })?;
    let vault_path = PathBuf::from(&vault_path_str);

    let cleaned = crate::writer::cleanup_temp_files(&vault_path)?;
    if cleaned > 0 {
        eprintln!(
            "Cleaned up {} incomplete write(s) from previous run",
            cleaned
        );
    }

    let orphans = crate::writer::verify_index_integrity(&store, &vault_path)?;
    if orphans > 0 {
        eprintln!("Cleaned up {} orphan DB entries for missing files", orphans);
    }

    let profile = Config::load_vault_profile().ok().flatten();

    // Load the cross-encoder if enabled
    let reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>> = if config.intelligence_enabled()
    {
        match crate::llm::LlamaRerank::new(&models_dir, &config) {
            Ok(rerank) => Some(Arc::new(Mutex::new(
                Box::new(rerank) as Box<dyn RerankModel + Send>
            ))),
            Err(e) => {
                tracing::warn!("failed to load reranker: {e}, reranking disabled");
                None
            }
        }
    } else {
        None
    };

    // Refuse to serve an index this build did not produce (issue #31).
    //
    // The startup reconciliation below would fix a stale store, but it runs in a
    // spawned task and only warns on failure — so the server would answer from
    // the old index in the meantime, which is exactly the silent wrong answer
    // fingerprints exist to prevent. Checked here, synchronously, beside the
    // dimension guard that already refuses to start for the same reason.
    {
        let rerank_fp = match &reranker {
            Some(model) => Some(model.lock().await.fingerprint()),
            None => None,
        };
        let fingerprints = crate::fingerprint::Fingerprints::compute(
            &config,
            &EmbedModel::fingerprint(&embedder),
            rerank_fp.as_deref(),
        );
        crate::fingerprint::verify(&store, &fingerprints)?;
    }

    let store_arc = Arc::new(Mutex::new(store));
    let embedder_arc: Arc<Mutex<Box<dyn EmbedModel + Send>>> =
        Arc::new(Mutex::new(Box::new(embedder) as Box<dyn EmbedModel + Send>));
    let vault_path_arc = Arc::new(vault_path);
    let profile_arc = Arc::new(profile);
    let recent_writes: RecentWrites = Arc::new(Mutex::new(HashMap::new()));

    // Clone Arcs for HTTP server before MCP consumes them
    let http_store = store_arc.clone();
    let http_embedder = embedder_arc.clone();
    let http_vault_path = vault_path_arc.clone();
    let http_profile = profile_arc.clone();
    let http_reranker = reranker.as_ref().map(Arc::clone);
    let http_recent_writes = recent_writes.clone();

    // Start file watcher for real-time index updates
    let mut exclude = config.exclude.clone();
    if let Some(ref prof) = *profile_arc
        && let Some(ref archive) = prof.structure.folders.archive
    {
        let pattern = format!("{}/", archive);
        if !exclude.contains(&pattern) {
            exclude.push(pattern);
        }
    }
    // Capture retrieval settings before the watcher takes ownership of `config`.
    let max_chunks_per_file = config.max_chunks_per_file;
    let group_by = config.group_by;
    let rerank = config.rerank;
    let ranking = config.ranking;
    let lane_weights = config.lane_weights;
    let fts = config.fts;
    let embed = crate::prefix::EmbedComposition::from_config(&config);
    let chunk_opts = config.chunk_options();

    let (watcher_handle, watcher_shutdown) = crate::watcher::start_watcher(
        store_arc.clone(),
        embedder_arc.clone(),
        vault_path_arc.clone(),
        profile_arc.clone(),
        config,
        exclude,
        recent_writes.clone(),
    )?;

    if read_only {
        eprintln!("Read-only mode: write tools disabled");
    }

    let server = EngraphServer {
        store: store_arc,
        embedder: embedder_arc,
        vault_path: vault_path_arc,
        profile: profile_arc,
        tool_router: EngraphServer::tool_router(),
        reranker,
        recent_writes,
        read_only,
        max_chunks_per_file,
        group_by,
        rerank,
        ranking,
        lane_weights,
        fts,
        embed,
        chunk_opts,
    };

    // Cancellation token for coordinated shutdown of HTTP + MCP
    let cancel_token = tokio_util::sync::CancellationToken::new();

    // Spawn HTTP server as a background task (before MCP blocks on stdio)
    if let Some(ref opts) = http_opts {
        let config = Config::load()?;
        let api_state = crate::http::ApiState {
            store: http_store,
            embedder: http_embedder,
            vault_path: http_vault_path,
            profile: http_profile,
            reranker: http_reranker,
            http_config: Arc::new(config.http.clone()),
            no_auth: opts.no_auth,
            recent_writes: http_recent_writes,
            rate_limiter: Arc::new(crate::http::RateLimiter::new(config.http.rate_limit)),
            read_only,
            max_chunks_per_file,
            group_by,
            rerank,
            ranking,
            lane_weights,
            fts,
            embed,
            chunk_opts,
        };
        let router = crate::http::build_router(api_state);
        let addr = format!("{}:{}", opts.host, opts.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let cancel = cancel_token.clone();
        eprintln!("HTTP server listening on http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(cancel.cancelled_owned())
                .await
                .ok();
        });
    }

    eprintln!("engraph MCP server starting...");

    let transport = rmcp::transport::io::stdio();
    match server.serve(transport).await {
        Ok(server_handle) => {
            server_handle.waiting().await?;
        }
        Err(e) => {
            if http_opts.is_some() {
                // MCP transport failed (e.g., no stdin) but HTTP is running — stay alive
                eprintln!("MCP transport unavailable ({e:#}), HTTP server still running...");
                cancel_token.cancelled().await;
            } else {
                return Err(anyhow::anyhow!("{e}"));
            }
        }
    }

    cancel_token.cancel(); // triggers HTTP graceful shutdown

    // Shut down watcher cleanly after MCP transport exits
    let _ = watcher_shutdown.send(());
    if let Err(e) = watcher_handle.join() {
        tracing::warn!("Watcher thread panicked: {:?}", e);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for <https://github.com/devwhodevs/engraph/issues/32>,
    /// carried onto `update`'s edit list (#62). `edits` is the one array of
    /// objects an MCP tool takes, so it is the one place the schema can
    /// publish an `items` that OpenAI refuses.
    #[test]
    fn update_edits_schema_has_object_items() {
        let schema = schemars::schema_for!(crate::params::Update);
        let json = serde_json::to_value(&schema).unwrap();

        let items = &json["properties"]["edits"]["items"];
        assert!(
            items.is_object(),
            "edits.items must be an object schema, got: {items}"
        );

        // schemars may inline properties or use a $ref to $defs; both are
        // valid object schemas that OpenAI accepts.
        let has_properties = items.get("properties").is_some();
        let has_ref = items.get("$ref").is_some();
        assert!(
            has_properties || has_ref,
            "edits.items must define properties or $ref, got: {items}"
        );
    }

    /// `migrate` is one tool for three operations (#62), so the mode is the
    /// one parameter a caller must always send, and the preview it may hold
    /// from a `preview` call stays reachable.
    #[test]
    fn the_migrate_schema_requires_a_mode_and_still_accepts_a_preview() {
        let schema = schemars::schema_for!(crate::params::Migrate);
        let json = serde_json::to_value(&schema).unwrap();

        let required: Vec<&str> = json["required"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        assert_eq!(required, vec!["mode"], "got {json}");
        assert!(
            json["properties"].get("preview").is_some(),
            "the preview an apply acts on is not in the schema: {json}"
        );
    }
}
