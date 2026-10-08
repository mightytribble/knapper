use knapper::cli::{Cli, Command, ModelsAction};
use knapper::config;
use knapper::indexer;
use knapper::profile::VaultProfile;
use knapper::search;
use knapper::store;

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, BufRead, Read as _, Write};
use std::path::{Path, PathBuf};

use config::Config;

/// Prompt user to enable intelligence, download models if yes.
fn prompt_intelligence(data_dir: &std::path::Path) -> Result<bool> {
    eprint!(
        "\nEnable AI-powered search intelligence?\n\n\
         This downloads ~650MB of additional models for:\n\
         \x20 - Result reranking (a cross-encoder scores each result for relevance)\n\n\
         Enable now? [y/N] "
    );
    io::stderr().flush()?;
    let mut answer = String::new();
    io::stdin().lock().read_line(&mut answer)?;
    let enable = answer.trim().eq_ignore_ascii_case("y");

    if enable {
        let models_dir = data_dir.join("models");
        let defaults = knapper::llm::ModelDefaults::default();
        let rerank_uri = knapper::llm::HfModelUri::parse(&defaults.rerank_uri)?;
        println!("Downloading the cross-encoder ({})...", rerank_uri.repo);
        knapper::llm::ensure_model(&rerank_uri, &models_dir)?;
        println!("Done.");
    } else {
        println!(
            "Intelligence disabled. You can enable later with: knapper configure --enable-intelligence"
        );
    }

    Ok(enable)
}

/// Check whether an index has been built by looking for the store file in data_dir.
fn index_exists(data_dir: &std::path::Path) -> bool {
    config::db_path(data_dir).exists()
}

/// Remove a file, ignoring NotFound errors.
fn remove_if_exists(path: &std::path::Path) -> Result<bool> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Remove a directory recursively, ignoring NotFound errors.
fn remove_dir_if_exists(path: &std::path::Path) -> Result<bool> {
    match std::fs::remove_dir_all(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// The index bar `knapper index` draws, over the sink the library reports to.
struct BarProgress(Option<indicatif::ProgressBar>);

impl knapper::indexer::IndexProgress for BarProgress {
    fn begin(&mut self, total: usize) {
        let pb = indicatif::ProgressBar::new(total as u64);
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "  [{bar:40.cyan/blue}] {pos}/{len} {msg} ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        self.0 = Some(pb);
    }
    fn file(&mut self, rel_path: &str) {
        if let Some(pb) = &self.0 {
            pb.set_message(rel_path.to_string());
            pb.inc(1);
        }
    }
    fn end(&mut self) {
        if let Some(pb) = self.0.take() {
            pb.finish_with_message("done");
        }
    }
}

/// The store, the vault it indexed and that vault's profile.
///
/// Every capability that reads or writes the vault opens these three the same
/// way. The two command groups used to open them once for a whole group; the
/// commands are flat now, so one function is what keeps the twelve arms below
/// from each spelling it out (#62).
fn open_vault(data_dir: &Path) -> Result<(store::Store, PathBuf, Option<VaultProfile>)> {
    if !index_exists(data_dir) {
        eprintln!("No index found. Run 'knapper index <path>' first.");
        std::process::exit(1);
    }
    let profile = config::Config::load_vault_profile().ok().flatten();
    let store = store::Store::open(&config::db_path(data_dir))?
        .with_archive_folder(knapper::profile::archive_folder(profile.as_ref()));
    let vault_path = store.get_meta("vault_path")?.ok_or_else(|| {
        anyhow::anyhow!("No vault path in index. Run 'knapper index <path>' first.")
    })?;
    Ok((store, PathBuf::from(&vault_path), profile))
}

/// The vault root for `validate`: `--vault` if given, else the configured
/// vault when one is indexed, else an error. Unlike `open_vault`, it does not
/// require an index — `validate` runs pre-`init`.
fn resolve_validate_root(vault_arg: Option<PathBuf>, data_dir: &Path) -> Result<PathBuf> {
    if let Some(v) = vault_arg {
        return Ok(v);
    }
    if index_exists(data_dir) {
        let store = store::Store::open(&config::db_path(data_dir))?;
        if let Some(vp) = store.get_meta("vault_path")? {
            return Ok(PathBuf::from(vp));
        }
    }
    anyhow::bail!("no vault root: pass --vault <dir>, or index a vault first")
}

fn render_validate_report(report: &knapper::validate::ValidateReport) {
    use knapper::validate::Severity;
    println!("Files checked: {}", report.files_checked);
    println!("Errors:        {}", report.error_count);
    println!("Warnings:      {}", report.warning_count);
    for f in &report.findings {
        let sev = match f.severity {
            Severity::Error => "error",
            Severity::Warning => "warn",
        };
        let loc = match f.line {
            Some(n) => format!("{}:{}", f.file, n),
            None => f.file.clone(),
        };
        let rule = serde_json::to_value(f.rule)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        println!("  {sev} {loc} [{rule}] {}", f.message);
    }
}

/// The embedding model, checked against the store it is about to write into.
///
/// A command that indexes what it wrote has to produce rows the rest of the
/// index agrees with: the same vector width (issue #12), and the same code
/// that built the index (issue #31). Mixing two chunkings in one store is
/// worse than either of them.
fn open_indexing_embedder(
    cfg: &Config,
    data_dir: &Path,
    store: &store::Store,
) -> Result<Box<dyn knapper::llm::EmbedModel + Send>> {
    let models_dir = data_dir.join("models");
    let embedder = knapper::llm::load_embedder(&models_dir, cfg)?;
    store.verify_embedding_dim(knapper::llm::EmbedModel::dim(&embedder))?;
    knapper::fingerprint::verify(
        store,
        &knapper::fingerprint::Fingerprints::compute(
            cfg,
            &knapper::llm::EmbedModel::fingerprint(&embedder),
            None,
        ),
    )?;
    Ok(embedder)
}

/// The content a write takes, read from stdin when the argument is omitted.
/// The CLI is the one surface that has a stdin, so this fallback is its own.
fn content_or_stdin(content: Option<String>) -> Result<String> {
    match content {
        Some(c) => Ok(c),
        None => {
            let mut buf = String::new();
            io::stdin().lock().read_to_string(&mut buf)?;
            Ok(buf)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Before clap, which prints `--help` and `--version` to stdout (#128).
    knapper::pipe::restore_default_sigpipe();

    let cli = Cli::parse();

    // A `--data-dir` flag sets the override before any `data_dir()` read, so
    // config, store, and models all resolve under it (issue #77).
    if let Some(dir) = &cli.data_dir {
        config::set_data_dir_override(dir.clone());
    }

    // Set up tracing. The library reports through `tracing`, and a warning
    // is one the user has to see: a dimension change that re-indexes the
    // vault, a file left unembedded, a config table this version does not
    // read. --verbose enables debug for knapper, info for everything else.
    let filter = if cli.verbose {
        "knapper=debug,info"
    } else {
        "warn"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_writer(std::io::stderr)
        .init();

    let mut cfg = Config::load()?;
    let data_dir = Config::data_dir()?;

    match cli.command {
        Command::Index { args, path } => {
            let (rebuild, no_gitignore) = (args.rebuild, args.no_gitignore);
            // Merge CLI vault path over config.
            cfg.merge_vault_path(path);
            if no_gitignore {
                cfg.respect_gitignore = false;
            }

            // Fall back to current directory if neither CLI nor config provides a vault path.
            let vault_path = match &cfg.vault_path {
                Some(p) => p.clone(),
                None => {
                    let cwd = std::env::current_dir()?;
                    cfg.vault_path = Some(cwd.clone());
                    cwd
                }
            };

            // Canonicalize to resolve symlinks and relative paths.
            let vault_path = vault_path.canonicalize().unwrap_or(vault_path);

            // Ensure data directory exists.
            std::fs::create_dir_all(&data_dir)?;

            // Check for vault mismatch: if store has a different vault path, warn.
            let db_path = config::db_path(&data_dir);
            if db_path.exists() && !rebuild {
                let store = store::Store::open(&db_path)?;
                if let Some(stored_vault) = store.get_meta("vault_path")? {
                    let stored = PathBuf::from(&stored_vault);
                    if stored != vault_path {
                        eprint!(
                            "Warning: Index was built for '{}'. Re-indexing will replace it. Continue? [y/N] ",
                            stored.display()
                        );
                        io::stderr().flush()?;
                        let mut answer = String::new();
                        io::stdin().lock().read_line(&mut answer)?;
                        if !answer.trim().eq_ignore_ascii_case("y") {
                            println!("Aborted.");
                            return Ok(());
                        }
                    }
                }
            }

            // First-run intelligence prompt (only if not yet configured)
            if cfg.intelligence.is_none() {
                let enable = prompt_intelligence(&data_dir)?;
                cfg.intelligence = Some(enable);
                cfg.save()?;
            }

            let result = indexer::run_index(
                &vault_path,
                &cfg,
                indexer::IndexSettings::from_config(&cfg),
                rebuild,
                &mut BarProgress(None),
            )?;

            println!(
                "Indexed {} new, {} updated, {} deleted files ({} chunks) in {:.1}s",
                result.new_files,
                result.updated_files,
                result.deleted_files,
                result.total_chunks,
                result.duration.as_secs_f64(),
            );
        }

        Command::Search(args) => {
            let scope = knapper::search::parse_scope(&args)?;

            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'knapper index <path>' first.");
                std::process::exit(1);
            }

            search::run_search(args, scope, cli.json, &data_dir, &cfg)?;
        }

        Command::Match(args) => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'knapper index <path>' first.");
                std::process::exit(1);
            }
            let profile = config::Config::load_vault_profile().ok().flatten();
            let store = store::Store::open(&config::db_path(&data_dir))?
                .with_archive_folder(knapper::profile::archive_folder(profile.as_ref()));
            let report = knapper::matching::run(&store, &args)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", knapper::matching::render_text(&report));
            }
        }

        Command::Status(_) => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'knapper index <path>' first.");
                std::process::exit(1);
            }

            search::run_status(cli.json, &data_dir)?;
        }

        Command::Read(args) => {
            let (store, vault_path, profile) = open_vault(&data_dir)?;
            let params = knapper::context::ContextParams {
                store: &store,
                vault_path: &vault_path,
                profile: profile.as_ref(),
            };
            let result = knapper::context::context_read(
                &params,
                &args.file,
                args.section.as_deref(),
                args.include,
            )?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                use knapper::context::ReadResult;
                // A note with no docid is its path alone: one space
                // separates the two, and nothing trails a path that is on
                // its own.
                let ident = |path: &str, docid: &Option<String>| match docid.as_deref() {
                    Some(d) => format!("{path} (#{d})"),
                    None => path.to_string(),
                };
                let via = |names: &[String]| {
                    if names.is_empty() {
                        String::new()
                    } else {
                        format!(" via {}", names.join(", "))
                    }
                };
                // The YAML as a note writes it. `split_frontmatter` hands
                // back the block without its fences, so they are put back
                // here: they mark where the frontmatter ends, and a caller
                // copying the output back into a note needs them. A note
                // with no frontmatter prints none, rather than empty fences.
                let yaml = |fm: &str| {
                    if fm.trim().is_empty() {
                        String::new()
                    } else {
                        format!("---\n{}\n---\n", fm.trim_end())
                    }
                };
                match result {
                    ReadResult::Content(note) => {
                        // The content is the section's body, so the heading is
                        // printed here or it is not printed at all — in the
                        // markup the outline uses, `#` depth or the bold form
                        // of a promoted line (#68, #96).
                        let heading = note.section.as_ref().map(|span| match span.level {
                            Some(level) => {
                                format!("{} {}", "#".repeat(level as usize), span.heading)
                            }
                            None => format!("**{}**", span.heading),
                        });
                        match heading {
                            Some(heading) => {
                                println!("{} > {heading}", ident(&note.path, &note.docid))
                            }
                            None => println!("{}", ident(&note.path, &note.docid)),
                        }
                        // `--include all` prints the frontmatter above the
                        // prose, which is the order a note writes them in.
                        if let Some(fm) = note.frontmatter.as_deref() {
                            print!("{}", yaml(fm));
                        }
                        println!("{}", note.content);
                    }
                    ReadResult::Frontmatter(note) => {
                        println!("{}", ident(&note.path, &note.docid));
                        print!("{}", yaml(&note.frontmatter));
                    }
                    ReadResult::Metadata(meta) => {
                        println!("{}", ident(&meta.path, &meta.docid));
                        println!("Bytes: {}", meta.byte_count);
                        println!("Outgoing links: {}", meta.outgoing_links.len());
                        for l in &meta.outgoing_links {
                            println!("  {}{}", ident(&l.path, &l.docid), via(&l.properties));
                        }
                        println!("Incoming links: {}", meta.incoming_links.len());
                        for l in &meta.incoming_links {
                            println!("  {}{}", ident(&l.path, &l.docid), via(&l.properties));
                        }
                        if !meta.properties.is_empty() {
                            println!("Properties: {}", meta.properties.len());
                            for p in &meta.properties {
                                let at = match &p.heading_path {
                                    Some(h) => format!(" @ {h}"),
                                    None => String::new(),
                                };
                                println!("  {} = {} ({}){at}", p.name, p.value, p.kind.as_str());
                            }
                        }
                        if !meta.frontmatter.is_empty() {
                            println!("Frontmatter:\n{}", meta.frontmatter);
                        }
                    }
                }
            }
        }

        Command::List(args) => {
            let (store, vault_path, profile) = open_vault(&data_dir)?;
            let params = knapper::context::ContextParams {
                store: &store,
                vault_path: &vault_path,
                profile: profile.as_ref(),
            };
            let all_terms = knapper::tags::merge_scope_alias(args.scope, args.all);
            let filter = knapper::tags::Scope::parse(&all_terms, &args.any, &args.none)?
                .with_filters(
                    args.property.as_deref(),
                    args.links_to.as_deref(),
                    args.linked_from.as_deref(),
                )?;
            let items = knapper::context::context_list(
                &params,
                &filter,
                args.created_by.as_deref(),
                args.limit,
                args.after.as_deref(),
                args.sort.into(),
                args.detailed,
            )?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&items)?);
            } else if args.detailed {
                // The path, then the note's headings as their own `#`
                // markers, with a blank line between notes (#68).
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        println!();
                    }
                    println!("{}", item.path);
                    for h in item.headings.iter().flatten() {
                        match h.level {
                            Some(level) => println!("{} {}", "#".repeat(level as usize), h.text),
                            // A promoted line has no `#` depth, so it prints
                            // in the bold form the file holds, which is one of
                            // the spellings `--section` takes (#69).
                            None => println!("**{}**", h.text),
                        }
                    }
                }
            } else {
                // One path per line and nothing else: that is what makes
                // the listing pipeable, and `wc -l` is the total. The path
                // is as the store holds it, relative to the vault root, so
                // it can be pasted into `read`, `update` or `move` (#68).
                for item in &items {
                    println!("{}", item.path);
                }
            }
        }

        Command::Tags(args) => {
            let (store, _vault_path, _profile) = open_vault(&data_dir)?;
            let prefix = args.under.as_deref().and_then(knapper::tags::parse_term);
            let rows = store.tags_under(prefix.as_ref())?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else {
                for row in &rows {
                    println!("{} ({})", row.as_written(), row.note_count);
                }
            }
        }

        Command::Properties(args) => {
            let (store, vault_path, _profile) = open_vault(&data_dir)?;
            let report = knapper::properties::run(&store, &vault_path, &args)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", knapper::properties::render_text(&report));
            }
        }

        Command::VaultMap(_) => {
            let (store, vault_path, profile) = open_vault(&data_dir)?;
            let params = knapper::context::ContextParams {
                store: &store,
                vault_path: &vault_path,
                profile: profile.as_ref(),
            };
            let map = knapper::context::vault_map(&params)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&map)?);
            } else {
                println!("Vault: {}", map.vault_path);
                println!("Type: {}, Structure: {}", map.vault_type, map.structure);
                println!(
                    "Files: {}, Chunks: {}, Edges: {}\n",
                    map.total_files, map.total_chunks, map.total_edges
                );
                if !map.top_notes.is_empty() {
                    println!("Most linked notes:");
                    for note in &map.top_notes {
                        println!("  {}: {} links in", note.path, note.links_in);
                    }
                    println!();
                }
                println!("Folders:");
                for f in &map.folders {
                    println!("  {}: {} notes", f.path, f.note_count);
                }
                println!(
                    "\nTop tags ({} of {} notes tagged, {} {}):",
                    map.tagged_notes,
                    map.total_files,
                    map.tag_axes,
                    if map.tag_axes == 1 { "axis" } else { "axes" }
                );
                for (tag, count) in &map.top_tags {
                    println!("  {}: {}", tag, count);
                }
                println!("\nRecently changed:");
                for path in &map.recent_files {
                    println!("  {}", path);
                }
            }
        }

        Command::Health(_) => {
            let (store, _vault_path, profile) = open_vault(&data_dir)?;
            let health_config = knapper::health::HealthConfig {
                daily_folder: profile
                    .as_ref()
                    .and_then(|p| p.structure.folders.daily.clone()),
                inbox_folder: profile
                    .as_ref()
                    .and_then(|p| p.structure.folders.inbox.clone()),
            };
            let report = knapper::health::generate_health_report(&store, &health_config)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("Files:          {}", report.total_files);
                println!("Orphans:        {}", report.orphans.len());
                println!("Broken links:   {}", report.broken_links.len());
                println!("Stale headings: {}", report.stale_headings.len());
                println!("Stale notes:    {}", report.stale_notes.len());
                println!("Inbox:          {} pending", report.inbox_pending.len());
                println!("Tag issues:     {}", report.tag_issues.len());
                println!("Index age:      {}s", report.index_age_seconds);
                for link in &report.broken_links {
                    println!("  broken: {} -> {}", link.source, link.target);
                }
                for link in &report.stale_headings {
                    println!(
                        "  stale:  {} -> {}#{}",
                        link.source, link.target, link.heading
                    );
                }
                for issue in &report.tag_issues {
                    println!("  tag: {} — {}", issue.file, issue.issue);
                }
            }
        }

        Command::Validate { args, vault } => {
            let root = resolve_validate_root(vault, &data_dir)?;
            let target = args.target()?;
            let limits = knapper::validate::ChunkLimits {
                min_chars: cfg.chunk_min_chars,
                target_tokens: knapper::chunker::limits::TARGET_TOKENS,
            };
            let report = knapper::validate::validate_target(&root, &target, &limits, args.strict)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                render_validate_report(&report);
            }
            if !report.ok {
                std::process::exit(1);
            }
        }

        Command::Clear { all } => {
            if all {
                // Delete entire ~/.knapper/ directory.
                if remove_dir_if_exists(&data_dir)? {
                    println!("Removed {}", data_dir.display());
                } else {
                    println!("Nothing to clear (data directory does not exist).");
                }
            } else {
                // Delete only index files.
                let db_path = config::db_path(&data_dir);
                if remove_if_exists(&db_path)? {
                    println!("Removed {}", db_path.display());
                } else {
                    println!("Nothing to clear (no index files found).");
                }
            }
        }

        Command::Init { args, path } => {
            cfg.merge_vault_path(path);
            let vault_path = match &cfg.vault_path {
                Some(p) => p.clone(),
                None => std::env::current_dir()?,
            };
            let vault_path = vault_path.canonicalize().unwrap_or(vault_path);
            let result = match args.mode {
                knapper::params::InitMode::Detect => {
                    knapper::onboarding::run_detect_json(&vault_path)?
                }
                knapper::params::InitMode::Apply => {
                    let settings = knapper::indexer::IndexSettings::from_config(&cfg);
                    knapper::onboarding::run_apply_json(
                        &vault_path,
                        &cfg,
                        settings,
                        &data_dir,
                        &mut BarProgress(None),
                    )?
                }
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Command::Configure {
            enable_intelligence,
            disable_intelligence,
            model,
            add_api_key,
            key_name,
            key_permissions,
            list_api_keys,
            revoke_api_key,
            setup_chatgpt,
        } => {
            let mut cfg = Config::load()?;

            if enable_intelligence {
                cfg.intelligence = Some(true);
                println!("Intelligence enabled. Models will be downloaded on first search.");
                let models_dir = data_dir.join("models");
                let defaults = knapper::llm::ModelDefaults::default();
                let rerank_uri = knapper::llm::HfModelUri::parse(
                    cfg.models.rerank.as_deref().unwrap_or(&defaults.rerank_uri),
                )?;
                println!("Downloading the cross-encoder ({})...", rerank_uri.repo);
                knapper::llm::ensure_model(&rerank_uri, &models_dir)?;
                println!("Done.");
            } else if disable_intelligence {
                cfg.intelligence = Some(false);
                println!("Intelligence disabled. Models remain cached.");
            }

            if let Some(parts) = model
                && parts.len() == 2
            {
                let model_type = &parts[0];
                let uri = &parts[1];
                knapper::llm::HfModelUri::parse(uri)?;
                match model_type.as_str() {
                    "embed" => {
                        cfg.models.embed = Some(uri.clone());
                        println!("Embedding model set to: {uri}");
                        println!("Warning: Next 'knapper index' will re-embed your entire vault.");
                    }
                    "rerank" => {
                        cfg.models.rerank = Some(uri.clone());
                        println!("Reranker model set to: {uri}");
                    }
                    other => {
                        anyhow::bail!("Unknown model type: {other}. Use: embed or rerank.");
                    }
                }
            }

            if add_api_key {
                let name = key_name.unwrap_or_else(|| "default".into());
                let perms = key_permissions.unwrap_or_else(|| "read".into());
                if perms != "read" && perms != "write" {
                    anyhow::bail!("Permissions must be 'read' or 'write', got: {perms}");
                }
                let key = knapper::http::generate_api_key();
                cfg.http.api_keys.push(knapper::config::ApiKeyConfig {
                    key: key.clone(),
                    name: name.clone(),
                    permissions: perms.clone(),
                });
                cfg.save()?;
                println!("API key created:");
                println!("  Name: {name}");
                println!("  Permissions: {perms}");
                println!("  Key: {key}");
                println!("\nSave this key — it won't be shown again.");
            }

            if list_api_keys {
                if cfg.http.api_keys.is_empty() {
                    println!("No API keys configured.");
                } else {
                    println!("API keys:");
                    for k in &cfg.http.api_keys {
                        println!("  {} ({})", k.name, k.permissions);
                    }
                }
            }

            if let Some(ref name) = revoke_api_key {
                let before = cfg.http.api_keys.len();
                cfg.http.api_keys.retain(|k| k.name != *name);
                if cfg.http.api_keys.len() < before {
                    cfg.save()?;
                    println!("Revoked API key: {name}");
                } else {
                    println!("No API key found with name: {name}");
                }
            }

            if setup_chatgpt {
                println!("Setting up knapper for ChatGPT Actions...\n");

                if !cfg.http.enabled {
                    cfg.http.enabled = true;
                    println!("\u{2713} HTTP server enabled");
                } else {
                    println!("\u{2713} HTTP server already enabled");
                }

                if cfg.http.api_keys.is_empty() {
                    let key = knapper::http::generate_api_key();
                    cfg.http.api_keys.push(knapper::config::ApiKeyConfig {
                        key: key.clone(),
                        name: "chatgpt".into(),
                        permissions: "read".into(),
                    });
                    println!("\u{2713} API key created: {key}");
                    println!("  Save this \u{2014} you'll need it for ChatGPT Action setup.");
                } else {
                    println!("\u{2713} API key already configured");
                }

                for origin in ["https://chat.openai.com", "https://chatgpt.com"] {
                    if cfg.http.cors_origins.iter().any(|o| o == origin) {
                        println!("\u{2713} CORS origin already configured: {origin}");
                    } else {
                        cfg.http.cors_origins.push(origin.to_string());
                        println!("\u{2713} CORS origin added: {origin}");
                    }
                }

                cfg.save()?;
                println!("\nSetup complete. Next steps:");
                println!("1. knapper serve --http");
                println!(
                    "2. Expose via tunnel: cloudflared tunnel --url http://localhost:{}",
                    cfg.http.port
                );
                println!(
                    "3. Add public_url = \"<your-tunnel-url>\" inside the existing [http] table of config.toml, above any [[http.api_keys]], and restart serve"
                );
                println!(
                    "4. ChatGPT \u{2192} Create GPT \u{2192} Add Action \u{2192} Import from: <your-tunnel-url>/openapi.json"
                );
                println!("5. Auth: API Key, Bearer, paste your key");
            }

            cfg.save()?;
            println!(
                "Configuration saved to {}",
                data_dir.join("config.toml").display()
            );
        }

        Command::Serve {
            http,
            port,
            host,
            no_auth,
            read_only,
        } => {
            if !index_exists(&data_dir) {
                eprintln!("No index found. Run 'knapper index <path>' first.");
                std::process::exit(1);
            }
            let http_opts = http.then(|| knapper::serve::HttpServeOpts {
                port: port.unwrap_or(cfg.http.port),
                host: host.unwrap_or_else(|| cfg.http.host.clone()),
                no_auth,
            });
            knapper::serve::run_serve(&data_dir, cfg, http_opts, read_only).await?;
        }

        Command::Create(args) => {
            let (store, vault_path, profile) = open_vault(&data_dir)?;
            let mut embedder = open_indexing_embedder(&cfg, &data_dir, &store)?;
            // The CLI is the one surface with a stdin, so an omitted content
            // is read from it here and is an error on the other two.
            let content = content_or_stdin(args.content)?;
            let input = knapper::writer::CreateNoteInput {
                content,
                filename: args.filename,
                tags: args.tags,
                folder: args.folder,
                created_by: "cli".into(),
                auto_link: args.auto_link,
            };
            let result = knapper::writer::create_note(
                input,
                &store,
                &mut embedder,
                knapper::prefix::EmbedComposition::from_config(&cfg),
                cfg.chunk_options(),
                &vault_path,
                profile.as_ref(),
            )?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("Created: {} (#{})", result.path, result.docid);
                if !result.links_added.is_empty() {
                    println!("Links: {}", result.links_added.join(", "));
                }
                if !result.links_suggested.is_empty() {
                    println!("Suggested: {}", result.links_suggested.join(", "));
                }
            }
        }

        Command::Update {
            file,
            section,
            property,
            heading,
            after,
            before,
            mode,
            content,
            edits,
        } => {
            let (store, vault_path, _profile) = open_vault(&data_dir)?;
            // The model loads before the write, not after it: a store this
            // build must not index into is a refusal, and a refusal has to
            // come while the file is still untouched (issues #12 and #31).
            let mut embedder = open_indexing_embedder(&cfg, &data_dir, &store)?;
            // The whole list is read before anything is written, so a request
            // that names an impossible target writes nothing (#62).
            let request = knapper::params::Update::from_cli(
                file,
                knapper::params::CliEdit {
                    section,
                    property,
                    heading,
                    after,
                    before,
                    mode,
                    content,
                },
                edits,
                || content_or_stdin(None),
            )?;
            let edits = request.to_writer_edits()?;
            let input = knapper::writer::UpdateInput {
                file: request.file,
                edits,
            };
            let result = knapper::writer::update_note(&store, &vault_path, &input)?;
            // `update_note` stores the new content hash and writes no chunks,
            // so nothing else will re-derive them: `diff_vault` sees a hash
            // that already matches disk. Re-index here or the note stays
            // searchable only as the text it held before the edit (#62). Both
            // servers do the same after their own `update`.
            //
            // A failure here happens after the write, so the message says what
            // did happen rather than reading as "nothing did".
            knapper::indexer::reindex_written_file(
                &result.path,
                &store,
                &mut embedder,
                &vault_path,
                &cfg,
            )
            .with_context(|| {
                format!(
                    "the file was written; its index rows were not updated for {}",
                    result.path
                )
            })?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("Updated: {}", result.path);
                for link in &result.stale_links {
                    println!(
                        "  stale link: {} still names \"{}\"",
                        link.source, link.heading
                    );
                }
            }
        }

        Command::Move(args) => {
            let (store, vault_path, _profile) = open_vault(&data_dir)?;
            // A move changes a path and no content, so it re-indexes nothing
            // and needs no model.
            let result =
                knapper::writer::move_note(&args.file, &args.new_folder, &store, &vault_path)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("Moved: {} → {}", args.file, result.path);
            }
        }

        Command::Archive(args) => {
            let (store, vault_path, profile) = open_vault(&data_dir)?;
            // Archiving and restoring are one operation and its reverse, so
            // they are one capability with a flag rather than two names (#62).
            // Only the restore indexes anything, so only it loads a model.
            let result = if args.undo {
                let mut embedder = open_indexing_embedder(&cfg, &data_dir, &store)?;
                knapper::writer::unarchive_note(
                    &args.file,
                    &store,
                    &mut embedder,
                    knapper::prefix::EmbedComposition::from_config(&cfg),
                    cfg.chunk_options(),
                    &vault_path,
                    profile.as_ref(),
                )?
            } else {
                knapper::writer::archive_note(&args.file, &store, &vault_path, profile.as_ref())?
            };
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else if args.undo {
                println!("Restored: {} → {}", args.file, result.path);
            } else {
                println!("Archived: {} → {}", args.file, result.path);
            }
        }

        Command::Delete(args) => {
            let (store, vault_path, _profile) = open_vault(&data_dir)?;
            // Deleting and restoring are one operation and its reverse, as
            // `archive` is (#151). Only the restore indexes anything, so only
            // it loads a model.
            if args.restores()? {
                let mut embedder = open_indexing_embedder(&cfg, &data_dir, &store)?;
                let result = knapper::writer::undelete_note(
                    &args.file,
                    &store,
                    &mut embedder,
                    knapper::prefix::EmbedComposition::from_config(&cfg),
                    cfg.chunk_options(),
                    &vault_path,
                )?;
                if cli.json {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                } else {
                    println!("Restored: {} → {}", args.file, result.path);
                }
            } else {
                let delete_mode = knapper::writer::DeleteMode::from(args.mode);
                knapper::writer::delete_note(&store, &vault_path, &args.file, delete_mode)?;
                if cli.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "deleted": args.file,
                            "mode": args.mode
                        }))?
                    );
                } else {
                    println!("Deleted: {} ({})", args.file, args.mode);
                }
            }
        }

        Command::ReindexFile(args) => {
            let (store, vault_path, _profile) = open_vault(&data_dir)?;
            let mut embedder = open_indexing_embedder(&cfg, &data_dir, &store)?;
            let result = knapper::indexer::reindex_written_file(
                &args.file,
                &store,
                &mut embedder,
                &vault_path,
                &cfg,
            )?;
            let output = serde_json::json!({
                "file": args.file,
                "chunks": result.total_chunks,
                "docid": result.docid,
            });
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!(
                    "Re-indexed: {} ({} chunks, #{})",
                    args.file, result.total_chunks, result.docid
                );
            }
        }

        Command::Models { action } => {
            let defaults = knapper::llm::ModelDefaults::default();
            // Dimensionality belongs to the model, not to a table here, and
            // reading it means loading the GGUF (issue #12). Report what the
            // index was actually built at instead — the number that matters
            // operationally — and say so plainly.
            let indexed_dim = store::Store::open(&config::db_path(&data_dir))
                .ok()
                .and_then(|s| s.vec_table_dim().ok().flatten());
            // What `models.embed` resolves to right now, so a row can be
            // marked as the one in use rather than merely offered.
            let selected = Config::load()
                .ok()
                .and_then(|c| c.models.embed.clone())
                .unwrap_or_else(|| defaults.embed_uri.clone());
            match action {
                ModelsAction::List => {
                    for e in knapper::llm::known_embedders() {
                        let mark = if e.uri == selected { " (in use)" } else { "" };
                        println!("{}{mark}", e.uri);
                        println!(
                            "  {} dim, {} token context, {} download",
                            e.dim, e.context, e.download
                        );
                        println!("  {}", e.note);
                        println!();
                    }
                    println!(
                        "Set one with: knapper configure --model <NAME>\n\
                         Any other hf:<repo>/<file>.gguf works too; the width and the\n\
                         prompt format come from the model itself."
                    );
                }
                ModelsAction::Info { name } => {
                    match knapper::llm::known_embedders()
                        .iter()
                        .find(|e| e.uri == name)
                    {
                        Some(e) => {
                            println!("Name:        {}", e.uri);
                            println!("Format:      GGUF");
                            println!("Dimensions:  {} (native)", e.dim);
                            // The store's width is the operational number: it
                            // is what the index was actually built at, and it
                            // disagrees with the native one whenever the model
                            // changed and no re-index has run yet (issue #12).
                            match indexed_dim {
                                Some(d) => println!("Indexed at:  {d}"),
                                None => println!("Indexed at:  nothing indexed yet"),
                            }
                            println!("Context:     {} tokens", e.context);
                            println!("Download:    {}", e.download);
                            println!(
                                "In use:      {}",
                                if e.uri == selected { "yes" } else { "no" }
                            );
                            println!("Description: {}", e.note);
                        }
                        None => {
                            eprintln!("Unknown model: {name}");
                            eprintln!("Run 'knapper models list' to see available models.");
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
