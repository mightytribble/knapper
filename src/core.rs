//! What `knapper serve` holds once and every surface shares: the store, the
//! models and the configuration read at start, behind the two rules every
//! handler follows. Blocking work runs off the async runtime, and a read never
//! waits for a search, a write or the startup reconciliation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::SystemTime;

use anyhow::Result;
use tokio::sync::Mutex;

use crate::config::{Config, db_path};
use crate::indexer::IndexSettings;
use crate::llm::{EmbedModel, RerankModel};
use crate::profile::VaultProfile;
use crate::store::Store;

/// Paths the write tools wrote, with the mtime they wrote. The watcher skips a
/// change whose mtime matches, because the pipeline indexed it already.
pub type RecentWrites = Arc<Mutex<HashMap<PathBuf, SystemTime>>>;

/// What a core call is handed: the writer connection, the embedder, and the
/// cross-encoder when one is configured.
pub struct CoreGuards<'a> {
    pub store: &'a Store,
    pub embedder: &'a mut Box<dyn EmbedModel + Send>,
    pub reranker: Option<&'a mut dyn RerankModel>,
}

/// The shared state of one `serve` process. Every field is an `Arc` or `Copy`,
/// so a clone is cheap and every clone is the same core.
#[derive(Clone)]
pub struct Core {
    writer: Arc<Mutex<Store>>,
    reader: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
    reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>>,
    /// `config.toml` as it was when `serve` started. A later edit takes effect
    /// on restart.
    pub config: Arc<Config>,
    /// The index-time settings, read once off `config`, so every write tool
    /// and every full index this server runs shares one chunking and one
    /// vector space with the vault (#72).
    pub index_settings: IndexSettings,
    pub vault_path: Arc<PathBuf>,
    pub profile: Arc<Option<VaultProfile>>,
    pub recent_writes: RecentWrites,
    /// Watcher events sent and not yet applied. Every sender adds before it
    /// sends; the consumer subtracts after it applies. `status` reports it.
    pub pending_events: Arc<AtomicUsize>,
    pub read_only: bool,
}

impl Core {
    /// Open everything `serve` needs and refuse what this build must not
    /// serve: a store at another embedding width, or one whose fingerprints
    /// name code that did not build it (#12, #31).
    pub fn open(data_dir: &Path, config: Config, read_only: bool) -> Result<Core> {
        let db = db_path(data_dir);
        let models_dir = data_dir.join("models");

        let store = Store::open(&db)?;
        let embedder = crate::llm::load_embedder(&models_dir, &config)?;
        store.verify_embedding_dim(embedder.dim())?;

        let vault_path = PathBuf::from(store.get_meta("vault_path")?.ok_or_else(|| {
            anyhow::anyhow!("No vault path in index. Run 'knapper index <path>' first.")
        })?);

        let cleaned = crate::writer::cleanup_temp_files(&vault_path)?;
        if cleaned > 0 {
            eprintln!("Cleaned up {cleaned} incomplete write(s) from previous run");
        }
        let orphans = crate::writer::verify_index_integrity(&store, &vault_path)?;
        if orphans > 0 {
            eprintln!("Cleaned up {orphans} orphan DB entries for missing files");
        }

        let profile = Config::load_vault_profile().ok().flatten();

        let reranker: Option<Box<dyn RerankModel + Send>> = if config.intelligence_enabled() {
            match crate::llm::LlamaRerank::new(&models_dir, &config) {
                Ok(rerank) => Some(Box::new(rerank)),
                Err(e) => {
                    tracing::warn!("failed to load reranker: {e}, reranking disabled");
                    None
                }
            }
        } else {
            None
        };

        // Checked here, synchronously, so the server never answers from a
        // stale index while a background task repairs it (issue #31).
        let fingerprints = crate::fingerprint::Fingerprints::compute(
            &config,
            &EmbedModel::fingerprint(&embedder),
            reranker.as_ref().map(|r| r.fingerprint()).as_deref(),
        );
        crate::fingerprint::verify(&store, &fingerprints)?;

        let reader = Store::open_reader(&db)?;

        Ok(Self::from_parts(
            Arc::new(Mutex::new(store)),
            reader,
            Arc::new(Mutex::new(embedder)),
            reranker.map(|r| Arc::new(Mutex::new(r))),
            config,
            Arc::new(vault_path),
            Arc::new(profile),
            Arc::new(Mutex::new(HashMap::new())),
            read_only,
        ))
    }

    /// Assemble a core from parts a caller already holds.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        writer: Arc<Mutex<Store>>,
        reader: Store,
        embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
        reranker: Option<Arc<Mutex<Box<dyn RerankModel + Send>>>>,
        config: Config,
        vault_path: Arc<PathBuf>,
        profile: Arc<Option<VaultProfile>>,
        recent_writes: RecentWrites,
        read_only: bool,
    ) -> Core {
        let index_settings = IndexSettings::from_config(&config);
        Core {
            writer,
            reader: Arc::new(Mutex::new(reader)),
            embedder,
            reranker,
            config: Arc::new(config),
            index_settings,
            vault_path,
            profile,
            recent_writes,
            pending_events: Arc::new(AtomicUsize::new(0)),
            read_only,
        }
    }

    /// Run `f` against the reader connection, off the runtime thread.
    ///
    /// The reader is its own connection, so this waits for no search, write
    /// or re-index. A panic inside `f` is the `Err`; the server keeps running.
    pub async fn with_reader<R, F>(&self, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&Store) -> Result<R> + Send + 'static,
    {
        let store = self.reader.clone().lock_owned().await;
        match tokio::task::spawn_blocking(move || f(&store)).await {
            Ok(result) => result,
            Err(e) => Err(anyhow::anyhow!("reader call panicked: {e}")),
        }
    }

    /// Run `f` with the writer, the embedder and the reranker, off the runtime
    /// thread.
    ///
    /// The one place two of these locks are held at once, in one order:
    /// writer, embedder, reranker. The guards move into the blocking task and
    /// drop when `f` returns, or during unwinding if it panics, so a panic
    /// answers one call with an error and releases every lock.
    pub async fn with_core<R, F>(&self, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: for<'a> FnOnce(CoreGuards<'a>) -> Result<R> + Send + 'static,
    {
        let store = self.writer.clone().lock_owned().await;
        let mut embedder = self.embedder.clone().lock_owned().await;
        let mut reranker = match &self.reranker {
            Some(r) => Some(r.clone().lock_owned().await),
            None => None,
        };
        let joined = tokio::task::spawn_blocking(move || {
            let guards = CoreGuards {
                store: &store,
                embedder: &mut embedder,
                reranker: reranker
                    .as_mut()
                    .map(|g| g.as_mut() as &mut dyn RerankModel),
            };
            f(guards)
        })
        .await;
        match joined {
            Ok(result) => result,
            Err(e) => Err(anyhow::anyhow!("core call panicked: {e}")),
        }
    }

    /// Record a path the pipeline just wrote, with its mtime, so the watcher
    /// skips the change event the write itself raises.
    pub async fn record_write(&self, path: &Path) {
        if let Ok(meta) = std::fs::metadata(path)
            && let Ok(mtime) = meta.modified()
        {
            self.recent_writes
                .lock()
                .await
                .insert(path.to_path_buf(), mtime);
        }
    }
}

#[cfg(test)]
impl Core {
    /// A core over a temp-file store, for tests. The writer and the reader are
    /// two connections to `db`; an in-memory store cannot be opened twice.
    pub fn for_test(
        db: &Path,
        embedder: Box<dyn EmbedModel + Send>,
        config: Config,
        vault_path: PathBuf,
    ) -> Core {
        let writer = Store::open(db).expect("writer");
        let reader = Store::open_reader(db).expect("reader");
        Self::from_parts(
            Arc::new(Mutex::new(writer)),
            reader,
            Arc::new(Mutex::new(embedder)),
            None,
            config,
            Arc::new(vault_path),
            Arc::new(None),
            Arc::new(Mutex::new(HashMap::new())),
            false,
        )
    }

    /// The writer, for a test that seeds or inspects rows directly.
    pub fn writer(&self) -> Arc<Mutex<Store>> {
        self.writer.clone()
    }

    /// The captured config, for a test that changes a per-call default.
    pub fn config_mut(&mut self) -> &mut Config {
        Arc::make_mut(&mut self.config)
    }

    pub fn set_reranker(&mut self, reranker: Box<dyn RerankModel + Send>) {
        self.reranker = Some(Arc::new(Mutex::new(reranker)));
    }
}

/// Fixtures the server and watcher tests share.
#[cfg(test)]
pub mod testing {
    use super::*;
    use crate::llm::{EmbedDoc, MockLlm};

    /// The config the server tests run under: the defaults, with the
    /// calibrated sort off so the pre-calibration paths are what is asserted.
    /// The calibrated sort has its own tests.
    pub fn test_config() -> Config {
        let mut config = Config::default();
        config.calibrated.enabled = false;
        config
    }

    /// Write `notes` under a fresh vault and index them with `MockLlm` into a
    /// store beside it. Returns the temp dir, the vault root and the db path.
    pub fn indexed_vault(
        notes: &[(&str, &str)],
        config: &Config,
    ) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let tmp = tempfile::TempDir::new().unwrap();
        let vault = tmp.path().join("vault");
        let db = tmp.path().join("knapper.db");
        std::fs::create_dir_all(&vault).unwrap();
        for (rel, body) in notes {
            let path = vault.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }
        let store = Store::open(&db).unwrap();
        crate::indexer::run_index_shared(
            &vault,
            config,
            IndexSettings::from_config(config),
            &store,
            &mut MockLlm::new(256),
            false,
            None,
        )
        .unwrap();
        (tmp, vault, db)
    }

    /// An indexed vault and a core over it, embedding with `MockLlm`.
    pub fn indexed_core(notes: &[(&str, &str)], config: Config) -> (tempfile::TempDir, Core) {
        let (tmp, vault, db) = indexed_vault(notes, &config);
        let core = Core::for_test(&db, Box::new(MockLlm::new(256)), config, vault);
        (tmp, core)
    }

    /// A `MockLlm` whose first embed call parks until released, so a test can
    /// hold the embedder and prove what else still answers.
    pub struct GatedEmbed {
        inner: MockLlm,
        gate: Option<std::sync::mpsc::Receiver<()>>,
        entered: std::sync::mpsc::Sender<()>,
    }

    impl GatedEmbed {
        /// The embedder, the sender that opens its gate, and the receiver
        /// that fires once the first embed call has reached the gate.
        pub fn new(
            dim: usize,
        ) -> (
            Self,
            std::sync::mpsc::Sender<()>,
            std::sync::mpsc::Receiver<()>,
        ) {
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let (entered_tx, entered_rx) = std::sync::mpsc::channel();
            let gated = Self {
                inner: MockLlm::new(dim),
                gate: Some(release_rx),
                entered: entered_tx,
            };
            (gated, release_tx, entered_rx)
        }

        fn pass_gate(&mut self) {
            if let Some(gate) = self.gate.take() {
                let _ = self.entered.send(());
                let _ = gate.recv();
            }
        }
    }

    impl EmbedModel for GatedEmbed {
        fn embed_batch(&mut self, docs: &[EmbedDoc<'_>]) -> Result<Vec<Vec<f32>>> {
            self.pass_gate();
            self.inner.embed_batch(docs)
        }
        fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
            self.pass_gate();
            self.inner.embed_query(text)
        }
        fn token_count(&self, text: &str) -> usize {
            self.inner.token_count(text)
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn max_context(&self) -> usize {
            self.inner.max_context()
        }
        fn fingerprint(&self) -> String {
            EmbedModel::fingerprint(&self.inner)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The rule the whole design rests on: a reader call answers while a core
    /// call holds the writer and the models.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reader_call_answers_while_a_core_call_holds_the_models() {
        let (_tmp, core) =
            testing::indexed_core(&[("a.md", "# A\n\nBody.\n")], testing::test_config());
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        let held = {
            let core = core.clone();
            tokio::spawn(async move {
                core.with_core(move |_guards| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
            })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("the core call started");

        let count = tokio::time::timeout(
            Duration::from_secs(2),
            core.with_reader(|store| store.file_count()),
        )
        .await
        .expect("a read waited on the core call")
        .unwrap();
        assert_eq!(count, 1);

        release_tx.send(()).unwrap();
        held.await.unwrap().unwrap();
    }

    /// Two core calls serialize: the second runs after the first releases.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn core_calls_run_one_at_a_time() {
        let (_tmp, core) = testing::indexed_core(&[], testing::test_config());
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        let held = {
            let core = core.clone();
            tokio::spawn(async move {
                core.with_core(move |_guards| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
            })
        };
        tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .unwrap();

        let second = tokio::time::timeout(
            Duration::from_millis(300),
            core.with_core(|guards| guards.store.file_count()),
        )
        .await;
        assert!(
            second.is_err(),
            "the second core call must wait for the first"
        );

        release_tx.send(()).unwrap();
        held.await.unwrap().unwrap();
        assert_eq!(
            core.with_core(|guards| guards.store.file_count())
                .await
                .unwrap(),
            0
        );
    }

    /// A panic inside a call is that call's error. The locks drop during
    /// unwinding, so the next call runs.
    #[tokio::test]
    async fn a_panic_inside_a_core_call_is_an_error_and_the_core_survives() {
        let (_tmp, core) = testing::indexed_core(&[], testing::test_config());
        let err = core
            .with_core(|_guards| -> Result<()> { panic!("boom") })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("panicked"), "got {err:#}");
        assert_eq!(
            core.with_reader(|store| store.file_count()).await.unwrap(),
            0
        );
        assert!(
            core.with_core(|guards| guards.store.file_count())
                .await
                .is_ok(),
            "the writer lock is released after a panic"
        );
    }
}
