use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use notify::{PollWatcher, RecursiveMode, Watcher};
use notify_debouncer_full::{
    DebounceEventResult, DebouncedEvent, Debouncer, FileIdCache, RecommendedCache, new_debouncer,
    new_debouncer_opt,
};
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::config::{Config, WatcherBackend};
use crate::core::{Core, CoreGuards, RecentWrites};
use crate::exclude::ExcludeMatcher;
use crate::indexer;
use crate::placement;
use crate::store::Store;

/// The concrete watcher backend after config, env, and filesystem are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedWatcher {
    Native,
    Poll,
}

/// The backend the operator asked for: the `KNAPPER_WATCHER_BACKEND` override
/// if it named one, else the config value.
fn requested_backend(
    config_backend: WatcherBackend,
    env: Option<WatcherBackend>,
) -> WatcherBackend {
    env.unwrap_or(config_backend)
}

/// Resolve the concrete backend. `fs_needs_poll` is consulted only for `Auto`;
/// `None` there — detection did not run or could not tell — resolves to native,
/// the safe default on a local disk.
fn resolve_watcher(requested: WatcherBackend, fs_needs_poll: Option<bool>) -> ResolvedWatcher {
    match requested {
        WatcherBackend::Native => ResolvedWatcher::Native,
        WatcherBackend::Poll => ResolvedWatcher::Poll,
        WatcherBackend::Auto => match fs_needs_poll {
            Some(true) => ResolvedWatcher::Poll,
            _ => ResolvedWatcher::Native,
        },
    }
}

/// Linux `statfs` `f_type` magics for filesystems whose change notifications
/// inotify cannot deliver, so a warm watcher on them must poll (issue #83).
/// Values from `linux/magic.h`.
fn fs_magic_needs_poll(magic: i64) -> bool {
    const OVERLAYFS_SUPER_MAGIC: i64 = 0x794c_7630;
    const FUSE_SUPER_MAGIC: i64 = 0x6573_5546;
    const V9FS_MAGIC: i64 = 0x0102_1997; // 9p — Docker Desktop / WSL2 mounts
    const NFS_SUPER_MAGIC: i64 = 0x6969;
    const SMB_SUPER_MAGIC: i64 = 0x517b;
    const CIFS_MAGIC_NUMBER: i64 = 0xff53_4d42; // cifs / smb2 / smb3
    matches!(
        magic,
        OVERLAYFS_SUPER_MAGIC
            | FUSE_SUPER_MAGIC
            | V9FS_MAGIC
            | NFS_SUPER_MAGIC
            | SMB_SUPER_MAGIC
            | CIFS_MAGIC_NUMBER
    )
}

/// Whether the filesystem under `path` needs the poll backend. `None` when
/// detection did not run (non-Linux) or `statfs` failed — [`resolve_watcher`]
/// reads that as native, the safe default on a local disk.
#[cfg(target_os = "linux")]
fn fs_needs_poll(path: &Path) -> Option<bool> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut buf = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `statfs` writes a full `struct statfs` into `buf` when it returns
    // 0; `f_type` is read only on that path, after `assume_init`.
    let rc = unsafe { libc::statfs(c_path.as_ptr(), buf.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let buf = unsafe { buf.assume_init() };
    // `f_type` is `__fsword_t`, whose width is platform-dependent — i64 here,
    // i32 on 32-bit targets — so the widening cast is needed for portability
    // even where this target makes it a no-op.
    #[allow(clippy::unnecessary_cast)]
    let magic = buf.f_type as i64;
    Some(fs_magic_needs_poll(magic))
}

#[cfg(not(target_os = "linux"))]
fn fs_needs_poll(_path: &Path) -> Option<bool> {
    None
}

/// Start the file watcher and consumer. Returns a thread handle for the
/// producer and a shutdown sender.
///
/// Three tasks: the producer thread watches the vault; one task diffs the
/// vault against the store and queues what changed while the server was
/// down, per file, so the handshake answers at once and a read answers
/// throughout; the consumer applies events one file at a time.
pub fn start_watcher(
    core: Core,
    exclude: Vec<String>,
) -> anyhow::Result<(std::thread::JoinHandle<()>, oneshot::Sender<()>)> {
    let (tx, rx) = mpsc::channel::<Vec<WatchEvent>>(64);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Compile the exclude globs once, here, so a bad pattern fails server
    // startup rather than every event batch.
    let matcher = ExcludeMatcher::new(&exclude)?;

    let producer_handle = start_producer(
        core.vault_path.as_ref().clone(),
        matcher,
        tx.clone(),
        shutdown_rx,
        core.config.watcher.backend,
        Duration::from_secs(core.config.watcher.poll_interval_secs),
        core.pending_events.clone(),
    );

    {
        let core = core.clone();
        let exclude = exclude.clone();
        tokio::spawn(async move {
            // A failed diff sent nothing, so the counter holds nothing of it.
            if let Err(e) = enqueue_diff(&core, &exclude, &tx).await {
                tracing::warn!("Startup reconciliation failed: {:#}", e);
            }
        });
    }

    tokio::spawn(async move {
        run_consumer(rx, core, exclude).await;
    });

    Ok((producer_handle, shutdown_tx))
}

/// What the vault holds that the store does not, as the events the consumer
/// applies: a `Deleted` for each record the disk no longer holds, a `Changed`
/// for each new or changed path. Reads the file records through the reader,
/// then walks and hashes the vault holding no lock, so a read is not delayed
/// by the walk.
pub async fn diff_events(core: &Core, exclude: &[String]) -> anyhow::Result<Vec<WatchEvent>> {
    let stored = core.with_reader(|store| store.get_all_files()).await?;
    let vault = core.vault_path.clone();
    let exclude = exclude.to_vec();
    let respect_gitignore = core.config.respect_gitignore;
    crate::core::blocking(move || {
        let files = indexer::walk_vault(&vault, &exclude, respect_gitignore)?;
        let (new_files, changed_files, deleted) = indexer::diff_files(&files, &vault, stored)?;
        let mut events = Vec::with_capacity(new_files.len() + changed_files.len() + deleted.len());
        events.extend(
            deleted
                .into_iter()
                .map(|record| WatchEvent::Deleted(vault.join(&record.path))),
        );
        events.extend(
            new_files
                .into_iter()
                .chain(changed_files)
                .map(WatchEvent::Changed),
        );
        Ok(events)
    })
    .await
}

/// Queue the vault diff for the consumer, in batches of at most 64 events.
/// The whole diff is counted into `pending_events` before the first send, so
/// the counter does not reach zero between batches. A failed send settles
/// every event not yet sent.
pub async fn enqueue_diff(
    core: &Core,
    exclude: &[String],
    tx: &mpsc::Sender<Vec<WatchEvent>>,
) -> anyhow::Result<()> {
    let events = diff_events(core, exclude).await?;
    tracing::info!(events = events.len(), "startup reconciliation queued");
    core.pending_events
        .fetch_add(events.len(), Ordering::Relaxed);
    let mut sent = 0;
    for batch in events.chunks(64) {
        if tx.send(batch.to_vec()).await.is_err() {
            settle(&core.pending_events, events.len() - sent);
            break;
        }
        sent += batch.len();
    }
    Ok(())
}

/// Take `k` applied or dropped events off the counter. Saturates at zero, so
/// a sender that did not count cannot wrap it.
fn settle(pending: &AtomicUsize, k: usize) {
    let _ = pending.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(k))
    });
}

/// Events sent from the watcher producer to the consumer.
#[derive(Debug, Clone)]
pub enum WatchEvent {
    /// File content was modified or a new file was created.
    Changed(PathBuf),
    /// File was deleted.
    Deleted(PathBuf),
    /// File was moved/renamed (detected via content hash or inode tracking).
    Moved { from: PathBuf, to: PathBuf },
    /// macOS FSEvents buffer overflow — full rescan needed.
    FullRescan,
}

/// Start the producer thread. Returns thread handle.
/// The producer watches the vault, debounces events, and sends batches to tx.
pub fn start_producer(
    vault_path: PathBuf,
    exclude: ExcludeMatcher,
    tx: mpsc::Sender<Vec<WatchEvent>>,
    shutdown_rx: oneshot::Receiver<()>,
    backend: WatcherBackend,
    poll_interval: Duration,
    pending: Arc<AtomicUsize>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        // Create std channel for debouncer events
        let (debouncer_tx, debouncer_rx) = std::sync::mpsc::channel();

        // Resolve which backend runs: the env override wins over config, and
        // `Auto` probes the filesystem under the vault (issue #83).
        let requested = requested_backend(
            backend,
            std::env::var("KNAPPER_WATCHER_BACKEND")
                .ok()
                .and_then(|v| WatcherBackend::from_env_value(&v)),
        );
        let resolved = resolve_watcher(
            requested,
            if requested == WatcherBackend::Auto {
                fs_needs_poll(&vault_path)
            } else {
                None
            },
        );
        tracing::info!(backend = ?resolved, "warm-sync watcher backend selected");

        match resolved {
            ResolvedWatcher::Native => {
                match new_debouncer(Duration::from_secs(2), None, debouncer_tx) {
                    Ok(d) => drive_producer(
                        d,
                        vault_path,
                        exclude,
                        tx,
                        shutdown_rx,
                        debouncer_rx,
                        pending,
                    ),
                    Err(e) => tracing::error!("Failed to create file watcher: {}", e),
                }
            }
            ResolvedWatcher::Poll => {
                let cfg = notify::Config::default().with_poll_interval(poll_interval);
                match new_debouncer_opt::<_, PollWatcher, RecommendedCache>(
                    Duration::from_secs(2),
                    None,
                    debouncer_tx,
                    RecommendedCache::new(),
                    cfg,
                ) {
                    Ok(d) => drive_producer(
                        d,
                        vault_path,
                        exclude,
                        tx,
                        shutdown_rx,
                        debouncer_rx,
                        pending,
                    ),
                    Err(e) => tracing::error!("Failed to create poll watcher: {}", e),
                }
            }
        }
    })
}

/// Watch the vault and forward debounced batches until shutdown. Generic over
/// the watcher backend so the native and poll paths share one loop; the only
/// thing that differs is the `Debouncer` handed in, which stays alive — and so
/// keeps watching — for as long as this runs.
fn drive_producer<T, C>(
    mut debouncer: Debouncer<T, C>,
    vault_path: PathBuf,
    exclude: ExcludeMatcher,
    tx: mpsc::Sender<Vec<WatchEvent>>,
    mut shutdown_rx: oneshot::Receiver<()>,
    debouncer_rx: std::sync::mpsc::Receiver<DebounceEventResult>,
    pending: Arc<AtomicUsize>,
) where
    T: Watcher,
    C: FileIdCache + Send + 'static,
{
    if let Err(e) = debouncer.watch(&vault_path, RecursiveMode::Recursive) {
        tracing::error!("Failed to watch {:?}: {}", vault_path, e);
        return;
    }

    tracing::info!("File watcher started for {:?}", vault_path);

    loop {
        // Check shutdown (non-blocking)
        if shutdown_rx.try_recv().is_ok() {
            tracing::info!("Watcher shutting down");
            break;
        }

        match debouncer_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(Ok(events)) => {
                let watch_events = process_debounced_events(&events, &vault_path, &exclude);
                if watch_events.is_empty() {
                    continue;
                }
                let sent = watch_events.len();
                pending.fetch_add(sent, Ordering::Relaxed);
                if tx.blocking_send(watch_events).is_err() {
                    settle(&pending, sent);
                    tracing::info!("Consumer gone, watcher exiting");
                    break;
                }
            }
            Ok(Err(errors)) => {
                for e in errors {
                    tracing::warn!("Watcher error: {:?}", e);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Convert `DebouncedEvent`s to `WatchEvent`s, filtering to `.md` files.
fn process_debounced_events(
    events: &[DebouncedEvent],
    vault_path: &Path,
    exclude: &ExcludeMatcher,
) -> Vec<WatchEvent> {
    let mut result = Vec::new();

    for debounced in events {
        let event = &debounced.event; // Access the inner notify::Event

        let paths: Vec<&PathBuf> = event
            .paths
            .iter()
            .filter(|p| p.extension().map(|e| e == "md").unwrap_or(false))
            .filter(|p| !exclude.matches_under(p, vault_path))
            .collect();

        if paths.is_empty() {
            continue;
        }

        use notify::EventKind;
        match &event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in paths {
                    result.push(WatchEvent::Changed(path.clone()));
                }
            }
            EventKind::Remove(_) => {
                for path in paths {
                    // A removal whose path still holds a file is not a
                    // deletion. A rename that replaces a file — every atomic
                    // save, `writer::atomic_write` included — makes the
                    // debouncer synthesize one for the target path ahead of the
                    // events that describe the new file. Passing it on drops
                    // the note from the index, and the change behind it is the
                    // writer's own, which `is_recent_write` suppresses: the
                    // note is left on disk and out of search until something
                    // re-indexes it (#93). `Deleted` means the file is gone,
                    // and disk is what says so.
                    if path.exists() {
                        continue;
                    }
                    result.push(WatchEvent::Deleted(path.clone()));
                }
            }
            EventKind::Other => {
                result.push(WatchEvent::FullRescan);
            }
            _ => {}
        }
    }

    result
}

/// Detect file moves by matching `Deleted` + `Changed` pairs via content hash.
///
/// When a file is moved, the OS reports a delete at the old path and a create at
/// the new path. We match these by comparing the stored content hash (for the
/// deleted file) against the on-disk content hash (for the new file). Matched
/// pairs are replaced with `Moved { from, to }` events.
fn detect_moves(events: &mut Vec<WatchEvent>, store: &Store, vault_path: &Path) {
    // Collect deletion paths and their stored content hashes.
    let mut deletion_hashes: HashMap<String, PathBuf> = HashMap::new();
    for event in events.iter() {
        if let WatchEvent::Deleted(path) = event {
            let rel = path
                .strip_prefix(vault_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            if let Ok(Some(record)) = store.get_file(&rel) {
                deletion_hashes.insert(record.content_hash.clone(), path.clone());
            }
        }
    }

    if deletion_hashes.is_empty() {
        return;
    }

    // Collect creation paths (Changed events for files NOT already in store = new files).
    let mut creation_hashes: HashMap<String, PathBuf> = HashMap::new();
    for event in events.iter() {
        if let WatchEvent::Changed(path) = event {
            let rel = path
                .strip_prefix(vault_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            // Only consider files not already in the store (truly new files).
            if store.get_file(&rel).ok().flatten().is_none()
                && let Ok(hash) = indexer::compute_file_hash(path)
            {
                creation_hashes.insert(hash, path.clone());
            }
        }
    }

    // Match deletions to creations by content hash.
    let mut moves: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (hash, del_path) in &deletion_hashes {
        if let Some(create_path) = creation_hashes.get(hash) {
            moves.push((del_path.clone(), create_path.clone()));
        }
    }

    if moves.is_empty() {
        return;
    }

    // Replace matched pairs with Moved events.
    let move_from_set: std::collections::HashSet<PathBuf> =
        moves.iter().map(|(from, _)| from.clone()).collect();
    let move_to_set: std::collections::HashSet<PathBuf> =
        moves.iter().map(|(_, to)| to.clone()).collect();

    events.retain(|event| match event {
        WatchEvent::Deleted(p) => !move_from_set.contains(p),
        WatchEvent::Changed(p) => !move_to_set.contains(p),
        _ => true,
    });

    for (from, to) in moves {
        tracing::info!(from = %from.display(), to = %to.display(), "detected file move");
        events.push(WatchEvent::Moved { from, to });
    }
}

/// Check if a file was recently written by an MCP tool (so the watcher should skip it).
/// Returns true if the file's current mtime matches the recorded write mtime.
async fn is_recent_write(recent_writes: &RecentWrites, path: &Path) -> bool {
    let mut map = recent_writes.lock().await;
    if let Some(recorded_mtime) = map.get(path) {
        if let Ok(meta) = std::fs::metadata(path)
            && let Ok(current_mtime) = meta.modified()
            && current_mtime == *recorded_mtime
        {
            // Match — this file was written by us; remove entry and skip
            map.remove(path);
            return true;
        }
        // mtime doesn't match (file was modified again externally) — remove stale entry
        map.remove(path);
    }
    false
}

/// How many sent events one applied event accounts for. A `Moved` replaced a
/// `Deleted` and a `Changed` in `detect_moves`.
fn credits(event: &WatchEvent) -> usize {
    match event {
        WatchEvent::Moved { .. } => 2,
        _ => 1,
    }
}

fn rel_of(vault_path: &Path, path: &Path) -> String {
    path.strip_prefix(vault_path)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn folder_of(rel: &str) -> String {
    Path::new(rel)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
    let dim = vectors[0].len();
    let mut mean = vec![0.0f32; dim];
    for v in vectors {
        for (i, val) in v.iter().enumerate() {
            mean[i] += val;
        }
    }
    let n = vectors.len() as f32;
    for val in &mut mean {
        *val /= n;
    }
    mean
}

fn unix_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

/// Index one changed or new file and, for a new one, move its folder's
/// centroid toward it. Returns the file's id for the edge pass.
fn index_changed_file(
    g: CoreGuards<'_>,
    rel: &str,
    content: &str,
    hash: &str,
    vault_path: &Path,
    config: &Config,
) -> anyhow::Result<i64> {
    let is_new_file = g.store.get_file(rel).ok().flatten().is_none();
    let result = indexer::index_file(rel, content, hash, g.store, g.embedder, vault_path, config)?;
    if is_new_file
        && let Ok(vectors) = g.store.get_chunk_vectors_for_file(result.file_id)
        && !vectors.is_empty()
        && let Err(e) =
            g.store
                .adjust_folder_centroid(&folder_of(rel), &mean_vector(&vectors), true)
    {
        tracing::warn!(path = %rel, error = %e, "failed to adjust centroid for new file");
    }
    Ok(result.file_id)
}

/// Remove a deleted file and move its folder's centroid away from it.
fn remove_deleted_file(g: CoreGuards<'_>, rel: &str, vault_path: &Path) -> anyhow::Result<()> {
    let centroid = g.store.get_file(rel).ok().flatten().and_then(|file| {
        let vectors = g.store.get_chunk_vectors_for_file(file.id).ok()?;
        if vectors.is_empty() {
            return None;
        }
        Some((mean_vector(&vectors), folder_of(rel)))
    });
    indexer::remove_file(rel, g.store, vault_path)?;
    if let Some((mean, folder)) = centroid
        && let Err(e) = g.store.adjust_folder_centroid(&folder, &mean, false)
    {
        tracing::warn!(path = %rel, error = %e, "failed to adjust centroid for deleted file");
    }
    Ok(())
}

/// Rename a moved file in the index and learn from the move. Returns the
/// file's id for the edge pass, and the note's text with the placement
/// frontmatter stripped when that write is owed; the write happens outside
/// the lock.
fn rename_moved_file(
    g: CoreGuards<'_>,
    old_rel: &str,
    new_rel: &str,
    to: &Path,
    vault_path: &Path,
) -> anyhow::Result<(Option<i64>, Option<String>)> {
    indexer::rename_file(old_rel, new_rel, g.store, vault_path)?;
    let file_id = g.store.get_file(new_rel)?.map(|record| record.id);
    let Ok(content) = std::fs::read_to_string(to) else {
        return Ok((file_id, None));
    };
    let actual_folder = folder_of(new_rel);
    let stripped = match placement::detect_correction_from_frontmatter(&content, &actual_folder) {
        Some(correction) => {
            tracing::info!(
                file = %new_rel,
                suggested = %correction.suggested_folder,
                actual = %correction.actual_folder,
                "placement correction detected"
            );
            if let Some(file) = g.store.get_file(new_rel)?
                && let Ok(vectors) = g.store.get_chunk_vectors_for_file(file.id)
                && !vectors.is_empty()
            {
                let mean = mean_vector(&vectors);
                if let Err(e) =
                    g.store
                        .adjust_folder_centroid(&correction.actual_folder, &mean, true)
                {
                    tracing::warn!(error = %e, "failed to adjust actual folder centroid");
                }
                if let Err(e) =
                    g.store
                        .adjust_folder_centroid(&correction.suggested_folder, &mean, false)
                {
                    tracing::warn!(error = %e, "failed to adjust suggested folder centroid");
                }
            }
            if let Err(e) = g.store.insert_placement_correction(
                new_rel,
                &correction.suggested_folder,
                &correction.actual_folder,
            ) {
                tracing::warn!(error = %e, "failed to log placement correction");
            }
            let stripped = placement::strip_placement_frontmatter(&content);
            (stripped != content).then_some(stripped)
        }
        None if content.contains("suggested_folder:") => {
            let stripped = placement::strip_placement_frontmatter(&content);
            (stripped != content).then_some(stripped)
        }
        None => None,
    };
    Ok((file_id, stripped))
}

/// Apply one event. Returns the file id the edge pass should revisit.
async fn apply_event(core: &Core, event: WatchEvent) -> anyhow::Result<Option<i64>> {
    match event {
        WatchEvent::Changed(path) => {
            if is_recent_write(&core.recent_writes, &path).await {
                tracing::debug!(path = %path.display(), "skipping re-index for a file the pipeline wrote");
                return Ok(None);
            }
            let rel = rel_of(&core.vault_path, &path);
            let content = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("failed to read changed file: {e}"))?;
            let hash = indexer::compute_file_hash(&path)?;
            let vault = core.vault_path.clone();
            let config = core.config.clone();
            let rel_for_call = rel.clone();
            let file_id = core
                .with_core(move |g| {
                    index_changed_file(g, &rel_for_call, &content, &hash, &vault, &config)
                })
                .await?;
            tracing::info!(path = %rel, file_id, "indexed changed file");
            Ok(Some(file_id))
        }
        WatchEvent::Deleted(path) => {
            // A path on disk is not deleted. The startup diff runs beside the
            // live watcher, so its `Deleted` can arrive after a `Changed` that
            // restored the note (#93).
            if path.exists() {
                tracing::debug!(path = %path.display(), "skipping deletion for a file on disk");
                return Ok(None);
            }
            let rel = rel_of(&core.vault_path, &path);
            let vault = core.vault_path.clone();
            let rel_for_call = rel.clone();
            core.with_core(move |g| remove_deleted_file(g, &rel_for_call, &vault))
                .await?;
            tracing::info!(path = %rel, "removed deleted file from index");
            Ok(None)
        }
        WatchEvent::Moved { from, to } => {
            let old_rel = rel_of(&core.vault_path, &from);
            let new_rel = rel_of(&core.vault_path, &to);
            let vault = core.vault_path.clone();
            let (old_for_call, new_for_call, to_for_call) =
                (old_rel.clone(), new_rel.clone(), to.clone());
            let (file_id, stripped) = core
                .with_core(move |g| {
                    rename_moved_file(g, &old_for_call, &new_for_call, &to_for_call, &vault)
                })
                .await?;
            tracing::info!(from = %old_rel, to = %new_rel, "renamed file in index");
            // The frontmatter write happens outside the lock. It raises a
            // `Changed` event that re-indexes the note.
            if let Some(stripped) = stripped {
                let tmp = to.with_extension("md.tmp");
                if let Err(e) =
                    std::fs::write(&tmp, &stripped).and_then(|_| std::fs::rename(&tmp, &to))
                {
                    tracing::warn!(error = %e, "failed to strip placement frontmatter");
                    let _ = std::fs::remove_file(&tmp);
                }
            }
            Ok(file_id)
        }
        // Replaced by the vault diff before this is reached.
        WatchEvent::FullRescan => Ok(None),
    }
}

/// Consumer task: applies batches of watch events one file at a time.
///
/// Two passes per batch: mutations, then an edge rebuild for the files the
/// batch touched. A batch that carries a `FullRescan` is replaced by the vault
/// diff, which covers every event beside it.
pub async fn run_consumer(
    mut rx: mpsc::Receiver<Vec<WatchEvent>>,
    core: Core,
    exclude: Vec<String>,
) {
    tracing::info!("Watcher consumer started");

    while let Some(mut events) = rx.recv().await {
        tracing::info!(count = events.len(), "processing event batch");

        if events.iter().any(|e| matches!(e, WatchEvent::FullRescan)) {
            tracing::info!("performing full rescan");
            // The diff is counted before the batch is settled, so the counter
            // does not pass through zero.
            let replaced = events.len();
            match diff_events(&core, &exclude).await {
                Ok(diff) => {
                    core.pending_events.fetch_add(diff.len(), Ordering::Relaxed);
                    settle(&core.pending_events, replaced);
                    events = diff;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "full rescan failed");
                    settle(&core.pending_events, replaced);
                    continue;
                }
            }
        }

        // Move detection reads the store and nothing else.
        let sent = events.len();
        let vault = core.vault_path.clone();
        let events = match core
            .with_reader(move |store| {
                let mut events = events;
                detect_moves(&mut events, store, &vault);
                Ok(events)
            })
            .await
        {
            Ok(events) => events,
            Err(e) => {
                tracing::warn!(error = %e, "move detection failed; batch dropped");
                settle(&core.pending_events, sent);
                continue;
            }
        };

        // Pass 1: mutations, one file per lock.
        let mut affected_file_ids: Vec<i64> = Vec::new();
        for event in events {
            let credit = credits(&event);
            match apply_event(&core, event).await {
                Ok(Some(file_id)) => affected_file_ids.push(file_id),
                Ok(None) => {}
                Err(e) => tracing::warn!(error = %e, "failed to apply watch event"),
            }
            settle(&core.pending_events, credit);
        }

        // Pass 2: the files this batch touched, and the notes whose broken
        // links those files now satisfy (#108). A deletion repairs itself
        // inside `indexer::remove_file`.
        if !affected_file_ids.is_empty() {
            tracing::info!(
                count = affected_file_ids.len(),
                "rebuilding edges for affected files"
            );
            let vault = core.vault_path.clone();
            if let Err(e) = core
                .with_core(move |g| {
                    indexer::reconcile_links(g.store, &vault, &affected_file_ids).map(|_| ())
                })
                .await
            {
                tracing::warn!(error = %e, "failed to rebuild edges");
            }
        }

        if core.pending_events.load(Ordering::Relaxed) == 0
            && let Err(e) = core
                .with_core(|g| g.store.set_meta("last_indexed_at", &unix_now()))
                .await
        {
            tracing::warn!(error = %e, "failed to record last_indexed_at");
        }

        tracing::info!("batch processing complete");
    }

    tracing::info!("Watcher consumer shutting down (channel closed)");
}

#[cfg(test)]
mod tests {
    use super::{ResolvedWatcher, fs_magic_needs_poll, requested_backend, resolve_watcher};
    use crate::config::WatcherBackend;

    /// A note the watcher indexes resolves the links other notes wrote before
    /// it existed (#108).
    ///
    /// The watcher rebuilds the edges of the files it touched. The note that
    /// wrote the link is not one of them — it did not change — so its broken
    /// link outlived the condition it described.
    #[tokio::test]
    async fn a_file_the_watcher_indexes_resolves_the_links_that_waited_for_it() {
        use super::{WatchEvent, run_consumer};
        use crate::config::Config;
        use crate::llm::MockLlm;
        use crate::store::Store;
        use std::sync::atomic::Ordering;
        use tokio::sync::mpsc;

        let tmp = tempfile::tempdir().unwrap();
        let vault = tmp.path().to_path_buf();
        std::fs::write(vault.join("a.md"), "# A\n\nSee [[b]].\n").unwrap();

        let db = tmp.path().join("knapper.db");
        let config = Config::default();
        {
            let store = Store::open(&db).unwrap();
            crate::indexer::run_index_shared(
                &vault,
                &config,
                crate::indexer::IndexSettings::from_config(&config),
                &store,
                &mut MockLlm::new(256),
                false,
                None,
            )
            .unwrap();
            assert_eq!(store.get_unresolved_links().unwrap().len(), 1);
        }

        std::fs::write(vault.join("b.md"), "# B\n\nBody.\n").unwrap();

        let core =
            crate::core::Core::for_test(&db, Box::new(MockLlm::new(256)), config, vault.clone());
        let (tx, rx) = mpsc::channel(4);
        core.pending_events.fetch_add(1, Ordering::Relaxed);
        tx.send(vec![WatchEvent::Changed(vault.join("b.md"))])
            .await
            .unwrap();
        drop(tx);

        run_consumer(rx, core.clone(), Vec::new()).await;
        assert_eq!(core.pending_events.load(Ordering::Relaxed), 0);

        let writer = core.writer();
        let store = writer.lock().await;
        assert!(
            store.get_unresolved_links().unwrap().is_empty(),
            "the link A wrote resolves now that B is indexed: {:?}",
            store.get_unresolved_links().unwrap()
        );
        let f_a = store.get_file("a.md").unwrap().unwrap();
        let f_b = store.get_file("b.md").unwrap().unwrap();
        let out = store.get_outgoing(f_a.id, Some("wikilink")).unwrap();
        assert_eq!(out.len(), 1, "and the graph holds the edge");
        assert_eq!(out[0].0, f_b.id);
    }

    /// A note the watcher sees move records the path-shaped links it breaks
    /// (#108).
    ///
    /// A wikilink resolves by basename, so most links survive a move. One
    /// written as `[[folder/note]]` names a path the vault no longer has.
    #[tokio::test]
    async fn a_move_the_watcher_sees_records_the_path_shaped_links_it_breaks() {
        use super::{WatchEvent, run_consumer};
        use crate::config::Config;
        use crate::llm::MockLlm;
        use crate::store::Store;
        use std::sync::atomic::Ordering;
        use tokio::sync::mpsc;

        let tmp = tempfile::tempdir().unwrap();
        let vault = tmp.path().to_path_buf();
        std::fs::create_dir_all(vault.join("inbox")).unwrap();
        std::fs::create_dir_all(vault.join("lore")).unwrap();
        std::fs::write(vault.join("a.md"), "# A\n\nSee [[inbox/n]].\n").unwrap();
        std::fs::write(vault.join("inbox/n.md"), "# N\n\nBody.\n").unwrap();

        let db = tmp.path().join("knapper.db");
        let config = Config::default();
        {
            let store = Store::open(&db).unwrap();
            crate::indexer::run_index_shared(
                &vault,
                &config,
                crate::indexer::IndexSettings::from_config(&config),
                &store,
                &mut MockLlm::new(256),
                false,
                None,
            )
            .unwrap();
            assert!(store.get_unresolved_links().unwrap().is_empty());
        }

        std::fs::rename(vault.join("inbox/n.md"), vault.join("lore/n.md")).unwrap();

        let core =
            crate::core::Core::for_test(&db, Box::new(MockLlm::new(256)), config, vault.clone());
        let (tx, rx) = mpsc::channel(4);
        // The sender counts the one event it sends. The consumer credits a
        // `Moved` as two, and the counter stops at zero.
        core.pending_events.fetch_add(1, Ordering::Relaxed);
        tx.send(vec![WatchEvent::Moved {
            from: vault.join("inbox/n.md"),
            to: vault.join("lore/n.md"),
        }])
        .await
        .unwrap();
        drop(tx);

        run_consumer(rx, core.clone(), Vec::new()).await;
        assert_eq!(core.pending_events.load(Ordering::Relaxed), 0);

        let writer = core.writer();
        let store = writer.lock().await;
        assert_eq!(
            store.get_unresolved_links().unwrap(),
            vec![("a.md".to_string(), "inbox/n".to_string())],
            "the link names the path the note has left"
        );
    }

    #[test]
    fn the_env_override_wins_over_config() {
        assert_eq!(
            requested_backend(WatcherBackend::Auto, Some(WatcherBackend::Poll)),
            WatcherBackend::Poll
        );
        assert_eq!(
            requested_backend(WatcherBackend::Poll, None),
            WatcherBackend::Poll
        );
    }

    #[test]
    fn resolve_honours_explicit_backends_and_ignores_the_filesystem() {
        assert_eq!(
            resolve_watcher(WatcherBackend::Native, Some(true)),
            ResolvedWatcher::Native
        );
        assert_eq!(
            resolve_watcher(WatcherBackend::Poll, Some(false)),
            ResolvedWatcher::Poll
        );
    }

    #[test]
    fn auto_polls_only_when_the_filesystem_needs_it() {
        assert_eq!(
            resolve_watcher(WatcherBackend::Auto, Some(true)),
            ResolvedWatcher::Poll
        );
        assert_eq!(
            resolve_watcher(WatcherBackend::Auto, Some(false)),
            ResolvedWatcher::Native
        );
        assert_eq!(
            resolve_watcher(WatcherBackend::Auto, None),
            ResolvedWatcher::Native
        );
    }

    #[test]
    fn bind_mount_filesystems_want_polling() {
        // overlay, fuse, 9p, nfs, smbfs, cifs
        for magic in [
            0x794c7630_i64,
            0x65735546,
            0x0102_1997,
            0x6969,
            0x517b,
            0xff53_4d42,
        ] {
            assert!(fs_magic_needs_poll(magic), "magic {magic:#x} should poll");
        }
    }

    #[test]
    fn local_filesystems_use_native_notifications() {
        // ext4, btrfs, xfs, tmpfs
        for magic in [0xEF53_i64, 0x9123_683E, 0x5846_5342, 0x0102_1994] {
            assert!(
                !fs_magic_needs_poll(magic),
                "magic {magic:#x} should not poll"
            );
        }
    }

    /// The debouncer reports an atomic save as a removal of the target path
    /// followed by the events that describe the new file, and the removal is
    /// what a `serve` session's own writes produce: `writer::atomic_write`
    /// renames a temp file over the note. Taking that removal at face value
    /// drops the note from the index, and the change that follows it is the
    /// writer's own, which `is_recent_write` suppresses — so nothing puts the
    /// note back and it stays on disk and out of search (#93).
    #[tokio::test(flavor = "multi_thread")]
    async fn an_atomic_save_over_a_note_is_not_a_deletion() {
        use super::{WatchEvent, start_producer};
        use crate::exclude::ExcludeMatcher;
        use std::time::Duration;

        let tmp = tempfile::TempDir::new().unwrap();
        let vault = tmp.path().to_path_buf();
        let note = vault.join("note.md");
        std::fs::write(&note, "# Note\n\nfirst\n").unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<WatchEvent>>(64);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let producer = start_producer(
            vault.clone(),
            ExcludeMatcher::new(&[]).unwrap(),
            tx,
            shutdown_rx,
            WatcherBackend::Native,
            Duration::from_millis(200),
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        );
        tokio::time::sleep(Duration::from_millis(1500)).await;

        // The debouncer synthesizes the removal only when the target path is
        // already in its queue, which is any write inside the debounce window
        // — a second `update` to the same note, or an editor that saved it.
        std::fs::write(&note, "# Note\n\nsecond\n").unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        // What `writer::atomic_write` does.
        let temp = note.with_extension("md.tmp");
        std::fs::write(&temp, "# Note\n\nthird\n").unwrap();
        std::fs::rename(&temp, &note).unwrap();

        let mut seen: Vec<WatchEvent> = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(6);
        while let Ok(Some(batch)) = tokio::time::timeout(
            deadline.saturating_duration_since(tokio::time::Instant::now()),
            rx.recv(),
        )
        .await
        {
            seen.extend(batch);
        }
        let _ = shutdown_tx.send(());
        let _ = producer.join();

        assert!(
            !seen
                .iter()
                .any(|e| matches!(e, WatchEvent::Deleted(p) if p == &note)),
            "a note the rename replaced is still on disk: {seen:?}"
        );
        assert!(
            seen.iter()
                .any(|e| matches!(e, WatchEvent::Changed(p) if p == &note)),
            "the write itself has to reach the consumer: {seen:?}"
        );
    }

    /// The other half of the same rule: a path that is gone is gone, and the
    /// store has to let go of it (#93).
    #[test]
    fn a_removal_is_a_deletion_when_the_path_is_gone() {
        use super::{WatchEvent, process_debounced_events};
        use crate::exclude::ExcludeMatcher;
        use notify::{
            Event,
            event::{EventKind, RemoveKind},
        };
        use notify_debouncer_full::DebouncedEvent;

        let tmp = tempfile::TempDir::new().unwrap();
        let vault = tmp.path().to_path_buf();
        let kept = vault.join("kept.md");
        let gone = vault.join("gone.md");
        std::fs::write(&kept, "# Kept\n").unwrap();

        let removal = |path: &std::path::Path| {
            DebouncedEvent::new(
                Event {
                    kind: EventKind::Remove(RemoveKind::Any),
                    paths: vec![path.to_path_buf()],
                    attrs: Default::default(),
                },
                std::time::Instant::now(),
            )
        };

        let events = process_debounced_events(
            &[removal(&kept), removal(&gone)],
            &vault,
            &ExcludeMatcher::new(&[]).unwrap(),
        );

        assert!(
            matches!(events.as_slice(), [WatchEvent::Deleted(p)] if p == &gone),
            "only the path that is gone is a deletion: {events:?}"
        );
    }

    /// Startup reconciliation runs per file through the consumer: a read
    /// answers while it runs, and the tables it leaves equal a full index of
    /// the same vault (serve-core spec).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_reconciliation_indexes_per_file() {
        use super::{diff_events, enqueue_diff, run_consumer};
        use crate::core::Core;
        use crate::core::testing::{GatedEmbed, indexed_vault, test_config};
        use crate::indexer::edge_snapshot;
        use crate::llm::MockLlm;
        use crate::store::Store;
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        use tokio::sync::mpsc;

        let config = test_config();
        let mut notes: Vec<(String, String)> = (0..10)
            .map(|i| {
                (
                    format!("n{i}.md"),
                    format!(
                        "# Note {i}\n\nThe body of note {i}, long enough to be its own chunk and then some more words.\n"
                    ),
                )
            })
            .collect();
        notes[1].1.push_str("See [[n2]].\n");
        notes[3].1.push_str("See [[n4]].\n");
        let borrowed: Vec<(&str, &str)> = notes
            .iter()
            .map(|(p, b)| (p.as_str(), b.as_str()))
            .collect();
        let (tmp, vault, db) = indexed_vault(&borrowed, &config);

        // The server is "down": three notes change, one goes, one arrives.
        for i in [1, 2, 3] {
            std::fs::write(
                vault.join(format!("n{i}.md")),
                format!(
                    "# Note {i}\n\nRewritten while the server was down, still long enough to be a chunk of its own.\n"
                ),
            )
            .unwrap();
        }
        std::fs::remove_file(vault.join("n4.md")).unwrap();
        std::fs::write(
            vault.join("n10.md"),
            "# Note 10\n\nA new note that links to [[n0]] and is long enough to be a chunk.\n",
        )
        .unwrap();

        let (embed, release, entered) = GatedEmbed::new(256);
        let core = Core::for_test(&db, Box::new(embed), config.clone(), vault.clone());

        let diff = diff_events(&core, &[]).await.unwrap();
        assert_eq!(diff.len(), 5, "3 changed + 1 new + 1 deleted, got {diff:?}");

        let (tx, rx) = mpsc::channel(64);
        enqueue_diff(&core, &[], &tx).await.unwrap();
        drop(tx);
        assert_eq!(core.pending_events.load(Ordering::Relaxed), 5);

        let consumer = {
            let core = core.clone();
            tokio::spawn(async move { run_consumer(rx, core, Vec::new()).await })
        };
        tokio::task::spawn_blocking(move || entered.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .expect("the consumer reached the embedder");

        // The first file is parked on the embedder; a read still answers.
        let count = tokio::time::timeout(
            Duration::from_secs(2),
            core.with_reader(|store| store.file_count()),
        )
        .await
        .expect("a read waited on the reconciliation")
        .unwrap();
        assert!(count >= 9, "got {count}");
        assert!(core.pending_events.load(Ordering::Relaxed) > 0);

        release.send(()).unwrap();
        consumer.await.unwrap();
        assert_eq!(core.pending_events.load(Ordering::Relaxed), 0);

        // What the per-file path left equals a full index of the same vault.
        let fresh_db = tmp.path().join("fresh.db");
        let fresh = Store::open(&fresh_db).unwrap();
        crate::indexer::run_index_shared(
            &vault,
            &config,
            crate::indexer::IndexSettings::from_config(&config),
            &fresh,
            &mut MockLlm::new(256),
            false,
            None,
        )
        .unwrap();

        type Snapshot = (
            Vec<(String, String)>,
            Vec<(String, i64, String)>,
            Vec<String>,
        );
        fn snapshot(store: &Store) -> Snapshot {
            let mut files: Vec<(String, String)> = store
                .get_all_files()
                .unwrap()
                .into_iter()
                .map(|f| (f.path, f.content_hash))
                .collect();
            files.sort();
            let mut chunks = Vec::new();
            for f in store.get_all_files().unwrap() {
                for c in store.get_chunks_by_file(f.id).unwrap() {
                    chunks.push((f.path.clone(), c.seq, c.text));
                }
            }
            chunks.sort();
            (files, chunks, edge_snapshot(store))
        }
        let reconciled = {
            let writer = core.writer();
            let store = writer.lock().await;
            snapshot(&store)
        };
        let full = snapshot(&fresh);
        assert_eq!(reconciled.0, full.0, "files differ");
        assert_eq!(reconciled.1, full.1, "chunks differ");
        assert_eq!(reconciled.2, full.2, "edges differ");
    }

    /// A `Deleted` for a path that is on disk leaves the row alone. The
    /// startup diff runs beside the consumer, so a live `Changed` for a
    /// restored note can land before the diff's stale `Deleted`.
    #[tokio::test]
    async fn a_deletion_for_a_note_on_disk_keeps_its_row() {
        use super::{WatchEvent, run_consumer};
        use crate::core::testing::{indexed_core, test_config};
        use std::sync::atomic::Ordering;
        use tokio::sync::mpsc;

        let body = "# A\n\nA note that is long enough to be a chunk of its own and then some.\n";
        let (_tmp, core) = indexed_core(&[("a.md", body)], test_config());
        let note = core.vault_path.join("a.md");
        assert!(note.exists());

        let (tx, rx) = mpsc::channel(4);
        core.pending_events.fetch_add(1, Ordering::Relaxed);
        tx.send(vec![WatchEvent::Deleted(note)]).await.unwrap();
        drop(tx);

        run_consumer(rx, core.clone(), Vec::new()).await;
        assert_eq!(core.pending_events.load(Ordering::Relaxed), 0);
        let row = core
            .with_reader(|store| store.get_file("a.md"))
            .await
            .unwrap();
        assert!(row.is_some(), "the note on disk keeps its row");
    }

    /// A batch that carries a `FullRescan` is replaced by the vault diff: the
    /// counter ends at zero and the store matches the vault.
    #[tokio::test]
    async fn a_full_rescan_settles_the_counter_and_matches_the_vault() {
        use super::{WatchEvent, run_consumer};
        use crate::core::testing::{indexed_core, test_config};
        use std::sync::atomic::Ordering;
        use tokio::sync::mpsc;

        let body = |n: &str| {
            format!(
                "# {n}\n\nThe body of note {n}, long enough to be its own chunk and then some.\n"
            )
        };
        let (_tmp, core) =
            indexed_core(&[("a.md", &body("a")), ("b.md", &body("b"))], test_config());
        let vault = core.vault_path.as_ref().clone();
        std::fs::write(vault.join("a.md"), body("a, rewritten")).unwrap();
        std::fs::remove_file(vault.join("b.md")).unwrap();
        std::fs::write(vault.join("c.md"), body("c")).unwrap();

        let (tx, rx) = mpsc::channel(4);
        core.pending_events.fetch_add(2, Ordering::Relaxed);
        tx.send(vec![
            WatchEvent::FullRescan,
            WatchEvent::Changed(vault.join("c.md")),
        ])
        .await
        .unwrap();
        drop(tx);

        run_consumer(rx, core.clone(), Vec::new()).await;
        assert_eq!(core.pending_events.load(Ordering::Relaxed), 0);

        let files = core
            .with_reader(|store| store.get_all_files())
            .await
            .unwrap();
        let mut stored: Vec<(String, String)> = files
            .into_iter()
            .map(|f| (f.path, f.content_hash))
            .collect();
        stored.sort();
        let on_disk: Vec<(String, String)> = ["a.md", "c.md"]
            .iter()
            .map(|rel| {
                let hash = crate::indexer::compute_file_hash(&vault.join(rel)).unwrap();
                (rel.to_string(), hash)
            })
            .collect();
        assert_eq!(stored, on_disk);
    }
}
