use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::{PollWatcher, RecursiveMode, Watcher};
use notify_debouncer_full::{
    DebounceEventResult, DebouncedEvent, Debouncer, FileIdCache, RecommendedCache, new_debouncer,
    new_debouncer_opt,
};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::config::{Config, WatcherBackend};
use crate::exclude::ExcludeMatcher;
use crate::indexer;
use crate::llm::EmbedModel;
use crate::placement;
use crate::profile::VaultProfile;
use crate::serve::RecentWrites;
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

/// Start the file watcher and consumer. Returns a thread handle for the producer
/// and a shutdown sender. On startup, runs a reconciliation index to catch any
/// changes that occurred while the server was down, then begins watching for
/// real-time file changes.
pub fn start_watcher(
    store: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
    vault_path: Arc<PathBuf>,
    profile: Arc<Option<VaultProfile>>,
    config: Config,
    exclude: Vec<String>,
    recent_writes: RecentWrites,
) -> anyhow::Result<(std::thread::JoinHandle<()>, oneshot::Sender<()>)> {
    let (tx, rx) = mpsc::channel::<Vec<WatchEvent>>(64);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Compile the exclude globs once, here, so a bad pattern fails server startup
    // rather than every event batch.
    let matcher = ExcludeMatcher::new(&exclude)?;

    // Start producer (begins buffering events immediately)
    let producer_handle = start_producer(
        vault_path.as_ref().clone(),
        matcher,
        tx,
        shutdown_rx,
        config.watcher.backend,
        Duration::from_secs(config.watcher.poll_interval_secs),
    );

    // Spawn consumer task
    let store_clone = store.clone();
    let embedder_clone = embedder.clone();
    let vault_clone = vault_path.clone();
    let profile_clone = profile.clone();
    let config_clone = config.clone();
    tokio::spawn(async move {
        // Startup reconciliation: run index to catch changes since last shutdown
        {
            let store_lock = store_clone.lock().await;
            let mut embedder_lock = embedder_clone.lock().await;
            // The startup config is the session's own, captured once and never
            // reloaded, so reading the index-time settings off it here yields
            // the same values `knapper serve` captured — not a fresh load that
            // could drift (#72).
            let settings = crate::indexer::IndexSettings::from_config(&config_clone);
            if let Err(e) = crate::indexer::run_index_shared(
                &vault_clone,
                &config_clone,
                settings,
                &store_lock,
                &mut *embedder_lock,
                false,
                profile_clone.as_ref().as_ref(),
            ) {
                tracing::warn!("Startup reconciliation failed: {:#}", e);
            }
        }

        // Then consume events
        run_consumer(
            rx,
            store_clone,
            embedder_clone,
            vault_clone,
            profile_clone,
            config_clone,
            recent_writes,
        )
        .await;
    });

    Ok((producer_handle, shutdown_tx))
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
                    Ok(d) => drive_producer(d, vault_path, exclude, tx, shutdown_rx, debouncer_rx),
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
                    Ok(d) => drive_producer(d, vault_path, exclude, tx, shutdown_rx, debouncer_rx),
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
                if !watch_events.is_empty() && tx.blocking_send(watch_events).is_err() {
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

/// Consumer async task that processes batches of watch events.
///
/// Two-pass processing:
/// - Pass 1: Apply mutations (index/remove/rename files)
/// - Pass 2: Rebuild edges for affected files
pub async fn run_consumer(
    mut rx: mpsc::Receiver<Vec<WatchEvent>>,
    store: Arc<Mutex<Store>>,
    embedder: Arc<Mutex<Box<dyn EmbedModel + Send>>>,
    vault_path: Arc<PathBuf>,
    profile: Arc<Option<VaultProfile>>,
    config: Config,
    recent_writes: RecentWrites,
) {
    tracing::info!("Watcher consumer started");

    while let Some(mut events) = rx.recv().await {
        tracing::info!(count = events.len(), "processing event batch");

        // Move detection (needs store lock briefly)
        {
            let store_guard = store.lock().await;
            detect_moves(&mut events, &store_guard, &vault_path);
        }

        let mut affected_file_ids: Vec<i64> = Vec::new();
        let mut had_full_rescan = false;

        // Pass 1: mutations (one event at a time)
        for event in &events {
            match event {
                WatchEvent::Changed(path) => {
                    // Skip files recently written by MCP tools to avoid redundant re-indexing
                    if is_recent_write(&recent_writes, path).await {
                        tracing::debug!(path = %path.display(), "skipping re-index for MCP-written file");
                        continue;
                    }

                    let rel = path
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    let content = match std::fs::read_to_string(path) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!(path = %path.display(), error = %e, "failed to read changed file, skipping");
                            continue;
                        }
                    };

                    let content_hash = match indexer::compute_file_hash(path) {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::warn!(path = %path.display(), error = %e, "failed to hash changed file, skipping");
                            continue;
                        }
                    };

                    let store_guard = store.lock().await;
                    // Check if file is new (not yet in store) before indexing
                    let is_new_file = store_guard.get_file(&rel).ok().flatten().is_none();

                    let mut embedder_guard = embedder.lock().await;
                    match indexer::index_file(
                        &rel,
                        &content,
                        &content_hash,
                        &store_guard,
                        &mut *embedder_guard,
                        &vault_path,
                        &config,
                    ) {
                        Ok(result) => {
                            tracing::info!(
                                path = %rel,
                                file_id = result.file_id,
                                chunks = result.total_chunks,
                                "indexed changed file"
                            );
                            affected_file_ids.push(result.file_id);

                            // Adjust folder centroid for newly added files
                            if is_new_file
                                && let Ok(vectors) =
                                    store_guard.get_chunk_vectors_for_file(result.file_id)
                                && !vectors.is_empty()
                            {
                                let dim = vectors[0].len();
                                let mut mean = vec![0.0f32; dim];
                                for v in &vectors {
                                    for (i, val) in v.iter().enumerate() {
                                        mean[i] += val;
                                    }
                                }
                                let n = vectors.len() as f32;
                                for val in &mut mean {
                                    *val /= n;
                                }

                                let folder = std::path::Path::new(&rel)
                                    .parent()
                                    .map(|p| p.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                if let Err(e) =
                                    store_guard.adjust_folder_centroid(&folder, &mean, true)
                                {
                                    tracing::warn!(
                                        path = %rel,
                                        error = %e,
                                        "failed to adjust centroid for new file"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(path = %rel, error = %e, "failed to index changed file");
                        }
                    }
                    drop(embedder_guard);
                    drop(store_guard);
                }

                WatchEvent::Deleted(path) => {
                    let rel = path
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    let store_guard = store.lock().await;

                    // Capture mean vector BEFORE removal for centroid adjustment
                    let mean_vec_and_folder =
                        store_guard.get_file(&rel).ok().flatten().and_then(|file| {
                            let vectors = store_guard.get_chunk_vectors_for_file(file.id).ok()?;
                            if vectors.is_empty() {
                                return None;
                            }
                            let dim = vectors[0].len();
                            let mut mean = vec![0.0f32; dim];
                            for v in &vectors {
                                for (i, val) in v.iter().enumerate() {
                                    mean[i] += val;
                                }
                            }
                            let n = vectors.len() as f32;
                            for val in &mut mean {
                                *val /= n;
                            }
                            let folder = std::path::Path::new(&rel)
                                .parent()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_default();
                            Some((mean, folder))
                        });

                    match indexer::remove_file(&rel, &store_guard) {
                        Ok(()) => {
                            tracing::info!(path = %rel, "removed deleted file from index");

                            // Adjust folder centroid after successful removal
                            if let Some((mean, folder)) = mean_vec_and_folder
                                && let Err(e) =
                                    store_guard.adjust_folder_centroid(&folder, &mean, false)
                            {
                                tracing::warn!(
                                    path = %rel,
                                    error = %e,
                                    "failed to adjust centroid for deleted file"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(path = %rel, error = %e, "failed to remove deleted file");
                        }
                    }
                    drop(store_guard);
                }

                WatchEvent::Moved { from, to } => {
                    let old_rel = from
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(from)
                        .to_string_lossy()
                        .to_string();
                    let new_rel = to
                        .strip_prefix(vault_path.as_ref())
                        .unwrap_or(to)
                        .to_string_lossy()
                        .to_string();

                    // Phase 1: Store operations under lock
                    let needs_frontmatter_strip = {
                        let store_guard = store.lock().await;
                        match indexer::rename_file(&old_rel, &new_rel, &store_guard) {
                            Ok(()) => {
                                tracing::info!(from = %old_rel, to = %new_rel, "renamed file in index");
                                // Track the file_id for edge rebuild
                                if let Ok(Some(record)) = store_guard.get_file(&new_rel) {
                                    affected_file_ids.push(record.id);
                                }

                                // Placement correction detection
                                if let Ok(content) = std::fs::read_to_string(to) {
                                    let actual_folder = std::path::Path::new(&new_rel)
                                        .parent()
                                        .map(|p| p.to_string_lossy().to_string())
                                        .unwrap_or_default();

                                    match placement::detect_correction_from_frontmatter(
                                        &content,
                                        &actual_folder,
                                    ) {
                                        Some(correction) => {
                                            tracing::info!(
                                                file = %new_rel,
                                                suggested = %correction.suggested_folder,
                                                actual = %correction.actual_folder,
                                                "placement correction detected"
                                            );

                                            // Compute mean vector from file chunks
                                            if let Ok(Some(file)) = store_guard.get_file(&new_rel)
                                                && let Ok(vectors) =
                                                    store_guard.get_chunk_vectors_for_file(file.id)
                                                && !vectors.is_empty()
                                            {
                                                let dim = vectors[0].len();
                                                let mut mean = vec![0.0f32; dim];
                                                for v in &vectors {
                                                    for (i, val) in v.iter().enumerate() {
                                                        mean[i] += val;
                                                    }
                                                }
                                                let n = vectors.len() as f32;
                                                for val in &mut mean {
                                                    *val /= n;
                                                }

                                                // Adjust centroids: boost actual, decay suggested
                                                if let Err(e) = store_guard.adjust_folder_centroid(
                                                    &correction.actual_folder,
                                                    &mean,
                                                    true,
                                                ) {
                                                    tracing::warn!(error = %e, "failed to adjust actual folder centroid");
                                                }
                                                if let Err(e) = store_guard.adjust_folder_centroid(
                                                    &correction.suggested_folder,
                                                    &mean,
                                                    false,
                                                ) {
                                                    tracing::warn!(error = %e, "failed to adjust suggested folder centroid");
                                                }
                                            }

                                            // Log the correction
                                            if let Err(e) = store_guard.insert_placement_correction(
                                                &new_rel,
                                                &correction.suggested_folder,
                                                &correction.actual_folder,
                                            ) {
                                                tracing::warn!(error = %e, "failed to log placement correction");
                                            }

                                            // Signal that frontmatter strip is needed (done outside lock)
                                            let stripped =
                                                placement::strip_placement_frontmatter(&content);
                                            if stripped != content {
                                                Some(stripped)
                                            } else {
                                                None
                                            }
                                        }
                                        None => {
                                            // Check if it's a confirmation (suggested == actual) — just strip
                                            let has_suggested =
                                                content.contains("suggested_folder:");
                                            if has_suggested {
                                                let stripped =
                                                    placement::strip_placement_frontmatter(
                                                        &content,
                                                    );
                                                if stripped != content {
                                                    Some(stripped)
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            }
                                        }
                                    }
                                } else {
                                    None
                                }
                            }
                            Err(e) => {
                                tracing::warn!(from = %old_rel, to = %new_rel, error = %e, "failed to rename file");
                                None
                            }
                        }
                    }; // store_guard dropped here

                    // Phase 2: Frontmatter file I/O without store lock.
                    // The write triggers a Changed event that gets re-indexed anyway.
                    if let Some(stripped) = needs_frontmatter_strip {
                        let tmp = to.with_extension("md.tmp");
                        if let Err(e) =
                            std::fs::write(&tmp, &stripped).and_then(|_| std::fs::rename(&tmp, to))
                        {
                            tracing::warn!(error = %e, "failed to strip placement frontmatter");
                            let _ = std::fs::remove_file(&tmp);
                        }
                    }
                }

                WatchEvent::FullRescan => {
                    // FullRescan: holds both locks for the entire rescan duration.
                    // This blocks MCP tool calls but is acceptable since FullRescan
                    // is rare (macOS FSEvents buffer overflow). Future optimization:
                    // process files one-at-a-time with per-file lock release.
                    tracing::info!("performing full rescan");
                    let store_guard = store.lock().await;
                    let mut embedder_guard = embedder.lock().await;
                    // Off the session's own startup config, not a fresh load —
                    // the same settings `knapper serve` captured (#72).
                    let settings = crate::indexer::IndexSettings::from_config(&config);
                    match indexer::run_index_shared(
                        &vault_path,
                        &config,
                        settings,
                        &store_guard,
                        &mut *embedder_guard,
                        false,
                        profile.as_ref().as_ref(),
                    ) {
                        Ok(result) => {
                            tracing::info!(
                                new = result.new_files,
                                updated = result.updated_files,
                                deleted = result.deleted_files,
                                chunks = result.total_chunks,
                                duration_secs = result.duration.as_secs_f64(),
                                "full rescan complete"
                            );
                            had_full_rescan = true;
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "full rescan failed");
                        }
                    }
                    drop(embedder_guard);
                    drop(store_guard);
                }
            }
        }

        // Pass 2: edge rebuild for affected files (skip if full rescan already rebuilt everything)
        if !had_full_rescan && !affected_file_ids.is_empty() {
            tracing::info!(
                count = affected_file_ids.len(),
                "rebuilding edges for affected files"
            );
            let store_guard = store.lock().await;
            for file_id in &affected_file_ids {
                // Delete old edges first
                // Outgoing only: backlinks into this file belong to other
                // files' content, and re-indexing this one must not touch them
                // (issue #27).
                if let Err(e) = store_guard.delete_outgoing_edges_for_file(*file_id) {
                    tracing::warn!(file_id, error = %e, "failed to delete old edges");
                    continue;
                }

                if let Ok(Some(file)) = store_guard.get_file_by_id(*file_id) {
                    let content =
                        std::fs::read_to_string(vault_path.join(&file.path)).unwrap_or_default();
                    if let Err(e) = indexer::build_edges_for_file(&store_guard, *file_id, &content)
                    {
                        tracing::warn!(
                            file_id,
                            path = %file.path,
                            error = %e,
                            "failed to rebuild edges"
                        );
                    }
                }
            }
            drop(store_guard);
        }

        tracing::info!("batch processing complete");
    }

    tracing::info!("Watcher consumer shutting down (channel closed)");
}

#[cfg(test)]
mod tests {
    use super::{ResolvedWatcher, fs_magic_needs_poll, requested_backend, resolve_watcher};
    use crate::config::WatcherBackend;

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
}
