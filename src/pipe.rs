//! What the CLI does when the reader of its output goes away (#128).
//!
//! Rust sets `SIGPIPE` to `SIG_IGN` before `main`, so a write to a closed pipe
//! returns `EPIPE` and `println!` panics on it. `knapper tags | head -3` then
//! printed a panic and exited 101 where every other Unix CLI ends quietly.

/// Put back the `SIGPIPE` disposition the process would have without Rust.
///
/// Call it before any output, `--help` and `--version` included.
#[cfg(unix)]
pub fn restore_default_sigpipe() {
    // SAFETY: `SIG_DFL` is the disposition a process starts with. It installs
    // no handler, so no Rust code runs in a signal context, and the call reads
    // and writes nothing but the process's own signal table.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

/// A platform with no `SIGPIPE` needs nothing put back.
#[cfg(not(unix))]
pub fn restore_default_sigpipe() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_process_default_replaces_the_ignore_rust_installs() {
        // Rust sets SIG_IGN before `main`, so a write to a closed pipe returns
        // EPIPE and `println!` panics instead of the process ending on the
        // signal. Reading the disposition means setting one, so the read puts
        // SIG_IGN back and the rest of the test process runs as it was.
        restore_default_sigpipe();
        let previous = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
        assert_eq!(previous, libc::SIG_DFL);
    }
}
