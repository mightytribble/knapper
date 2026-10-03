//! The faults a caller can repair, as one kind each.
//!
//! A `Fault` is built where the fault is known, carried inside `anyhow` like
//! any other error, and read once at each server. The server maps the kind
//! to its own vocabulary: HTTP to a status, MCP to a code. Everything that is
//! not a `Fault` is internal.

/// A fault the caller can repair, or a state the caller must know about.
///
/// A string variant carries the message its site writes, so converting a
/// site to a `Fault` changes no text. `ReadOnly` has one text for both
/// servers.
#[derive(Debug, thiserror::Error)]
pub enum Fault {
    /// The caller's text named nothing or asked two things at once: a scope
    /// term, a cursor, a link filter, a contradictory pair of flags.
    #[error("{0}")]
    InvalidInput(String),
    /// The `file` or `section` a capability addresses is absent.
    #[error("{0}")]
    NotFound(String),
    /// One name, several notes.
    #[error("{0}")]
    Ambiguous(String),
    /// The write would clobber: the note moved under the caller, the target
    /// exists, the note is already archived.
    #[error("{0}")]
    Conflict(String),
    /// The index cannot answer this until `knapper index` runs.
    #[error("{0}")]
    StaleIndex(String),
    /// The server was started with `--read-only`.
    #[error(
        "write operations are disabled in read-only mode; start `serve` without --read-only to enable writes"
    )]
    ReadOnly,
}

impl Fault {
    /// Every kind, in variant order. The OpenAPI document's `kind` enum
    /// reads it, so it agrees with `kind()` by test.
    pub const KINDS: &'static [&'static str] = &[
        "invalid_input",
        "not_found",
        "ambiguous",
        "conflict",
        "stale_index",
        "read_only",
    ];

    /// The kind as a snake_case word, for the `kind` field of an error body.
    pub fn kind(&self) -> &'static str {
        match self {
            Fault::InvalidInput(_) => "invalid_input",
            Fault::NotFound(_) => "not_found",
            Fault::Ambiguous(_) => "ambiguous",
            Fault::Conflict(_) => "conflict",
            Fault::StaleIndex(_) => "stale_index",
            Fault::ReadOnly => "read_only",
        }
    }

    /// The `Fault` anywhere in `e`'s chain, if one is there.
    ///
    /// `anyhow` downcasts through every `.context()` layer to the error that
    /// was wrapped, so a site may add context freely. What it must not do is
    /// `anyhow!("{e} ...")`, which starts a new chain with no source.
    pub fn of(e: &anyhow::Error) -> Option<&Fault> {
        e.downcast_ref::<Fault>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn a_fault_survives_two_contexts() {
        // The servers read the kind after the pipeline has added context, so
        // the downcast must reach through the chain.
        let inner: anyhow::Result<()> =
            Err(Fault::Conflict("mtime conflict: note.md".into()).into());
        let err = inner
            .context("editing note.md")
            .context("the update tool")
            .unwrap_err();
        let fault = Fault::of(&err).expect("the kind is reachable through context");
        assert!(matches!(fault, Fault::Conflict(_)), "{fault:?}");
        assert_eq!(fault.kind(), "conflict");
        assert_eq!(
            format!("{err:#}"),
            "the update tool: editing note.md: mtime conflict: note.md"
        );
    }

    #[test]
    fn a_formatted_rewrap_loses_the_kind() {
        // `anyhow!("{e} ...")` is a new error with no source. This is the
        // rule every site a `Fault` passes through follows: add context,
        // never reformat.
        let err: anyhow::Error = Fault::NotFound("file not found: a.md".into()).into();
        let rewrapped = anyhow::anyhow!("{err} in a.md");
        assert!(Fault::of(&rewrapped).is_none());
    }

    #[test]
    fn each_kind_has_its_name() {
        let cases = [
            (Fault::InvalidInput("x".into()), "invalid_input"),
            (Fault::NotFound("x".into()), "not_found"),
            (Fault::Ambiguous("x".into()), "ambiguous"),
            (Fault::Conflict("x".into()), "conflict"),
            (Fault::StaleIndex("x".into()), "stale_index"),
            (Fault::ReadOnly, "read_only"),
        ];
        for (fault, kind) in cases {
            assert_eq!(fault.kind(), kind, "{fault:?}");
        }
    }

    /// `KINDS` is what the OpenAPI document publishes as the `kind` enum, so
    /// it has to agree with `kind()`.
    #[test]
    fn kinds_lists_every_variant_once() {
        let variants = [
            Fault::InvalidInput("x".into()),
            Fault::NotFound("x".into()),
            Fault::Ambiguous("x".into()),
            Fault::Conflict("x".into()),
            Fault::StaleIndex("x".into()),
            Fault::ReadOnly,
        ];
        assert_eq!(Fault::KINDS.len(), variants.len());
        for (i, fault) in variants.iter().enumerate() {
            assert_eq!(Fault::KINDS[i], fault.kind(), "{fault:?}");
        }

        // A new variant must be added to `variants` above and to `KINDS`.
        // This match has no wildcard, so adding one is a compile error here.
        for fault in &variants {
            match fault {
                Fault::InvalidInput(_)
                | Fault::NotFound(_)
                | Fault::Ambiguous(_)
                | Fault::Conflict(_)
                | Fault::StaleIndex(_)
                | Fault::ReadOnly => {}
            }
        }
    }

    #[test]
    fn a_string_variant_displays_its_message_alone() {
        let fault = Fault::InvalidInput("no such tag 'x'".into());
        assert_eq!(fault.to_string(), "no such tag 'x'");
        assert_eq!(
            Fault::ReadOnly.to_string(),
            "write operations are disabled in read-only mode; start `serve` without --read-only to enable writes"
        );
    }
}
