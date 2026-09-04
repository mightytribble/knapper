//! Custom properties (#66): a named value on a note, read from frontmatter
//! under Obsidian's rules and from Dataview inline fields under Dataview's.
//!
//! This module owns the kind vocabulary and the two extractors. Nothing else
//! in the crate reads a property out of a note.

/// The kind of a property value, read from the value's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Text,
    Number,
    Checkbox,
    Link,
    Empty,
}

impl Kind {
    /// The name the store writes into `properties.kind`.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Number => "number",
            Kind::Checkbox => "checkbox",
            Kind::Link => "link",
            Kind::Empty => "empty",
        }
    }

    /// The inverse of [`Kind::as_str`]. `None` for a name this build does
    /// not know.
    pub fn parse(name: &str) -> Option<Kind> {
        match name {
            "text" => Some(Kind::Text),
            "number" => Some(Kind::Number),
            "checkbox" => Some(Kind::Checkbox),
            "link" => Some(Kind::Link),
            "empty" => Some(Kind::Empty),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kind_round_trips_through_its_name() {
        for kind in [
            Kind::Text,
            Kind::Number,
            Kind::Checkbox,
            Kind::Link,
            Kind::Empty,
        ] {
            assert_eq!(Kind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(Kind::parse("date"), None);
    }
}
