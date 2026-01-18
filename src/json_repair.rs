use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RepairError {
    #[error("empty input")]
    Empty,
    #[error("parse error: {0}")]
    Parse(String),
}

/// A minimal, safe JSON "repair" that:
/// - Ensures the snippet is wrapped in braces if it isn't already.
/// - Replaces single quotes with double quotes.
/// - Removes trailing commas in objects/arrays.
pub fn repair_json_snippet(input: &str) -> Result<String, RepairError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(RepairError::Empty);
    }

    // Heuristic replacements; conservative to avoid mangling valid JSON.
    let mut s = trimmed.replace('\'', "\"");
    s = remove_trailing_commas(&s);

    // If it doesn't start with { or [, assume it's an object.
    if !s.starts_with('{') && !s.starts_with('[') {
        s = format!("{{{s}}}");
    }

    // Validate round-trip.
    serde_json::from_str::<Value>(&s).map_err(|e| RepairError::Parse(e.to_string()))?;
    Ok(s)
}

fn remove_trailing_commas(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for (i, ch) in s.chars().enumerate() {
        if ch == ',' {
            // Look ahead to see if next non-space is closing brace/bracket.
            let mut j = i + 1;
            let bytes = s.as_bytes();
            while j < s.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < s.len() {
                let next = bytes[j] as char;
                if next == '}' || next == ']' {
                    // skip this trailing comma
                    continue;
                }
            }
        }
        out.push(ch);
    }
    out
}
