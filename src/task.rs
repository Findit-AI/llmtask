//! `Task` trait and `ParseError` — the cross-engine abstraction.

use serde_json::Value;

/// A structured-output task description.
///
/// Implementations supply the prompt, the JSON schema for constrained
/// decoding, and a parser that turns the model's raw text into a typed
/// `Output`. The trait is `Send + Sync` and `Output: Send` so trait
/// objects (`dyn Task<Output = ...>`) and concurrent call sites work
/// without extra bounds at the call site.
///
/// Implementations should cache their schema (build it once in `new`)
/// rather than rebuilding it per call — `schema` returns a borrow.
pub trait Task: Send + Sync {
  /// The typed result of a successful run.
  type Output: Send;

  /// The user-message prompt sent alongside the images.
  fn prompt(&self) -> &str;

  /// JSON schema used for constrained decoding.
  fn schema(&self) -> &Value;

  /// Parse the model's raw text output into a typed `Output`.
  fn parse(&self, raw: &str) -> Result<Self::Output, ParseError>;
}

/// Errors returned by [`Task::parse`].
#[derive(thiserror::Error, Debug)]
pub enum ParseError {
  /// `serde_json` failed to parse the response as valid JSON.
  #[error("invalid JSON: {0}")]
  Json(#[from] serde_json::Error),
  /// JSON parsed but one or more required schema fields are absent or
  /// present as JSON `null`. Both cases are treated as missing because
  /// the schema requires every listed field to carry a string or array
  /// value, never null. (Per-field deserializers silently coerce `null`
  /// into defaults, which would otherwise hide constrained-decoder
  /// drift; this check fails fast before deserialization runs.)
  #[error("schema violation: required fields missing or null: {0:?}")]
  MissingFields(Vec<&'static str>),
  /// JSON parsed and had no missing fields, but every value was empty.
  #[error("structured response had no usable fields")]
  NoUsableFields,
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::OnceLock;

  /// `Task` is dyn-compatible with `Output` carrying through.
  #[test]
  fn task_is_dyn_compatible() {
    struct Dummy;
    impl Task for Dummy {
      type Output = ();
      fn prompt(&self) -> &str { "" }
      fn schema(&self) -> &Value {
        static V: OnceLock<Value> = OnceLock::new();
        V.get_or_init(|| Value::Null)
      }
      fn parse(&self, _raw: &str) -> Result<(), ParseError> { Ok(()) }
    }
    let _: Box<dyn Task<Output = ()>> = Box::new(Dummy);
    fn _assert_send_sync(_: &(impl Send + Sync + ?Sized)) {}
    _assert_send_sync(&*Box::new(Dummy) as &dyn Task<Output = ()>);
  }
}
