//! `ImageAnalysis` — the canonical single-image VLM output type — and
//! [`ImageAnalysisTask`] (behind the `json` feature) — the canonical
//! `Task` implementation that produces it: prompt, JSON Schema, and a
//! resilient parser tolerant of constrained-decoder output drift.
//!
//! Both live here so every engine (`lfm`, `qwen3-vl`, and future
//! consumers such as `mediagraph`'s VLM node) runs the exact same
//! task instead of maintaining parallel copies. Before this module
//! absorbed it, `lfm` and `qwen3-vl` each carried their own
//! `ImageAnalysisTask` — byte-for-byte equivalent down to the
//! parser's resilience rules, both still pinned to `llmtask = "0.1"`
//! and its pre-rename nine-field `ImageAnalysis` (`mood`, no
//! `categories`). Retiring those local copies in favor of this one is
//! each engine's own follow-up, not done here.
//!
//! `ImageAnalysis` itself only needs `alloc`; `ImageAnalysisTask`
//! additionally needs `serde_json::Value` (the schema) and
//! [`crate::JsonParseError`], so it's gated on `json`.
//!
//! The type is named for what it holds (analysis of an image) rather
//! than the upstream use case (representing a video scene via a
//! keyframe). The `scene` field still carries the scene-category
//! label within the analysis.

use smol_str::SmolStr;
// Bring `Vec` into scope under both std (resolves via the
// `extern crate std`) and alloc-only (resolves via the
// `extern crate alloc as std` alias in lib.rs).
use std::vec::Vec;

/// Structured single-image VLM output. Construct via an engine's
/// `ImageAnalysisTask::parse` (the `Task::parse` impl) or, for
/// tests/builders, [`ImageAnalysis::new`]
/// followed by `with_*` chains. All fields are private; the accessor
/// surface follows the rest of the crate's `scenesdetect`-style getter /
/// `with_*` / `set_*` convention.
///
/// Detection-array fields (`subjects` / `objects` / `actions` / `emotion` /
/// `lighting`) are `Vec<SmolStr>` — flat label lists, no per-detection
/// confidence. Wrapping each label in a `Detection { label, confidence }`
/// would require a confidence source the VLM can't reliably provide —
/// VLM self-reported confidence is poorly calibrated, and a hardcoded
/// placeholder is a no-op for both UX and search-time ranking. If a
/// downstream consumer needs per-detection scoring, the practical
/// sources are search-time embedding similarity or scene-aggregation
/// metrics, not VLM self-report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageAnalysis {
  scene: SmolStr,
  description: SmolStr,
  subjects: Vec<SmolStr>,
  objects: Vec<SmolStr>,
  actions: Vec<SmolStr>,
  emotion: Vec<SmolStr>,
  shot_type: SmolStr,
  lighting: Vec<SmolStr>,
  tags: Vec<SmolStr>,
  categories: Vec<SmolStr>,
}

impl ImageAnalysis {
  /// Construct an empty `ImageAnalysis` (all fields default).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new() -> Self {
    Self::default()
  }

  // --- scene (SmolStr: empty = absent) ---

  /// Short scene category (e.g. `"office"`, `"airport arrivals hall"`).
  /// Returns the empty string when the model didn't classify the scene
  /// (SCENE_PROMPT instructs the model to use empty strings for unknown
  /// fields). Check `scene().is_empty()` to test for absence.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn scene(&self) -> &str {
    &self.scene
  }

  /// Builder-style setter for `scene`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_scene(mut self, val: impl Into<SmolStr>) -> Self {
    self.scene = val.into();
    self
  }

  /// In-place setter for `scene`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_scene(&mut self, val: impl Into<SmolStr>) -> &mut Self {
    self.scene = val.into();
    self
  }

  // --- description ---

  /// 1-2 sentence free-form scene description, or empty when the model
  /// produced no description (e.g., on a low-information frame).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn description(&self) -> &str {
    &self.description
  }

  /// Builder-style setter for `description`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_description(mut self, val: impl Into<SmolStr>) -> Self {
    self.description = val.into();
    self
  }

  /// In-place setter for `description`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_description(&mut self, val: impl Into<SmolStr>) -> &mut Self {
    self.description = val.into();
    self
  }

  // --- subjects ---

  /// Distinct people or animals visible in the scene.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn subjects(&self) -> &[SmolStr] {
    &self.subjects
  }

  /// Builder-style setter for `subjects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_subjects(mut self, val: Vec<SmolStr>) -> Self {
    self.subjects = val;
    self
  }

  /// In-place setter for `subjects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_subjects(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.subjects = val;
    self
  }

  // --- objects ---

  /// Notable, search-relevant objects.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn objects(&self) -> &[SmolStr] {
    &self.objects
  }

  /// Builder-style setter for `objects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_objects(mut self, val: Vec<SmolStr>) -> Self {
    self.objects = val;
    self
  }

  /// In-place setter for `objects`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_objects(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.objects = val;
    self
  }

  // --- actions ---

  /// Visible actions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn actions(&self) -> &[SmolStr] {
    &self.actions
  }

  /// Builder-style setter for `actions`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_actions(mut self, val: Vec<SmolStr>) -> Self {
    self.actions = val;
    self
  }

  /// In-place setter for `actions`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_actions(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.actions = val;
    self
  }

  // --- emotion ---

  /// Scene-level emotion terms.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn emotion(&self) -> &[SmolStr] {
    &self.emotion
  }

  /// Builder-style setter for `emotion`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_emotion(mut self, val: Vec<SmolStr>) -> Self {
    self.emotion = val;
    self
  }

  /// In-place setter for `emotion`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_emotion(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.emotion = val;
    self
  }

  // --- shot_type ---

  /// One short camera-shot label (e.g. `"wide shot"`, `"close-up"`),
  /// or empty when the model didn't pick one.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn shot_type(&self) -> &str {
    &self.shot_type
  }

  /// Builder-style setter for `shot_type`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_shot_type(mut self, val: impl Into<SmolStr>) -> Self {
    self.shot_type = val.into();
    self
  }

  /// In-place setter for `shot_type`. Pass an empty string to clear.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_shot_type(&mut self, val: impl Into<SmolStr>) -> &mut Self {
    self.shot_type = val.into();
    self
  }

  // --- lighting ---

  /// Lighting terms.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn lighting(&self) -> &[SmolStr] {
    &self.lighting
  }

  /// Builder-style setter for `lighting`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_lighting(mut self, val: Vec<SmolStr>) -> Self {
    self.lighting = val;
    self
  }

  /// In-place setter for `lighting`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_lighting(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.lighting = val;
    self
  }

  // --- tags ---

  /// 8-12 short English search tags in lowercase.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn tags(&self) -> &[SmolStr] {
    &self.tags
  }

  /// Builder-style setter for `tags`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_tags(mut self, val: Vec<SmolStr>) -> Self {
    self.tags = val;
    self
  }

  /// In-place setter for `tags`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_tags(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.tags = val;
    self
  }

  // --- categories ---

  /// Broad content-category labels, coarser-grained than `tags`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn categories(&self) -> &[SmolStr] {
    &self.categories
  }

  /// Builder-style setter for `categories`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_categories(mut self, val: Vec<SmolStr>) -> Self {
    self.categories = val;
    self
  }

  /// In-place setter for `categories`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_categories(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.categories = val;
    self
  }
}

// Tests run under both std (default) and `--no-default-features
// --features alloc`: `vec!` / `format!` are alloc macros, not
// std-only — bring them into scope explicitly so the std prelude
// isn't required.
#[cfg(test)]
mod tests {
  use super::*;
  use std::vec;

  #[test]
  fn default_is_empty() {
    let s = ImageAnalysis::new();
    assert!(s.scene().is_empty());
    assert!(s.description().is_empty());
    assert!(s.subjects().is_empty());
    assert_eq!(s, ImageAnalysis::default());
  }

  #[test]
  fn builder_chains() {
    let s = ImageAnalysis::new()
      .with_scene("airport")
      .with_description("travelers walking through terminal")
      .with_subjects(vec!["middle-aged woman".into(), "child".into()])
      .with_emotion(vec!["busy".into()])
      .with_tags(vec!["airport".into(), "travel".into(), "indoor".into()])
      .with_categories(vec!["travel".into()]);
    assert_eq!(s.scene(), "airport");
    assert_eq!(s.subjects().len(), 2);
    assert_eq!(s.emotion().len(), 1);
    assert_eq!(s.tags().len(), 3);
    assert_eq!(s.categories().len(), 1);
  }

  #[test]
  fn set_in_place() {
    let mut s = ImageAnalysis::new();
    s.set_scene("plaza");
    s.set_emotion(vec!["calm".into()]);
    s.set_categories(vec!["landscape".into()]);
    assert_eq!(s.scene(), "plaza");
    assert_eq!(s.emotion().len(), 1);
    assert_eq!(s.categories().len(), 1);
  }
}

// ===== ImageAnalysisTask (json feature only) =====

#[cfg(feature = "json")]
#[cfg_attr(docsrs, doc(cfg(feature = "json")))]
pub use image_analysis_task::ImageAnalysisTask;

/// The `ImageAnalysisTask` implementation. Grouped in its own private
/// module (mirroring `task::json`'s pattern) so the whole surface —
/// struct, impls, the prompt/schema consts, and the parser's helper
/// functions — shares one `#[cfg(feature = "json")]` gate on the `mod`
/// line instead of repeating it on every item.
#[cfg(feature = "json")]
mod image_analysis_task {
  use serde_json::{Value, json};
  use smol_str::SmolStr;
  use std::vec::Vec;

  use super::ImageAnalysis;
  use crate::{
    grammar::Grammar,
    task::{JsonParseError, Task},
  };

  /// The image-analysis prompt. Ported verbatim (structure and
  /// resilience-relevant wording) from the two engine-side copies
  /// this task replaces — `lfm/src/image_analysis.rs` and
  /// `qwen3-vl/src/image_analysis.rs`, themselves a verbatim port of
  /// the legacy `findit-qwen` service's prompt — with three changes:
  /// `mood` renamed to `emotion` (matching `ImageAnalysis` 0.2's
  /// rename), a `categories` field added, and the per-field word-count
  /// guidance (which varied field-to-field: "2-6 words", "1-4 words",
  /// "single-word or two-word") collapsed into one sealed discipline
  /// rule applied uniformly to every array field — see the "Rules"
  /// section below.
  ///
  /// Intentionally written WITHOUT enumerated example values. Both of
  /// today's `llmtask` consumers (`lfm`, `qwen3-vl`) run on mistralrs,
  /// whose deterministic/greedy sampling applies a `presence_penalty`
  /// over the full context (prompt + generated tokens) — any
  /// value-token the prompt enumerates as an example (e.g. "office",
  /// "wide shot", "birthday cake with candles") gets a negative logit
  /// shift before the model emits anything, biasing generation away
  /// from that exact term even when a scene legitimately matches it.
  /// `llmtask` itself has no engine dependency and doesn't assume this
  /// mechanism is universal, but avoiding enumerated examples costs
  /// nothing and is free insurance for any engine that behaves this
  /// way — so format guidance stays in descriptive constraints (word
  /// counts, lowercase, singular) instead of `e.g. "..."` examples.
  const IMAGE_ANALYSIS_PROMPT: &str = r#"Analyze the following video keyframes (in chronological order) from a single scene.

Return ONLY a valid JSON object with exactly these fields:
scene: a single short scene-category label in lowercase English, 1-3 words, no full sentence.
description: 1-2 concise sentences in English describing the stable visual facts across the scene. Cover who is present, what they are doing, the setting, and the overall mood or visual style. If readable on-screen text appears, quote that text first, then continue the description.
subjects: array of distinct people or animals with visible distinguishing features.
objects: array of notable, search-relevant objects.
actions: array of visible actions.
emotion: array of descriptors for the scene's overall emotional tone.
shot_type: a single short camera-shot label in lowercase English, 1-2 words (a cinematography term).
lighting: array of lighting descriptors.
tags: array of 8-12 short English search tags. Prefer high-confidence search terms, complementary synonyms, style words, and culture-specific terms only when visually supported.
categories: array of broad content categories, coarser-grained than tags.

Rules:
- Use only information supported by the keyframes.
- Prefer concrete visual facts over speculation.
- Every array field's elements are lowercase, singular, 1-3-word phrases, with no trailing punctuation.
- Keep arrays deduplicated.
- Use empty arrays or empty strings when a field is unknown.
- Do not return markdown or any text outside the JSON object."#;

  /// Names of the nine REQUIRED schema fields, in `ImageAnalysis`'s
  /// field order. `categories` is deliberately absent: it's the one
  /// field this Task marks optional (see the module docs and
  /// [`build_schema`]) — an object missing `categories` entirely still
  /// parses, defaulting to an empty list. The other nine were already
  /// required in both engine copies this task replaces (`tags`
  /// included), so that convention carries forward unchanged; extending
  /// it to `categories` too was the one available choice this census
  /// didn't dictate, and "optional, defaults empty" was chosen because
  /// `categories` is the newly-added field with no prior engine
  /// producing it — treating its total absence as a schema violation
  /// would reject every response from a still-unmigrated prompt/engine
  /// pairing instead of just leaving the list empty.
  const REQUIRED_FIELDS: &[&str] = &[
    "scene",
    "description",
    "subjects",
    "objects",
    "actions",
    "emotion",
    "shot_type",
    "lighting",
    "tags",
  ];

  /// The image-analysis task. Construct via [`ImageAnalysisTask::new`].
  ///
  /// # Example
  ///
  /// ```
  /// use llmtask::{Task, image_analysis::ImageAnalysisTask};
  ///
  /// let task = ImageAnalysisTask::new();
  /// assert!(!task.prompt().is_empty());
  /// assert!(task.grammar().is_json_schema());
  ///
  /// let raw = r#"{
  ///   "scene": "office", "description": "two people talking",
  ///   "subjects": ["person"], "objects": [], "actions": [],
  ///   "emotion": [], "shot_type": "wide shot", "lighting": [],
  ///   "tags": ["office", "meeting"]
  /// }"#;
  /// let analysis = task.parse(raw).expect("parse should succeed");
  /// assert_eq!(analysis.scene(), "office");
  /// assert_eq!(analysis.tags().len(), 2);
  /// ```
  #[derive(Clone)]
  pub struct ImageAnalysisTask {
    schema: Value,
    accept_empty: bool,
  }

  impl ImageAnalysisTask {
    /// Construct with `accept_empty = false` (a payload that lacks the
    /// required indexable content — `description` AND `tags` both
    /// populated, OR at least one of the substantive detection buckets
    /// `subjects` / `objects` / `actions` non-empty — is treated as a
    /// model regression and rejected; see [`Self::with_accept_empty`]
    /// for the full predicate and the opt-in alternative).
    pub fn new() -> Self {
      Self {
        schema: build_schema(),
        accept_empty: false,
      }
    }

    /// Returns whether the parser accepts payloads that lack the
    /// required indexable content (`description` AND `tags` both
    /// non-empty). See [`Self::with_accept_empty`] for the trade-off.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn accept_empty(&self) -> bool {
      self.accept_empty
    }

    /// Builder-style setter for `accept_empty`.
    ///
    /// When `false` (default), the parser rejects payloads that lack
    /// the required indexable content as [`JsonParseError::NoUsableFields`].
    /// The composite threshold accepts a payload when **either**:
    ///
    /// - `description` AND `tags` are both populated (the prose +
    ///   keyword path), OR
    /// - at least one of the **substantive** detection buckets —
    ///   `subjects`, `objects`, or `actions` — is non-empty (the
    ///   substantive-detection path; preserves who/what/where search
    ///   metadata even when the model fails to summarize).
    ///
    /// Style/attribute buckets (`emotion`, `lighting`, `categories`)
    /// and single-label fields (`scene`, `shot_type`) are intentionally
    /// NOT in the substantive path. A payload like `lighting: ["natural
    /// light"]` or `emotion: ["calm"]` alone (description and tags
    /// empty, no substantive detections) is more often a regression
    /// than a legitimate weak-but-real scene; rejecting it surfaces the
    /// failure instead of writing a single-attribute stub to the search
    /// index.
    ///
    /// When `true`, the parser bypasses the indexable-content check and
    /// returns whatever round-trips through the schema.
    /// `IMAGE_ANALYSIS_PROMPT` explicitly tells the model to "Use empty
    /// arrays or empty strings when a field is unknown", so on truly
    /// low-information frames (blank, fade-to-black, plain color)
    /// compliant model output can legitimately be sparse or
    /// fully-empty. Use this knob if your pipeline distinguishes
    /// "low-information scene" from "no useful content" via something
    /// other than the parser.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn with_accept_empty(mut self, val: bool) -> Self {
      self.accept_empty = val;
      self
    }

    /// In-place setter for `accept_empty`. See
    /// [`Self::with_accept_empty`] for the trade-off.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub const fn set_accept_empty(&mut self, val: bool) -> &mut Self {
      self.accept_empty = val;
      self
    }
  }

  impl Default for ImageAnalysisTask {
    fn default() -> Self {
      Self::new()
    }
  }

  impl Task for ImageAnalysisTask {
    type Output = ImageAnalysis;
    type Value = Value;
    type ParseError = JsonParseError;

    fn prompt(&self) -> &str {
      IMAGE_ANALYSIS_PROMPT
    }

    fn schema(&self) -> &Value {
      &self.schema
    }

    fn grammar(&self) -> Grammar {
      // Clone the cached JSON Schema once per call. Cheap relative to
      // constraint compilation, and matches the `schema()` contract.
      Grammar::JsonSchema(self.schema.clone())
    }

    fn parse(&self, raw: &str) -> Result<Self::Output, JsonParseError> {
      let value: Value = serde_json::from_str(raw.trim())?;
      let Some(object) = value.as_object() else {
        // Not a JSON object at all: by definition every required field
        // is absent. Naming all nine via `MissingFields` is more
        // informative than a generic "expected top-level object"
        // error, and needs nothing beyond `serde_json` to construct.
        return Err(JsonParseError::MissingFields(REQUIRED_FIELDS.to_vec()));
      };
      let unusable = unusable_fields(object);
      if !unusable.is_empty() {
        return Err(JsonParseError::MissingFields(unusable));
      }
      // Every field `unusable_fields` didn't flag is now known to carry
      // a JSON shape its `extract_*` helper can consume, so extraction
      // itself is infallible from here.
      let result = ImageAnalysis::new()
        .with_scene(extract_label(object, "scene"))
        .with_description(extract_label(object, "description"))
        .with_subjects(extract_detection_array(object, "subjects"))
        .with_objects(extract_detection_array(object, "objects"))
        .with_actions(extract_detection_array(object, "actions"))
        .with_emotion(extract_detection_array(object, "emotion"))
        .with_shot_type(extract_shot_type(object))
        .with_lighting(extract_detection_array(object, "lighting"))
        .with_tags(extract_tags(object))
        .with_categories(extract_detection_array(object, "categories"));
      // Indexable-content gate. IMAGE_ANALYSIS_PROMPT instructs the model
      // to "Use empty arrays or empty strings when a field is unknown",
      // so a truly compliant response on a blank/fade-to-black frame can
      // be partially or fully empty. But a decoder/model regression on a
      // normal frame also produces sparse output, and silently
      // overwriting real search metadata with that is worse than
      // failing. See `with_accept_empty` for the full predicate.
      if !self.accept_empty && lacks_indexable_content(&result) {
        return Err(JsonParseError::NoUsableFields);
      }
      Ok(result)
    }
  }

  /// JSON Schema for [`ImageAnalysisTask`]. `categories` is the only
  /// `properties` entry absent from `required` — see [`REQUIRED_FIELDS`].
  /// `additionalProperties: false` rejects any field outside this list.
  fn build_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "scene": { "type": "string" },
            "description": { "type": "string" },
            "subjects": { "type": "array", "items": { "type": "string" } },
            "objects": { "type": "array", "items": { "type": "string" } },
            "actions": { "type": "array", "items": { "type": "string" } },
            "emotion": { "type": "array", "items": { "type": "string" } },
            "shot_type": { "type": "string" },
            "lighting": { "type": "array", "items": { "type": "string" } },
            "tags": { "type": "array", "items": { "type": "string" } },
            "categories": { "type": "array", "items": { "type": "string" } }
        },
        "required": REQUIRED_FIELDS,
        "additionalProperties": false
    })
  }

  /// Names the fields in `object` that are unusable: a required field
  /// (per [`REQUIRED_FIELDS`]) absent or JSON `null`, or ANY listed
  /// field — required or the optional `categories` — present with a
  /// JSON type its shape can't hold (e.g. a number where a string or
  /// array of strings is expected).
  ///
  /// Folding "wrong type" into the same named-field list as
  /// "missing"/"null" is a deliberate hardening over the two engine
  /// copies this task replaces: those only pre-checked null/absent,
  /// then let a wrong-type field fall through to `serde`'s untagged-enum
  /// deserialization, which fails with a generic "data did not match
  /// any variant" message that never names the offending field. Every
  /// unusable field is now caught in this one pass and named.
  fn unusable_fields(object: &serde_json::Map<String, Value>) -> Vec<&'static str> {
    let mut unusable = Vec::new();
    for &field in REQUIRED_FIELDS {
      let well_shaped = match object.get(field) {
        None | Some(Value::Null) => false,
        Some(v) => field_is_well_shaped(field, v),
      };
      if !well_shaped {
        unusable.push(field);
      }
    }
    // `categories` is optional: absent or null is fine (extraction
    // defaults it to an empty list). A *present* value still has to be
    // shape-valid — silently dropping a wrong-type `categories` would
    // hide the same drift `unusable_fields` catches for every other
    // field.
    if let Some(v) = object.get("categories")
      && !matches!(v, Value::Null)
      && !field_is_well_shaped("categories", v)
    {
      unusable.push("categories");
    }
    unusable
  }

  /// `true` iff `value`'s JSON type is one `field`'s extractor can
  /// consume. Presence/null is checked separately by the caller (`null`
  /// means different things for required vs. optional fields); this
  /// only judges the shape of a value that's actually present.
  fn field_is_well_shaped(field: &str, value: &Value) -> bool {
    match field {
      // `scene` / `description`: bare string only — no array-wrapping
      // tolerance (unlike `shot_type` below).
      "scene" | "description" => value.is_string(),
      // `shot_type`: bare string, or the sole element of a one-element
      // array (tolerates a constrained decoder that wraps a scalar).
      "shot_type" => match value {
        Value::String(_) => true,
        Value::Array(items) => items.len() == 1 && items[0].is_string(),
        _ => false,
      },
      // Every other schema field (subjects/objects/actions/emotion/
      // lighting/tags/categories) is array-shaped: a JSON array of
      // strings, or a bare string tolerated as a single-element array.
      _ => match value {
        Value::String(_) => true,
        Value::Array(items) => items.iter().all(Value::is_string),
        _ => false,
      },
    }
  }

  /// Extracts the trimmed string at `field` (`scene` / `description`),
  /// or an empty `SmolStr` when absent/null. Caller must have already
  /// confirmed via [`unusable_fields`] that a present value is a JSON
  /// string.
  fn extract_label(object: &serde_json::Map<String, Value>, field: &str) -> SmolStr {
    match object.get(field) {
      Some(Value::String(s)) => SmolStr::new(s.trim()),
      _ => SmolStr::default(),
    }
  }

  /// Extracts `shot_type`: a bare string, or the sole element of a
  /// one-element array. Caller must have already confirmed the shape.
  fn extract_shot_type(object: &serde_json::Map<String, Value>) -> SmolStr {
    match object.get("shot_type") {
      Some(Value::String(s)) => SmolStr::new(s.trim()),
      Some(Value::Array(items)) => match items.first() {
        Some(Value::String(s)) => SmolStr::new(s.trim()),
        _ => SmolStr::default(),
      },
      _ => SmolStr::default(),
    }
  }

  /// Extracts a detection-style label array (`subjects`, `objects`,
  /// `actions`, `emotion`, `lighting`, `categories`): a JSON array
  /// trims and dedupes each element verbatim — no comma-splitting,
  /// because detection labels can themselves contain commas (e.g. "red,
  /// white, and blue flag") — or a single JSON string wraps as one
  /// label. Absent/null yields an empty list. Caller must have already
  /// confirmed the shape via [`unusable_fields`].
  fn extract_detection_array(object: &serde_json::Map<String, Value>, field: &str) -> Vec<SmolStr> {
    let mut values = Vec::new();
    match object.get(field) {
      Some(Value::String(s)) => push_label(&mut values, s),
      Some(Value::Array(items)) => {
        for item in items {
          if let Value::String(s) = item {
            push_label(&mut values, s);
          }
        }
      }
      _ => {}
    }
    values
  }

  /// Extracts `tags`: like [`extract_detection_array`], but a JSON
  /// string form is additionally split on commas / semicolons /
  /// newlines — tag-list drift (model dropped the array around a flat
  /// comma-separated string) is the historically common case for this
  /// specific field. The array form is never split (a tag like "july 4,
  /// 2026" must stay one entry).
  fn extract_tags(object: &serde_json::Map<String, Value>) -> Vec<SmolStr> {
    let mut values = Vec::new();
    match object.get("tags") {
      Some(Value::String(s)) => {
        for part in s.split([',', ';', '\n']) {
          push_label(&mut values, part);
        }
      }
      Some(Value::Array(items)) => {
        for item in items {
          if let Value::String(s) = item {
            push_label(&mut values, s);
          }
        }
      }
      _ => {}
    }
    values
  }

  /// Trims `raw`; pushes it onto `values` if non-empty and not already
  /// present (verbatim-case dedup — case-folding happens nowhere in
  /// this parser; see `parse_does_not_lowercase_labels` in the tests
  /// below).
  fn push_label(values: &mut Vec<SmolStr>, raw: &str) {
    let trimmed = raw.trim();
    if !trimmed.is_empty() && !values.iter().any(|existing| existing.as_str() == trimmed) {
      values.push(SmolStr::new(trimmed));
    }
  }

  /// `true` if `analysis` lacks the minimum content required to produce
  /// a useful indexing record. See [`ImageAnalysisTask::with_accept_empty`]
  /// for the full predicate this implements.
  fn lacks_indexable_content(analysis: &ImageAnalysis) -> bool {
    let has_prose_and_keywords = !analysis.description().is_empty() && !analysis.tags().is_empty();
    let has_substantive_detection = !analysis.subjects().is_empty()
      || !analysis.objects().is_empty()
      || !analysis.actions().is_empty();
    !has_prose_and_keywords && !has_substantive_detection
  }

  #[cfg(test)]
  mod tests {
    use super::*;

    // ===== ported from both engine copies (lfm + qwen3-vl; the two
    // copies' test suites were substantively identical) =====

    /// `IMAGE_ANALYSIS_PROMPT` must not enumerate value tokens — see
    /// the const's doc comment for the presence-penalty rationale.
    #[test]
    fn scene_prompt_does_not_enumerate_value_tokens() {
      let prompt_lower = IMAGE_ANALYSIS_PROMPT.to_lowercase();
      let banned_tokens = [
        "stage performance",
        "middle-aged man",
        "golden retriever",
        "birthday cake",
        "vintage red sports car",
        "cutting cake",
        "taking photos",
        "wide shot",
        "close-up",
        "medium shot",
        "over-the-shoulder",
        "celebratory",
        "natural light",
        "low light",
        "backlit",
      ];
      for token in banned_tokens {
        assert!(
          !prompt_lower.contains(&token.to_lowercase()),
          "IMAGE_ANALYSIS_PROMPT must not enumerate value token {token:?} \
           (prompt-vocabulary tokens get a negative logit shift on \
           mistralrs-backed engines in deterministic mode); use \
           descriptive format guidance instead of `e.g. \"...\"` examples"
        );
      }
    }

    #[test]
    fn parse_valid_json() {
      let json = r#"{"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"emotion":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"],"categories":["nature"]}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(result.scene(), "beach");
      assert_eq!(result.description(), "Sunset over the ocean");
      assert_eq!(result.emotion().len(), 1);
      assert_eq!(result.subjects().len(), 1);
      assert_eq!(result.categories(), &[SmolStr::from("nature")][..]);
    }

    #[test]
    fn reject_json_with_wrapper_text() {
      let text =
        "Here is the analysis:\n{\"scene\":\"office\",\"description\":\"People working\"}\nDone.";
      let task = ImageAnalysisTask::new();
      assert!(task.parse(text).is_err());
    }

    #[test]
    fn reject_plain_text_output() {
      let text = "A beautiful sunset over the ocean.";
      let task = ImageAnalysisTask::new();
      assert!(task.parse(text).is_err());
    }

    /// A fenced markdown block around an otherwise-valid JSON object is
    /// rejected the same way as prose-wrapped JSON: this parser expects
    /// `raw.trim()` to already be a bare JSON object, and does not strip
    /// wrapping of any kind (fences included).
    #[test]
    fn reject_fenced_json() {
      let text = "```json\n{\"scene\":\"office\",\"description\":\"People working\"}\n```";
      let task = ImageAnalysisTask::new();
      assert!(task.parse(text).is_err());
    }

    #[test]
    fn parse_comma_separated_tag_string() {
      let json = r#"{"scene":"stage performance","description":"A singer on stage","subjects":[],"objects":["microphone"],"actions":["singing"],"emotion":["energetic"],"shot_type":"medium shot","lighting":["spotlight"],"tags":"concert, live music, spotlight"}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(
        result.tags(),
        &[
          SmolStr::from("concert"),
          SmolStr::from("live music"),
          SmolStr::from("spotlight"),
        ][..]
      );
    }

    #[test]
    fn reject_empty_json_payload() {
      let task = ImageAnalysisTask::new();
      assert!(task.parse("{}").is_err());
    }

    #[test]
    fn reject_unknown_json_fields() {
      let json = r#"{"description":"A singer on stage","extra":"unexpected"}"#;
      let task = ImageAnalysisTask::new();
      assert!(task.parse(json).is_err());
    }

    #[test]
    fn reject_missing_required_fields() {
      let json = r#"{"description":"A singer on stage","tags":["concert"]}"#;
      let task = ImageAnalysisTask::new();
      assert!(task.parse(json).is_err());
    }

    #[test]
    fn parse_array_form_subjects() {
      let json_list = r#"{"scene":"x","description":"y","subjects":["a","b"],"objects":[],"actions":[],"emotion":[],"shot_type":"x","lighting":[],"tags":["t"]}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json_list).expect("list-form parse");
      assert_eq!(result.subjects().len(), 2);
      assert_eq!(result.subjects()[0], "a");
      assert_eq!(result.subjects()[1], "b");
    }

    #[test]
    fn subjects_string_form_treated_as_single_label() {
      let json = r#"{"scene":"x","description":"y","subjects":"middle-aged man, in red jacket","objects":[],"actions":[],"emotion":[],"shot_type":"x","lighting":[],"tags":["t"]}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect("string-form parse");
      assert_eq!(
        result.subjects().len(),
        1,
        "string-form must wrap as a single label, not comma-split"
      );
      assert_eq!(result.subjects()[0], "middle-aged man, in red jacket");
    }

    #[test]
    fn reject_all_required_fields_empty_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject all-empty payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn accept_all_required_fields_empty_payload_when_opted_in() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new().with_accept_empty(true);
      let result = task
        .parse(json)
        .expect("opt-in must accept the all-empty payload");
      assert!(result.scene().is_empty());
      assert!(result.description().is_empty());
      assert!(result.subjects().is_empty());
      assert!(result.objects().is_empty());
      assert!(result.actions().is_empty());
      assert!(result.emotion().is_empty());
      assert!(result.shot_type().is_empty());
      assert!(result.lighting().is_empty());
      assert!(result.tags().is_empty());
      assert!(result.categories().is_empty());
    }

    #[test]
    fn reject_tags_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": ["concert", "live music"]
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject tags-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_scene_only_payload_by_default() {
      let json = r#"{
            "scene": "office",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject scene-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_description_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "People working in an office",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject description-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn accept_minimal_indexable_payload() {
      let json = r#"{
            "scene": "",
            "description": "Two people talking",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": ["conversation"]
          }"#;
      let task = ImageAnalysisTask::new();
      let result = task
        .parse(json)
        .expect("description+tags must clear the indexable threshold");
      assert_eq!(result.description(), "Two people talking");
      assert_eq!(result.tags(), &[SmolStr::from("conversation")][..]);
      assert!(result.subjects().is_empty());
      assert!(result.objects().is_empty());
      assert!(result.scene().is_empty());
    }

    #[test]
    fn accept_detection_rich_payload_with_empty_description_and_tags() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": ["middle-aged woman in red dress"],
            "objects": ["wedding cake"],
            "actions": ["cutting cake"],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect(
        "detection-rich payload must clear the indexable threshold via \
               the detection-bucket path even when description+tags are empty",
      );
      assert_eq!(result.subjects().len(), 1);
      assert_eq!(result.objects().len(), 1);
      assert_eq!(result.actions().len(), 1);
      assert!(result.description().is_empty());
      assert!(result.tags().is_empty());
    }

    #[test]
    fn accept_subjects_only_payload() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": ["a single subject label"],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let result = task
        .parse(json)
        .expect("subjects-only must clear the indexable threshold");
      assert_eq!(result.subjects().len(), 1);
    }

    #[test]
    fn accept_objects_only_payload() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": ["a single object label"],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let result = task
        .parse(json)
        .expect("objects-only must clear the indexable threshold");
      assert_eq!(result.objects().len(), 1);
    }

    #[test]
    fn accept_actions_only_payload() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": ["a single action label"],
            "emotion": [],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let result = task
        .parse(json)
        .expect("actions-only must clear the indexable threshold");
      assert_eq!(result.actions().len(), 1);
    }

    #[test]
    fn reject_emotion_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": ["calm"],
            "shot_type": "",
            "lighting": [],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject emotion-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_lighting_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "",
            "lighting": ["natural light"],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("default ImageAnalysisTask must reject lighting-only payload");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_attribute_only_payload_by_default() {
      let json = r#"{
            "scene": "",
            "description": "",
            "subjects": [],
            "objects": [],
            "actions": [],
            "emotion": ["tense"],
            "shot_type": "",
            "lighting": ["low light"],
            "tags": []
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("style-attribute-only payload must reject regardless of bucket count");
      assert!(
        matches!(err, JsonParseError::NoUsableFields),
        "expected NoUsableFields, got {err:?}"
      );
    }

    #[test]
    fn reject_null_required_array() {
      let json = r#"{
            "scene": "office",
            "description": "people working",
            "subjects": null,
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "wide",
            "lighting": [],
            "tags": ["work"]
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("null required field must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => {
          assert!(
            fields.contains(&"subjects"),
            "expected 'subjects' in MissingFields, got {fields:?}"
          );
        }
        other => panic!("expected MissingFields, got {other:?}"),
      }
    }

    #[test]
    fn reject_null_required_string() {
      let json = r#"{
            "scene": null,
            "description": "people working",
            "subjects": ["person"],
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "wide",
            "lighting": [],
            "tags": ["work"]
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("null required field must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => {
          assert!(
            fields.contains(&"scene"),
            "expected 'scene' in MissingFields, got {fields:?}"
          );
        }
        other => panic!("expected MissingFields, got {other:?}"),
      }
    }

    #[test]
    fn reject_multiple_null_required_fields() {
      let json = r#"{
            "scene": null,
            "description": null,
            "subjects": null,
            "objects": [],
            "actions": [],
            "emotion": [],
            "shot_type": "wide",
            "lighting": [],
            "tags": ["work"]
          }"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("null required fields must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => {
          assert!(fields.contains(&"scene"), "missing 'scene' in {fields:?}");
          assert!(
            fields.contains(&"description"),
            "missing 'description' in {fields:?}"
          );
          assert!(
            fields.contains(&"subjects"),
            "missing 'subjects' in {fields:?}"
          );
        }
        other => panic!("expected MissingFields, got {other:?}"),
      }
    }

    #[test]
    fn array_elements_are_not_comma_split() {
      let json = r#"{
            "scene": "patriotic event",
            "description": "Flag display",
            "subjects": ["middle-aged man, in red jacket"],
            "objects": ["red, white, and blue flag", "birthday cake with candles, balloons"],
            "actions": ["waving"],
            "emotion": ["festive"],
            "shot_type": "wide shot",
            "lighting": ["natural, dramatic backlight"],
            "tags": ["july 4, 2026"]
          }"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(result.subjects().len(), 1);
      assert_eq!(result.subjects()[0], "middle-aged man, in red jacket");
      assert_eq!(result.objects().len(), 2);
      assert_eq!(result.objects()[0], "red, white, and blue flag");
      assert_eq!(result.objects()[1], "birthday cake with candles, balloons");
      assert_eq!(result.lighting().len(), 1);
      assert_eq!(result.lighting()[0], "natural, dramatic backlight");
      assert_eq!(result.tags().len(), 1);
      assert_eq!(result.tags()[0].as_str(), "july 4, 2026");
    }

    #[test]
    fn parse_shot_type_list_form() {
      // shot_type accepts the list form `["wide shot"]` (one element).
      let json_one = r#"{"scene":"x","description":"y","subjects":[],"objects":[],"actions":[],"emotion":[],"shot_type":["wide shot"],"lighting":[],"tags":["t"]}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json_one).expect("single-element list parse");
      assert_eq!(result.shot_type(), "wide shot");
    }

    /// A multi-element `shot_type` array is now a named
    /// `MissingFields(["shot_type"])` error rather than the generic
    /// serde message the two engine copies produced ("expected a
    /// single shot_type label, got multiple values", wrapped as
    /// `JsonParseError::Json`) — folded into `unusable_fields` like
    /// every other shape violation.
    #[test]
    fn reject_shot_type_multi_element_array_with_named_error() {
      let json_many = r#"{"scene":"x","description":"y","subjects":[],"objects":[],"actions":[],"emotion":[],"shot_type":["wide","close-up"],"lighting":[],"tags":["t"]}"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json_many)
        .expect_err("multi-element shot_type array must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => assert!(fields.contains(&"shot_type")),
        other => panic!("expected MissingFields naming shot_type, got {other:?}"),
      }
    }

    // ===== new for the llmtask 0.3 merge: categories, the sealed label
    // discipline, and the wrong-type / non-object resilience cases the
    // brief calls for =====

    #[test]
    fn categories_absent_defaults_to_empty() {
      // `categories` omitted entirely (not merely empty) — it's the one
      // optional field; this must not raise MissingFields.
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"]}"#;
      let task = ImageAnalysisTask::new();
      let result = task
        .parse(json)
        .expect("categories is optional; absence must not fail parsing");
      assert!(result.categories().is_empty());
    }

    #[test]
    fn categories_present_populates() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"],"categories":["business","corporate"]}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(
        result.categories(),
        &[SmolStr::from("business"), SmolStr::from("corporate")][..]
      );
    }

    #[test]
    fn reject_wrong_type_required_field_with_named_error() {
      let json = r#"{"scene":"office","description":"people working","subjects":42,"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"]}"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("wrong-type required field must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => assert!(
          fields.contains(&"subjects"),
          "expected 'subjects' named in {fields:?}"
        ),
        other => panic!("expected MissingFields naming subjects, got {other:?}"),
      }
    }

    #[test]
    fn reject_wrong_type_optional_field_with_named_error() {
      let json = r#"{"scene":"office","description":"people working","subjects":["person"],"objects":[],"actions":[],"emotion":[],"shot_type":"wide","lighting":[],"tags":["work"],"categories":42}"#;
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(json)
        .expect_err("wrong-type optional field must still be rejected, not silently dropped");
      match err {
        JsonParseError::MissingFields(fields) => assert!(
          fields.contains(&"categories"),
          "expected 'categories' named in {fields:?}"
        ),
        other => panic!("expected MissingFields naming categories, got {other:?}"),
      }
    }

    #[test]
    fn reject_non_object_top_level_value() {
      let task = ImageAnalysisTask::new();
      let err = task
        .parse(r#"["not", "an", "object"]"#)
        .expect_err("a JSON array at the top level must be rejected");
      match err {
        JsonParseError::MissingFields(fields) => {
          assert_eq!(fields.len(), REQUIRED_FIELDS.len());
          for required in REQUIRED_FIELDS {
            assert!(fields.contains(required));
          }
        }
        other => panic!("expected MissingFields naming every required field, got {other:?}"),
      }
    }

    /// Sealed discipline (see `IMAGE_ANALYSIS_PROMPT`'s "Rules"
    /// section): every array field's elements must be lowercase,
    /// singular, 1-3-word phrases with no trailing punctuation — as a
    /// PROMPT instruction. This test pins the prompt text; the parser
    /// itself must never enforce it by transforming output (see
    /// `parse_does_not_lowercase_labels` below) — that's the model's
    /// job, not the parser's.
    #[test]
    fn label_array_discipline_is_stated_in_prompt() {
      assert!(
        IMAGE_ANALYSIS_PROMPT.contains("lowercase, singular, 1-3-word phrases"),
        "prompt must instruct the sealed per-array label discipline verbatim"
      );
      assert!(
        IMAGE_ANALYSIS_PROMPT.contains("no trailing punctuation"),
        "prompt must instruct no trailing punctuation on array elements"
      );
    }

    /// Verbatim law: the parser must NEVER lowercase (or otherwise
    /// case-fold) label text at parse time, even though the prompt asks
    /// the model for lowercase output. The discipline is prompt-only;
    /// enforcing it in the parser would silently mask a
    /// non-compliant model instead of surfacing the drift.
    #[test]
    fn parse_does_not_lowercase_labels() {
      let json = r#"{"scene":"Office","description":"Desc","subjects":["MidCase Person"],"objects":[],"actions":[],"emotion":[],"shot_type":"Wide Shot","lighting":[],"tags":["MixedCase"]}"#;
      let task = ImageAnalysisTask::new();
      let result = task.parse(json).expect("parse should succeed");
      assert_eq!(result.scene(), "Office");
      assert_eq!(result.shot_type(), "Wide Shot");
      assert_eq!(result.subjects()[0], "MidCase Person");
      assert_eq!(result.tags()[0], "MixedCase");
    }

    /// The grammar/regex face constrains array items to plain strings
    /// only — no `pattern` regex forcing lowercase/word-count at the
    /// schema level. The discipline lives in the prompt (pinned above),
    /// never in the constrained-decoding grammar.
    #[test]
    fn array_schema_items_are_plain_strings_only() {
      let task = ImageAnalysisTask::new();
      let schema = task.schema();
      for field in [
        "subjects",
        "objects",
        "actions",
        "emotion",
        "lighting",
        "tags",
        "categories",
      ] {
        let items = &schema["properties"][field]["items"];
        assert_eq!(
          items["type"], "string",
          "field {field} items must be plain strings"
        );
        assert!(
          items.get("pattern").is_none(),
          "field {field} items must not carry a regex pattern constraint"
        );
      }
    }

    // ===== Task contract round-trip =====

    #[test]
    fn prompt_is_non_empty() {
      assert!(!ImageAnalysisTask::new().prompt().is_empty());
    }

    #[test]
    fn schema_is_valid_json_schema_shape() {
      let task = ImageAnalysisTask::new();
      let schema = task.schema();
      assert_eq!(schema["type"], "object");
      assert_eq!(schema["additionalProperties"], false);
      let properties = schema["properties"]
        .as_object()
        .expect("properties must be an object");
      assert_eq!(properties.len(), 10, "ten ImageAnalysis fields");
      let required = schema["required"]
        .as_array()
        .expect("required must be an array");
      assert_eq!(required.len(), REQUIRED_FIELDS.len());
      assert!(
        !required.iter().any(|v| v == "categories"),
        "categories must be optional, not required"
      );
    }

    #[test]
    fn grammar_wraps_the_cached_schema() {
      let task = ImageAnalysisTask::new();
      let grammar = task.grammar();
      assert!(grammar.is_json_schema());
      assert_eq!(grammar.as_json_schema(), Some(task.schema()));
    }
  }
}
