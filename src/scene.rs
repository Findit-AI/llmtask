//! `SceneAnalysis` — the canonical scene-analysis output type, shared
//! across `qwen` and `lfm` engines. Each engine's `SceneTask` constructs
//! values of this type; downstream consumers can pass `&SceneAnalysis`
//! references between engine outputs without conversion.

use smol_str::SmolStr;

/// Structured scene-level VLM output. Construct via an engine's
/// `SceneTask::parse` (the `Task::parse` impl) or, for tests/builders,
/// [`SceneAnalysis::new`]
/// followed by `with_*` chains. All fields are private; the accessor
/// surface follows the rest of the crate's `scenesdetect`-style getter /
/// `with_*` / `set_*` convention.
///
/// Detection-array fields (`subjects` / `objects` / `actions` / `mood` /
/// `lighting`) are `Vec<SmolStr>` — flat label lists, no per-detection
/// confidence. The previous design wrapped each label in a
/// `Detection { label, confidence }` and stamped a hardcoded `0.8`
/// per entry; round-14 dropped that placeholder because flat-confidence
/// is a no-op for both UX and search-time ranking, and VLM self-reported
/// confidence is poorly calibrated. If a downstream consumer needs
/// per-detection scoring, the practical sources are search-time
/// embedding similarity or scene-aggregation metrics — not VLM
/// self-report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneAnalysis {
  scene: SmolStr,
  description: SmolStr,
  subjects: Vec<SmolStr>,
  objects: Vec<SmolStr>,
  actions: Vec<SmolStr>,
  mood: Vec<SmolStr>,
  shot_type: SmolStr,
  lighting: Vec<SmolStr>,
  tags: Vec<SmolStr>,
}

impl SceneAnalysis {
  /// Construct an empty `SceneAnalysis` (all fields default).
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

  // --- mood ---

  /// Scene-level mood terms.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn mood(&self) -> &[SmolStr] {
    &self.mood
  }

  /// Builder-style setter for `mood`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_mood(mut self, val: Vec<SmolStr>) -> Self {
    self.mood = val;
    self
  }

  /// In-place setter for `mood`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_mood(&mut self, val: Vec<SmolStr>) -> &mut Self {
    self.mood = val;
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
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_is_empty() {
    let s = SceneAnalysis::new();
    assert!(s.scene().is_empty());
    assert!(s.description().is_empty());
    assert!(s.subjects().is_empty());
    assert_eq!(s, SceneAnalysis::default());
  }

  #[test]
  fn builder_chains() {
    let s = SceneAnalysis::new()
      .with_scene("airport")
      .with_description("travelers walking through terminal")
      .with_subjects(vec!["middle-aged woman".into(), "child".into()])
      .with_tags(vec!["airport".into(), "travel".into(), "indoor".into()]);
    assert_eq!(s.scene(), "airport");
    assert_eq!(s.subjects().len(), 2);
    assert_eq!(s.tags().len(), 3);
  }

  #[test]
  fn set_in_place() {
    let mut s = SceneAnalysis::new();
    s.set_scene("plaza");
    assert_eq!(s.scene(), "plaza");
  }
}
