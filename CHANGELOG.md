# Changelog

## [Unreleased]

## [0.3.0] - 2026-08-31

### Added
- `ImageAnalysisTask` (`llmtask::image_analysis::ImageAnalysisTask`), behind
  the existing `json` feature: the canonical `Task` implementation for
  `ImageAnalysis` — prompt, JSON Schema, and a resilient parser — merged up
  from the two byte-for-byte-equivalent engine copies (`lfm/src/image_analysis.rs`
  and `qwen3-vl/src/image_analysis.rs`, both still pinned to `llmtask = "0.1"`
  and its pre-rename nine-field `ImageAnalysis`). Every engine now runs the
  same task instead of maintaining parallel copies; retiring the two
  downstream copies is each engine's own follow-up.
- Sealed label-prompt discipline: `IMAGE_ANALYSIS_PROMPT` now instructs, for
  every array field, lowercase/singular/1-3-word phrases with no trailing
  punctuation (previously the per-field word-count guidance varied field to
  field). The parser never lowercases at parse time — the discipline is
  prompt-only, pinned by `label_array_discipline_is_stated_in_prompt` and
  `parse_does_not_lowercase_labels`.
- `categories` joins the JSON Schema's `properties` as the merged task's one
  **optional** field (absent from `required`) — the newly-added
  `ImageAnalysis` field with no prior engine producing it; the other nine
  fields (`tags` included) stay required, matching both source copies.

### Changed
- `JsonParseError::MissingFields` now also names a listed field (required or
  the optional `categories`) that's present with a JSON type its schema entry
  can't satisfy (e.g. a number where a string or array of strings is
  expected) — previously only absent/null required fields were named, and a
  wrong-type field fell through to a generic, unnamed `serde` error.

## [0.2.0] - 2026-08-31

### Added
- `ImageAnalysis::categories: Vec<SmolStr>` field, with `categories()` /
  `with_categories()` / `set_categories()` accessors — broad content-category
  labels, coarser-grained than `tags`.

### Changed
- **Breaking:** `ImageAnalysis::mood` renamed to `ImageAnalysis::emotion`
  (`mood()` / `with_mood()` / `set_mood()` → `emotion()` / `with_emotion()` /
  `set_emotion()`). `llmtask` is published at `0.1.0`, so this rename needs a
  pre-1.0 semver-breaking version bump on release. The downstream `lfm`
  engine's `ImageAnalysisTask` (prompt text, JSON schema, and
  `LfmScenePayload`, all in `lfm/src/image_analysis.rs`) still names the
  field `mood` throughout and needs a follow-up patch before `lfm` can pick
  up the new `llmtask` version.

## [0.1.0] - 2026-05-10

### Added
- Initial release: `Task` trait, `ParseError` enum, `ImageAnalysis` data type.
- `ImageAnalysis` accessor surface ported verbatim from `qwen::scene::SceneAnalysis`
  to host the canonical type once both `qwen` and `lfm` engines depend on this crate.
