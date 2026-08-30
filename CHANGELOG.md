# Changelog

## [Unreleased]

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
