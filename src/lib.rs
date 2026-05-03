//! Shared types for findit-studio VLM engines.
//!
//! This crate hosts the cross-engine abstractions that both `qwen` and
//! `lfm` depend on: the [`Task`] trait, [`ParseError`], and the
//! canonical [`SceneAnalysis`] data type. Each engine ships its own
//! `SceneTask` implementation — the prompt, schema, and parser are
//! engine-specific (tuned to each model's drift patterns) — but they
//! all produce values of the same `SceneAnalysis` type.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

pub mod scene;
pub mod task;

pub use scene::SceneAnalysis;
pub use task::{ParseError, Task};
