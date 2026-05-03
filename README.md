# vlm-tasks

Shared types for findit-studio VLM engines.

This crate hosts the cross-engine abstractions that both [`qwen`] and
[`lfm`] depend on: the `Task` trait, `ParseError`, and the canonical
`SceneAnalysis` data type. Each engine ships its own `SceneTask`
implementation — prompt, schema, and parser are engine-specific —
but they all produce values of the same `SceneAnalysis` type.

## Status

Internal findit-studio crate. Not published to crates.io.

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE).
