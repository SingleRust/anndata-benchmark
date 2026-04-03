# Project Overview
This is a polyglot (Rust and Python) test or benchmark project focused on evaluating the `anndata` library, specifically testing the `sparse-rewrite` branch of the `anndata-rs` project. The workspace utilizes `pixi` for package and environment management alongside `cargo` for the Rust component. It relies heavily on data processing and scientific computing libraries, including `polars`, `ndarray`, and `sprs`.

# Building and Running

## Environment
The project uses `pixi` for managing the environment.
- **Install dependencies:** `pixi install` (TODO: define tasks in pixi.toml if needed)

## Rust
- **Directory:** `rust/`
- **Build:** `cd rust && cargo build`
- **Run:** `cd rust && cargo run`
- **Format & Lint:** `cd rust && cargo fmt && cargo clippy`

## Python
- **Directory:** `python/`
- **Run Script:** `pixi run python python/anndata-test.py` (Note: script is currently empty)

# Development Conventions
- **Rust Setup:** Uses standard Cargo toolchains (2024 edition). Error handling is performed via `anyhow`, and logging via the `log` crate.
- **Testing:** No explicit tests exist yet, but the structure implies benchmarking or integration tests comparing Rust and Python `anndata` implementations.
- **Data/Artifacts:** Outputs or resulting datasets should be placed in the `results/` directory.