# herbie-lint

A Dylint lint that detects numerically unstable floating-point expressions and suggests more stable alternatives.

## Overview

This lint analyses Rust code for `f64` expressions that may suffer from numerical instability (rounding errors, precision loss, etc.) and suggests mathematically equivalent but more stable alternatives.

## Modes

The lint operates in one of two modes:

| Mode | Description | Setup |
|------|-------------|-------|
| **Database mode** (default) | Uses a pre-computed database of 119 transformations embedded in the lint binary. Works out of the box with no configuration. | None required |
| **Live mode** | Calls the [Herbie](https://herbie.uwplse.org/) tool at lint-time to analyse expressions not in the database. Useful for domain-specific numerical patterns. | Requires Herbie 2.x installed; set `use_herbie = true` in `Herbie.toml` |

Most users should start with database mode. See [Using Herbie for dynamic analysis](#using-herbie-for-dynamic-analysis) for details on enabling live mode.

## Examples

| Original Expression | Suggested Replacement | Reason |
|---------------------|----------------------|--------|
| `(a*a + b*b).sqrt()` | `a.hypot(b)` | Built-in is more stable |
| `(a + 1.0).ln()` | `a.ln_1p()` | Avoids precision loss near zero |
| `a.exp() - 1.0` | `a.exp_m1()` | Avoids precision loss near zero |

## Installation

Requires [cargo-dylint](https://github.com/trailofbits/dylint):

```bash
cargo install cargo-dylint dylint-link
```

## Usage

### Via Git (Recommended)

Add to your project's `Cargo.toml`:

```toml
[workspace.metadata.dylint]
libraries = [
    { git = "https://github.com/urschrei/herbie-lint", tag = "v0.4.0" }
]
```

Then run:

```bash
cargo dylint --all
```

Dylint will automatically clone, build, and cache the lint.

### One-off check

```bash
cargo dylint --git https://github.com/urschrei/herbie-lint --tag v0.4.0
```

### Local build

If you've cloned the repository locally:

```bash
cd herbie-lint
cargo build --release
cargo dylint --lib-path target/release/libherbie_lint@nightly-2025-09-18-aarch64-apple-darwin.dylib \
    --manifest-path /path/to/your/project/Cargo.toml
```

Note: The library filename includes your toolchain version and architecture.

## Configuration (Optional)

The lint works out of the box with no configuration. If you want to customise behaviour, create a `Herbie.toml` file in your project root:

```toml
# Use a custom database instead of the embedded one (optional)
# db_path = "path/to/custom/Herbie.db"

# Whether to call the Herbie tool for unknown expressions
# - true: Require Herbie, fail if not found
# - false: Never call Herbie
use_herbie = false

# Maximum time in seconds for Herbie to process an expression
# Set to 0 for no timeout
timeout = 120
```

## Suppressing Warnings

To suppress the lint for a specific function or module:

```rust
#[allow(herbie)]
fn my_function() {
    // Lint warnings suppressed here
}
```

Or use the `#[herbie_ignore]` attribute on items.

## Supported Functions

The following mathematical functions are recognised and can be transformed:

| Function | Arity | Description |
|----------|-------|-------------|
| `abs` | 1 | Absolute value |
| `acos` | 1 | Arc cosine |
| `asin` | 1 | Arc sine |
| `atan` | 1 | Arc tangent |
| `atan2` | 2 | Two-argument arc tangent |
| `cos` | 1 | Cosine |
| `cosh` | 1 | Hyperbolic cosine |
| `exp` | 1 | Exponential |
| `exp_m1` | 1 | Exponential minus one |
| `hypot` | 2 | Hypotenuse |
| `ln` | 1 | Natural logarithm |
| `ln_1p` | 1 | Natural log of (1 + x) |
| `powf` | 2 | Power function |
| `sin` | 1 | Sine |
| `sinh` | 1 | Hyperbolic sine |
| `sqrt` | 1 | Square root |
| `tan` | 1 | Tangent |
| `tanh` | 1 | Hyperbolic tangent |

## Database

The lint includes an embedded database with 119 pre-computed transformations sourced from the [GHC Herbie Plugin](https://github.com/mikeizbicki/HerbiePlugin/blob/master/data/Herbie.db) (last updated October 2015). This covers the most common numerical stability improvements.

### Using Herbie for dynamic analysis

For expressions not in the database, you can optionally enable live analysis via the [Herbie](https://herbie.uwplse.org/) tool (version 2.0 or later). When enabled, the lint will:

1. **Analyse unknown expressions** - Spawn `herbie shell` to find improvements for expressions that don't match any pre-computed patterns
2. **Grow your database** - Save newly discovered transformations for future runs

This is useful for codebases with domain-specific numerical patterns not covered by the default database.

**Requirements:** Herbie 2.x installed. Install via:
- macOS: `brew install minimal-racket && raco pkg install --auto herbie`
- Linux/other: See [Herbie installation docs](https://herbie.uwplse.org/doc/latest/installing.html)

To enable, create a `Herbie.toml` in your project:

```toml
# Path to Herbie executable (required if not on PATH)
# macOS with Homebrew:
herbie_path = "/opt/homebrew/opt/minimal-racket/bin/herbie"

# Enable live Herbie analysis
use_herbie = true

# Timeout per expression in seconds (Herbie can be slow)
timeout = 120

# Custom database to save new discoveries (optional)
# db_path = "my-herbie.db"
```

Note: Live analysis adds significant time per expression. The default database covers the most impactful patterns without requiring Herbie.

## Troubleshooting

### Stale cache after upgrade

Dylint caches built lints. If you upgrade to a new version and experience issues, clear the cache:

```bash
# macOS
rm -rf ~/Library/Caches/dylint

# Linux
rm -rf ~/.cache/dylint
```

Then run the lint again to trigger a fresh build.

## Development

Build the lint:

```bash
cargo build --release
```

Run tests:

```bash
cargo test
```

## Background

This is a port of [rust-herbie-lint](https://github.com/mcarton/rust-herbie-lint) from the defunct rustc plugin system to the modern [Dylint](https://github.com/trailofbits/dylint) framework. The original plugin used `#[plugin_registrar]` which was removed in Rust 1.75.

## Licence

[Blue Oak Model License 1.0.0](https://blueoakcouncil.org/license/1.0.0)
