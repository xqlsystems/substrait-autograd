# ddx
[JAX](https://docs.jax.dev/en/latest/)-style [automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) in SQL

Write calculus directly in SQL and get derivatives back as ordinary columns,
evaluated row by row by the engine alongside everything else:

```sql
SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
```

`grad`/`jvp` are compile-time **markers**: they carry a differentiation request
through parsing and are rewritten away — to plain derivative SQL — before the
query ever runs. One engine-neutral Rust core, thin per-engine adapters.

_status_: **M0 landed** — the scalar core (`ddx-core`) is implemented. The rest
is in progress; see [docs/design.md](docs/design.md) §8 for the milestones.

## Layout

```
crates/
  ddx-core/         # v1 engine — differentiate sqlparser::ast::Expr + rewrite_sql   [M0 ✓]
  ddx-ad/           # v2 engine — query-level reverse-mode AD over Substrait          [M3/M4]
  ddx-datafusion/   # DataFusion adapter: ddx_sql (Path A) + AnalyzerRule (Path B)     [M2]
  ddx-duckdb/       # DuckDB community extension: ddx('<sql>')                          [M5]
python/ddxdb/       # PyO3/maturin wheel: rewrite_sql + Context.sql() shim             [M2]
tests/              # cross-engine numeric-agreement suites (vs JAX)                    [M2/M6]
spikes/             # runnable evidence for every load-bearing design claim
future/             # deferred: ddx-duckdb-cpp (bare grad() in DuckDB), ddx-pg
docs/design.md      # the design
```

## Try the core

```rust
use ddx_core::Ddx;
use ddx_core::sqlparser::dialect::GenericDialect;

let out = Ddx::new()
    .rewrite_sql("SELECT grad(sin(x), x) AS d FROM t", &GenericDialect {})
    .unwrap();
assert_eq!(out, "SELECT (cos(x)) AS d FROM t");
```

```
cargo test -p ddx-core
```

See [`crates/ddx-core/README.md`](crates/ddx-core/README.md) for what the engine
supports and the correctness properties worth knowing.
