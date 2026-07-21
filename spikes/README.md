# Spikes — the evidence behind the design

Every load-bearing claim in [`../docs/design.md`](../docs/design.md) that could be
checked with a small program was checked with one. These are those programs. Each is
self-contained and prints a pass/fail; they double as regression fixtures for the
crates once they exist.

| Spike | Verifies | Design ref |
| --- | --- | --- |
| `relational_ad_spike.py` | An MLP's whole backward pass = mechanical application of **four transpose rules** over relational primitives; all 6 param grads match `jax.grad` to ~1e-18. Reproduces xarray-sql#196's hand-written `delta*`/`g*` queries exactly. | §7.3 (true AD) |
| `attention_ad_spike.py` | Same four rules cover **scaled dot-product attention** (Q/K/V projections, QKᵀ, softmax over the key axis, A@V); grads w.r.t. Wq/Wk/Wv/X match `jax.grad` to ~1e-16. Generality beyond the MLP. | §7.3 |
| `attention_causal_mask_check.py` | The transformer **causal mask** is just elementwise — masked attention grads still match `jax.grad` to ~1e-16, no new rule. | §7.3 |
| `sqlparser-spike/` (Rust) | **G1:** `sqlparser`'s `Display` drops precedence parens on *constructed* trees (`(a+b)*c` → `a + b * c`), and `Nested`-wrapping fixes it. **G3:** spans are 1-based *characters*, not byte offsets. | §5.1, §8, M0 |
| `substrait_ad_marker_spike.py` | Adopting Substrait + custom extension-function markers (not a bespoke Rust IR) for v2: a `ddx_contract_mark(...)` marker wrapped around an aggregate's operand survives DuckDB's own `get_substrait`→`from_substrait` round-trip AND a genuine cross-engine hop (DataFusion produces the marker-tagged plan, DuckDB consumes and executes it) — numerically exact both ways. DuckDB→DataFusion deserializes cleanly (execution not yet exercised). DuckDB's `substrait` extension is community-maintained, not core, as of 1.5.4 (`INSTALL substrait` 404s; `INSTALL substrait FROM community` works). | design-relational-ad.md §1, §9 |
| `route_ad_spike.py` | The Route (argmax/max-pool) transpose rule vs. `jax.grad`: machine-exact (0.00e+00) away from ties. **At an exact tie**, our SQL-idiom's deterministic first-index tiebreak diverges from `jax.grad(jnp.max)`'s tie-splitting convention — both defensible, must be pinned explicitly (same treatment as the `abs`-at-0 kink, §8/F12), not assumed to agree with JAX. | design-relational-ad.md §3.4 |
| `duckdb_substrait_window_bug.py` | Route's forward SQL (`ROW_NUMBER()` top-1-per-group) through Substrait: a plain window column round-trips fine through DuckDB, but the **full top-1-per-group idiom silently returns the wrong (unfiltered) rows** — no exception — because DuckDB's own optimizer rewrites it into an `arg_max` join before Substrait export, and that rewritten form doesn't survive the round-trip. Reproduces with no ddx marker involved. DataFusion round-trips the identical idiom correctly, isolating this as a DuckDB-specific bug, not a general Substrait-window gap. A two-step workaround (Substrait-round-trip the window column, then filter with plain engine-native SQL) is verified to produce the correct result — Route does not need to wait on an upstream fix to ship. | design-relational-ad.md §3.4 |
| `duckdb_reentrancy_r1b.py` | **R1b:** a query on a 2nd connection to the same DuckDB DB, run during an outer query, is safe (reads, DML, no deadlock) but runs in its own transaction (can't see uncommitted state). | §5.4 |
| `substrait_limitation_repro.py` | **§6/ddx#1:** `datafusion-substrait`'s producer rejects recursive CTEs and DML (`Unsupported plan type: RecursiveQuery` / `DmlStatement`) — the reason Substrait isn't the transport. | §6, ddx#1 |

## Running them

Python spikes (a venv with the deps):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy jax duckdb datafusion pyarrow  # jax for the AD spikes; duckdb/datafusion/pyarrow for the engine ones
python spikes/relational_ad_spike.py             # → W2..b0 max|rule - jax.grad| ~1e-18  OK
python spikes/attention_ad_spike.py              # → Wq/Wk/Wv/X ~1e-16  OK
python spikes/attention_causal_mask_check.py     # → causal Wq/Wk/Wv ~1e-16  OK
python spikes/duckdb_reentrancy_r1b.py
python spikes/substrait_limitation_repro.py
python spikes/substrait_ad_marker_spike.py       # → 4/4 checks OK (DuckDB round-trip + cross-engine)
python spikes/route_ad_spike.py                  # → 0.00e+00 vs jax.grad; ties diverge (pin explicitly)
python spikes/duckdb_substrait_window_bug.py     # → A/C/D OK, B silently wrong (DuckDB bug, workaround verified)
```

Rust spike (`sqlparser` 0.62):

```bash
cd spikes/sqlparser-spike && cargo run
# G1 constructed (a+b)*c   Display => a + b * c      (WRONG — reparses as a+(b*c))
# G1 fixed  Nested(a+b)*c  Display => (a + b) * c    (correct)
# G3 'grad' byte offset=17, char offset=16           (spans are characters)
```

## Note on the AD spikes

`relational_ad_spike.py` and `attention_ad_spike.py` implement **only** the transpose
(VJP) rules for four relational primitives — contraction (`JOIN`+`GROUP BY SUM`),
elementwise map (whose local derivative is `ddx-core`'s scalar `grad`), per-group
`SUM`, and broadcast/bias — and compose them in reverse. Nothing else is hand-written;
the softmax/softmax-cross-entropy deltas *fall out* of the primitives. That they match
`jax.grad` to machine precision is the concrete evidence that query-level reverse-mode
AD (§7.3) is an engineering project, not research. The published precedent is
Tang et al., *Auto-Differentiation of Relational Computations …*, ICML 2023.
