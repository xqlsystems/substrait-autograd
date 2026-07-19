# Design Doc: `ddx` — portable autograd for composable databases

_author_: Alex Merose

_co-author_: Claude (Opus 4.8), via Claude Code

_created_: 2026-07-19

_last updated_: 2026-07-19

_status_: Design — iterating toward implementation.

> **Revision note (2026-07-19, v0.2 — strategic simplification).** After review, the
> design collapses along three axes that earlier drafts left open:
> 1. **One injection path for v1, not two.** Both v1 targets need the SQL
>    source-to-source rewrite ("Path A"); neither needs a native in-engine plan
>    rewrite ("Path B"). Path B is demoted to future work (§5.3).
> 2. **One IR: the `sqlparser` AST**, not a bespoke `DExpr` + adapters. We
>    differentiate directly on `sqlparser::ast::Expr`, parse per-dialect, and
>    unparse via `Display` (§5.1). The core then depends only on `sqlparser` — no
>    DataFusion `Expr`, no `protoc`.
> 3. **Substrait leaves the critical path** (§6). SQL is already the portable
>    surface for grad/jvp/vjp; Substrait would be a redundant second IR that does
>    not solve our real portability problem (rewrite injection).
>
> This supersedes the earlier "both paths as peers" and "Substrait as protocol +
> optional IR" decisions; the reasoning is recorded inline where each applies.

---

## 1. Goal & thesis

Build a **generic, portable component for XQL-style symbolic autograd** that can be
installed into composable database systems (DataFusion, DuckDB, later Postgres),
so a user can write calculus directly in SQL:

```sql
SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g
```

and get derivatives back as ordinary columns, evaluated row-by-row by the engine
alongside everything else.

**Data model (assumed throughout).** We assume the **XQL data model**: an
N-dimensional array is a long/tidy relational table — one row per coordinate
tuple, dimensions and variables as columns (`temp(time, lat, lon)` becomes rows of
`(time, lat, lon, temp)`). Every derivative is therefore just another column
aligned to the same coordinates, which is what makes `grad` compose with ordinary
SQL. See [xql.systems](https://xql.systems) and
[xarray-sql](https://github.com/xqlsystems/xarray-sql) for the model in depth;
this doc takes it as given.

**Thesis (the vmap insight, carried over from the [prototype](https://github.com/xqlsystems/xarray-sql/pull/192)).** Because each row
of a table is an independent evaluation point, differentiating a column
expression and letting the engine evaluate it per row is the relational
equivalent of `jax.vmap(jax.grad(f))` — *the rows are the batch dimension.* This
turns SQL into a place you can express gradients, directional derivatives, and
whole training loops. We think of trained ML models as [differentiable](https://www.youtube.com/watch?v=LNNU33TmBFk) [databases](https://www.youtube.com/watch?v=jUe3rvTmv7Q);
hence the name **`ddx`** (`INSTALL ddx FROM community;`, `pip install ddxdb`).

This doc is grounded in a working prototype:
[xarray-sql#192](https://github.com/xqlsystems/xarray-sql/pull/192), which
implements `grad`/`jvp`/`vjp` for DataFusion. §3 records what that prototype
taught us; the rest of the doc generalizes it into a reusable component.

### 1.1 Success criteria

This is not a toy. The design succeeds when:

- A single **engine-independent core** (`ddx-core`) implements the
  differentiation algorithm once, and each database integration is a thin
  adapter over it.
- It ships in two real, actively-used projects with no regressions:
  [xarray-sql](https://github.com/xqlsystems/xarray-sql) (DataFusion/Python) and
  [duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr) (Rust community
  extension).
- The `grad`/`jvp`/`vjp` surface is **portable** — the same SQL-level functions,
  with one shared `ddx-core` defining what they mean, adopted by every target
  engine (rather than each reimplementing differentiation). Portability lives at
  the SQL surface, not in a plan-interchange format (§6).

---

## 2. Non-goals (for the first cut)

- **Not** a runtime tensor library or a replacement for JAX/PyTorch. We
  differentiate SQL scalar expressions symbolically; we do not implement a tape,
  GPU kernels, or autodiff of arbitrary imperative UDFs.
- **Not** general `u^v` power, `CASE`/conditional subgradients, or non-smooth ops
  in v1 (tracked in §7 roadmap). The engine returns a clear `NotImplemented`
  rather than a silently-wrong derivative.
- **Not** a Substrait project. Substrait is off the critical path (§6): no
  whole-plan round-trip (§3.2 explains why the prototype removed it), and — as of
  v0.2 — no Substrait expression IR or dependency either.
- **Not** two injection paths in v1. The native in-engine plan rewrite ("Path B")
  is deferred; v1 ships the SQL rewrite ("Path A") only (§5.3).
- **Postgres is later** — it needs array/XQL support first (via `pgrx`), and its
  planner-hook story differs from the two first targets.

---

## 3. What the prototype taught us (review of xarray-sql#192)

Reviewing all 13 commits of the PR — especially what was *added and then
removed* — produced the design constraints below. These are the load-bearing
lessons.

### 3.1 `grad`/`jvp`/`vjp` are compile-time rewrites, not runtime UDFs

This is the single most important finding. A scalar UDF only sees *values* at
runtime; it never sees the *symbolic expression* of its argument. But
differentiation is a function of the symbolic form. So `grad(...)` cannot be a
real row function.

In the prototype, `grad`/`jvp`/`vjp` are **markers**: no-op `ScalarUDF`s whose
only job is to parse and carry the differentiation request through the pipeline.
They are **always rewritten away before execution**, and their `invoke` method
deliberately errors if a marker ever reaches execution
(`src/autograd.rs::MarkerUdf::invoke_with_args`).

**Consequence for this project:** "installing a UDF in each database" is the
wrong mental model. What each engine actually needs is a **rewrite hook** at (or
before) planning time. UDF *registration* is only there to make the marker
*parse*. Every integration decision flows from this.

### 3.2 Substrait as a whole-plan transport was tried and deliberately removed

> Reproduced and written up in **[ddx#1](https://github.com/xqlsystems/ddx/issues/1)**
> (references [xarray-sql#192](https://github.com/xqlsystems/xarray-sql/pull/192)
> and [#197](https://github.com/xqlsystems/xarray-sql/issues/197)).

The PR's intermediate design (commit `672e7d0`) round-tripped the entire logical
plan through Substrait to apply the rewrite in a separately-linked copy of
DataFusion:

> produce logical plan as Substrait → `grad_rewrite` consumes it → rewrite every
> `grad()` into the derivative → re-produce Substrait → Python consumes & executes.

The **final commit (`14b26971`) deleted this entirely** ("Differentiate grad() as
a SQL rewrite, dropping the Substrait bridge"), because Substrait's producer could
not represent the query shapes that make in-SQL *training loops* interesting
(Newton's method / gradient descent in a `WITH RECURSIVE`, or `INSERT`-ing updated
parameters), and it required a `protoc` build dependency plus per-engine schema
plumbing.

I re-verified the limitation against **datafusion 54.0.0** (ddx#1); the current
picture — note it has *shifted* since #197 was filed:

| Query shape | `to_substrait_plan` |
| --- | --- |
| Plain scalar projection | ✅ works |
| Recursive CTE (`WITH RECURSIVE`) | ❌ `Unsupported plan type: RecursiveQuery` |
| DML (`INSERT … SELECT`) | ❌ `Unsupported plan type: DmlStatement` |
| Scalar subquery | ✅ works now (**was** broken at #197; fixed upstream) |

So the disqualifying gaps are **recursive CTEs and DML** — exactly the training-loop
shapes. (Upstream tracking: [apache/datafusion#16248](https://github.com/apache/datafusion/issues/16248).)

The replacement — a **SQL source-to-source rewrite before planning**
(`rewrite_grad_in_sql`) — works for *any query shape the SQL parser accepts*,
needs no engine fork and no custom wheel, and runs against the stock published
package.

**Consequence:** Substrait is *not* a good fit as the mandatory plan transport.
The failure was specific to **whole plans** — a single scalar expression (the
argument to `grad()`) is well within Substrait's expressive core — so a Substrait
*scalar-expression* IR was, for a while, kept as an optional role. v0.2 drops even
that: SQL text already carries the scalar argument portably, so a second Substrait
IR earns nothing. See §6 for the full rationale.

### 3.3 The reusable crown jewel is a small, IR-shaped differentiation engine

`src/autograd.rs` is ~350 lines of actual algorithm:

- A **per-primitive rule registry** mirroring JAX's `defjvp` (`linearize` +
  `linearize_binary` + `linearize_scalar_function` + `linearize_power`).
- **Forward-mode linearization** with a pluggable *leaf rule*: `grad` is a
  one-hot seed, `jvp` an arbitrary per-input seed, `vjp` is `cotangent × grad`.
  One chain rule, three surfaces.
- A **0/1-folding simplifier** (`add`/`sub`/`mul`/`div`/`neg` smart
  constructors) that plays the role of JAX's `Zero` tangents — keeps output
  compact and short-circuits dead terms.
- Rules for `+ - * /`, unary chain rule for `sin/cos/tan`, `asin/acos/atan`,
  `exp/ln/log2/log10/sqrt`, `sinh/cosh/tanh`, `abs`, and `power()` with a
  constant base *or* exponent. Everything else → `NotImplemented`.

The catch: it is currently written **against DataFusion's `Expr` type**
(`datafusion::logical_expr::Expr`). The algorithm is engine-independent; the
*data type it walks* is not. **Re-pointing the rules off `Expr` onto the
engine-neutral `sqlparser::ast::Expr` is the central refactor of this project**
(§5.1).

### 3.4 Other inherited design decisions worth keeping

- **Long/tidy data model.** A gradient/Jacobian is several scalar columns
  (`grad(f,x) AS dfdx, grad(f,y) AS dfdy`), never a nested array. The PR added an
  array `jacobian()` and then removed it (commit `c952df1`) because a nested-array
  cell breaks the one-value-per-coordinate model. Keep scalar-only.
- **Higher-order for free** via bottom-up rewrite: `grad(grad(f,x),x)` just works.
- **Differentiation through aggregates is linearity**: put the marker *inside*
  the aggregate — `AVG(grad(loss, theta))` — not outside. This is what makes
  gradient descent expressible in SQL.
- **Also export a "calculus compiler"**: `differentiate_sql(expr, wrt, columns)`
  returns the derivative as SQL *text*, for embedding an update rule as a string
  where a marker can't reach (e.g. inside a recursive term).

---

## 4. Design principles

1. **Differentiate once, on the SQL AST.** The algorithm lives in `ddx-core` and
   operates directly on `sqlparser::ast::Expr` — the same parser DataFusion uses,
   with a `DuckDbDialect`. No bespoke IR, no per-representation adapters, no engine
   `Expr` dependency (§5.1). SQL text in, SQL text out.
2. **Rewrite, don't execute.** Markers are erased before execution. Every
   integration is fundamentally "parse, find the marker, differentiate its
   argument, splice the derivative back, hand plain SQL onward."
3. **One injection path in v1: the SQL source-to-source rewrite.** It reaches
   every v1 target and distribution channel (§5.3). A native in-engine plan
   rewrite ("Path B") is a *future* enhancement for engines that expose the hook,
   not a co-equal pillar.
4. **SQL is the portable surface; Substrait is off the critical path.** grad/jvp/vjp
   are ordinary SQL function calls syntactically, so portability is free at the SQL
   level. Substrait solves plan interchange, not our problem (rewrite injection),
   so it is neither a dependency nor a mechanism in v1 (§6).
5. **Fail loud, never silently wrong.** An unsupported node is a typed error, not
   an approximate derivative. This is a numerical-correctness product.
6. **Prove it in real projects.** xarray-sql and duckdb-zarr are acceptance
   tests, not demos.

---

## 5. Architecture

A monorepo (Cargo workspace + one Python wheel). Two crates and a wheel do the v1
job; everything else is explicitly future.

```
                    ┌──────────────────────────────────────────────┐
                    │                 ddx-core (Rust)              │
                    │  operates on sqlparser::ast::Expr            │
                    │  rule registry · linearize · 0/1 simplifier  │
                    │  grad / jvp / vjp  ·  statement rewriter      │
                    │  deps: sqlparser only (no protoc, no engine) │
                    └──────────────────────────────────────────────┘
                          SQL text in ▲        ▼ SQL text out
        ┌───────────────────────────┴─────────┴────────────────────────────┐
        │  The one path (v1): SQL source-to-source rewrite before planning  │
        │  parse (per-dialect) → find grad/jvp/vjp → differentiate arg →    │
        │  unparse (Display) → splice → hand plain SQL to the engine        │
        └───────────────────────────┬─────────┬────────────────────────────┘
                                     │ shipped as
        ┌────────────────────────────┴─────────┴───────────────────────────┐
        │ ddxdb (Python wheel)          │ ddx (DuckDB community ext, Rust)  │
        │ rewrite_sql + Context.sql()   │ `ddx('<sql>')` table function     │
        │ → xarray-sql, DuckDB-python   │ → duckdb-zarr                     │
        └───────────────────────────────┴───────────────────────────────────┘

   Future (not v1): ddx-datafusion (native AnalyzerRule, "Path B") · C++/cxx.rs
   hybrid for bare grad() in DuckDB (§5.4 opt 4) · ddx-pg (pgrx) · Substrait (§6)
```

**Why one path is enough (was: "do we need both A and B?").** Both v1 targets are
reached by the SQL rewrite alone: xarray-sql *must* use it (datafusion-python can't
inject an `AnalyzerRule`, R2), and the DuckDB `ddx('<sql>')` table function *is*
the SQL rewrite executed inside the extension (§5.4). A native in-engine rewrite
("Path B") buys nicer ergonomics (bare `grad()` with no wrapper) but reaches no
target the one path doesn't already reach. So v1 builds one path; Path B is a
later ergonomic upgrade per engine.

### 5.1 `ddx-core` — the engine (the refactor)

Port the prototype's `src/autograd.rs`, but change the type it walks: **off
DataFusion `Expr`, onto `sqlparser::ast::Expr`** — the [`sqlparser`
crate](https://docs.rs/sqlparser/) (Apache's `datafusion-sqlparser-rs`, the same
parser DataFusion uses; **not** the third-party `sqlparser-patched`). It ships
`DuckDbDialect`, `PostgreSqlDialect`, `GenericDialect`, …, and `ast::Expr: Display`
so a differentiated AST renders straight back to SQL. Surface:

```rust
// Differentiate one scalar expression w.r.t. a column, both as sqlparser AST.
pub fn differentiate(e: &ast::Expr, wrt: &ColRef) -> Result<ast::Expr, DiffError>; // grad
pub fn jvp(e: &ast::Expr, seeds: &HashMap<ColRef, ast::Expr>) -> Result<ast::Expr, DiffError>;
pub fn vjp(e: &ast::Expr, wrt: &ColRef, cotangent: &ast::Expr) -> Result<ast::Expr, DiffError>;

// Statement-level rewrite: parse, find every grad/jvp/vjp call, differentiate its
// argument, splice the derivative back, return SQL text. This is the whole path.
pub fn rewrite_sql(sql: &str, dialect: &dyn Dialect) -> Result<String, DiffError>;

// A column identity read straight off the AST (Identifier / CompoundIdentifier).
pub struct ColRef { pub qualifier: Option<String>, pub name: String }
```

The rules match on the `ast::Expr` variants we support and `NotImplemented` the
rest: `Expr::BinaryOp{left,op,right}` (`+ - * /`), `Expr::Function` (name-dispatched:
`sin`, `power`, …), `Expr::UnaryOp`(minus), `Expr::Cast`, `Expr::Nested`,
`Expr::Identifier`/`CompoundIdentifier` (leaves), `Expr::Value` (literals). This
is essentially what the earlier draft's `DExpr` enum *was* — a stripped-down SQL
expression tree — so we use `sqlparser`'s type directly and delete both the
bespoke IR and its four `From`/`To` adapters.

Design notes:

- **Rule registry keyed by function name string** (as the prototype does:
  `match name { "sin" => … }`). Engine-agnostic, since function names come straight
  off the parsed call. A small canonicalization table folds dialect spellings (e.g.
  `ln`/`log`, `pow`/`power`) to one canonical name before dispatch (§7).
- **Keep the smart constructors** (`add/sub/mul/div/neg/square`) — the
  `Zero`/`add_tangents` 0/1-folding simplifier — building `ast::Expr` nodes now.
- **Literals:** `sqlparser` stores numbers as strings (`Value::Number("0.0", _)`);
  parse to `f64` for constant folding, and **emit `DOUBLE`-typed literals/casts**
  in output (R1b finding: DuckDB types `0.0` as `DECIMAL`, which would pull
  derivative arithmetic into decimal). Let the engine coerce from there.
- **Qualifier-aware (§5.5):** `ColRef` carries the qualifier from
  `CompoundIdentifier`, so `grad(a.x + b.x, a.x)` differentiates the right column
  with no catalog. An unqualified `wrt` whose name collides under two qualifiers in
  the argument is a hard error (detectable from the AST alone) — never a silent
  mismatch (§5.5).
- Port the prototype's 15 rule unit tests; they pin the math unchanged.

**Deps: `sqlparser` only.** No DataFusion, no `protoc`, no engine crate. This is
the reusable component the goal calls for, and it is smaller than the earlier
core-plus-adapters design.

### 5.2 The rewrite driver (was: IR adapters)

There are no IR adapters anymore — the AST *is* the IR. What each integration
provides is a **thin driver** that (a) picks the right `sqlparser` `Dialect`,
(b) calls `ddx-core::rewrite_sql`, and (c) hands the resulting SQL to the engine.
The whole surface each integration wires up:

| Integration | Dialect | How the rewritten SQL reaches the engine |
| --- | --- | --- |
| **Rust DataFusion** (`ddx-datafusion` helper) | `GenericDialect` / DataFusion's | `ctx.sql(rewrite_sql(sql, dialect)?)` — a one-line wrapper; see below |
| `ddxdb` (Python → DataFusion) | `GenericDialect` / DataFusion-compatible | `Context.sql()` shim calls `rewrite_sql`, then the stock datafusion-python context plans it |
| `ddxdb` for DuckDB-python | `DuckDbDialect` | preprocess the string before `duckdb.sql(...)` |
| `ddx` (DuckDB ext) | `DuckDbDialect` | `ddx('<sql>')` table fn calls `rewrite_sql`, runs it on an inner connection (§5.4) |

**Native Rust DataFusion is the most direct consumer of `ddx-core`,** and in v0.2
it is *simpler* than before, not missing. DataFusion is built on the very
`sqlparser` crate `ddx-core` uses, so the integration is just: rewrite the SQL
string, then hand it to the stock `SessionContext`:

```rust
// The entire v1 "integration" for native Rust DataFusion.
// (Returns datafusion::Result; assumes `impl From<DiffError> for DataFusionError`
// so the `?` on rewrite_sql composes — a one-liner in the ddx-datafusion crate.)
pub async fn ddx_sql(ctx: &SessionContext, sql: &str) -> DataFusionResult<DataFrame> {
    let rewritten = ddx_core::rewrite_sql(sql, &GenericDialect {})?;
    ctx.sql(&rewritten).await
}
```

No AnalyzerRule, no fork, no custom build — this is the same one path, called from
Rust. It can live as a tiny `ddx-datafusion` convenience crate or just be inlined
by the caller; it is *not* a required v1 milestone (our two acceptance targets are
xarray-sql and duckdb-zarr), but it is essentially free to offer.

The prototype's `rewrite_grad_in_sql` / `GradSqlRewriter` is exactly this driver,
minus its detour through DataFusion `Expr` (it currently parses to DF `Expr` to
differentiate, then unparses; we differentiate the `ast::Expr` directly instead).

### 5.3 The one path, and the future Path B

**The path (v1).** Intercept the SQL string before it reaches the engine, rewrite
every `grad`/`jvp`/`vjp` call to derivative SQL, pass plain SQL onward. It runs
*before* planning, so it works for every query shape the parser accepts —
recursive CTEs, DML, subqueries — which is what lets a whole training loop live in
one query. This is what the prototype settled on and what all v1 integrations use.

**Path B (future, per engine).** A native in-engine plan rewrite so `grad()` works
bare with no wrapper (e.g. `SELECT grad(sin(x), x) FROM t` directly). It reaches no
new *target*, so it is deferred — but it is a real ergonomic upgrade, and its cost
differs sharply by engine:
- **DataFusion (Rust) — cheap, available now, likely the first Path B.** Register
  the markers, then install an `AnalyzerRule`/`FunctionRewrite` over the bound
  `LogicalPlan`. Uniquely, this needs **no second rule engine**: DataFusion already
  ships an `Expr`→SQL unparser (`expr_to_sql`) and a SQL→`Expr` planner, so the
  rule can lift each `grad()` argument to SQL, call the *same* `ddx-core` rewrite,
  and plan the result back — reusing the one rule set the sqlparser-IR choice gives
  us. (Alternatively, port the prototype's `differentiate(&Expr)` natively, but the
  bridge avoids duplicate logic.) Not reachable from datafusion-python (R2), so it
  serves native-Rust users only. Because it's so cheap here, `ddx-datafusion` may
  offer *both* the v1 helper (§5.2) and this rule.
- **DuckDB — expensive.** Needs the C++/cxx.rs hybrid (§5.4 option 4); the stable C
  API has no hook (R1). Post-v1.

Either way Path B differentiates the engine's *bound* expression tree; with the
`Expr↔SQL` bridge above it still funnels through `ddx-core`, so there is one rule
engine regardless of path. Out of scope for v1.

### 5.4 Per-engine integration & distribution

**DataFusion / `ddxdb` (Python) → xarray-sql.**
`datafusion-python` does not expose injecting an `AnalyzerRule` into its
`SessionContext` (R2) — which is *why* the SQL rewrite is the path here, not a
limitation to work around. `ddxdb` re-exports `ddx-core::rewrite_sql(sql, dialect)`
and a `Context.sql()` shim; xarray-sql consumes it, deleting its vendored
`autograd.rs` in favor of `ddx-core` (regression-tested against the PR's Python
suite). (Native Rust DataFusion is covered separately just below.)

**DataFusion (native Rust) / `ddx-datafusion`.**
The lowest-friction integration of all, because DataFusion is built on the same
`sqlparser` crate `ddx-core` differentiates over. v1 is a one-line helper —
`ctx.sql(ddx_core::rewrite_sql(sql, dialect)?)` (§5.2) — shippable as a small
`ddx-datafusion` crate or simply inlined by the caller; no fork, no custom build.
The only care point is parsing with the dialect DataFusion itself uses, so the
rewrite accepts exactly what `ctx.sql` would. This is *not* a required v1 milestone
(the acceptance targets are xarray-sql and duckdb-zarr) but is essentially free to
offer. Native Rust is also the one place **Path B** (bare `grad()`, no wrapper) is
cheap and available today — an `AnalyzerRule` that reuses `ddx-core` via
DataFusion's `Expr`↔SQL bridge (§5.3) — so if/when we build any Path B, it lands
here first, well before DuckDB's C++ route.

**DuckDB / `ddx` community extension (Rust) → duckdb-zarr.**
_Settled by spike R1 (2026-07-19)._ Reading DuckDB's actual C Extension API
header (`duckdb/src/include/duckdb_extension.h`, what the `duckdb` crate's
`loadable-extension` feature binds) confirms it exposes registration for **only**
scalar, aggregate, table, and cast functions, plus replacement scans — and
**zero** optimizer / parser / operator / logical-plan / bound-expression hooks.
This is corroborated by duckdb-zarr, a mature Rust extension that uses exactly
table functions + a replacement scan and nothing deeper.

**Verdict: a Rust community extension cannot do a native bare-`grad()` rewrite
(Path B).** A scalar `grad()` UDF only ever receives executed *values*, never the
symbolic argument tree — insufficient for differentiation (§3.1). The ranked
options that remain:

  1. **In-extension Path A via a table function (recommended primary).** The same
     header exposes `duckdb_bind_get_parameter` (read a **literal SQL string at
     bind time**) and `duckdb_connect` + `duckdb_query` (**execute a query from
     inside the extension**). So a `ddx('<sql>')` table function reads the query
     literal, rewrites markers via `ddx-core::rewrite_sql` (with `DuckDbDialect`),
     executes the plain SQL on a connection to the same database, and streams the
     result back:
     ```sql
     INSTALL ddx FROM community;
     SELECT * FROM ddx('SELECT grad(sin(x), x) AS d FROM t');
     ```
     Pure Rust, community-installable, honors the `INSTALL ddx` vision. Cost: the
     `ddx('…')` wrapper instead of bare `grad()`. Caveat to validate: re-entrancy
     of a query-within-a-query on the same DB (safe for reads under MVCC; DML
     needs care — see R1b below).
  2. **Client-side Path A for DuckDB-Python (ships day one).** `ddxdb` preprocesses
     the SQL string before `duckdb.sql(...)`. Zero engine hooks; the fastest path
     to a working duckdb-zarr integration and a useful fallback.
  3. **A scalar `ddx_rewrite(sql) → sql` helper.** Pure string→string, trivially
     safe (no inner connection), for users who want to inspect or run the
     rewritten SQL themselves.
  4. **A hybrid C++/Rust extension via an `OptimizerExtension` (stretch goal;
     spiked 2026-07-19).** The *only* route to bare `grad()` anywhere in a normal
     SELECT — and, as a bonus, it is **binding-aware for free** (§5.5), since the
     optimizer runs *after* binding. Architecture: the DuckDB **C++
     [extension-template](https://github.com/duckdb/extension-template)** (CMake +
     vcpkg + DuckDB submodule) registers an `OptimizerExtension` whose
     `optimize_function` walks the bound `LogicalOperator` plan, finds each `grad`
     `BoundFunctionExpression`, and replaces it with the derivative computed by our
     Rust `ddx-core`, called across a **[cxx.rs](https://cxx.rs/)** bridge
     (`ddx-core` built as a `staticlib`, linked into the C++ extension; `cxxbridge`
     CLI generates the C++ glue in the CMake build). Distributes via community
     extensions (`INSTALL ddx FROM community;`) like any C++ extension.

     _Spike verdict:_ **cxx.rs is the right FFI tool and makes the Rust↔C++ call
     itself trivial — but it does not remove the two real costs, so this stays
     post-v1.** (a) A C++ extension links DuckDB *internals* and must be rebuilt
     per DuckDB version against an unstable internal API (heavier build + CI than
     the stable C API). (b) **The hard part is the expression boundary, which cxx
     does not solve:** at the optimizer stage columns are bound *structurally by
     index* (`ColumnBinding`), so a bound `Expression` does **not** round-trip to
     re-parseable SQL — the SQL-text boundary we use elsewhere is unavailable here.
     The boundary must carry the expression *structurally*: either (i) serialize
     DuckDB's expression to bytes and write a Rust deserializer into `ddx-core`'s
     `sqlparser::ast::Expr` (clean, but couples to DuckDB's internal serialization
     format), or
     (ii) expose the C++ `Expression` tree to Rust as cxx opaque types with
     accessors and rebuild a bound expression from Rust (most code, tight version
     coupling). (i) is the likely first cut. `autocxx` (auto-binding DuckDB's
     headers) is tempting but DuckDB's headers are large/complex — prefer a narrow
     hand-written cxx bridge.
  5. **`CREATE MACRO`** — rejected: macros are fixed expansions and cannot perform
     general differentiation.

Plan: ship (1) as the Rust community extension for duckdb-zarr, with (2) as the
Python-side convenience. Keep (4) — the cxx.rs hybrid — as the documented route to
bare `grad()` + binding-aware DuckDB, revisited when bare `grad()` is prioritized;
its cost is the C++/internals build and the structural expression boundary, not
the FFI.

> **R1b — RESOLVED (2026-07-19, spike).** Tested DuckDB 1.5.4's behavior when a
> query runs on a second connection to the same database *during* execution of an
> outer query — a stricter model than `ddx('…')` itself (whose outer holds no
> user-table scan). Findings:
> - **Re-entrancy is safe.** An inner query on the same DB, run mid-execution of
>   an outer table scan, works with no deadlock (`inner_select` per row → OK).
> - **Inner reads of committed user tables work** (`read_param` → `[0, 10, 20]`).
> - **Inner DML (INSERT) works** — 3 rows written during the outer scan, no
>   conflict or deadlock.
> - **The inner connection runs in its own transaction:** it does **not** see the
>   outer connection's *uncommitted* writes (outer `UPDATE a=999` uncommitted →
>   inner still reads `a=1.0`). This is the one real semantic consequence.
>
> **Design consequence.** `ddx('…')` is safe for self-contained queries, including
> a whole recursive-CTE training loop passed as one string (it runs entirely on
> the inner connection). But because the inner connection can't see the caller's
> *uncommitted* state, a training loop that mutates parameters across statements
> inside an open `BEGIN…` block must either (a) keep the whole loop inside the
> `ddx('…')` string, (b) `COMMIT` between steps, or (c) use **client-side Path A**
> (option 2), which rewrites on the *user's own* connection and so preserves their
> session and transaction visibility. Recommendation: **`ddx('…')` for
> self-contained queries; client-side Path A for stateful/transactional loops.**
> (Engine-level re-entrancy/isolation is now established; a Rust-extension smoke
> test remains as a confirmation task in M3.)
>
> Side finding: DuckDB types `0.0`-style literals as `DECIMAL`, so the rewrite
> should emit `DOUBLE`-typed literals/casts (matching the prototype's `Float64`
> treatment) to avoid `DECIMAL` arithmetic surprises in derivative output.

**Postgres / `ddx-pg` (`pgrx`) — later.** Needs array/XQL support first; native
Path B would use a planner hook. Out of scope for v1.

### 5.5 Column identity: qualifier-aware is binding-correct

The earlier draft worried that a *syntactic* (pre-binding) rewrite forces
*unqualified* column names and can't tell `a.x` from `b.x`. Working it through, the
concern mostly dissolves — and the fix is small:

- **`ColRef` reads the qualifier straight off the AST.** `sqlparser` parses `a.x`
  as a `CompoundIdentifier`, so `ColRef { qualifier: Some("a"), name: "x" }` falls
  out for free. `grad(a.x + b.x, a.x)` differentiates the right column with no
  catalog. This is strictly better than the prototype's unqualified-only form.
- **Qualifier-aware differentiation is binding-correct — with one guard.** For a
  *qualified* `wrt` (`a.x`), or an unqualified `wrt` whose name is not reused under
  another qualifier in the argument, matching by `ColRef` picks exactly the column
  the engine would bind. The case that needs care: an **unqualified `wrt` whose
  bare name appears under two different qualifiers in the argument** — e.g.
  `grad(a.x * b.x, x)` in a self-join `FROM t a JOIN t b`. Treating both `a.x` and
  `b.x` as "the same `x`" is wrong, and because we rewrite *before* binding, the
  engine never gets the chance to reject the ambiguity. A naïve "match by bare
  name" would silently return a wrong derivative here.
- **We catch that case syntactically — no catalog needed.** The argument AST itself
  reveals the collision: if the `wrt` name occurs under ≥2 distinct qualifiers (or
  both qualified and bare) inside the expression being differentiated, an
  unqualified `wrt` is ambiguous → **hard error** asking for qualification
  (`grad(a.x * b.x, a.x)`). This detection needs only the expression, not a binder,
  so it keeps the "fail loud, never silently wrong" guarantee (§4.5).
- **The remaining unqualified cases fail loud downstream.** If the reference is
  *wholly* unqualified in a genuinely ambiguous scope (`grad(sin(x), x)` with `x`
  in two joined tables), the rewrite emits SQL that still contains the bare `x`,
  which the engine then rejects as ambiguous when it plans the rewritten query. So
  no silent wrong answer escapes; what catalog-driven **Path B** adds is
  *resolving* such cases (and `SELECT *` expansion) rather than erroring — deferred
  with Path B (§5.3).
- **Aliases need no catalog either.** In `SELECT a+b AS s, grad(s*s, s)`, treating
  the identifier `s` as the differentiation variable is exactly right
  (`d/ds (s*s) = 2s`); the surrounding query re-substitutes `s`.

**Net:** v1 is qualifier-aware syntactic differentiation — which honors "don't be
limited to unqualified names" while needing no binder. This is a deliberate,
partial revision of the earlier "invest fully in binding awareness now" decision:
the fully-bound path rides along with Path B rather than blocking v1. Open
sub-question in §12 Q2.

---

## 6. Substrait: off the critical path (revised)

**Decision (v0.2): Substrait is not a mechanism or a dependency in `ddx`.** This
supersedes the earlier "protocol + optional IR" role. The project name shed
"substrait" (→ `ddx`) for the same reason. Three questions settle it:

1. **Is Substrait needed for portability?** No. `grad`/`jvp`/`vjp` are ordinary SQL
   function calls *syntactically*, and every target speaks SQL, so **SQL is already
   the portable surface**. A Substrait `Expression` IR would be a *second*,
   redundant portable format — with a `protoc` build tax — carrying no expression
   we can't already carry as SQL text.
2. **Does Substrait solve our actual problem?** No. Our problem is *rewrite
   injection* — getting a plan-time hook in each engine so a marker is differentiated
   before execution (§3.1). Substrait standardizes *plan interchange*, not
   *plan-time rewriting*, and it has no notion of a "marker function that must not
   execute." A Substrait definition of `grad` would advertise a runtime function
   that no compliant consumer could actually run. Even with it, each engine still
   needs its own rewrite hook — so it saves no integration work.
3. **The whole-plan bridge is already rejected** (§3.2) for recursive CTEs / DML /
   subqueries.

So there is no `ddx-substrait` crate, no `ddx.yaml`, no `protoc`, and no R3 spike.

**Door left open (cheap, non-blocking).** If a Substrait-native engine (Velox,
etc.) ever wants to adopt `ddx`, the surface *can* be declared as a Substrait
simple-extension at that point, and `ddx-core`'s AST rules could be fronted by a
Substrait-`Expression`→AST adapter. That is a future adapter behind a feature
flag, written on demand — not part of the v1 architecture, and not what defines
this project.

---

## 7. The differentiation surface & math roadmap

**v1 surface (port of the prototype, unchanged semantics):**

- `grad(expr, column)` → `d(expr)/d(column)`.
- `jvp(expr, column, tangent)` → forward-mode `d(expr)/d(column) · tangent`.
  Multi-input directional derivative = sum of `jvp` terms.
- `vjp(expr, column, cotangent)` → reverse-mode `cotangent · d(expr)/d(column)`.
- `differentiate_sql(expr, wrt)` → derivative as SQL text (the "calculus
  compiler" escape hatch). The prototype's third `columns` argument (§3.4) is
  dropped: it existed only to synthesize a DataFusion schema for standalone
  parsing, which `sqlparser` does not need.
- Rules: `+ - * /`, unary chain rule for the trig/inverse-trig/exp/log/hyperbolic
  set + `abs`, `power` with constant base or exponent. Higher-order via nesting.
  Through-aggregate via linearity (`AGG(grad(...))`).

**Roadmap (each an explicit rule addition, fail-loud until then):**

- General `u^v` (both variable) via the `exp(v·ln u)` trick.
- `CASE`/conditional and `min`/`max`/`greatest`/`least` — subgradients, with a
  documented convention at kinks (JAX-style), mirroring how `abs` uses `signum`.
- `atan2`, `log(base, x)`, `cbrt`, `expm1`/`log1p`, `pow`/`^` operator spellings.
- Dialect name normalization table (canonical → per-engine spellings).
- Clear taxonomy of "not differentiable" (comparisons, string/temporal ops,
  window functions) that stays `NotImplemented`.

---

## 8. Testing & verification

Differentiation is a numerical-correctness feature; the test strategy is
layered and reuses the prototype's:

- **Unit (rule) tests in `ddx-core`** — port the prototype's 15 Rust tests;
  every rule pinned symbolically on `ast::Expr`.
- **Round-trip property tests** — `parse → differentiate → unparse` produces
  parseable SQL that re-parses to an equal AST; fuzz small random expression trees
  per dialect.
- **Numeric agreement (the ground truth) — against JAX.** For a battery of
  expressions, compare the engine-evaluated derivative against **`jax.grad`** (and
  `jax.jvp`/`jax.vjp` for those surfaces), which is the natural oracle: the whole
  design mirrors JAX's forward-mode structure, so JAX gives an exact analytic
  reference for the same seed/cotangent semantics — a closer match than hand-coded
  numpy derivatives. Keep a **finite-difference** check as a cheap independent
  cross-check where a JAX equivalent is awkward. (The prototype checked against
  numpy analytics; JAX is the upgrade.)
- **Cross-engine equivalence** — the *same* expression rewritten with
  `DuckDbDialect` vs. the DataFusion-compatible dialect must evaluate to
  numerically equal columns in DuckDB and DataFusion respectively.
- **Real-integration acceptance** — end-to-end gradient descent and a recursive-
  CTE training loop converging to closed-form solutions, run inside xarray-sql and
  duckdb-zarr.

**Open research spikes (do these early — they de-risk the plan):**

- **R1 — RESOLVED (2026-07-19).** DuckDB's C Extension API has no
  parser/optimizer/plan hooks; use the in-extension `ddx('<sql>')` table function.
  C++/cxx.rs hybrid is the future route to bare `grad()`. Full verdict in §5.4.
- **R1b — RESOLVED (2026-07-19).** `ddx('…')`'s inner-connection re-entrancy is
  safe (reads, DML, no deadlock); it runs in its own transaction (can't see the
  caller's *uncommitted* state), so self-contained queries use `ddx('…')` and
  stateful loops use the client-side rewrite. Remaining: a Rust-extension smoke
  test in M3.
- **R2:** Confirm `datafusion-python` still can't inject an `AnalyzerRule` — this is
  what keeps v1 on the SQL rewrite. (If a seam exists, it only *adds* a future
  Path B; it does not change v1.)
- ~~**R3** (Substrait round-trip)~~ — dropped with Substrait (§6).

---

## 9. Monorepo layout (proposed)

```
ddx/                               (repo; crates published under the ddx-* names)
├── crates/
│   ├── ddx-core/                   # the engine — differentiate sqlparser::ast::Expr
│   │                               #   + rewrite_sql; dep: sqlparser only
│   └── ddx-duckdb/                 # DuckDB community extension: `ddx('<sql>')` table fn
├── python/
│   └── ddxdb/                      # PyO3/maturin wheel: rewrite_sql + Context.sql() shim
├── docs/
│   └── design.md                   # this file
├── tests/                          # cross-engine numeric-agreement suites (vs JAX)
└── future/                         # not v1 — see §5.3 / §5.4 / §6
    ├── ddx-datafusion/             #   native AnalyzerRule (Path B) for Rust DataFusion
    ├── ddx-duckdb-cpp/             #   C++/cxx.rs hybrid for bare grad() (§5.4 opt 4)
    └── ddx-substrait/              #   only if a Substrait-native engine asks (§6)
```

Rationale: the v1 surface is just **`ddx-core` + `ddx-duckdb` + `ddxdb`**.
`ddx-core` publishes independently (dep: `sqlparser`) so anyone can drive it from a
new engine. No `protoc` anywhere. Future crates are physically separated so the v1
build stays trivial. (The `future/` directory is illustrative — those may live as
un-published workspace members or separate repos.)

---

## 10. Naming & distribution

**Decision: name the project `ddx` (not `autograd`).** Rationale:

- **It names the mechanism correctly.** "autograd" connotes a *runtime, tape-based*
  system (PyTorch/HIPS). This is *symbolic differentiation as a plan-time rewrite* —
  literally `d/dx` of an expression. `ddx` sets the right expectation; "autograd"
  would make users look for a tape and kernels (an explicit non-goal, §2).
- **It fits the XQL family** (`xql.systems`, `xarray-sql`, `duckdb-zarr`): short,
  lowercase, evocative. `ddx` / `ddxdb` / `INSTALL ddx` belong to that set.
- **The thesis is in the name:** "ML models as differentiable databases" → `d/dx`
  of a table.
- **Practical:** `autograd` is already taken on PyPI (the HIPS package); `ddxdb` is
  distinctive. (Availability of `ddx*` on crates.io / PyPI / the DuckDB community
  registry still to be confirmed — §12 Q5.)

Distribution:

- Rust crates (v1): `ddx-core`, `ddx-duckdb` on crates.io. (`ddx-datafusion`,
  `ddx-substrait` only if/when the future work lands.)
- Python: `pip install ddxdb`.
- DuckDB: `INSTALL ddx FROM community;` → the `ddx('<sql>')` table function (§5.4).
- Repo: renamed `substrait-autograd` → `ddx`, both locally and on GitHub
  (`github.com/xqlsystems/ddx`; the old name redirects). The local git remote URL
  may still read `substrait-autograd.git` until repointed — harmless, GitHub
  redirects it. Tagline: "SQL-portable autograd," *not* "Substrait" (§6).

---

## 11. Milestones

With one path, one IR, and no Substrait, the plan is short. `ddx-core` is the
critical dependency; the two integrations then go in parallel.

- **M0 — Extract the core.** Create the workspace; lift the prototype's
  `src/autograd.rs` into `ddx-core`, re-pointing the rules from DataFusion `Expr`
  onto `sqlparser::ast::Expr`, and implement `rewrite_sql(sql, dialect)`. Port the
  15 rule unit tests + round-trip tests. *Exit:* `ddx-core` reproduces every
  prototype rule and rewrites SQL end-to-end, depending only on `sqlparser`.
- **M1 — Confirm R2** (short). Verify datafusion-python still can't inject an
  `AnalyzerRule` (keeps v1 on the rewrite). *Exit:* v1 path confirmed; any Path B
  seam noted as future-only.
- **M2 — DataFusion / Python.** `ddxdb` wheel: `rewrite_sql` + `Context.sql()`
  shim. Re-integrate into xarray-sql, deleting its vendored `autograd.rs` in favor
  of `ddx-core`. *Exit:* xarray-sql green on `ddx-core`, no regressions, checked
  against JAX.
- **M3 — DuckDB.** `ddx-duckdb` = the `ddx('<sql>')` table function (read SQL
  literal → `rewrite_sql` with `DuckDbDialect` → execute on inner connection →
  stream), plus the `ddxdb` client-side path for DuckDB-python. Integrate with
  duckdb-zarr; run the R1b Rust-extension smoke test. *Exit:* `grad` works
  end-to-end via `SELECT * FROM ddx('…')` on a real duckdb-zarr dataset.
- **M4 — Math roadmap & hardening.** Extend rules (§7), cross-engine equivalence
  vs. JAX, dialect canonicalization table, docs, benchmarks.
- **Future (post-v1, demand-driven):** native DataFusion `AnalyzerRule` (Path B),
  the C++/cxx.rs hybrid for bare `grad()` in DuckDB (§5.4 opt 4), `ddx-pg`, and a
  Substrait front-end only if a Substrait-native engine asks (§6).

---

## 12. Open questions for review

1. **DuckDB ergonomics — DECIDED (Alex, 2026-07-19).** `SELECT * FROM ddx('…')` is
   an acceptable v1 UX. Bare `grad()` remains the aspiration: pursue it later via
   an upstream DuckDB change, a C++ `OptimizerExtension`, or another native hook —
   tracked as a stretch goal (§5.4 option 4), not a v1 blocker.
2. **Column-name binding — REVISED (v0.2, §5.5).** v1 is **qualifier-aware
   syntactic** differentiation on the `sqlparser` AST — binding-*correct* for every
   query the engine accepts (SQL forces qualification exactly where a bare name
   would be ambiguous), and strictly better than the prototype's unqualified-only
   form. This partially walks back the earlier "invest fully in binding awareness
   now" decision: full catalog-driven resolution rides along with the future Path B
   rather than blocking v1. **For your review:** is qualifier-aware-with-hard-error-
   on-ambiguity acceptable for v1, or is a case you care about (which?) forcing
   full binding sooner?
3. **Where does `ddx-core`'s canonical function vocabulary live** — hard-coded
   match arms (as today) vs. a small registry API consumers can extend (third-party
   rule packs)? (Substrait-YAML-driven is off the table with §6.)
4. **Should xarray-sql depend on `ddx-core` directly, or vendor-then-migrate?**
   Direct dependency is cleaner but couples release cadences.
5. **Naming collision check** — is `ddx`/`ddxdb` free on crates.io, PyPI, and the
   DuckDB community registry? (I can check this quickly on request.)
6. **Does dropping Substrait cost anything you value?** §6 argues no (SQL is the
   portable surface; Substrait doesn't solve rewrite injection). Flagging
   explicitly because the project began as "substrait-autograd" — veto if there's
   an ecosystem/relationship reason to keep a Substrait artifact in v1.

---

_Next step:_ Alex to review; then iterate this doc (with agents) before we start
M0. The prototype's `autograd.rs` and its Python test suite are the concrete
starting materials for the `ddx-core` extraction.
