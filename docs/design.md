# `ddx` — portable autograd for composable databases

_author_: Alex Merose

_co-author_: Claude (Opus 4.8), via Claude Code

_reviewed by_: Claude (Fable 5), 2026-07-19 — adversarial review in §13

_created_: 2026-07-19

_last updated_: 2026-07-19

_status_: Design — iterating toward implementation.

> **Revision note (2026-07-19, v0.2 — strategic simplification).** After review, the
> design collapses along three axes that earlier drafts left open:
> 1. **SQL rewrite ("Path A") is the universal path.** Both acceptance targets are
>    reached by it alone. We *also* ship one in-engine rewrite ("Path B") in v1 —
>    an `AnalyzerRule` for native Rust DataFusion — as the cheap reference proof
>    that `ddx-core` drives an in-engine rewrite (§5.3); other engines' Path B is
>    future.
> 2. **One IR: the `sqlparser` AST**, not a bespoke `DExpr` + adapters. We
>    differentiate directly on `sqlparser::ast::Expr`, parse per-dialect, and render
>    back with precedence-safe `Display` (§5.1, G1). The core depends only on
>    `sqlparser` — no DataFusion `Expr`, no `protoc` — but that public type ties
>    `ddx-core`'s version to `sqlparser`'s fast-moving 0.x (pin = DataFusion's; §9,
>    G2). The one virtue the deleted `DExpr` had — insulating the API from that churn
>    — is the price; recorded, not reversed.
> 3. **Substrait is dropped** entirely — not a transport, IR, or dependency (§6).
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

**Sweet spot vs. bounded demo (G5).** The honest framing leads with **calculus as
columns on tidy scientific/array data** — sensitivity columns, small Jacobians,
Newton steps, curve fitting, physical derivatives alongside gridded data — which is
exactly the XQL / xarray-sql / duckdb-zarr audience, and which no other SQL-native
tool does at all. The in-SQL *training loop* is a genuine and impressive demo, but
it is **bounded to small models by construction**: this is *symbolic* forward-mode
differentiation, so an SGD step over N parameters costs N independent full
derivations of the loss, each re-evaluated per row (F6 swell × F10 no-reverse-mode
— the two multiply). That compounding is *the* canonical reason ML moved from
symbolic to reverse-mode AD (Baydin et al., JMLR 2018). So: low-N calculus is the
product; training loops are the wow, with an explicit small-N caveat (§7.2) — not
a general autodiff engine (§2). This positioning is what M4's benchmark targets
(swell vs. *N*, not just vs. expression size).

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
- **Not** a Substrait project — considered and rejected (§6); no Substrait
  transport, IR, or dependency.
- **Not** two injection paths *everywhere* in v1. The SQL rewrite ("Path A") is
  the universal path; the in-engine plan rewrite ("Path B") ships in v1 for **one**
  engine only — native Rust DataFusion, as the reference in-engine integration
  (§5.3). Other engines' Path B (notably DuckDB's C++ hybrid) is deferred.
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

### 3.2 The reusable crown jewel is a small, IR-shaped differentiation engine

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

### 3.3 Other inherited design decisions worth keeping

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
3. **One universal path + one reference in-engine path.** The SQL source-to-source
   rewrite ("Path A") reaches every v1 target and channel. We *also* ship the
   in-engine `AnalyzerRule` ("Path B") for native Rust DataFusion in v1 — the
   cheapest proof that `ddx-core` drives an in-engine rewrite, not merely a text
   preprocess (§5.3). Other engines' Path B is future.
4. **SQL is the portable surface.** grad/jvp/vjp are ordinary SQL function calls,
   so portability is free at the SQL level — no plan-interchange format needed.
   (Substrait was considered and rejected; §6.)
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

   Also v1: ddx-datafusion — bare grad() via an in-engine AnalyzerRule (Path B),
   the reference native integration (§5.3).
   Future (not v1): C++/cxx.rs hybrid for bare grad() in DuckDB (§5.4 opt 4) ·
   ddx-pg (pgrx)
```

**Paths in v1 (was: "do we need both A and B?").** The **SQL rewrite (Path A)** is
universal and reaches both acceptance targets on its own: xarray-sql *must* use it
(datafusion-python can't inject an `AnalyzerRule`, R2) and the DuckDB `ddx('<sql>')`
table function *is* the rewrite inside the extension (§5.4). We *also* build the
**in-engine `AnalyzerRule` (Path B) for native Rust DataFusion** — not because a
target needs it, but because it is the cheapest way to validate that `ddx-core`
drives a real in-engine plan rewrite (the other half of the thesis, §3.1), and it
is nearly free via an unparse→core→reparse bridge that reuses the one rule engine
(§5.3). DuckDB's bare-`grad()` Path B (C++ hybrid) stays post-v1.

### 5.1 `ddx-core` — the engine (the refactor)

Port the prototype's `src/autograd.rs`, but change the type it walks: **off
DataFusion `Expr`, onto `sqlparser::ast::Expr`** — the [`sqlparser`
crate](https://docs.rs/sqlparser/) (Apache's `datafusion-sqlparser-rs`, the same
parser DataFusion uses; **not** the third-party `sqlparser-patched`). It ships
`DuckDbDialect`, `PostgreSqlDialect`, `GenericDialect`, …, and `ast::Expr: Display`.
But `Display` is **not precedence-safe** for *constructed* trees (G1, below), so
rendering back to SQL means "`Display` + explicit `Nested` wrapping," not raw
`Display`. Version note (G2): `ddx-core` **pins the same `sqlparser` as DataFusion**
and **re-exports it** (`pub use sqlparser`); a `sqlparser` bump is a breaking release
of `ddx-core` (§9). Surface:

```rust
// Public entry point is an OBJECT (F9), not free functions, so the user rule
// registry and dialect/identifier config have a home — no global mutable state,
// no API break when the registry lands.
pub struct Ddx { /* rules: RuleRegistry, ident/dialect policy, … */ }

impl Ddx {
    pub fn new() -> Self;                                    // built-in rules
    pub fn register(&mut self, name: &str, rule: Rule);      // user-extensible (§12 Q3)

    // The whole path: parse the statement, find every grad/jvp/vjp call,
    // differentiate its argument, splice the derivative back *by source span*
    // (F5), return SQL text. A statement with no marker returns byte-identical.
    pub fn rewrite_sql(&self, sql: &str, dialect: &dyn Dialect) -> Result<String, DiffError>;

    // Lower-level, on the AST directly (used by the DataFusion AnalyzerRule bridge).
    pub fn differentiate(&self, e: &ast::Expr, wrt: &ColRef) -> Result<ast::Expr, DiffError>;
    pub fn jvp(&self, e: &ast::Expr, seeds: &HashMap<ColRef, ast::Expr>) -> Result<ast::Expr, DiffError>;
    pub fn vjp(&self, e: &ast::Expr, wrt: &ColRef, cotangent: &ast::Expr) -> Result<ast::Expr, DiffError>;
}

// Column identity read off the AST. Stores sqlparser `Ident`s (which keep
// quote-style), and is compared with *per-dialect* identifier folding (F1), not
// raw-string equality — the folding rule is dialect-specific (see the F1 note).
pub struct ColRef { pub qualifier: Option<Ident>, pub name: Ident }
```

The rules match on the `ast::Expr` variants we support and `NotImplemented` the
rest: `Expr::BinaryOp{left,op,right}` (`+ - * /`), `Expr::Function` (name-dispatched:
`sin`, `power`, …), `Expr::UnaryOp`(minus), `Expr::Cast`, `Expr::Nested`,
`Expr::Identifier`/`CompoundIdentifier` (leaves), `Expr::Value` (literals). This
is essentially what the earlier draft's `DExpr` enum *was* — a stripped-down SQL
expression tree — so we use `sqlparser`'s type directly and delete both the
bespoke IR and its four `From`/`To` adapters.

Design notes:

- **Extensible rule registry, keyed by function name.** The engine dispatches on
  the parsed function name (the prototype's `match name { "sin" => … }`), but
  exposed as a *registry users can extend*: `registry.register("myfn", rule)` adds a
  differentiation rule for a custom function. Built-ins populate it; for a unary
  `f(u)` a user rule supplies just `f'(u)` and the engine applies the chain rule
  (`· du`) automatically — so adding a function is a few lines, no fork (§12 Q3).
  (Binary/n-ary custom rules are a richer trait, likely post-v1.) A small
  canonicalization table folds dialect spellings (`ln`/`log`, `pow`/`power`) to one
  canonical name before dispatch.
- **Smart constructors own the numeric-type policy (R1b/F4).** `add/sub/mul/div/neg/
  square` do the `Zero`/`add_tangents` 0/1-folding *and* enforce floating-point
  arithmetic — because differentiation runs *pre-binding*, operand types are always
  unknown, and SQL integer `/` truncates on some engines (DataFusion: `grad(x/y,y)`
  → `(-x)/(y*y)`; on `BIGINT`, `1/4 = 0` not `-0.25`; DuckDB's `/` is float, so the
  two engines would silently disagree). So `div()` (and any construction that can
  hit integer inputs) wraps in `CAST(… AS DOUBLE)`, and literals are emitted
  `DOUBLE`-typed (DuckDB otherwise types `0.0` as `DECIMAL`). Slightly noisier
  output, correct on every engine. Pin with an integer-column test in M0.
- **Smart constructors also own *precedence* — the silent-wrong bug in "unparse via
  `Display`" (G1).** `sqlparser`'s `Display` for a binary op is literally
  `{left} {op} {right}` — **no precedence parentheses**. Parsed trees round-trip
  because the parser keeps explicit parens as `Expr::Nested`, but the differentiator
  *constructs* new trees, which have none. Verified (sqlparser 0.62.0): a constructed
  `(a+b)*c` Displays as `a + b * c` (reparses as `a+(b·c)`), and `a−(b+c)` as
  `a − b + c`. This hits the product/quotient rules immediately (`mul(cos(u), da+db)`
  → `cos(u) * da + db`) — a wrong number, valid SQL, nothing fails downstream: exactly
  the principle-5 class. The prototype dodged it only by using DataFusion's unparser,
  which wraps every op in `Nested` and runs a `remove_unnecessary_nesting` pass. Fix:
  the smart constructors **wrap composite operands in `Expr::Nested`** (verified to
  render `(a + b) * c` correctly); a later tidy pass can drop redundant parens.
  M0 exit criterion alongside F4.
- **Identifier folding, not raw-string equality (F1) — and the fold is
  *per-dialect*.** SQL unquoted identifiers are case-insensitive, so
  `grad(Temp*Temp, temp)` must match — otherwise it silently yields derivative `0`.
  But the exact rule differs by engine and **is not simply "casefold unquoted,
  exact-match quoted":**
  - **DataFusion / Postgres-style:** unquoted fold to lowercase; **quoted are
    case-sensitive** (`"Temp"` ≠ `"temp"`).
  - **DuckDB:** case-insensitive **throughout** — even *quoted* identifiers match
    case-insensitively (case is preserved for display only), so `"Temp"` = `"temp"`.
    Exact-matching quoted parts here would reintroduce the silent mismatch.

  So `ColRef` equality takes the dialect and applies its rule to each part. (The
  prototype got folding free via DataFusion's `parse_sql_expr`; the sqlparser-only
  core must do it explicitly — a regression risk for the M2 "no regressions" gate.)
  Output preserves original spelling.
- **Qualifier-aware, with an ambiguity guard on *uncertain* occurrences (§5.5, F2).**
  `ColRef` carries the qualifier from `CompoundIdentifier`. Hard-error only when an
  occurrence of the `wrt` base name can't be pinned syntactically — a *bare*
  occurrence when `wrt` is qualified, or a qualified occurrence when `wrt` is bare.
  A fully-qualified, unambiguous `wrt` like `grad(a.x*b.x, a.x)` is **accepted**.
  AST-only, no catalog (§5.5).
- **Reserve the marker names precisely (F8), case-insensitively (G7).** `grad`/`jvp`/
  `vjp` are claimed only as **unqualified** function calls (a user's `myschema.grad(…)`
  is left alone), but matched **case-folded** — `GRAD(x, x)` is a marker too. Miss the
  folding and `GRAD(x,x)` sails through unrewritten to a confusing engine-side
  "unknown function grad" error. (The pre-gate above folds too.)
- **Splice by source span, don't reprint the statement (F5) — and mind the span
  units (G3).** `rewrite_sql` first runs a **parse-free, case-insensitive pre-gate**
  — `(?i)(grad|jvp|vjp)\s*\(` (G7) — and returns the input **verbatim** if it
  doesn't hit. This is what makes the byte-identical guarantee real: a marker-free
  statement is *never parsed*, so sqlparser coverage gaps can't fail or reformat it
  (§5.3 F5). When the gate hits, it parses and replaces only the marker call's
  region. **But `sqlparser`'s `Spanned` gives `Location { line, column }` in 1-based
  *characters*, not byte offsets** (verified: in `SELECT 'héllo', grad(x,x) …` the
  `grad` is at byte 17 / column 16). So the splice is a small subsystem, not a
  one-liner: line/column→byte conversion (UTF-8- and multi-line-aware), splice
  multiple markers in **reverse source order**, rewrite **nested** markers bottom-up
  on the AST and splice only the outermost span, and fall back safely on empty spans
  (`Location` documents "Line 0 … for empty spans", so partial `Spanned` coverage is
  designed-in). The pre-gate may over-match inside string literals — harmless, the
  real parse decides; never fold the gate into the rewrite. Named M0 task with a
  multibyte + multi/nested-marker test.
- **0/1-folding is a stated NULL convention, not an accident (F11).** Folding drops
  structurally-zero terms, so `0 * (NULL expr)` becomes `0` where unfolded SQL would
  give `NULL`. This matches JAX's `Zero`-tangent semantics (a structural zero is a
  real `0`), but folded vs. unfolded derivatives then differ on NULL-bearing rows —
  so it is a documented decision with a test, not a quirk of the simplifier.
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
| **Rust DataFusion** (`ddx-datafusion` helper) | `GenericDialect` / DataFusion's | `ctx.sql(ddx.rewrite_sql(sql, dialect)?)` — a one-line wrapper; see below |
| `ddxdb` (Python → DataFusion) | `GenericDialect` / DataFusion-compatible | `Context.sql()` shim calls `rewrite_sql`, then the stock datafusion-python context plans it |
| `ddxdb` for DuckDB-python | `DuckDbDialect` | preprocess the string before `duckdb.sql(...)` |
| `ddx` (DuckDB ext) | `DuckDbDialect` | `ddx('<sql>')` table fn calls `rewrite_sql`, runs it on an inner connection (§5.4) |

**Native Rust DataFusion is the most direct consumer of `ddx-core`,** and in v0.2
it is *simpler* than before, not missing. DataFusion is built on the very
`sqlparser` crate `ddx-core` uses, so the integration is just: rewrite the SQL
string, then hand it to the stock `SessionContext`:

```rust
// The entire v1 "integration" for native Rust DataFusion. `ddx: &Ddx` carries the
// rule registry (F9), so callers can register custom rules once and reuse it.
// (Returns datafusion::Result; assumes `impl From<DiffError> for DataFusionError`
// so the `?` composes — a one-liner in the ddx-datafusion crate.)
pub async fn ddx_sql(ddx: &Ddx, ctx: &SessionContext, sql: &str) -> DataFusionResult<DataFrame> {
    ctx.sql(&ddx.rewrite_sql(sql, &GenericDialect {})?).await
}
```

No AnalyzerRule, no fork, no custom build — this is the same one path, called from
Rust. It can live as a tiny `ddx-datafusion` convenience crate or just be inlined
by the caller; it is *not* a required v1 milestone (our two acceptance targets are
xarray-sql and duckdb-zarr), but it is essentially free to offer.

The prototype's `rewrite_grad_in_sql` / `GradSqlRewriter` is exactly this driver,
minus its detour through DataFusion `Expr` (it currently parses to DF `Expr` to
differentiate, then unparses; we differentiate the `ast::Expr` directly instead).

### 5.3 The two v1 paths: universal SQL rewrite + DataFusion in-engine rule

**Path A — SQL source-to-source rewrite (universal, every target).** Intercept the
SQL string before it reaches the engine, rewrite every `grad`/`jvp`/`vjp` call to
derivative SQL, pass plain SQL onward. It runs *before* planning, so it works for
every query shape the parser accepts — recursive CTEs, DML, subqueries — which is
what lets a whole training loop live in one query. xarray-sql and the DuckDB
extension rely on it.

*Known bound (F5): `sqlparser` is a whole-query gatekeeper.* When a query *does*
contain a marker, Path A parses the **entire** statement, so applicability is capped
by sqlparser's per-dialect coverage — not by what `grad` touches. The gaps are
narrower than earlier drafts claimed (spike, `DuckDbDialect` @ sqlparser 0.62.0,
G9): `SELECT * EXCLUDE`, `FROM`-first (`FROM t SELECT 1`), bare `FROM t`, lambdas
(`x -> x+1`), and `t.* REPLACE (…)` **all parse**; the real misses are **`PIVOT`**
and **`#1` positional columns**. So the treadmill is real (PIVOT proves it) but not
dire, and it applies *only to marker-bearing queries*. We limit the damage two ways
(both v1): a **parse-free pre-gate** returns any
marker-free statement **byte-identical without parsing it at all** (so unsupported
syntax only fails a query that *actually contains* a marker), and when a marker is
present the derivative is spliced **by source span** rather than by reprinting the
statement (§5.1, F5) — so parse coverage bounds only marker-bearing queries, and
reprint fidelity is not a separate risk.

**Path B — in-engine plan rewrite, shipped for native DataFusion in v1.** A marker
UDF + `AnalyzerRule` so `grad()` works bare, with no wrapper (`SELECT grad(sin(x),
x) FROM t` directly), across both the SQL and DataFrame APIs. We promote this into
v1 for exactly one engine — native Rust DataFusion — because it is the cheapest
possible **validation of the core architectural claim**: that `ddx-core` can drive
an *in-engine* plan-time rewrite, not just a text preprocess. Both Path-A targets
exercise only the text path, so without this the "portable rewrite hook" half of
the thesis (§3.1) would ship unproven. (It does **not** de-risk the DuckDB C++ path
— G6: that path's hard part, rebuilding a *bound* expression with `ColumnBinding`
indices and catalog entries, is exactly what the DataFusion bridge *avoids* by
leaning on DF's unparse/re-plan utilities; the two share only the shallow "walk
plan, find marker, substitute" pattern. The honest DuckDB de-risker is the
M3-adjacent spike, §11/§12 Q6.)

Implementation — the **bridge**, not a second rule engine. The rule walks the bound
`LogicalPlan`, and for each `grad()` `ScalarFunction`:

1. unparse its argument with DataFusion's `expr_to_sql`, which emits a
   `sqlparser::ast::Expr` — *exactly* `ddx-core`'s input type (**iff the versions
   are pinned identical**, G2; if they ever diverge, the bridge degrades to a
   string-level round-trip: `Display`-with-parens → DataFusion's own parser — still
   one rule engine, less elegant);
2. differentiate via `ddx-core`;
3. re-plan the resulting `ast::Expr` back to a DataFusion `Expr` against the node's
   schema; replace and recompute the schema.

Two day-one details (G7): (a) `add_analyzer_rule` runs *after* the default analyzer,
so `TypeCoercion` may already have injected `Cast`s into the marker's argument by the
time the rule sees it — fine (`Expr::Cast` has a rule) but it changes the symbolic
form and swells output, and the marker UDF must be coercion-tolerant (variadic-any);
(b) step 3's re-plan needs a `ContextProvider`/function registry — the seam is
`SessionState::create_logical_expr`. One rule engine, shared with Path A; and because
the input `Expr` is already **bound**, its columns unparse *qualified*, so this path
is binding-aware for free (the §5.5 guard never fires). We deliberately do **not**
resurrect the prototype's native `differentiate(&Expr)`: that would reintroduce the
duplicate rule set v0.2 removed, taxing every future rule (§7) twice.

**Still future.** DataFusion Path B is *not* reachable from datafusion-python (R2),
so xarray-sql keeps Path A. DuckDB's bare-`grad()` Path B needs the C++/cxx.rs
hybrid (§5.4 option 4; the stable C API has no hook, R1) and stays post-v1.

### 5.4 Per-engine integration & distribution

**DataFusion / `ddxdb` (Python) → xarray-sql.**
`datafusion-python` does not expose injecting an `AnalyzerRule` into its
`SessionContext` (R2) — which is *why* the SQL rewrite is the path here, not a
limitation to work around. `ddxdb` wraps a `Ddx` (F9) and exposes `rewrite_sql`
plus a `Context.sql()` shim. **xarray-sql pulls it in as an optional extra —
`pip install "xarray-sql[ddx]"`** (the `[ddx]` extra depends on `ddxdb`), so
autograd is opt-in and xarray-sql carries no autograd weight for users who don't
ask for it (§12 Q4). With the extra installed, xarray-sql routes `grad()` queries
through `ddxdb` rather than its old vendored `autograd.rs`. (Native Rust DataFusion
is covered separately just below.)

**DataFusion (native Rust) / `ddx-datafusion`.**
The reference in-engine integration, and a v1 deliverable. The `ddx-datafusion`
crate (deps: `ddx-core` + `datafusion`) exposes two entry points:
- **`ddx_sql(ctx, sql)` helper (Path A):** one line —
  `ctx.sql(ddx.rewrite_sql(sql, dialect)?)` (§5.2, F9). Parse with the dialect
  DataFusion uses so the rewrite accepts exactly what `ctx.sql` would.
- **Marker UDFs + `AnalyzerRule` (Path B):** bare `grad()` with no wrapper, across
  the SQL *and* DataFrame APIs, via the unparse→`ddx-core`→reparse bridge (§5.3) —
  one rule engine, binding-aware for free. It ships in v1 as the cheapest proof
  that `ddx-core` drives an in-engine rewrite, even though neither acceptance
  target needs it (xarray-sql is Python → Path A; duckdb-zarr is DuckDB). (It does
  *not* stand in for the DuckDB C++ boundary — G6, §5.3.)

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

  1. **In-extension Path A via a table function (recommended primary — but
     *transitional*).** Brand it, in both this doc and the extension's own docs, as
     "the portable form until bare `grad()` lands," so it never hardens into a
     compatibility contract (its real risk is social, not technical — §12 Q6). The
     same header exposes `duckdb_bind_get_parameter` (read a **literal SQL string at
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
     `ddx('…')` wrapper instead of bare `grad()`. Re-entrancy is validated (R1b).
     **Three mechanics still to nail down (F7) — these are the actual substance of
     M3, not details:**
     - **Bind-time schema (the hard part).** A DuckDB table function must declare
       its result columns at *bind* time, so `ddx('…')` must prepare/`DESCRIBE` the
       *rewritten* query on the inner connection during bind to learn its schema.
       Feasible-looking but unspiked — named as an explicit M3 task.
     - **Connection-scoped state is lost.** The inner `duckdb_connect` is a *new
       session*: the caller's **temp tables, session `SET`s, and prepared
       statements are invisible** inside `ddx('…')` (R1b covered only transaction
       visibility; this is broader and more common). Folds into the "when to use
       client-side Path A" guidance below.
     - **DML policy — decide on purpose.** DuckDB *does* ship a built-in
       `query('sql')` table function (a shape precedent), but — correcting an
       earlier claim here — it is **not** SELECT-only: its docs warn it runs
       *arbitrary* queries and can alter database state, and it accepts **only a
       literal string** argument ([DuckDB docs](https://duckdb.org/docs/stable/guides/sql_features/query_and_query_table_functions)).
       So precedent neither forces SELECT-only nor offers a constant-folding
       shortcut. R1b showed inner DML *works*; we still default `ddx('…')` to
       **read/SELECT-only**, but on our *own* grounds — re-entrancy caution and
       community-review conservatism — and route DML loops through client-side
       Path A.
     - **Ergonomics.** SQL-in-a-string quoting is unpleasant for the flagship
       recursive-CTE examples; document **dollar-quoting** — `ddx($$ … $$)` — as the
       house style.
  2. **Client-side Path A for DuckDB-Python (ships day one).** `ddxdb` preprocesses
     the SQL string before `duckdb.sql(...)`. Zero engine hooks; the fastest path
     to a working duckdb-zarr integration and a useful fallback.
  3. **A scalar `ddx_rewrite(sql) → sql` helper.** Pure string→string, trivially
     safe (no inner connection), for users who want to inspect or run the
     rewritten SQL themselves.
  4. **A hybrid C++/Rust extension via an `OptimizerExtension` (stretch goal;
     spiked 2026-07-19).** The *only* route to bare `grad()` anywhere in a normal
     SELECT — and its advantage is **structural, not cosmetic**: running *after
     binding*, it is immune to the entire silent-wrong class Path A must guard
     against syntactically. Columns arrive as resolved bindings, so **F1**
     (identifier case), **F2/F3** (qualification ambiguity, projection boundaries)
     simply cannot arise, and **F5** (sqlparser as whole-query gatekeeper)
     disappears because DuckDB parses the query and only the marker's expression
     crosses to Rust. That is the real long-term case for it (§12 Q6). Architecture:
     the DuckDB **C++
     [extension-template](https://github.com/duckdb/extension-template)** (CMake +
     vcpkg + DuckDB submodule) registers an `OptimizerExtension` whose
     `optimize_function` walks the bound `LogicalOperator` plan, finds each `grad`
     `BoundFunctionExpression`, and replaces it with the derivative computed by our
     Rust `ddx-core`, called across a **[cxx.rs](https://cxx.rs/)** bridge
     (`ddx-core` built as a `staticlib`, linked into the C++ extension; `cxxbridge`
     CLI generates the C++ glue in the CMake build). Distributes via community
     extensions (`INSTALL ddx FROM community;`) like any C++ extension.

     _Spike verdict:_ **cxx.rs is the right FFI tool and makes the Rust↔C++ call
     itself trivial — but it does not remove the real cost, so this stays
     post-v1.** (a) A C++ extension links DuckDB *internals* and is rebuilt per
     DuckDB version against an unstable internal API — though this tax is *partly
     absorbed* by the community-extensions CI, which already rebuilds C++ extensions
     each release (so it's lighter than earlier drafts implied; most community
     extensions live here). (b) **The hard part is the expression boundary, which
     cxx does not solve — and it is bidirectional:** at the optimizer stage columns
     are bound *structurally by index* (`ColumnBinding`), so a bound `Expression`
     does **not** round-trip to re-parseable SQL. Getting the argument *out* to
     `ddx-core`'s `ast::Expr` (serialize DuckDB's expression → Rust deserializer) is
     the easy direction; **rebuilding a *bound* derivative expression on the way
     back — with correct `ColumnBinding` indices and catalog function entries for
     `cos`, `power`, … — is the gnarly part** (the 20% that is 80% of the work, and
     DuckDB-version-coupled forever). Its miniature is exactly the M3-adjacent spike
     in §12 Q6. `autocxx` (auto-binding DuckDB's headers) is tempting but the headers
     are large/complex — prefer a narrow hand-written cxx bridge.
  5. **`CREATE MACRO`** — rejected: macros are fixed expansions and cannot perform
     general differentiation.

Plan: ship (1) as the Rust community extension for duckdb-zarr (**branded
transitional**), with (2) as the Python-side convenience. Keep (4) — the cxx.rs
hybrid — as the documented, correctness-superior route to bare `grad()`, revisited
when prioritized. **Don't build (4) up front** (§12 Q6), but **pull its risk
forward** with a miniature spike (round-trip one `BoundFunctionExpression` through
`ddx-core` and back), so the full extension becomes a known quantity schedulable on
demand rather than a multi-week unknown.

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

### 5.5 Column identity and the projection boundary

Two things a pre-binding syntactic rewrite has to get right about columns. The
first — telling `a.x` from `b.x` — mostly dissolves with a small guard. The second
— that `grad` treats any *upstream-computed* column as a constant (F3) — does
**not** dissolve; it is a real semantic contract that lands on the pitched use case,
and it is the more important half of this section.

- **`ColRef` reads the qualifier straight off the AST.** `sqlparser` parses `a.x`
  as a `CompoundIdentifier`, so `ColRef { qualifier: Some("a"), name: "x" }` falls
  out for free. `grad(a.x + b.x, a.x)` differentiates the right column with no
  catalog. This is strictly better than the prototype's unqualified-only form.
- **Binding-correct given a guard keyed on *uncertain* occurrences (F2).** Match by
  `ColRef` (with identifier folding, §5.1 F1). The danger is an occurrence of the
  `wrt`'s base name whose binding we can't pin syntactically — and which occurrences
  those are depends on how `wrt` is written:
  - **Qualified `wrt`** (`a.x`): safe *unless a bare occurrence of the base name*
    (`x`) also appears in the argument — a bare `x` might bind to `a.x`, so treating
    it as a constant leaf could be silently wrong (`grad(x * a.x, a.x)` should be
    `2x`; naive matching gives `x`, a wrong number whose SQL still binds). But an
    occurrence under a *different* qualifier (`b.x`) is an unambiguously distinct
    column and is fine — so **`grad(a.x * b.x, a.x)` → `b.x` is accepted** (exactly
    the form §7.1 recommends; the guard must not reject it).
  - **Unqualified `wrt`** (`x`): safe only if *every* occurrence of the base name in
    the argument is also bare. If the base name appears qualified, or under ≥2
    qualifiers, we can't tell which column `x` means (`grad(a.x * b.x, x)`).

  When it fires, hard-error and demand full qualification. AST-only, no catalog.
  Keeps "fail loud, never silently wrong" (§4.5) *without* rejecting a
  fully-qualified, unambiguous `wrt`.
- **Wholly-unqualified ambiguity fails loud downstream.** `grad(sin(x), x)` with `x`
  in two joined tables emits SQL that still contains bare `x`, which the engine
  rejects as ambiguous. Catalog-driven **Path B** *resolves* these (and `SELECT *`
  expansion) instead of erroring; deferred (§5.3).
- **Columns are leaves — `grad` does not see through CTEs/views (F3).**
  Differentiation stops at column references, so a column *computed upstream* (a
  CTE/subquery/view select-list expression) is an opaque constant — every projection
  boundary is an implicit `stop_gradient`. **Contract:** `grad` differentiates the
  expression *as written, against the relation it directly queries*, never through
  view/CTE definitions. This is defensible relational semantics but a real trap for
  the pitched use case — factoring a loss through a CTE silently drops terms:
  ```sql
  WITH v AS (SELECT x, sin(x) AS s FROM t)
  SELECT grad(s * x, x) FROM v       -- ds/dx treated as 0 → result = s = sin(x)
  SELECT grad(sin(x) * x, x) FROM t  -- inlined by hand → cos(x)*x + sin(x)
  ```
  Because `rewrite_sql` sees the whole statement, we add a **best-effort guard**: if
  a marker argument references an identifier that is a *computed* select-list alias
  of a CTE/derived table **in the same statement**, error with "differentiate inside
  the CTE instead." It cannot see catalog views (no schema) — that residual is
  documentation-only.
  - **Carve-out — differentiating *w.r.t.* the alias is allowed (G4).** When the
    computed alias *is* the `wrt` (`grad(s*s, s)`), *every* occurrence of it is the
    differentiation leaf, so no gradient term can be silently dropped — the F3 danger
    structurally cannot occur, and `d/ds (s*s) = 2s` is exactly right (subject to the
    engine allowing the alias in scope). The guard must therefore fire **only when a
    computed alias appears as a *non-`wrt`* term** in the argument, never when it is
    the `wrt`. (Without this carve-out the guard would reject `grad(s*s, s)`, a case
    §5.5/§7.1 endorse — a self-contradiction.)

**Net:** v1 is qualifier-aware syntactic differentiation (needing no binder), plus
the **projection-boundary contract** (`grad` differentiates the expression as
written, treating upstream-computed columns as constants — F3). This honors "don't
be limited to unqualified names" while being honest that factoring through a CTE
drops gradient terms. The fully-bound path (which would erase both the
qualification guard *and* the projection-boundary caveat — §5.4 opt 4, §12 Q6)
rides along with Path B rather than blocking v1. Open sub-question in §12 Q2.

---

## 6. Substrait: considered and rejected

The project began life as "substrait-autograd," so for the record: **Substrait is
deliberately not used** — not as a plan transport, an expression IR, or a
dependency.

- **It was tried as a plan transport and removed.** The prototype round-tripped the
  whole `LogicalPlan` through Substrait; its producer can't represent recursive
  CTEs or DML (reproduced on datafusion 54.0.0 — `Unsupported plan type:
  RecursiveQuery` / `DmlStatement`), which are exactly the training-loop shapes we
  need. Written up with a repro in
  **[ddx#1](https://github.com/xqlsystems/ddx/issues/1)** (refs
  [xarray-sql#192](https://github.com/xqlsystems/xarray-sql/pull/192),
  [#197](https://github.com/xqlsystems/xarray-sql/issues/197)).
- **It isn't needed for portability.** `grad`/`jvp`/`vjp` are ordinary SQL function
  calls, and every target speaks SQL, so **SQL is already the portable surface**. A
  Substrait expression IR would be a redundant second format (plus a `protoc` tax)
  carrying nothing SQL text doesn't.
- **It doesn't solve our actual problem** — *rewrite injection* (a plan-time hook
  per engine, §3.1). Substrait standardizes plan interchange, not plan-time
  rewriting, and has no notion of a marker that must not execute.

Door left open, on demand only: if a Substrait-native engine ever wants `ddx`, a
`Substrait::Expression → ast::Expr` adapter could front `ddx-core` behind a feature
flag. Not part of v1, and not what defines this project.

---

## 7. The differentiation surface & math roadmap

**v1 surface (port of the prototype, unchanged semantics):**

- `grad(expr, column)` → `d(expr)/d(column)`.
- `jvp(expr, column, tangent)` → forward-mode `d(expr)/d(column) · tangent`.
  Multi-input directional derivative = sum of `jvp` terms.
- `vjp(expr, column, cotangent)` → `cotangent · d(expr)/d(column)`, the vjp
  *contraction* for a single input. **Honest caveat (F10):** this is a
  cotangent-scaled *forward* computation per column, **not** reverse-mode
  accumulation — there is no shared backward pass, so an N-input gradient still
  costs N independent differentiations and does not amortize the way JAX's `vjp`
  does. For a scalar output it numerically coincides with `jvp`; its value is the
  *seeding surface* (seed a cotangent on the output), not reverse-mode efficiency.
- `differentiate_sql(expr, wrt)` → derivative as SQL text (the "calculus
  compiler" escape hatch). The prototype's third `columns` argument (§3.3) is
  dropped: it existed only to synthesize a DataFusion schema for standalone
  parsing, which `sqlparser` does not need.
- Rules: `+ - * /`, unary chain rule for the trig/inverse-trig/exp/log/hyperbolic
  set + `abs`, `power` with constant base or exponent. Higher-order via nesting.
  Through-aggregate via linearity (`AGG(grad(...))`).

### 7.1 Concretely: what you can and can't write

A `grad(...)` call is rewritten *in place* into ordinary SQL, so anywhere a scalar
expression is legal, `grad` is legal. Worked rewrites (what the user types → what
the engine actually runs):

| You write | Rewrites to | Works? |
| --- | --- | --- |
| `SELECT grad(sin(x)*y, x) FROM g` | `SELECT (cos(x)*y) FROM g` | ✅ |
| `SELECT grad(x*y,x) AS dfdx, grad(x*y,y) AS dfdy FROM g` | `SELECT y AS dfdx, x AS dfdy FROM g` | ✅ full gradient as tidy columns |
| `SELECT grad(grad(power(x,3),x),x) FROM g` | `… (6*power(x,1)) …` | ✅ higher-order (nesting) |
| `SELECT grad(a.v * b.w, a.v) FROM t a JOIN u b …` | `… (b.w) …` | ✅ qualified across joins |
| `SELECT jvp(sin(x),x,dx), vjp(sin(x),x,w) FROM g` | `(cos(x)*dx)`, `(w*cos(x))` | ✅ forward / cotangent-seeded (not reverse-accum — F10) |
| `SELECT AVG(grad(loss, theta)) FROM batch` | `AVG( d(loss)/d(theta) )` | ✅ one gradient-descent step (linearity) |
| `SELECT a+b AS s, grad(s*s, s) FROM t` | `…, (s + s)` | ✅ differentiate w.r.t. a computed alias (G4; `s` is the leaf) |
| `WITH RECURSIVE n AS (… x-(x*x-2)/grad(x*x-2,x) …) …` | `… /(x+x) …` | ✅ training loop in one query (Path A) |
| `INSERT INTO p SELECT theta-lr*grad(loss,theta) FROM …` | rewritten SELECT | ✅ DML update rule (Path A) |
| `SELECT grad(sin(x),x) FROM t` in **DuckDB** | needs `SELECT * FROM ddx('…')` (v1); bare works only in native DataFusion (Path B) | ⚠️ wrapper (§5.4) |

What it will **refuse** (a clear error, never a wrong number — §4 principle 5):

| You write | Result |
| --- | --- |
| `grad(atan2(x,y), x)` | ❌ `NotImplemented` — `atan2` has no rule yet (roadmap) |
| `grad(power(x,x), x)` | ❌ `NotImplemented` — general `u^v` not yet (roadmap) |
| `grad(CASE WHEN x>0 THEN x END, x)` | ❌ `NotImplemented` — conditionals not yet (roadmap) |
| `grad(x > 0, x)` / string / date exprs | ❌ `NotImplemented` — not differentiable (permanent) |
| `grad(a.x * b.x, x)` in a self-join | ❌ hard error — ambiguous unqualified `wrt`; write `a.x` (§5.5 F2) |
| `grad(x * a.x, a.x)` where bare `x` also binds `a.x` | ❌ hard error — bare `x` may be the `wrt` column; qualify it as `a.x` (§5.5 F2) |
| `WITH v AS (SELECT sin(x) AS s …) SELECT grad(s*x, x) FROM v` | ❌ error (best-effort) — computed CTE alias `s` used as a *non-`wrt`* term; differentiate inside the CTE (§5.5 F3/G4) |
| `grad(x*y, x+y)` | ❌ error — the `wrt` argument must be a bare column, not an expression |
| `grad(SUM(f), x)` | ❌ rejected by SQL scoping — `x` is gone after aggregation; write `SUM(grad(f,x))` |

The mental model: **if every function in the expression has a rule and the `wrt` is
an unambiguous column, it works in any query shape; otherwise you get a typed
error at rewrite time, before the query runs.** Expanding the first table's left
column (more functions, `u^v`, conditionals) is exactly the §7 roadmap below.

**Roadmap (each an explicit rule addition, fail-loud until then):**

- General `u^v` (both variable) via the `exp(v·ln u)` trick.
- `CASE`/conditional and `min`/`max`/`greatest`/`least` — subgradients, with a
  documented convention at kinks (JAX-style), mirroring how `abs` uses `signum`.
- `atan2`, `log(base, x)`, `cbrt`, `expm1`/`log1p`, `pow`/`^` operator spellings.
- Dialect name normalization table (canonical → per-engine spellings).
- Clear taxonomy of "not differentiable" (comparisons, string/temporal ops,
  window functions) that stays `NotImplemented`.

### 7.2 Known limitation: symbolic expression swell (F6)

Product/quotient rules duplicate their operands (`|d(f·g)| ≈ |f|+|g|+|df|+|dg|`),
so an n-factor product yields an O(n²) derivative, `grad(grad(…))` compounds
multiplicatively, and an N-parameter gradient is N columns each re-deriving the
whole loss. This is the classic symbolic-diff blowup; 0/1-folding trims easy zeros
but does not share subexpressions, so a subexpression appearing k times is
recomputed per row k times (unless the engine's CSE catches it — partial,
engine-dependent). And `vjp` (F10) offers no reverse-mode amortization, so the two
compound (G5): an N-parameter SGD step is N independent full derivations of the loss,
per row, per iteration — which is precisely why ML abandoned symbolic diff for
reverse-mode AD (Baydin et al., JMLR 2018). **v1 accepts this and positions around
it** (low-N scientific calculus is the sweet spot, §1), but: (a) M4 benchmarks
**swell vs. N** (not just vs. expression size) so the cliff is measured, not
discovered, and so nothing is promised to duckdb-zarr users past where it holds; (b)
the post-v1 remedy is a **let-binding pass** that factors shared subexpressions into
projected columns (`…, cos(x) AS __ddx_t1`) or a CTE. (Reverse-mode proper is a
larger, later item — see §7/§12 on `vjp`.)

---

## 8. Testing & verification

Differentiation is a numerical-correctness feature; the test strategy is
layered and reuses the prototype's:

- **Unit (rule) tests in `ddx-core`** — port the prototype's 15 Rust tests;
  every rule pinned symbolically on `ast::Expr`.
- **Round-trip property tests — semantic, not just "parseable" (G1).** `construct →
  Display → reparse` must **equal the constructed AST modulo `Nested`** (normalize
  away parentheses on both sides, then compare). A test that only checks the output
  *parses*, or compares the reparse to itself, sails right past the precedence bug
  (`(a+b)*c` → `a + b * c` reparses fine but *wrong*). Fuzz small random trees per
  dialect; assert the reparsed tree matches the differentiator's tree.
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
- **Convention-pinning tests (F11, F12) — not blind comparison.** Several points
  need an *agreed* answer pinned by a test rather than a naïve oracle compare:
  - **Kinks (F12i):** `abs` at 0 — our `signum` rule gives 0, but JAX's convention
    at the kink differs (`jax.grad(jnp.abs)(0.0)` — verify the exact value, ~1.0).
    Pin the convention point explicitly; don't compare blindly at kinks.
  - **Domain-widening (F12ii/iii):** a derivative can fail where the primal
    doesn't — `sqrt(x)` is fine at `x=0` but `1/(2*sqrt(x))` divides by zero, and
    engines disagree (IEEE `inf` vs `NULL` vs error); likewise `tan` near π/2, `ln`
    near 0. Cross-engine equivalence needs a stated **domain-edge policy**: sample
    away from edges, *or* pin per-engine expected behavior at them.
  - **NULL/folding (F11):** test that the 0-folding NULL convention (§5.1) is the
    decided one — folded and unfolded derivatives agree everywhere *except* the
    documented NULL-row cases.
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
  what keeps xarray-sql on the SQL rewrite. (If a seam exists, it only *adds* a
  Path B option for xarray-sql; it does not change v1.)

---

## 9. Monorepo layout (proposed)

```
ddx/                               (repo; crates published under the ddx-* names)
├── crates/
│   ├── ddx-core/                   # the engine — differentiate sqlparser::ast::Expr
│   │                               #   + rewrite_sql; dep: sqlparser only
│   ├── ddx-datafusion/             # markers + AnalyzerRule (Path B) + ddx_sql helper
│   │                               #   deps: ddx-core, datafusion
│   └── ddx-duckdb/                 # DuckDB community extension: `ddx('<sql>')` table fn
├── python/
│   └── ddxdb/                      # PyO3/maturin wheel: rewrite_sql + Context.sql() shim
├── docs/
│   └── design.md                   # this file
├── tests/                          # cross-engine numeric-agreement suites (vs JAX)
└── future/                         # not v1 — see §5.4
    ├── ddx-duckdb-cpp/             #   C++/cxx.rs hybrid for bare grad() (§5.4 opt 4)
    └── ddx-pg/                     #   Postgres via pgrx (needs array/XQL support first)
```

Rationale: the v1 surface is **`ddx-core` + `ddx-datafusion` + `ddx-duckdb` +
`ddxdb`**. `ddx-core` publishes independently (dep: `sqlparser` only, no `protoc`,
**no DataFusion**) so anyone can drive it from a new engine; the heavy `datafusion`
dependency is quarantined in `ddx-datafusion`. Future crates are physically
separated so the v1 build stays light. (The `future/` directory is illustrative —
those may live as un-published workspace members or separate repos.)

**`sqlparser` version policy (G2) — a real cost of the "one IR" choice.** `ddx-core`'s
public API takes and returns `sqlparser::ast::Expr`, and the Path B bridge (§5.3)
requires `ddx-core` and DataFusion to resolve the **identical** `sqlparser` 0.x —
Cargo treats `0.62` and `0.63` as incompatible, so a mismatch makes them two unrelated
types and the bridge won't compile. `sqlparser` has shipped ~14 breaking 0.x releases
in ~24 months and DataFusion adopts each with a lag, while §5.3/G9 wants the newest for
`DuckDbDialect` coverage — the two paths pull the pin in opposite directions. Policy:
1. **Pin `sqlparser` to DataFusion's requirement** (today `^0.62`). The bridge is a v1
   deliverable and a broken bridge is a compile failure; the DuckDB-coverage cost is
   bounded (G9) and the F5 pre-gate limits blast radius.
2. **Re-export it** — `pub use sqlparser` from `ddx-core` — so downstream consumers
   can't accidentally link a mismatched version.
3. **A `sqlparser` bump is a breaking release of `ddx-core`** (standard Rust
   public-dependency semver). Say so in the changelog policy.
4. **Degraded mode, if the pins ever must diverge:** the bridge falls back to a
   string-level round-trip (§5.3) — one rule engine still, less elegant.
This is the one virtue the deleted `DExpr` had (API insulation from `sqlparser` churn);
recorded as a trade, not reversed.

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
- **Practical (availability confirmed 2026-07-19):** `autograd` is taken on PyPI
  (HIPS). The bare `ddx` crate on crates.io is taken (a dead project) — so no
  umbrella `ddx` crate, which we don't need. `ddx-core`, `ddx-datafusion`,
  `ddx-duckdb`, and `ddxdb` are all free on crates.io; `ddxdb` is free on PyPI;
  `ddx` is free on the DuckDB community registry.

Distribution:

- Rust crates (v1): `ddx-core`, `ddx-datafusion`, `ddx-duckdb` on crates.io (all
  confirmed available). No bare `ddx` crate.
- Python: `pip install ddxdb` (standalone), and `pip install "xarray-sql[ddx]"` —
  a coordinated optional extra that pulls in `ddxdb` so xarray-sql users opt into
  autograd without it becoming a hard dependency (§5.4, §12 Q4).
- DuckDB: `INSTALL ddx FROM community;` → the `ddx('<sql>')` table function (§5.4).
- Repo: renamed `substrait-autograd` → `ddx`, both locally and on GitHub
  (`github.com/xqlsystems/ddx`; the old name redirects). The local git remote URL
  may still read `substrait-autograd.git` until repointed — harmless, GitHub
  redirects it. Tagline: "SQL-portable autograd," *not* "Substrait" (§6).

---

## 11. Milestones

With a lean core and one IR, the plan is short. `ddx-core` is the critical
dependency; the integrations then go in parallel.

- **M0 — Extract the core.** Create the workspace; lift the prototype's
  `src/autograd.rs` into `ddx-core`, re-pointing the rules from DataFusion `Expr`
  onto `sqlparser::ast::Expr`, and implement `rewrite_sql`. Port the 15 rule unit
  tests. **Also lands the correctness/API fixes now, before publish:** (a) the `Ddx`
  object API (F9); (b) per-dialect identifier folding + case tests (F1); (c) the
  uncertain-occurrence ambiguity guard (F2); (d) the numeric-type policy in the smart
  constructors + integer-column tests (F4); (e) **precedence-safe construction (F5's
  sibling, G1): `Nested`-wrapping smart constructors + a `construct→Display→reparse→
  compare-modulo-Nested` property test** — the one that actually catches the
  precedence bug; (f) **span→byte splicing (G3): line/column→byte conversion with a
  multibyte + multi/nested-marker test**, plus the parse-free case-insensitive
  pre-gate (F5/G7); (g) pin+re-export `sqlparser` (G2). *Exit:* `ddx-core` reproduces
  every prototype rule, rewrites SQL end-to-end, and **passes the F1/F2/F4/G1/G3
  tests** — depends only on `sqlparser`.
- **M1 — Confirm R2** (short). Verify datafusion-python still can't inject an
  `AnalyzerRule` (keeps v1 on the rewrite). *Exit:* v1 path confirmed; any Path B
  seam noted as future-only.
- **M2 — DataFusion (Python + native Rust).** (a) `ddxdb` wheel: `rewrite_sql` +
  `Context.sql()` shim; re-integrate into xarray-sql, deleting its vendored
  `autograd.rs` in favor of `ddx-core`. (b) `ddx-datafusion`: marker UDFs + the
  `AnalyzerRule` bridge (unparse→`ddx-core`→reparse), plus the `ddx_sql` helper —
  mind the `TypeCoercion` ordering and `create_logical_expr` seam (G7). **Prereq
  (G7): pull a *minimal* JAX-oracle numeric-agreement harness forward from M4** —
  M2's "green vs JAX" gate is unenforceable without it. *Exit:* xarray-sql green on
  `ddx-core` (vs JAX, no regressions) **and** bare `grad()` runs end-to-end through
  the `AnalyzerRule` in a native DataFusion test — the first proof of an in-engine
  rewrite.
- **M3 — DuckDB.** `ddx-duckdb` = the `ddx('<sql>')` table function (read SQL
  literal → `rewrite_sql` with `DuckDbDialect` → execute on inner connection →
  stream), plus the `ddxdb` client-side path for DuckDB-python. Integrate with
  duckdb-zarr; run the R1b Rust-extension smoke test. **Named tasks, not
  discoveries (F7):** the bind-time-schema spike (declare result columns by
  `DESCRIBE`-ing the rewritten query on the inner connection) and the DML policy
  decision (SELECT-only by default). Also brand the extension **transitional** in
  its docs. *Exit:* `grad` works end-to-end via `SELECT * FROM ddx('…')` on a real
  duckdb-zarr dataset, with the schema/DML behavior documented.
- **M3-adjacent spike (de-risks §5.4 opt 4; schedule with M3, not after).** The
  miniature of the whole C++ hybrid: from an `OptimizerExtension`, serialize **one**
  `grad` `BoundFunctionExpression` out to `ddx-core`, differentiate, and **rebuild
  one bound derivative expression back in** — correct `ColumnBinding` indices,
  catalog entries for `cos`/`power`. That reconstruction is the entire risk of opt 4
  in miniature (§12 Q6). *Exit:* a yes/no on tractability in *days*, so the full
  extension becomes schedulable-on-demand rather than a multi-week unknown — without
  putting it on the duckdb-zarr critical path.
- **M4 — Math roadmap & hardening.** Extend rules (§7), cross-engine equivalence
  vs. JAX, dialect canonicalization table, docs. **Plus (F6/F11/F12):** the
  expression-swell size/latency **benchmark** (§7.2), and the convention-pinning
  tests for NULL-folding, kinks, and domain edges (§8).
- **Future (post-v1, demand-driven):** the C++/cxx.rs hybrid for bare `grad()` in
  DuckDB (§5.4 opt 4) and `ddx-pg` (Postgres).

---

## 12. Decisions & remaining questions

Answers from Alex's review (2026-07-19), folded in:

1. **DuckDB ergonomics — accepted, pending a second opinion.** `SELECT * FROM
   ddx('…')` is fine for v1; bare `grad()` stays the aspiration via the C++ hybrid
   (§5.4 opt 4). *Still open:* get the other duckdb-zarr maintainer to weigh in
   before we lock the extension's surface.
2. **Column binding — accepted.** Qualifier-aware syntactic differentiation, with a
   hard error on an ambiguous unqualified `wrt` (§5.5). §7.1 shows concretely which
   queries this handles and which it refuses.
3. **User-registrable rules — YES, adopted.** `ddx-core` exposes a rule registry so
   users can add differentiation rules for their own functions (§5.1). Feasibility:
   easy for the common (unary) case — the engine already dispatches on function
   name, so it's turning a `match` into a registry lookup where a user supplies
   `f'(u)` and the engine applies the chain rule. *Sub-question:* ship unary custom
   rules only in v1, or binary/n-ary too?
4. **xarray-sql integration — optional extra.** Ship as `pip install
   "xarray-sql[ddx]"`: a coordinated optional dependency where the `[ddx]` extra
   pulls in `ddxdb`, so autograd is opt-in and xarray-sql stays lean without it
   (§5.4).
5. **Naming — resolved.** The bare `ddx` crate is taken (a dead project), so there
   is no umbrella `ddx` crate — we don't need one. `ddx-core`, `ddx-datafusion`,
   `ddx-duckdb`, and `ddxdb` are all **free on crates.io**; `ddxdb` is **free on
   PyPI**; `ddx` is **free on the DuckDB community registry** (`INSTALL ddx`). (§10.)
6. **C++ hybrid up front vs. `ddx('…')` first — keep `ddx('…')`, but transitional
   (Fable follow-up, 2026-07-19).** The honest case for going C++ first is stronger
   than earlier drafts credited: the bound path is *structurally* immune to the
   F1/F2/F3/F5 silent-wrong class (§5.4 opt 4), not merely nicer UX. We still don't
   flip, for three reasons: (a) the hard part of the C++ path (rebuilding a *bound*
   derivative expression with correct `ColumnBinding` indices + catalog entries) is
   orthogonal to what v1 must prove and version-coupled forever; (b) **DataFusion
   Path B already buys the in-engine validation ~10× cheaper**; (c) `ddx('…')` is
   additive — its only real risk is *social* (hardening into a contract), fixed by
   branding it transitional, not by architecture. **What we changed instead:** brand
   `ddx('…')` transitional (§5.4 opt 1, M3); pull the *risk* forward with the
   M3-adjacent `BoundFunctionExpression` round-trip spike (§11), not the extension.
   *Resolved sub-spike:* the tempting `SELECT * FROM query(ddx_rewrite('…'))`
   collapse-to-a-scalar-function trick does **not** work — DuckDB's `query()` takes
   a **literal string only** (and isn't SELECT-only) — so the table function's
   bind-time-schema machinery (F7) is unavoidable.
   *Tripwire:* if the F1/F2 syntactic guards prove messier than expected in M0, that
   shifts the balance toward accelerating the C++ work (the bound path makes those
   guards unnecessary) — decide then, not now.
7. **Should `vjp` ship in v1? — decided *keep*, against the F10 caveat (G8); your
   call to overrule.** For the scalar expressions v1 supports, `vjp(e, x, ct)` is
   definitionally `mul(ct, grad(e, x))` — no reverse accumulation (F10). The cut
   case (Fable): shipping a function named `vjp` that isn't reverse-mode sets JAX
   users' expectations and defeats them, and a v1 name is a forever compat surface
   (the same "social hardening" logic as Q6). The keep case (my recommendation): the
   jvp/vjp *seeding* symmetry (seed an input tangent vs. an output cotangent) is a
   real, teachable surface; for scalar output the value it returns is *correct*, and
   F10 already documents honestly that it doesn't amortize; removing and re-adding is
   also an API event. **Recommendation: keep `grad`+`jvp`+`vjp`, with the F10 caveat
   loud in the docs.** Recorded here — per Fable — as a decision made *against* the
   F10 caveat, not before it. Overrule to `grad`+`jvp`-only if you'd rather not
   publish `vjp` until it means reverse accumulation.

---

## 13. Adversarial review (2026-07-19, Fable 5)

_Reviewed: this doc (v0.2) plus the prototype it is grounded in —
[xarray-sql#192](https://github.com/xqlsystems/xarray-sql/pull/192), all 13
commits and the final `autograd.rs` / `sql.py` / `lib.rs`. Scout's mindset: the
point is an accurate map, so this section records both where the terrain matches
the doc and where it doesn't._

**What holds up under attack** (briefly, because it's load-bearing): the
markers-not-UDFs finding (§3.1) is correct and the prototype's
`invoke`-must-error design is the right enforcement of it. The Substrait
rejection (§6) is earned — it was tried, it failed on exactly the query shapes
the thesis needs, and there's a repro. The v0.2 collapse to one IR is good
taste: deleting `DExpr` + four adapters removes a whole class of drift bugs.
R1/R1b are real spikes with falsifiable findings, not vibes.

The findings below are ordered by severity within each group. **F1–F4 can
produce a silently wrong number** — each is a violation of principle 5 ("fail
loud, never silently wrong") hiding inside the current plan, and I'd treat all
four as M0-blocking. F5–F7 are systems risks that bound how far the design
carries. F8–F12 are API and semantic debts that are cheap to fix before
`ddx-core` publishes and expensive after.

### 13.1 Silent-wrong findings (principle-5 violations)

**F1 — Column identity is raw-string equality, but SQL identifiers aren't.**
`ColRef` matching as specified compares the strings the parser saw. SQL
unquoted identifiers are case-insensitive (DuckDB is case-insensitive
throughout; DataFusion lowercases unquoted identifiers at planning). So in
DuckDB, `grad(Temp * Temp, temp)` differentiates w.r.t. a variable the matcher
never finds in the expression → derivative **0**, silently. This is also a
concrete **regression risk for the M2 "no regressions" gate**: the prototype
parses marker calls through DataFusion's own `parse_sql_expr`
(`autograd.rs::rewrite_call`), which applies identifier normalization, so
`X`/`x` match today; the sqlparser-only core loses that for free. *Fix:*
per-dialect identifier folding when constructing/comparing `ColRef` (casefold
unquoted, exact-match quoted; preserve original spelling in output). Add the
case tests in M0, before xarray-sql swaps internals.

**F2 — The §5.5 ambiguity guard is one-sided.** It fires only for an
*unqualified* `wrt`. The mirror case is silently wrong: a **qualified `wrt`
with a bare occurrence of the same name in the argument**. Repro: `FROM t a
JOIN u b` where only `t` has column `x`; the engine binds bare `x` to `a.x`, so
`grad(x * a.x, a.x)` is d/d(a.x) of `x²` = `2x` — but syntactic matching treats
bare `x` as a different (constant) leaf and returns `x`. Wrong number, no
error, and the emitted SQL is perfectly bindable so nothing fails downstream
either. *Fix:* extend the guard to a symmetric rule — if the `wrt` base name
occurs with **mixed qualification** (both bare and qualified, or under ≥2
qualifiers) anywhere in the argument, hard-error and demand full qualification,
regardless of whether `wrt` itself is qualified. Same cost as the existing
guard: AST-only, no catalog. Add a row to the §7.1 refusal table.

**F3 — Derivatives don't commute with query composition; the doc never says
so.** Differentiation stops at column leaves, so any column that is *computed*
upstream — a CTE, subquery, or view projection — is an opaque constant. Repro:

```sql
WITH v AS (SELECT x, sin(x) AS s FROM t)
SELECT grad(s * x, x) FROM v      -- ddx: ds/dx = 0  →  s      = sin(x)
-- inline v by hand:
SELECT grad(sin(x) * x, x) FROM t -- ddx: cos(x)*x + sin(x)
```

Two refactorings any SQL user considers equivalent give different derivatives.
As relational semantics this is *defensible* — every projection boundary is a
`stop_gradient` — but it is currently an **undocumented convention with a
silent failure mode**, and it lands exactly on the pitched use case (users
factoring a loss into CTEs will silently drop gradient terms). §5.5's alias
paragraph ("treating `s` as the differentiation variable is exactly right")
asserts the convention without naming it. *Fix, two parts:* (a) state the
contract loudly in §5.5/§7.1: *columns are leaves; `grad` differentiates the
expression as written against the relation it queries, never through view/CTE
definitions*. (b) A cheap syntactic guard is available for the worst subcase:
`rewrite_sql` sees the whole statement, so when a marker argument references an
identifier that is a **computed select-list alias of a CTE/derived table in the
same statement**, error (or warn) with "differentiate inside the CTE instead."
That guard can't see catalog views — say so; the residual risk is
documentation-only.

**F4 — The DOUBLE-literal rule doesn't cover literal-free derivatives; integer
division truncates.** R1b's "emit `DOUBLE` literals" fix only helps
expressions that *contain* literals. The quotient rule routinely emits
literal-free SQL: `grad(x / y, y)` → `(-x) / (y * y)` after 0/1-folding. On
`BIGINT` columns in DataFusion, `/` is **integer division**: x=1, y=2 gives
`-1/4 = 0` instead of `-0.25` — silently, and only on some engines (DuckDB's
`/` is float division, so the cross-engine equivalence suite would diverge here
too). *Fix:* the type policy belongs in the **smart constructors**, not the
literals — e.g. `div()` wraps its numerator in `CAST(… AS DOUBLE)` whenever
operand types are unknown (they always are, pre-binding). Slightly noisier
output; correct everywhere. Pin with an integer-column test in M0.

### 13.2 Systems risks

**F5 — `sqlparser` becomes a whole-query gatekeeper, and reprinting amplifies
it.** Path A must parse the *entire* statement, so `ddx-core`'s applicability
is bounded by sqlparser's per-dialect coverage — not by what the marker
touches. Any DuckDB surface sqlparser lags on (`FROM`-first queries, `SELECT *
EXCLUDE`, lambdas, `PIVOT`, next release's syntax…) fails the **whole** query
inside `ddx('…')`, even when the `grad` itself is `grad(x*x, x)`. DuckDB moves
fast; sqlparser's `DuckDbDialect` follows with a lag; this is a permanent
version-treadmill the doc should name as such. Two mitigations, one of which
should be v1: (a) **no-marker queries must pass through byte-identical** — never
parse-and-reprint a query you aren't changing (the `ddxdb` regex gate happens to
give this; make it a stated invariant of `rewrite_sql` too). (b) **Splice by
source span instead of reprinting the statement**: sqlparser's `Spanned` gives
the marker call's byte range in the original text; replace just that range and
leave every other byte alone. Parsing coverage remains the hard bound, but
reprint fidelity (comments stripped, formatting lost, canonical-form drift
across the whole statement — all of which today's `Display`-based plan incurs)
stops being a risk surface. It's cheap; I'd do it in M0.

**F6 — Symbolic expression swell, and nothing shares.** Product/quotient rules
duplicate their operands: `|d(f·g)| ≈ |f|+|g|+|df|+|dg|`, so a product chain of
n factors yields an O(n²) derivative, and repeated differentiation
(`grad(grad(…))` — advertised) compounds multiplicatively; this is the classic
symbolic-differentiation blowup, and 0/1-folding does not prevent it, only
trims the easy zeros. Consequences are systemic, not cosmetic: SQL text size,
parse/plan time, and **per-row recomputation** of repeated subexpressions
(`tanh(x)` appearing k times is k evaluations unless the engine's CSE catches
it — engine-dependent, partial). An N-parameter gradient is N `grad` columns
that each re-derive the whole loss, multiplying the swell by N. *Fix:* accept
for v1 but (a) add a size/latency benchmark to M4 so the cliff is measured, not
discovered; (b) note the post-v1 remedy — a "let-binding" pass that factors
shared subexpressions into projected columns (`…, cos(x) AS __ddx_t1`) or a
CTE. See also F10: the surface has no reverse-mode accumulation to amortize
this.

**F7 — `ddx('…')` has unvalidated mechanics and loses session state.** Three
gaps R1/R1b did not cover: (a) **Bind-time schema**: a DuckDB table function
must declare result columns at bind; deriving them requires
preparing/describing the *rewritten* query on the inner connection during bind.
Feasible-looking, but it's the actual heart of M3 and hasn't been spiked. (b)
**Connection-scoped state**: the inner `duckdb_connect` is a new session —
the caller's **temporary tables, session `SET`s, and prepared statements are
invisible** inside `ddx('…')`. R1b covered transaction visibility only; this is
a broader (and more common) surprise and belongs in the same "when to use
client-side Path A" guidance. (c) **Precedent & policy**: DuckDB ships a
built-in `query('sql')` table function with the same shape — deliberately
restricted to SELECT. R1b proudly notes inner DML "works"; decide *on purpose*
whether `ddx('…')` permits DML (community-extension review may object, and the
client-side path already covers stateful loops). Also: SQL-in-a-string quoting
is genuinely unpleasant for the flagship recursive-CTE examples — document
dollar-quoting (`ddx($$ … $$)`) as the house style.

**F8 — The markers hijack every function spelled `grad`/`jvp`/`vjp`.** The
prototype's `is_marker_name` (and the design's rewrite) claims those names
unconditionally — including a user's own UDF or a qualified `myschema.grad(…)`.
Fine to reserve the names, but *reserve them explicitly*: match only unqualified
spellings, and document the reservation. (The `ddxdb` regex gate also matches
inside string literals — harmless today because the real parse decides, but
worth a comment so nobody "optimizes" the gate into the rewrite itself.)

### 13.3 API & semantic debts (cheap now, expensive after publishing)

**F9 — The rule registry has no seam in the public API.** §5.1 sells
user-registrable rules, but every signature in the doc is a free function —
`rewrite_sql(sql, dialect)`, `differentiate(e, wrt)`. Where does the user's
registry go? As drafted this forces global mutable state (or an API break)
later. *Fix in M0, before crates.io:* make the entry point an object —
`Ddx::new(registry).rewrite_sql(…)` (default registry = built-ins) — which also
gives dialect canonicalization config a home.

**F10 — `vjp` is not reverse mode; say so.** As specified,
`vjp(expr, col, ct) = ct · d(expr)/d(col)` is a cotangent-scaled *forward*
pass, one per column — there is no reverse accumulation, so the thing reverse
mode is *for* (all N input sensitivities in one pass) is absent, and N-parameter
gradients pay N forward passes (compounding F6). The surface is fine; the
"reverse-mode" framing in §7 oversells it and will mislead exactly the JAX
users the project courts. One honest sentence fixes it.

**F11 — 0/1-folding changes NULL semantics; pin the convention.** Folding
`d/dx(x + y)` to `1` means the derivative is `1` even on rows where `y IS NULL`
and the primal is `NULL`. That matches JAX's treatment of NaN-contaminated
tangents and is a defensible convention — but folded and unfolded derivatives
now *disagree* on NULL-bearing rows, so it must be a documented decision with a
test, not an accident of the simplifier.

**F12 — Kinks and domain edges will make the oracle and the engines disagree.**
Three predictable flaps in the §8 test plan: (i) `abs` at 0 — the `signum` rule
gives 0; JAX's convention at the kink differs (verify: `jax.grad(jnp.abs)(0.0)`
is reportedly 1.0), so pin convention points explicitly rather than comparing
blindly. (ii) Derivatives **widen the domain of failure**: `sqrt(x)` at `x=0`
evaluates fine, but its derivative `1/(2*sqrt(x))` divides by zero — and
engines disagree on what that does (IEEE `inf` vs `NULL` vs error). The
derivative query can fail where the primal didn't, differently per engine;
cross-engine equivalence needs a stated policy at domain edges (sample away
from them, or pin per-engine expectations). (iii) Same for `tan` near π/2,
`ln` near 0.

### 13.4 Suggested plan deltas

- **M0** grows four exit criteria: identifier normalization + case tests (F1),
  the symmetric mixed-qualification guard (F2), the numeric-type policy in the
  smart constructors + integer-column tests (F4), and the `Ddx`-object API shape
  (F9). Decide span-splicing vs. reprint (F5) here too — it changes `rewrite_sql`'s
  internals.
- **§5.5/§7.1**: state the projection-boundary contract and add refusal-table
  rows for F2 and the F3 same-statement alias guard.
- **M3**: add the bind-time-schema spike and the DML policy decision (F7) as
  named tasks, not discoveries.
- **M4**: add the swell benchmark (F6) and convention-pinning tests (F11, F12).

**Overall:** the architecture is sound and the v0.2 simplification is the right
call — none of the findings above argue for a different shape, and several
(F1, F2, F9) are only visible *because* the design is now simple enough to
attack precisely. But four of them produce wrong numbers under the current
text, and this is a numerical-correctness product; they're small fixes, and
they belong in M0, not in a postmortem.

### 13.5 Resolution (Claude Opus 4.8, 2026-07-19)

**All twelve findings accepted and folded into the design above** — none rejected;
the review was right on every count. Where each now lives:

| # | Accepted change | Where |
| --- | --- | --- |
| F1 | Per-dialect identifier folding in `ColRef` (rule differs by dialect — DuckDB folds quoted too) | §5.1, M0 |
| F2 | Ambiguity guard keyed on *uncertain* occurrences (a bare occurrence of the `wrt` base name) | §5.5, §7.1, M0 |
| F3 | Projection-boundary (`stop_gradient`) contract + same-statement CTE-alias guard | §5.5, §7.1 |
| F4 | Numeric-type policy moved into smart constructors (`CAST … AS DOUBLE`) | §5.1, M0 |
| F5 | Byte-identical passthrough + splice-by-source-span; sqlparser-gatekeeper named | §5.1, §5.3, M0 |
| F6 | Expression-swell named as a known limitation; benchmark + let-binding remedy | §7.2, M4 |
| F7 | Bind-time schema, lost session state, DML policy, dollar-quoting | §5.4, M3 |
| F8 | Markers reserved as **unqualified** spellings only | §5.1 |
| F9 | Public API is now a `Ddx` object carrying the registry | §5.1, M0 |
| F10 | `vjp` reframed honestly — cotangent-seeded, *not* reverse accumulation | §7 |
| F11 | 0/1-folding NULL convention documented + pinned by test | §5.1, §8, M4 |
| F12 | Kink/domain-edge convention pinning + domain policy in the test plan | §8, M4 |

Two places I *refined* rather than adopted verbatim: F3's same-statement guard is
documented as **best-effort** (fully detecting "identifier is a computed alias"
across derived tables is subtle; the loud **contract** is the primary fix, the
guard is the safety net), and F5's span-splicing carries a "confirm `Spanned`
coverage for the pinned sqlparser in M0" caveat.

**Second-pass corrections (2026-07-19, after a self-review of the fixes).** Three of
the fixes above were themselves wrong on first cut and were corrected: **F2** — the
"symmetric ≥2-qualifiers" guard over-fired and would have rejected
`grad(a.x*b.x, a.x)`, the very form §7.1 recommends; retargeted to fire only on an
*uncertain* (bare) occurrence of the `wrt` base name. **F1** — "exact-match quoted"
is Postgres/DataFusion behavior; DuckDB folds *quoted* identifiers too, so the rule
is now explicitly per-dialect. **F5** — the byte-identical guarantee needs a
*parse-free* marker pre-gate (a marker-free query is never parsed), else it's
vacuous exactly when sqlparser can't parse the statement. And the Fable follow-up on
C++-vs-`ddx('…')` is folded in at §12 Q6 + §5.4 + M3.

### 13.6 Second adversarial round — resolution (G1–G9)

A fresh Fable pass ran four evidence spikes (sqlparser 0.62.0 behavior, DataFusion 54
pins, crates.io history). **G1 and G3 I re-verified myself** (Rust spike, sqlparser
0.62.0): constructed `(a+b)*c` Displays as `a + b * c` — wrong — and `Nested`-wrapping
fixes it; `grad` in `SELECT 'héllo', grad(x,x) …` is at byte 17 / column 16 (columns
are characters). **Accepted 8 of 9; partial on G8.**

| # | Finding | Resolution | Where |
| --- | --- | --- | --- |
| G1 | `Display` drops precedence parens → wrong math (**confirmed**) | Smart constructors `Nested`-wrap composite operands; semantic round-trip test | §5.1, §8, M0 |
| G2 | `sqlparser` version lockstep undermines "one IR" + the bridge | Pin to DataFusion's `sqlparser`; re-export it; bumps = breaking `ddx-core`; bridge string fallback | §5.1, §5.3, §9 |
| G3 | Spans are line/column *characters*, not byte ranges (**confirmed**) | Span→byte conversion subsystem; multibyte/multi/nested-marker M0 task | §5.1, M0 |
| G4 | F3 CTE-alias guard forbade the doc's own `grad(s*s, s)` | Carve-out: fire only when a computed alias is a *non-`wrt`* term | §5.5, §7.1 |
| G5 | Pitch: headline (training loops) is where limits bite hardest | Invert framing — low-N scientific calculus is the product; training loops a bounded demo | §1, §7.2, M4 |
| G6 | Contradiction: DF Path B "de-risks" DuckDB C++ (it doesn't) | Deleted the de-risk clause; the M3-adjacent spike is the honest de-risker | §5.3, §5.4 |
| G7 | Pre-gate/marker case+whitespace; `TypeCoercion` order; oracle-in-M2 | Case-fold gate+marker; named the `create_logical_expr` seam; pulled oracle to M2 | §5.1, §5.3, M2 |
| G8 | Cut `vjp`? (it's a 2-token macro in v1) | **Partial pushback** — recorded as an explicit decision (keep, w/ F10 caveat), your call | §12 Q7 |
| G9 | `sqlparser`-gap examples stale (3 of 4 actually parse) | Refreshed to `PIVOT`/`#1`; version-pinned the coverage claim | §5.3 |

The pattern worth keeping: this design is now simple enough that its remaining risks
live in dependency details (`Display`, `Spanned`, "the same type") — each checkable
with a 20-line spike before production. Two rounds in, the architecture has not moved
under attack.

_Next step:_ Alex to review; then iterate this doc (with agents) before we start
M0. The prototype's `autograd.rs` and its Python test suite are the concrete
starting materials for the `ddx-core` extraction.
