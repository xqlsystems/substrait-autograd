# Design Doc: `ddx` — portable autograd for composable databases

_author_: Alex Merose

_co-author_: Claude (Opus 4.8), via Claude Code

_created_: 2026-07-19

_last updated_: 2026-07-19

_status_: Design — iterating toward implementation.

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

[//]: # (TODO&#40;claude&#41; Make a note up front that we assume the XQL data model for arrays, i.e. ND arrays are treated as 2d tables. Please link to an appropriate doc &#40;e.g. https://xql.systems or Xarray-SQL docs&#41;&#41; if readers want to hear more.)

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
- The `grad`/`jvp`/`vjp` surface is **standardized** — one portable definition of
  what these functions mean, expressed as a Substrait simple-extension, that any
  engine can adopt.

---

## 2. Non-goals (for the first cut)

- **Not** a runtime tensor library or a replacement for JAX/PyTorch. We
  differentiate SQL scalar expressions symbolically; we do not implement a tape,
  GPU kernels, or autodiff of arbitrary imperative UDFs.
- **Not** general `u^v` power, `CASE`/conditional subgradients, or non-smooth ops
  in v1 (tracked in §7 roadmap). The engine returns a clear `NotImplemented`
  rather than a silently-wrong derivative.
- **Not** whole-plan Substrait round-tripping (§3 explains why the prototype
  removed it).
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

[//]: # (TODO&#40;claude&#41;: Please file an issue in this GH project that explains the bug in substrait that makes us go with this approach and not a substrait-centric approach. Please add appropriate links to the Xarray-SQL project and attempt steps to reproduce the error. After you've done this, please refer back to that issue in this design section.)

The PR's intermediate design (commit `672e7d0`) round-tripped the entire logical
plan through Substrait to apply the rewrite in a separately-linked copy of
DataFusion:

> produce logical plan as Substrait → `grad_rewrite` consumes it → rewrite every
> `grad()` into the derivative → re-produce Substrait → Python consumes & executes.

The **final commit (`14b26971`) deleted this entirely** ("Differentiate grad() as
a SQL rewrite, dropping the Substrait bridge"), because:

- Substrait's plan representation **could not carry recursive CTEs, DML, or
  subqueries** — precisely the query shapes that make in-SQL *training loops*
  interesting (e.g. Newton's method / gradient descent in a `WITH RECURSIVE`).
- It required a `protoc` build dependency and per-engine schema plumbing.

The replacement — a **SQL source-to-source rewrite before planning**
(`rewrite_grad_in_sql`) — works for *any query shape the SQL parser accepts*,
needs no engine fork and no custom wheel, and runs against the stock published
package.

**Consequence:** Substrait is *not* a good fit as the mandatory plan transport.
But note the failure was specific to **whole plans**. A single **scalar
expression** (the argument to `grad()`) is well within Substrait's expressive
core. This is what lets us still use Substrait as a *protocol* and *optional
expression IR* (§6) without repeating the mistake.

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
*data type it walks* is not. **Generalizing off `Expr` onto a neutral IR is the
central refactor of this project** (§5.1).

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

1. **Differentiate once, in a neutral IR.** The algorithm lives in `ddx-core`
   over an engine-agnostic scalar-expression IR (`DExpr`). Everything else is
   adapters.
2. **Rewrite, don't execute.** Markers are erased before execution. Every
   integration is fundamentally "find the marker, hand its argument to the core,
   splice the derivative back."
3. **Two peer injection paths** (per design decision): a portable **SQL
   source-to-source rewrite** *and* an **in-engine plan/expression rewrite**.
   `ddx-core` is agnostic to which one calls it; the difference is only the
   frontend/backend adapter used (§5.3).

[//]: # (   TODO&#40;claude&#41;: I'm starting to be skeptical of this! Will discuss a concrete question below in 5.)

4. **Substrait is the protocol, not the pipe.** Standardize *what the functions
   mean* via a Substrait simple-extension; offer Substrait as one *optional*
   expression IR. Never require a whole-plan round trip (§6).
5. **Fail loud, never silently wrong.** An unsupported node is a typed error, not
   an approximate derivative. This is a numerical-correctness product.
6. **Prove it in real projects.** xarray-sql and duckdb-zarr are acceptance
   tests, not demos.

---

## 5. Architecture

A monorepo (Cargo workspace + published Python wheels) with three layers: a pure
core, IR adapters, and per-engine integrations.

```
                         ┌───────────────────────────────────────┐
                         │              ddx-core (Rust)           │
                         │  DExpr  ·  rule registry  ·  linearize │
                         │  grad / jvp / vjp  ·  0/1 simplifier   │
                         │  (no engine deps, no protoc)           │
                         └───────────────────────────────────────┘
                             ▲  DExpr in            DExpr out ▼
        ┌────────────────────┴───────────────┬──────────────────┬───────────────┐
        │ ddx-sql                            │ ddx-substrait     │ ddx-datafusion │  ... ddx-duckdb
        │ sqlparser AST ⇄ DExpr              │ Substrait Expr ⇄  │ DF Expr ⇄ DExpr │  DuckDB expr ⇄ DExpr
        │ (dialect-configurable)             │ DExpr (protoc)    │                │
        └────────────────────┬───────────────┴──────────────────┴───────────────┘
                             │ used by
   ┌─────────────────────────┴───────────────────────────────────────────────────────┐
   │ Injection path A: SQL rewrite            │ Injection path B: in-engine plan rewrite │
   │ (portable, client/wrapper, pre-planning) │ (native, per-engine hook)                │
   └──────────────────────────────────────────┴──────────────────────────────────────────┘
                             │ shipped as
   ┌─────────────────────────┴───────────────────────────────────────────────────────┐
   │ ddxdb (Python wheel)      │ ddx (DuckDB community ext, Rust)  │ ddx-pg (pgrx, later)│
   │ → xarray-sql, DuckDB-py   │ → duckdb-zarr                     │                     │
   └───────────────────────────┴───────────────────────────────────┴─────────────────────┘
```

TODO(claude): At a high level, do we still need both Plan A and Plan B to reach all the distribution channels that we want to target? Is this dual plan still advised? Could we get away with implementing less and still reach all the targets? What do you think?

### 5.1 `ddx-core` — the engine (the refactor)

Port `src/autograd.rs` out of xarray-sql, generalized off DataFusion `Expr` onto
a small owned IR. Sketch:

```rust
pub struct ColRef {              // qualified column identity (see §5.5)
    pub qualifier: Option<String>,   // e.g. Some("era5") for era5.temp
    pub name: String,
}

pub enum DExpr {
    Const(f64),
    Column(ColRef),               // matched against `wrt` by qualified identity
    Neg(Box<DExpr>),
    Binary(BinOp, Box<DExpr>, Box<DExpr>),   // + - * /
    Call(String, Vec<DExpr>),     // "sin", "power", ... — name-dispatched rules
    Cast(Box<DExpr>, DType),      // locally linear
}

pub fn differentiate(e: &DExpr, wrt: &ColRef) -> Result<DExpr, DiffError>; // grad
pub fn jvp(e: &DExpr, seeds: &HashMap<ColRef, DExpr>) -> Result<DExpr, DiffError>;
pub fn vjp(e: &DExpr, wrt: &ColRef, cotangent: &DExpr) -> Result<DExpr, DiffError>;
```

[//]: # (TODO&#40;claude&#41;: I think a better design would be to adopt a common SQL Parser and IR. DataFusion's own parser is https://docs.rs/sqlparser-patched/latest/sqlparser/index.html and I would prefer to use that. It even comes with a DuckDB SQL dialect if I understand correctly: https://docs.rs/sqlparser-patched/latest/sqlparser/dialect/struct.DuckDbDialect.html)

Design notes:

- The **rule registry is keyed by function *name* (a string)**, exactly as the
  prototype already does (`match name { "sin" => ... }`). This is what makes the
  core engine-agnostic: every engine names `sin` the same, and name-dispatch
  needs no engine types. Adapters normalize dialect-specific spellings (e.g.
  DuckDB `ln` vs `log`) into canonical names before handing off.
- Keep the **smart constructors** (`add/sub/mul/div/neg/square`) verbatim — they
  are the `Zero`/`add_tangents` analog and are already well-tested.
- **No `f64`-only assumption baked into the surface**: types don't change the
  derivative's *form*, and casts are locally linear, so the symbolic pass is
  type-agnostic. But emit **`DOUBLE`-typed literals/casts** in the output (R1b side
  finding: DuckDB types `0.0` as `DECIMAL`, which would drag derivative arithmetic
  into decimal). Let the engine's type system handle final coercion.
- **Prefer binding-aware inputs (§5.5):** feed the core *bound* expressions with
  resolved `ColRef`s where the engine can supply them; the synthesized-schema
  syntactic path is a documented fallback mode, not the default.
- Port the prototype's 15 Rust unit tests directly; they pin the rules.

**This crate has zero engine dependencies and does not need `protoc`.** That is
the whole point — it is the reusable component the goal calls for.

### 5.2 IR adapters (frontends + backends)

Each adapter is a `From`/`To` between `DExpr` and one representation. They are
independent, feature-gated, and individually testable via round-trip properties.

| Adapter | In (frontend) | Out (backend) | Needs `protoc`? | Primary use |
| --- | --- | --- | --- | --- |
| `ddx-sql` | parse SQL scalar expr (sqlparser-rs) → `DExpr` | `DExpr` → SQL text | no | Path A (portable SQL rewrite) |
| `ddx-datafusion` | DataFusion `Expr` → `DExpr` | `DExpr` → `Expr` | no | Path B in DataFusion-Rust |
| `ddx-duckdb` | DuckDB bound/parsed expr → `DExpr` | `DExpr` → DuckDB expr / SQL | no | `ddx('…')` table fn (§5.4) |
| `ddx-substrait` | Substrait `Expression` → `DExpr` | `DExpr` → `Expression` | yes | protocol conformance + optional IR |

[//]: # (TODO&#40;Claude&#41;: What is the use of Substrait at this point? Is it still worthwile? Should we change trying to adhere to it if we slim down from two plans to one plan? What value does it provide us and it is essential? )

`ddx-sql` is effectively the prototype's `rewrite_grad_in_sql` /
`GradSqlRewriter` generalized: parse the statement, visit for marker calls,
differentiate the argument via `ddx-core`, unparse, splice in place. Dialect is a
parameter (`GenericDialect` today; per-engine dialects later).

### 5.3 The two injection paths (peers)

Both paths call the *same* `ddx-core`. They differ only in *when/where* the
rewrite fires and *which adapter* feeds it.

**Path A — SQL source-to-source rewrite (portable, proven).**
The caller intercepts the SQL string before it reaches the engine, rewrites
markers to derivative SQL, and passes plain SQL onward. Works for *every* engine
and every query shape the parser accepts (recursive CTEs, DML, subqueries).
Lives in a thin wrapper (`ctx.sql()` shim in Python; a helper/pragma in an
extension). This is exactly what the prototype settled on and what
`ddxdb`/xarray-sql use.

**Path B — in-engine plan/expression rewrite (native).**
The rewrite runs *inside* the engine as a plan-time pass over its own expression
trees, so `INSTALL ddx` "just works" with no client wrapper.

- **DataFusion (Rust):** register the markers, then install an `AnalyzerRule`
  (or `FunctionRewrite`/`ExprPlanner`) that walks the logical plan and applies
  the prototype's already-written `rewrite_grad_calls` (`transform_up` over
  `Expr`). This path is well-supported and low-risk in Rust.
- **DuckDB:** see §5.4 — the hook availability is the project's biggest open
  question.

Because `ddx-core` is injection-agnostic, an engine can offer **both**: Path B
for the common case, Path A as a fallback for query shapes an engine's own hook
can't reach.

### 5.4 Per-engine integration & distribution

**DataFusion / `ddxdb` (Python) → xarray-sql.**
`datafusion-python` does not expose injecting an `AnalyzerRule` into its
`SessionContext` — which is *why the prototype uses Path A*. So:
- `ddxdb` ships Path A (the SQL rewrite) as its default for Python/DataFusion,
  re-exporting `rewrite_sql(sql, dialect)` and a `Context.sql()` shim. xarray-sql
  consumes this — ideally deleting its vendored `autograd.rs` and depending on
  `ddx-core` instead (regression-tested against the PR's Python suite).
- `ddx-datafusion` (Path B) targets *native Rust* DataFusion users and any future
  datafusion-python hook.

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
     literal, rewrites markers via `ddx-core`+`ddx-sql`, executes the plain SQL on
     a connection to the same database, and streams the result back:
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
     DuckDB's expression to bytes and write a `ddx-duckdb` Rust deserializer into
     `DExpr` (clean, but couples to DuckDB's internal serialization format), or
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

### 5.5 Binding-aware differentiation (design decision)

The prototype differentiates **syntactically** — before name binding — so marker
arguments must use *unqualified* column names (`grad(y - a*x - b, a)`), and
`grad(a.x + b.x, a.x)` is ambiguous. **Decision: `ddx` invests in
binding-aware differentiation and is not limited to the syntactic form.** We won't
be bound by the prototype's shortcut where we can do better; where an engine can't
give us bound columns, we degrade explicitly rather than silently.

What this means concretely:

- **The core IR carries qualified column identity** (`ColRef`, §5.1). `wrt` and the
  leaf/seed maps match on `(qualifier, name)`, not a bare string — so
  `grad(a.x + b.x, a.x)` differentiates w.r.t. the right column.
- **Differentiate *bound* expression trees where possible.** This is exactly the
  domain of **Path B**: the prototype already ships a plan-level rewriter
  (`rewrite_grad_calls`, `transform_up` over bound `Expr`) that sees resolved,
  qualified columns. In native DataFusion, the `AnalyzerRule` gets binding for
  free. Note the prototype's move to SQL-text was to escape *Substrait's* shape
  limits (§3.2), **not** a limitation of plan-level rewriting — DataFusion binds
  recursive CTEs, subqueries, and DML fine.
- **Make Path A binding-aware too, when the engine can bind.** Instead of parsing
  each marker argument standalone against a synthesized `Float64` schema (the
  prototype's `call_schema`), the binding-aware Path A **plans the whole query to a
  bound logical plan, runs the plan-level rewrite over resolved columns, and
  executes the plan** — no Substrait, no re-unparse to SQL. For DataFusion this is
  available today; for the DuckDB `ddx('…')` table function the inner connection
  can bind and rewrite before executing.
- **Degrade explicitly for un-bindable inputs.** If a target truly can't hand us
  bound columns (a pure text preprocessor with no catalog), fall back to the
  syntactic rewrite and **document the unqualified-name constraint as a mode**, not
  the default. Ambiguous references in that mode are a typed error, never a guess.

Open sub-questions this raises are tracked in §12 Q2 (name resolution across
joins, correlated subqueries, and how `jvp`/`vjp` seed maps key on `ColRef`).

---

## 6. Substrait's role (resolved)

Per design decision, Substrait is **protocol + optional IR**, never the mandatory
plan pipe. Concretely:

1. **Standardized function definitions (the main contribution).** Publish a
   Substrait *simple-extension* YAML (`extensions/ddx.yaml`) declaring
   `grad`/`jvp`/`vjp` with a URN (`extension:ddx:autograd`), argument/return
   types, and semantics. This is the portable, engine-neutral *definition* of the
   surface — exactly the mechanism
   [Substrait UDFs](https://substrait.io/expressions/user_defined_functions/) and
   [extensions](https://substrait.io/extensions/) are designed for. Any engine
   that speaks Substrait can discover and agree on what these functions mean,
   even though the *implementation* is a rewrite rather than a runtime kernel.
   (Note: these are "marker" functions; we'll document that they are compile-time
   rewrites and must not reach execution — a semantics an engine's Substrait
   consumer should honor.)

2. **Optional expression interchange IR.** `ddx-substrait` maps a *scalar*
   Substrait `Expression` ⇄ `DExpr`. An engine that already produces Substrait for
   expressions can differentiate a `grad()` argument via the core without a SQL
   round trip. This stays away from §3.2's failure because it round-trips only the
   *scalar argument*, never a whole plan with recursion/DML.

3. **Explicitly NOT** a whole-plan produce→rewrite→consume bridge. That path is
   documented as a rejected alternative (§3.2) so we don't rebuild it.

This honors the substrait.io goal at the level where Substrait is genuinely
strong (a shared vocabulary for functions and scalar expressions) while using the
prototype-proven SQL rewrite as the default portable mechanism.

---

## 7. The differentiation surface & math roadmap

**v1 surface (port of the prototype, unchanged semantics):**

- `grad(expr, column)` → `d(expr)/d(column)`.
- `jvp(expr, column, tangent)` → forward-mode `d(expr)/d(column) · tangent`.
  Multi-input directional derivative = sum of `jvp` terms.
- `vjp(expr, column, cotangent)` → reverse-mode `cotangent · d(expr)/d(column)`.
- `differentiate_sql(expr, wrt)` → derivative as SQL text (the "calculus
  compiler" escape hatch).
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
  every rule pinned symbolically.
- **Adapter round-trip property tests** — `DExpr → repr → DExpr` is identity;
  `parse → differentiate → unparse` produces parseable SQL. Fuzz small random
  expression trees.
- **Numeric agreement (the ground truth)** — for a battery of expressions,
  compare the engine-evaluated derivative against a **finite-difference**
  estimate *and* against numpy analytic derivatives, per engine. This is what the
  prototype's `tests/test_autograd.py` does and it is the acceptance bar.
 
[//]: # (  TODO&#40;claude&#41;: I recommend comparing against JAX instead of numpy.)
- **Cross-engine equivalence** — the *same* expression differentiated through
  `ddx-sql` (Path A) and through the DataFusion/DuckDB native adapters (Path B)
  must produce numerically equal columns.
- **Real-integration acceptance** — end-to-end gradient descent and a recursive-
  CTE training loop converging to closed-form solutions, run inside xarray-sql and
  duckdb-zarr.

**Open research spikes (do these early — they de-risk the plan):**

- **R1 — RESOLVED (2026-07-19).** A DuckDB Rust community extension *cannot* do a
  native bare-`grad()` rewrite: the C Extension API has no parser/optimizer/plan
  hooks. Use an in-extension `ddx('<sql>')` table function (Path A inside the
  extension) as the primary route; C++ `OptimizerExtension` is the stretch goal
  for bare `grad()`. Full verdict in §5.4.
- **R1b — RESOLVED (2026-07-19).** `ddx('…')`'s inner-connection re-entrancy is
  safe (reads, DML, no deadlock); the inner connection runs in its own transaction
  and can't see the caller's *uncommitted* state → self-contained queries use
  `ddx('…')`, stateful/transactional loops use client-side Path A. Full findings in
  §5.4. Remaining: a Rust-extension smoke test in M3.
- **R2:** Confirm `datafusion-python` still can't inject an `AnalyzerRule` (keeps
  Path A as the Python default), or find the seam if it can.
- **R3:** Validate `ddx-substrait` scalar-expression round-trip fidelity for the
  full v1 rule set (does the simplified derivative survive Substrait ⇄ back?).

---

## 9. Monorepo layout (proposed)

[//]: # (TODO&#40;claude&#41;: This repo structure looks good, but the project title should be `ddx`.)
```
substrait-autograd/                (repo; crates published under the ddx-* names)
├── crates/
│   ├── ddx-core/                   # the engine — DExpr, rules, linearize (no engine deps)
│   ├── ddx-sql/                    # sqlparser frontend/backend  → Path A
│   ├── ddx-substrait/              # Substrait Expression ⇄ DExpr (protoc, feature-gated)
│   ├── ddx-datafusion/             # DataFusion Expr adapter + AnalyzerRule → Path B
│   └── ddx-duckdb/                 # DuckDB adapter + community extension → Path B/A
├── python/
│   └── ddxdb/                      # PyO3/maturin wheel: Path A rewrite + Context shim
├── extensions/
│   └── ddx.yaml                    # Substrait simple-extension: grad/jvp/vjp definitions
├── docs/
│   └── design.md                   # this file
└── tests/                          # cross-engine numeric-agreement suites
```

Rationale: `ddx-core` publishes independently so anyone can build a new engine
adapter without our integration crates. The `protoc` dependency is quarantined to
`ddx-substrait` (opt-in), so the common build stays light — directly addressing a
pain point the prototype hit.

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

- Rust crates: `ddx-core`, `ddx-sql`, `ddx-substrait`, `ddx-datafusion`,
  `ddx-duckdb` on crates.io.
- Python: `pip install ddxdb`.
- DuckDB: `INSTALL ddx FROM community;` → the `ddx('<sql>')` table function (§5.4).
- Repo: rename `substrait-autograd` → `ddx` (keep "Substrait-portable" in the
  tagline rather than the name).

---

## 11. Milestones

Given the decision to build **both injection paths and both integrations in
parallel**, the plan front-loads the shared core and the de-risking spikes so the
two tracks don't diverge. (R1 is already resolved — see §5.4.)

- **M0 — Extract the core.** Create the workspace; lift `src/autograd.rs` into
  `ddx-core` over `DExpr`; port all rule unit tests. `ddx-sql` frontend/backend
  with round-trip tests. *Exit:* `ddx-core` + `ddx-sql` reproduce every
  prototype rule and the SQL-rewrite behavior, engine-free.
- **M1 — Remaining spikes** (short, decision-making). R1 and R1b are **done**
  (§5.4): DuckDB gets an in-extension `ddx('<sql>')` table function (re-entrancy
  safe), not native `grad()`. Still to settle: R2 (datafusion-python AnalyzerRule
  seam) and R3 (Substrait scalar round-trip). *Exit:* Path B feasibility per engine
  final.
- **M2 — DataFusion track.** `ddx-datafusion` (Path B, native Rust) + `ddxdb`
  wheel (Path A) ; re-integrate into xarray-sql, replacing its vendored engine.
  *Exit:* xarray-sql green on `ddx-core`, no regressions.
- **M3 — DuckDB track.** `ddx-duckdb` = the `ddx('<sql>')` table function
  (reads SQL literal → rewrite via `ddx-core` → execute on inner connection →
  stream) + the `ddxdb` client-side path for DuckDB-Python ; integrate with
  duckdb-zarr. *Exit:* `grad` works end-to-end in DuckDB against a real
  duckdb-zarr dataset via `SELECT * FROM ddx('…')`.
- **M4 — Substrait protocol.** Publish `ddx.yaml`; implement `ddx-substrait`
  scalar round-trip (R3). *Exit:* an engine can consume the standardized
  definitions; optional Substrait IR path validated.
- **M5 — Math roadmap & hardening.** Extend rules (§7), cross-engine equivalence
  suite, docs, benchmarks.

---

## 12. Open questions for review

1. **DuckDB ergonomics — DECIDED (Alex, 2026-07-19).** `SELECT * FROM ddx('…')` is
   an acceptable v1 UX. Bare `grad()` remains the aspiration: pursue it later via
   an upstream DuckDB change, a C++ `OptimizerExtension`, or another native hook —
   tracked as a stretch goal (§5.4 option 4), not a v1 blocker.
2. **Column-name binding — DECIDED (Alex, 2026-07-19): invest in binding
   awareness (§5.5).** `ddx` is not limited to the prototype's syntactic,
   unqualified-name form; the core carries qualified `ColRef` identity and we
   differentiate bound expressions where the engine supplies them, degrading to a
   documented syntactic *mode* only where binding is unavailable. **Remaining
   sub-questions:** name resolution across joins / correlated subqueries; how far
   to push binding-aware Path A (plan-then-rewrite-then-execute) vs. leaving the
   syntactic fallback; and whether ambiguity is always a hard error.
3. **Where does `ddx-core`'s canonical function vocabulary live** — hard-coded
   match arms (as today), a registry API consumers can extend, or driven by the
   Substrait YAML? A registry makes third-party rule packs possible.
4. **Should xarray-sql depend on `ddx-core` directly, or vendor-then-migrate?**
   Direct dependency is cleaner but couples release cadences.
5. **Naming collision check** — is `ddx`/`ddxdb` free on crates.io, PyPI, and the
   DuckDB community registry?

---

_Next step:_ Alex to review; then iterate this doc (with agents) before we start
M0. The prototype's `autograd.rs` and its Python test suite are the concrete
starting materials for the `ddx-core` extraction.
