# Supplement: designing M3/M4 — query-level reverse-mode AD

_author_: Claude (Fable 5)

_status_: Proposed design, v2 — **revised after Alex's review to adopt Substrait as
the relational IR, replacing v1 of this document's bespoke `RelGraph` Rust type.**
The pivot, the reasoning, and the spike that grounds it are in §1; everything
downstream (§2–§8) is rewritten around it. This document answers the gap flagged
against `design.md`: "§7.3 proves true AD is possible and names the four rules —
but it's a feasibility argument, not a design." It is meant to be folded into
`design.md` (replacing/expanding §7.3 and the M3/M4 entries in §11) by whoever
synthesizes it.

_grounded in_: `spikes/relational_ad_spike.py`, `spikes/attention_ad_spike.py`,
`spikes/attention_causal_mask_check.py`, `spikes/substrait_ad_marker_spike.py`
(new — the empirical basis for §1), and `xarray-sql#196`'s `nn.py`. Every claim
below either cites a line in one of these or is flagged as unverified.

---

## 0. What question this answers

`design.md` §7.3 establishes *that* query-level reverse-mode AD works (verified:
machine-exact gradients on an MLP and on attention) and names four transpose rules.
It does not answer:

1. What does `ddx.vjp(query, wrt=table)` actually take as input, concretely — is
   "query" SQL text, and if so, how does the system recover which parts are
   contractions vs. elementwise maps?
2. How does a user express a "loss query"?
3. What is "the tape," concretely — named how, materialized when?
4. `nn.py` also computes an argmax (`ROW_NUMBER() OVER (... ORDER BY z DESC) = 1`,
   `nn.py:440`) — not currently differentiated, but any real CNN needs max-pooling,
   which is exactly this operation *inside* the gradient path. Is there a fifth
   rule?
5. How does cotangent accumulation work when one relation feeds two downstream
   consumers (fan-out)?

The first version of this document answered these with a **bespoke Rust IR** (a
`RelGraph`/`RelNode` builder API). Alex's review pushed back, correctly: that's a
new embedded DSL, not SQL, and it repeats a mistake this project already paid down
once — v0.2's headline simplification was deleting `DExpr` + four adapters in favor
of "one IR" (`sqlparser::ast::Expr`) precisely to stop inventing representations.
A bespoke v2 graph type violates the same principle (`design.md` §4.4, "SQL is the
portable surface") one layer up. This revision fixes that by adopting **Substrait**
as the relational IR instead, with the "don't infer, tag explicitly" requirement
(still correct, and kept) realized as **Substrait extension-function markers**
called from ordinary SQL — closer in spirit to how `grad()`/`jvp()` already work in
v1, not further from it.

---

## 1. The central decision, revised: Substrait + extension-function markers

### 1.1 What's kept from the first draft, and what's not

**Kept, unchanged:** the non-negotiable part of the original argument — a v2 AD
system cannot *infer* "this `JOIN`+`GROUP BY SUM` is a contraction" from an
arbitrary bound plan, because the same plan shape is used for things that are not
contractions (a plain aggregate report), and a misclassification produces a wrong
gradient silently, in the one place correctness matters most (`design.md`
principle 5). Every design here still requires **explicit, construction-time
tagging** — nothing is recovered by shape-matching.

**Not kept:** the conclusion that tagging requires a new Rust type to carry the
tags. It doesn't — SQL already has a mechanism for exactly this, and v1 already
uses it: **marker functions**. `grad(expr, x)` doesn't get its meaning inferred
from where it sits in a query; it's a function name ddx has claimed and every
occurrence is unambiguous. The same trick works one level up: instead of writing
`SUM(a.val * b.val)`, a user (or a higher-level model-definition helper) writes
`SUM(ddx_contract_mark(a.val * b.val))` — an **identity** scalar function whose
only job is to survive planning and mark that specific aggregate as a contraction.
No new vocabulary to learn beyond "wrap this one thing in a marker," which is
exactly v1's existing mental model.

### 1.2 Why not DataFusion's `LogicalPlan` as the carrier

Rejected, and not on taste. `design.md` §5.4/R1 already established, by reading
DuckDB's actual C extension header, that DuckDB's stable Rust-extension surface
has **zero** plan/optimizer/parser hooks. A v2 IR keyed to
`datafusion::logical_expr::LogicalPlan` is DataFusion-only by construction —
DuckDB can never produce or consume that Rust type, full stop. That breaks the
project's actual spine (`design.md` §1.1: "a single engine-independent core…
each database integration is a thin adapter"). This is a re-application of an
already-established fact, not a new argument.

### 1.3 Why Substrait — and why §6's rejection doesn't settle this

`design.md` §6 rejected Substrait as the **Path A whole-query transport**, for a
specific, evidenced reason: its producer can't represent `RecursiveQuery` or
`DmlStatement` (reproduced on datafusion 54.0.0, `spikes/substrait_limitation_repro.py`).
That finding doesn't transfer to v2, for two reasons:

1. **v2 doesn't need recursive CTEs.** §7.3 already concedes the training loop is
   Python-orchestrated per-step (`.cache()` between queries, matching `nn.py`
   exactly), not one recursive CTE. Every individual v2 step is a plain
   join/aggregate/project — squarely inside what Substrait's `RelType` handles.
2. **v1 and v2 need categorically different mechanisms, and only one of them can
   be text-splicing.** v1 differentiates a scalar expression and **splices text**
   into a source span (`design.md` §5.1, F5/G3) — that's exactly why v1 never
   needed a real plan IR. v2 has to **synthesize new joins and group-bys** (the
   backward contraction, the broadcast join) that don't exist anywhere in the
   forward query's text. That is not a text-splice problem. v2 structurally needs
   *some* real relational plan representation that v1 never did — so the question
   isn't "did we already reject Substrait," it's "now that we actually need a
   plan IR, is Substrait the right one" — a question §6 never asked, because at
   the time nothing in the design needed a plan IR at all.

**Substrait's extension mechanism is designed for exactly the tagging this needs.**
Every custom scalar/aggregate function gets a plan-local anchor via
`extension_uris` + `simple_extension_declaration`, independent of whatever base
relational shape it sits inside. This is mature and widely used for custom
*functions* specifically (distinct from custom *relation* types, which is a
genuinely less mature corner — there's an open DataFusion issue, apache/datafusion#6335,
asking for `LogicalPlan::Extension` → Substrait support). That distinction matters:
**ddx only needs tagged functions inside ordinary relations, never a new relation
kind**, so it sits in the well-trodden part of Substrait's extension surface, not
the immature part.

### 1.4 The spike: this is now verified, not just architecturally plausible

`spikes/substrait_ad_marker_spike.py` (new) tests the actual mechanism end to end:
wrap a contraction's multiplicand in an identity marker `ddx_contract_mark(...)`,
export via `get_substrait`/DataFusion's `Serde.serialize_bytes`, and check the
marker survives as a distinguishable extension-function anchor through both a
same-engine round-trip and a genuine cross-engine hop. All four checks pass,
**2026-07-20, DuckDB 1.5.4 + datafusion-python 54.0.0**:

1. **DuckDB round-trip** (`get_substrait` → `from_substrait`, same engine): marker
   preserved as `extensionFunction { functionAnchor: 3, name: "ddx_contract_mark" }`,
   correctly referenced from the aggregate's operand in the plan body, and the
   round-tripped plan **executes** and matches plain (unmarked) SQL exactly.
2. **Cross-engine, DataFusion → DuckDB**: a plan **produced by DataFusion**
   (`datafusion.substrait.Serde.serialize_bytes`), containing the same marker,
   is **consumed and executed by DuckDB's `from_substrait`**, producing numerically
   identical results to DataFusion's own execution of the same query. This is the
   actual portability claim — not "DuckDB doesn't corrupt its own plan," but
   "one engine's marker-tagged plan is directly usable by the other."
3. **Cross-engine, DuckDB → DataFusion**: a plan produced by DuckDB
   **deserializes cleanly** in DataFusion's own consumer (`Serde.deserialize_bytes`).
   Execution in this direction was **not** exercised — an honest scope boundary,
   named as a follow-up in §9, not claimed as verified.
4. **Sanity**: the base contraction shape (`JOIN` + `GROUP BY SUM`, no marker at
   all) round-trips through DuckDB correctly — confirms the underlying relational
   shape survives independent of the marker mechanism.

**One real nuance, not hidden.** Neither engine's producer emits a proper
`extension_uris` YAML-URI declaration for the custom marker — both represent it as
a bare `functionAnchor` + literal name, unlike builtin functions (`equal`,
`multiply`, `sum`), which *do* reference a real extension URN in the same plan.
This is looser than the Substrait spec's idealized intent (a well-formed custom
function should be declared against a real extension YAML), but it did not break
round-tripping between these two specific engines. It's unverified whether a
third engine's consumer would tolerate an anchor with no URI reference — named as
a risk to watch, not a blocker, since ddx's only two v1 targets are exactly the
two engines tested.

**This changes the empirical status of the whole approach.** Before this spike,
"Substrait + markers" was an architectural argument. After it, it's a checked
claim on the actual engines this project targets — the same epistemic bar R1/R1b
and the G-series findings were held to.

---

## 2. The relational IR: `substrait::proto`, not a bespoke Rust enum

`ddx-core` v2 operates directly on `substrait::proto::Plan` / `Rel` — the real,
generated Rust types from the `substrait` crate, the same way `ddx-core` v1
operates on the real `sqlparser::ast::Expr` rather than a bespoke `DExpr`. There is
no new user-facing type vocabulary to learn: **the IR the user's SQL compiles down
to (via the engine's own SQL→Substrait producer) already has every relation type
ddx needs** — `JoinRel`, `AggregateRel`, `ProjectRel`, `ReadRel` — and ddx's job is
to *recognize* its own marker functions inside that plan, not to define a
competing set of relation kinds.

```rust
// ddx-core v2's dependency, mirroring v1's `sqlparser`-only policy (design.md §9):
// substrait only — no datafusion, no duckdb crate, so the core stays a thin,
// reusable component and every engine integration is an adapter on top.
use substrait::proto::{Plan, Rel, rel::RelType};
```

**Marker functions, the actual new vocabulary (small and closed, on purpose):**

| Marker (SQL-visible name) | Wraps | Tags |
| --- | --- | --- |
| `ddx_contract_mark(x)` | the multiplicand inside a `SUM` | this `AggregateRel`'s measure is a **contraction** (§3.1) — the contracted dim and surviving dims are read off the surrounding `JoinRel`'s condition and the `AggregateRel`'s grouping set, which the user already wrote; the marker's only job is disambiguation, not carrying shape metadata |
| `ddx_reduce_mark(x)` | the operand inside a `SUM` with no accompanying join | a **group-reduce** (§3.3) |
| `ddx_route_mark(x)` | the `ORDER BY` key inside a windowed `ROW_NUMBER()`-then-filter idiom | a **Route** (argmax/argmin) node (§3.4) — **unverified whether Substrait's window-relation support covers this at all; see §3.4's flag** |
| `ddx_stop_gradient(x)` | any subexpression | an explicit `StopGradient` edge (§3.5) — no transpose rule fires past this point |

Ordinary `Elementwise`/`ElementwiseBinary` nodes (§3.2) need **no marker at all** —
any `ProjectRel` expression that isn't wrapped in one of the above is, by default,
an elementwise map over its inputs, and its local derivative is obtained by calling
`ddx-core`'s existing v1 scalar `differentiate` directly on the `Expression` message
(itself convertible to/from `ast::Expr` the same way DataFusion's `expr_to_sql`
bridge already does for Path B, `design.md` §5.3). This keeps the marker surface
minimal — only the operations whose *meaning* isn't recoverable from plan shape
need one.

---

## 3. The five rules — forward marker + SQL shape, VJP, provenance

Unchanged in substance from the first draft (the rules themselves were never the
problem; the carrier was). Restated here with the marker-based recognition
mechanism instead of graph-node construction.

### 3.1 Contraction

**Forward, as the user writes it (verified emittable/round-trippable,
`substrait_ad_marker_spike.py`):**
```sql
SELECT a.{batch...}, b.{out...}, SUM(ddx_contract_mark(a.val * b.val)) AS val
FROM {a} a JOIN {b} b ON a.j = b.j
GROUP BY a.{batch...}, b.{out...}
```
**Recognition:** ddx-core's plan walker finds an `AggregateRel` whose measure
expression contains a `ddx_contract_mark` extension-function call; the contracted
dim (`j`) and surviving dims are read directly off the enclosing `JoinRel`'s
condition and the `AggregateRel`'s `groupings` — **no separate shape argument is
threaded through the marker**, because the shape is already fully present in the
plan the user wrote. (This is a small but real simplification over the first
draft's `RelNode::Contract { a, b, contract_dim }`, which needed the dim passed
explicitly at construction time — here it's free.)

**Transpose**, unchanged: `Ā = Σ_out C̄·B` and `B̄ = Σ_batch A·C̄`, both themselves
contractions — verified in `relational_ad_spike.py:41-44`, matching `nn.py`'s
`g2`/`g1`/`g0` (`nn.py:307-314`). The backward SQL ddx-core emits for these is
**plain, unmarked** `JOIN`+`GROUP BY SUM` — ddx never needs to tag its *own*
output, only recognize the user's input, unless a second round of differentiation
is wanted (higher-order AD over queries — flagged open in §9).

### 3.2 Elementwise (unary and binary)

No marker (§2). Forward: an ordinary `ProjectRel` expression,
`f(val)` or `f(val1, val2)`. Transpose: `X̄ = Ȳ · f'(X)`, where **`f'` is one call
into `ddx-core`'s existing v1 scalar differentiator** — `Ddx::differentiate`,
unchanged (`design.md` §5.1). This remains the literal seam between v1 and v2:
nothing new is built here, one function from M0–M2 is reused. Verified against
`nn.py:332`, `dc.val * grad(tanh(fwd1.z), fwd1.z)`.

### 3.3 Reduce (+ broadcast)

**Forward:** `SUM(ddx_reduce_mark(val)) GROUP BY {surviving dims}`. Distinguished
from Contraction (§3.1) by the *absence* of an enclosing multi-relation join feeding
the aggregate — a `ddx_reduce_mark` wraps a plain reduction, a `ddx_contract_mark`
wraps a product-then-reduction. **Transpose:** broadcast join, unchanged from the
first draft, verified against `nn.py:325-333`.

**Mean-vs-sum note, unchanged recommendation:** don't add a `Mean` variant; require
`Reduce{Sum}` + a separate (unmarked, ordinary elementwise) `× 1/N` step, matching
`nn.py`'s actual pattern (`nn.py:308,318,339,348`) and keeping every transpose rule
independent of a global `N`.

### 3.4 Route (argmax/argmin) — still the fifth rule, now with a second open risk

**Forward, as `nn.py` already writes it** (`nn.py:437-445`, not yet wrapped in a
marker — this is the concrete migration needed to bring it under v2):
```sql
WITH ranked AS (
  SELECT {group_dims}, {route_dim}, ddx_route_mark(val) AS val,
         ROW_NUMBER() OVER (PARTITION BY {group_dims} ORDER BY val DESC) AS rk
  FROM {input}
)
SELECT {group_dims}, {route_dim}, val FROM ranked WHERE rk = 1
```
**Transpose:** scatter — cotangent flows only to the winning row, zero elsewhere —
unchanged from the first draft (§3.4 there), still the standard max-pool/hard-routing
subgradient convention.

**Two open risks now, not one.** The first draft flagged this rule as
**mathematically unverified** (no spike checks it against `jax.grad`, unlike the
other four). This revision adds a **second, more basic risk specific to choosing
Substrait**: it is not confirmed that Substrait's window-function support
(`ConsistentPartitionWindowRel` in the spec) is actually implemented by both
DataFusion's and DuckDB's Substrait producer/consumer — a DataFusion project search
turned up an explicit, current statement that "Substrait does not (yet) support the
full range of plans and expressions that DataFusion offers." This is exactly the
same *class* of risk `design.md` already names for sqlparser's `DuckDbDialect`
coverage (F5/G9) — a version-treadmill / coverage-gatekeeper pattern — just
recurring one layer up, on Substrait's own relation vocabulary instead of a SQL
dialect. **Recommend, as a named M3 task:** before trusting `Route`, spike whether
`ROW_NUMBER() OVER (PARTITION BY … ORDER BY …)` — the exact idiom `nn.py` already
uses — actually round-trips through both engines' Substrait producers/consumers at
all, *before* separately checking its gradient math. If window-relation support is
missing or partial, `Route` may need to stay a v1-style **text-splice escape
hatch** (bypass the plan-IR path entirely for this one rule, execute it as raw SQL
the way Path A already does) rather than a plan-level rule — a real, and now
concretely named, fallback design worth deciding on purpose rather than
discovering under deadline.

### 3.5 `StopGradient`

Unchanged in purpose from the first draft (needed so the softmax numerical-stability
max-shift doesn't receive spurious gradient, per the standard log-sum-exp identity).
Now realized as an explicit marker, `ddx_stop_gradient(x)`, rather than a graph-node
variant — the walker simply refuses to propagate a cotangent past a subexpression
wrapped in it. Same M3 test recommendation as before: assert the edge is actually
cut on the softmax fixture, not just that the math happens to work out.

---

## 4. The backward-program emitter

### 4.1 Algorithm — unchanged shape, different input type

The reverse-topological walk with fan-in cotangent accumulation from the first
draft carries over exactly; only the type being walked changes (a parsed-and-tagged
`substrait::proto::Rel` tree, built once from the input plan by locating marker
functions, rather than a `RelGraph` the caller constructs by hand):

```rust
pub fn vjp_query(
    &self,
    plan: &Plan,              // a substrait::proto::Plan, ALREADY containing
                               // ddx's marker functions (§2) — recognized, not built
    wrt: &[RelRef],            // which Source relations to differentiate w.r.t.
) -> Result<BackwardProgram, DiffError>;
```

Internally, `vjp_query` first walks `plan` once to locate every ddx marker
extension-function anchor and build a lightweight annotated-node index (which
`AggregateRel`s are contractions vs. reduces, where the `StopGradient` edges are)
— this index is purely an implementation detail of the walker, not a competing IR:
it's derived data over the real Substrait plan, the same relationship v1's `ColRef`
has to `sqlparser::ast::Expr`.

### 4.2 Fan-in accumulation — still verified, not hypothetical

Unchanged from the first draft: when a relation feeds more than one consumer
(attention's `X` feeding `Wq`/`Wk`/`Wv`), contributions are summed via an ordinary
(unmarked) elementwise-add step, verified against `attention_ad_spike.py:54`
(`Xbar = Xq + Xk + Xv`, matching `jax.grad` to 1e-16). The scheduling constraint —
a node is only processed once every consumer has contributed — is unchanged.

### 4.3 The tape: materialization, and how it differs from the RelGraph draft

**Naming scheme, unchanged:** `__ddx_fwd_{node_id}` / `__ddx_bwd_{node_id}`.

**What changed:** each backward step is now a **plain (unmarked) Substrait `Plan`**
— built by ddx-core using only base relations and builtin functions, since ddx is
authoring its own output and never needs to tag it for a later recognition pass
(barring higher-order AD, §9) — handed to **the engine's own Substrait consumer**,
not converted to SQL text by ddx-core itself. Materialization is then an ordinary
SQL/DataFrame operation wrapping that consumer call, e.g. for DuckDB:
```sql
CREATE TEMP TABLE __ddx_bwd_7 AS SELECT * FROM from_substrait($1)  -- $1 = the backward Plan's bytes
```
or for DataFusion, executing the plan DataFusion's own `datafusion-substrait`
consumer produces and registering the result as a named table. **This is a
meaningful simplification over the first draft**, which had ddx-core's emitter
responsible for generating raw SQL text for every rule from scratch (§3's SQL
templates in the first draft); now ddx-core only has to construct a *plan*, and
each engine's already-existing, already-tested Substrait consumer does the
plan-to-execution work — one less thing for ddx to get right per engine, and it
reuses machinery (`get_substrait`/`from_substrait`, `datafusion-substrait`) that
both target engines already ship and maintain.

`BackwardProgram`'s shape is otherwise as before:

```rust
pub struct BackwardProgram {
    pub forward_steps: Vec<(Ident, Plan)>,   // Plan, not SQL string — §4.3
    pub backward_steps: Vec<(Ident, Plan)>,
    pub gradients: HashMap<RelRef, Ident>,
}
```

Garbage collection note carries over unchanged (drop a temp table once its last
consumer in the walk order has run).

---

## 5. Expressing a "loss query"

Unchanged in substance: no special `LossQuery` type. A loss is the plan's
designated output relation — ordinarily a `ddx_reduce_mark`-tagged `SUM` over an
`ElementwiseBinary` residual or cross-entropy term (`nn.py:423-433`), seeded with
cotangent `1.0`. `vjp_query`'s `wrt` list names which `Source`/`ReadRel` relations
to differentiate against (the parameter tables); the output relation is simply
whichever node has no further consumers in the plan.

---

## 6. Worked trace: reproducing `nn.py`'s layer-2 backward — now with the actual verified SQL

The first draft's §6 hand-waved the forward SQL shape as plausible; this revision's
version is the **literal SQL exercised by `substrait_ad_marker_spike.py`**, adapted
to `nn.py`'s column names, so the correspondence is checked, not asserted:

```sql
-- what the user writes (the one change from nn.py's actual SQL: the marker)
SELECT a.out AS inp, d.out AS out, SUM(ddx_contract_mark(a.val * d.val)) AS val
FROM (SELECT sample, out, val FROM fwd1) a
JOIN delta2 d ON a.sample = d.sample
GROUP BY a.out, d.out
```

Spike-confirmed: this shape (marker wrapping a product inside a `JOIN`+`GROUP BY
SUM`) round-trips through both DuckDB and DataFusion's Substrait producers/consumers
and executes to the numerically correct contraction. ddx-core recognizes the
`ddx_contract_mark` anchor, reads `sample` as the contracted dim off the `JOIN`
condition and `out`/`out` as the surviving dims off the `GROUP BY`, and — for the
*backward* direction (not exercised by this particular query, since this SQL *is*
already `g2`, i.e. already a gradient computation in `nn.py`'s pipeline) — would
emit the transpose contractions per §3.1, as plain unmarked SQL. The concrete,
honest **migration cost** this surfaces: existing `nn.py`-style hand-written SQL
needs exactly one function-name change per contraction (`SUM(x)` →
`SUM(ddx_contract_mark(x))`) to opt into v2's automatic differentiation — a small,
mechanical, per-query cost worth stating plainly rather than implying v2 is a
drop-in replacement for the hand-written demo.

---

## 7. Consequences of the pivot, stated plainly

- **The v1/v2 dependency symmetry gets stronger, not weaker.** v1-core: `sqlparser`
  only. v2-core: `substrait` only. Neither depends on `datafusion` or `duckdb`;
  both are thin, reusable, engine-neutral components with per-engine adapters on
  top — the architecture pattern `design.md` §9 already commits to for v1 now
  extends cleanly to v2, which the `RelGraph` draft did not achieve (it was a
  bespoke type nothing else could produce or consume).
- **Principle 4 is restored.** The user-facing artifact is SQL with one marker
  function per semantically-special operation — not a Rust/Python builder API.
  This is strictly closer to how v1 already works (`grad()`/`jvp()` are also just
  marker functions) than the first draft was.
- **A new class of risk is introduced, and should be named as such, not
  downplayed:** ddx v2 is now bounded by whatever relation/expression vocabulary
  Substrait itself, and each engine's producer/consumer, actually implement — the
  same "coverage gatekeeper" pattern `design.md` already lives with for
  `sqlparser`'s `DuckDbDialect` (F5/G9), recurring one layer up. §3.4 already found
  one concrete instance (window-function support, unverified). This is a real cost
  of reuse versus building a bespoke IR (which would never refuse *because of a
  format's coverage gap* — only because a rule wasn't implemented yet) — worth the
  trade, given the portability payoff §1.4's spike demonstrates, but worth stating
  in `design.md` rather than treated as free.

---

## 8. Summary: direct answers to the five questions in §0

1. **What does `ddx.vjp` take/return?** `vjp_query(plan: &Plan, wrt: &[RelRef]) -> BackwardProgram` — a Substrait `Plan` in (already containing ddx's own marker functions, written by the user in ordinary SQL and compiled by the engine's own producer), a sequence of named, materializable `Plan`s out. §1, §4.
2. **How does a user express a loss query?** The plan's terminal (no-further-consumer) relation, ordinarily a `ddx_reduce_mark`-tagged `SUM`, seeded with cotangent `1.0`. §5.
3. **How does the emitter recognize primitives?** By locating ddx's own registered Substrait extension-function anchors (`ddx_contract_mark`, `ddx_reduce_mark`, `ddx_route_mark`, `ddx_stop_gradient`) — explicit tags surviving from user-written SQL through the engine's own SQL→Substrait planning, never inferred from plan shape. Verified end-to-end, including cross-engine, by `spikes/substrait_ad_marker_spike.py`. §1.4, §2.
4. **How is the tape named/materialized?** `__ddx_fwd_{node_id}`/`__ddx_bwd_{node_id}`, each a plain (unmarked) Substrait `Plan` executed and stored via the *engine's own* Substrait consumer (`from_substrait`, `datafusion-substrait`) — ddx-core builds plans, not SQL text, for this layer. §4.3.
5. **What was missing from the four-rule table, and what's newly at risk from the pivot?** `Route` (§3.4) remains the one rule unverified against `jax.grad`, and now carries a **second**, more basic open question — whether Substrait's window-relation support is even implemented well enough by both target engines to carry it — named as a dedicated M3 spike, with an explicit fallback (keep `Route` on a v1-style text-splice path) if it isn't.
