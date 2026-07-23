// SPDX-FileCopyrightText: 2026 Alex Merose <al@merose.com> & ddx Authors
// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & ddx Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Simulation / property-based tests for the v1 differentiation engine
//! (design.md §5: "numeric agreement" + "round-trip property tests").
//!
//! These are adversarial: instead of hand-picked expressions, they generate
//! random derivable SQL scalar expressions and hold the engine to three
//! properties any correct symbolic differentiator must satisfy:
//!
//! 1. **Numeric agreement (the finite-difference oracle).** The single
//!    strongest check on a derivative: for a random `f`, the symbolic `d/dv f`
//!    evaluated at a point must equal a central finite difference of `f` in the
//!    `v` direction there. A wrong rule (a sign flip, a missing chain factor, a
//!    bad power exponent) disagrees at *every* well-conditioned point, so it is
//!    caught even though a kink artifact (from `abs`) is tolerated as a lone
//!    outlier. Proven to have teeth by mutation testing (a corrupted `cos` rule
//!    fails with 8/8-points-disagree).
//! 2. **Render fidelity.** `reparse(render(d))` must be *value-equal* to `d`.
//!    This is the correctness-relevant form of the §5 round-trip invariant: a
//!    purely structural "== d modulo Nested" check is imprecise for `*`/`/`
//!    associativity (issue #50), but a value comparison still catches the G1
//!    precedence bug (`(a+b)*c` losing its parens → `a+b*c`).
//! 3. **Self-consumption / higher-order stability.** The engine must re-parse
//!    and re-differentiate its *own* text output repeatedly without panicking,
//!    erroring, or emitting unparseable SQL (e.g. a `--` line comment).
//!
//! No external fuzzing crate is used: the core is deliberately `sqlparser`-only,
//! and a dependency-free, deterministic generator keeps every failure perfectly
//! reproducible (each is reported with the seed that produced it).
//!
//! # Soak mode
//!
//! [`soak_continuous_property_fuzz`] is a long-running, `#[ignore]`-d variant
//! that explores far past the bounded tests' fixed seed ranges. It runs for a
//! wall-clock budget and keeps generating fresh expressions, so it can be left
//! running to hunt for rare bugs. Drive it with env vars:
//!
//! ```text
//! DDX_SOAK_SECS=300   cargo test -p ddx-core --test simulation \
//! DDX_SOAK_BASE=0       -- --ignored --nocapture soak_continuous_property_fuzz
//! DDX_SOAK_LOG=/path/to/soak.log
//! ```
//!
//! * `DDX_SOAK_SECS` — wall-clock budget in seconds (default 15).
//! * `DDX_SOAK_BASE` — starting seed offset; bump it between runs to cover new
//!   ground (default 0).
//! * `DDX_SOAK_LOG`  — if set, failures are appended immediately and a heartbeat
//!   line is written ~once a second, so a background run can be tailed live.

use std::fmt::Write as _;
use std::io::Write as _;

use ddx_core::sqlparser::ast::{BinaryOperator, Expr, UnaryOperator, Value};
use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::sqlparser::parser::Parser;
use ddx_core::{ColRef, Ddx, DiffError};

// ---------------------------------------------------------------------------
// A tiny deterministic PRNG (SplitMix64) — reproducible, no dependencies.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
}

/// The differentiation variable a check runs against — the generator uses two
/// free variables `x` and `y`, and the finite-difference oracle perturbs the
/// chosen one.
#[derive(Clone, Copy)]
enum Var {
    X,
    Y,
}

impl Var {
    fn name(self) -> &'static str {
        match self {
            Var::X => "x",
            Var::Y => "y",
        }
    }
}

// ---------------------------------------------------------------------------
// Random expression generator over the *derivable* v1 grammar.
// ---------------------------------------------------------------------------
//
// Everything produced here is inside the engine's supported surface, so
// `differentiate` never returns `NotImplemented`: vars {x, y}, numeric
// literals, `+ - * /`, unary minus, a numeric `CAST`, the unary-rule function
// set, and `power` with exactly one constant side. It emits SQL text
// (parenthesized to fix structure), which both exercises the parser and gives
// readable failures.

/// Unary functions that have a differentiation rule (design.md §3.6).
const UNARY_FNS: &[&str] = &[
    "sin", "cos", "tan", "asin", "acos", "atan", "exp", "ln", "log2", "log10", "sqrt", "sinh",
    "cosh", "tanh", "abs",
];

/// A *non-negative* constant, safe to place under a generated unary minus
/// without producing a `--` line-comment in the source text. (Negative literals
/// still appear — as `power` exponents, below — where they are direct function
/// arguments, and via the engine's own `num()` output.)
fn gen_const(rng: &mut Rng) -> String {
    let choices = ["2", "3", "0.5", "1.5", "2.5"];
    choices[rng.below(choices.len() as u64) as usize].to_string()
}

/// A constant `power` exponent, which *may* be negative — passed as a direct
/// call argument (`power(x, -2)`), never wrapped in a unary minus, so it never
/// forms a `--` in the generated text.
fn gen_exponent(rng: &mut Rng) -> String {
    let choices = ["2", "3", "0.5", "1.5", "-1", "-2", "-0.5", "2.5"];
    choices[rng.below(choices.len() as u64) as usize].to_string()
}

fn gen_expr(rng: &mut Rng, depth: u32) -> String {
    if depth == 0 || rng.below(100) < 30 {
        // Leaf: a variable or a constant.
        return match rng.below(5) {
            0 | 1 => "x".to_string(),
            2 => "y".to_string(),
            _ => gen_const(rng),
        };
    }
    match rng.below(11) {
        0 => format!(
            "({} + {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        1 => format!(
            "({} - {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        2 => format!(
            "({} * {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        3 => format!(
            "({} / {})",
            gen_expr(rng, depth - 1),
            gen_expr(rng, depth - 1)
        ),
        4 => format!("(-{})", gen_expr(rng, depth - 1)),
        5 | 6 => {
            let f = UNARY_FNS[rng.below(UNARY_FNS.len() as u64) as usize];
            format!("{f}({})", gen_expr(rng, depth - 1))
        }
        7 => format!("power({}, {})", gen_expr(rng, depth - 1), gen_exponent(rng)),
        8 => {
            // power(positive-const-base, variable-exponent)
            let base = ["2", "3", "1.5", "0.5"][rng.below(4) as usize];
            format!("power({base}, {})", gen_expr(rng, depth - 1))
        }
        _ => format!("CAST({} AS DOUBLE)", gen_expr(rng, depth - 1)),
    }
}

// ---------------------------------------------------------------------------
// A float evaluator for the emitted grammar (primal *and* derivative).
// ---------------------------------------------------------------------------

fn try_parse(text: &str) -> Result<Expr, String> {
    Parser::new(&GenericDialect {})
        .try_with_sql(text)
        .and_then(|mut p| p.parse_expr())
        .map_err(|e| e.to_string())
}

fn parse_expr(text: &str) -> Expr {
    try_parse(text).unwrap_or_else(|e| panic!("reparse of `{text}` failed: {e}"))
}

/// Evaluate a scalar expression at `(x, y)`. Returns `None` for anything not in
/// the numeric grammar (so an unexpected node fails a comparison loudly rather
/// than silently returning a bogus number).
fn eval(e: &Expr, x: f64, y: f64) -> Option<f64> {
    eval_mag(e, x, y).map(|(v, _)| v)
}

/// The largest absolute value taken by any subexpression of `e` at `(x, y)` —
/// the "how big did the intermediates get" probe. A point where this is huge is
/// unfit for the finite-difference oracle: f64 can no longer resolve an O(1)
/// perturbation against it (a huge additive term cancels the perturbation away;
/// a huge argument to `sin`/`cos` aliases), so *neither* the finite difference
/// *nor* the symbolic value is meaningful there — the point must be skipped.
fn max_intermediate_mag(e: &Expr, x: f64, y: f64) -> Option<f64> {
    eval_mag(e, x, y).map(|(_, m)| m)
}

/// Evaluate `e`, returning `(value, max_abs_intermediate)`.
fn eval_mag(e: &Expr, x: f64, y: f64) -> Option<(f64, f64)> {
    let here = |v: f64| Some((v, v.abs()));
    match e {
        Expr::Value(v) => match &v.value {
            Value::Number(s, _) => here(s.parse::<f64>().ok()?),
            _ => None,
        },
        Expr::Identifier(id) => match id.value.to_ascii_lowercase().as_str() {
            "x" => here(x),
            "y" => here(y),
            _ => None,
        },
        Expr::CompoundIdentifier(parts) => {
            match parts.last()?.value.to_ascii_lowercase().as_str() {
                "x" => here(x),
                "y" => here(y),
                _ => None,
            }
        }
        Expr::Nested(inner) => eval_mag(inner, x, y),
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let (v, m) = eval_mag(expr, x, y)?;
            Some((-v, m.max(v.abs())))
        }
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval_mag(expr, x, y),
        Expr::BinaryOp { left, op, right } => {
            let (a, ma) = eval_mag(left, x, y)?;
            let (b, mb) = eval_mag(right, x, y)?;
            let r = match op {
                BinaryOperator::Plus => a + b,
                BinaryOperator::Minus => a - b,
                BinaryOperator::Multiply => a * b,
                BinaryOperator::Divide => a / b,
                _ => return None,
            };
            Some((r, ma.max(mb).max(r.abs())))
        }
        // Numeric casts are the identity on f64.
        Expr::Cast { expr, .. } => eval_mag(expr, x, y),
        Expr::Function(f) => eval_function(f, x, y),
        // The `sign` CASE (the only CASE the engine emits).
        Expr::Case {
            operand: None,
            conditions,
            else_result,
            ..
        } => {
            let mut m = 0.0f64;
            for w in conditions {
                // Track the compared operand's magnitude too.
                if let Expr::BinaryOp { left, .. } = &w.condition {
                    if let Some((_, lm)) = eval_mag(left, x, y) {
                        m = m.max(lm);
                    }
                }
                if eval_bool(&w.condition, x, y)? {
                    let (v, rm) = eval_mag(&w.result, x, y)?;
                    return Some((v, m.max(rm)));
                }
            }
            let (v, rm) = eval_mag(else_result.as_deref()?, x, y)?;
            Some((v, m.max(rm)))
        }
        _ => None,
    }
}

fn eval_bool(e: &Expr, x: f64, y: f64) -> Option<bool> {
    if let Expr::BinaryOp { left, op, right } = e {
        let a = eval(left, x, y)?;
        let b = eval(right, x, y)?;
        return match op {
            BinaryOperator::Gt => Some(a > b),
            BinaryOperator::Lt => Some(a < b),
            BinaryOperator::GtEq => Some(a >= b),
            BinaryOperator::LtEq => Some(a <= b),
            BinaryOperator::Eq => Some(a == b),
            BinaryOperator::NotEq => Some(a != b),
            _ => None,
        };
    }
    None
}

fn eval_function(f: &ddx_core::sqlparser::ast::Function, x: f64, y: f64) -> Option<(f64, f64)> {
    use ddx_core::sqlparser::ast::{
        FunctionArg, FunctionArgExpr, FunctionArguments, ObjectNamePart,
    };
    let [ObjectNamePart::Identifier(id)] = f.name.0.as_slice() else {
        return None;
    };
    let name = id.value.to_ascii_lowercase();
    let FunctionArguments::List(list) = &f.args else {
        return None;
    };
    let mut args = Vec::new();
    let mut argmag = 0.0f64;
    for a in &list.args {
        match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => {
                let (v, m) = eval_mag(e, x, y)?;
                args.push(v);
                argmag = argmag.max(m);
            }
            _ => return None,
        }
    }
    let a0 = *args.first()?;
    let v = match name.as_str() {
        "sin" => a0.sin(),
        "cos" => a0.cos(),
        "tan" => a0.tan(),
        "asin" => a0.asin(),
        "acos" => a0.acos(),
        "atan" => a0.atan(),
        "exp" => a0.exp(),
        "ln" => a0.ln(),
        "log2" => a0.log2(),
        "log10" => a0.log10(),
        "sqrt" => a0.sqrt(),
        "sinh" => a0.sinh(),
        "cosh" => a0.cosh(),
        "tanh" => a0.tanh(),
        "abs" => a0.abs(),
        "power" | "pow" => {
            let e1 = *args.get(1)?;
            a0.powf(e1)
        }
        _ => return None,
    };
    Some((v, argmag.max(v.abs())))
}

// ---------------------------------------------------------------------------
// The three property checks, as reusable helpers.
// ---------------------------------------------------------------------------

/// A central finite difference of `f` in the `wrt` direction at `(x0, y0)`,
/// step `h`.
fn central_diff(f: &Expr, x0: f64, y0: f64, wrt: Var, h: f64) -> Option<f64> {
    let (fp, fm) = match wrt {
        Var::X => (eval(f, x0 + h, y0)?, eval(f, x0 - h, y0)?),
        Var::Y => (eval(f, x0, y0 + h)?, eval(f, x0, y0 - h)?),
    };
    Some((fp - fm) / (2.0 * h))
}

/// Property 1: symbolic `d` vs a central finite difference of `f` (`expr_text`)
/// in the `wrt` direction. Returns `Some(report)` when a strong majority of
/// well-conditioned points disagree (a real rule bug), tolerating a lone
/// `abs`-kink outlier.
///
/// **Richardson self-consistency gate.** A finite difference is only trusted at
/// a point where halving the step barely moves it (`fd(h) ≈ fd(h/2)`). This is
/// what makes the oracle sound at depth 5–6, where the generator reaches
/// pathological shapes a plain central difference mis-handles — proven necessary
/// by an earlier soak that flagged 16 *correct* derivatives (#54). It kills two
/// false-positive families. Catastrophic cancellation: `power(3, y…) + x`, where
/// the `3^96 ≈ 1e45` term swamps the `+x`, so `f(x+h) − f(x−h)` loses it to
/// float rounding (fd wrongly reads `0`) — halving `h` doubles that error, so
/// the two disagree and the point is skipped. Truncation / aliasing:
/// `sin(exp(9+x))` oscillates with period ≈ `h`, so the central difference is
/// out of its asymptotic regime — halving `h` changes it materially, so the
/// point is skipped. Only points where the difference is in its convergent
/// regime are compared to the symbolic derivative, so a surviving disagreement
/// is a real rule bug.
fn fd_failure(rng: &mut Rng, expr_text: &str, d: &Expr, wrt: Var) -> Option<String> {
    const H: f64 = 1e-4;
    const RTOL: f64 = 2e-3;
    const ATOL: f64 = 1e-5;
    const COND_CAP: f64 = 1e5; // skip near-singular points (huge slope)
                               // Max relative gap between fd(h) and fd(h/2) for the difference to count as
                               // "in its convergent regime" and therefore trustworthy as an oracle.
    const RICHARDSON_TOL: f64 = 1e-4;
    // Above this, some intermediate value is too large for f64 to resolve an
    // O(1) perturbation against — the point is unfit for numeric comparison
    // (total cancellation passes Richardson because *both* fd(h) and fd(h/2)
    // collapse to the same wrong value, so this magnitude gate is what catches
    // it — #54).
    const MAG_CAP: f64 = 1e8;

    let f = parse_expr(expr_text);
    let mut comparable = 0u32;
    let mut disagree = 0u32;
    let mut first_bad = String::new();

    for _ in 0..80 {
        if comparable >= 8 {
            break;
        }
        let x0 = rng.range(0.2, 1.8);
        let y0 = rng.range(0.2, 1.8);
        // Magnitude gate: skip points where f (or its derivative) exercises an
        // intermediate too large for f64 to resolve a perturbation against.
        let fmag = max_intermediate_mag(&f, x0, y0);
        let dmag = max_intermediate_mag(d, x0, y0);
        match (fmag, dmag) {
            (Some(fm), Some(dm)) if fm <= MAG_CAP && dm <= MAG_CAP => {}
            _ => continue,
        }
        let (Some(fd_h), Some(fd_h2), Some(dv)) = (
            central_diff(&f, x0, y0, wrt, H),
            central_diff(&f, x0, y0, wrt, H / 2.0),
            eval(d, x0, y0),
        ) else {
            continue;
        };
        if !fd_h.is_finite() || !fd_h2.is_finite() || !dv.is_finite() {
            continue;
        }
        if fd_h.abs() > COND_CAP || fd_h2.abs() > COND_CAP || dv.abs() > COND_CAP {
            continue;
        }
        // Richardson gate: skip points where the finite difference is not yet in
        // its convergent regime (cancellation- or truncation-dominated).
        if (fd_h - fd_h2).abs() > RICHARDSON_TOL * fd_h2.abs().max(1.0) {
            continue;
        }
        comparable += 1;
        // fd(h/2) is the more accurate estimate at a convergent point.
        let fd = fd_h2;
        if (fd - dv).abs() > ATOL + RTOL * dv.abs().max(fd.abs()) {
            disagree += 1;
            if first_bad.is_empty() {
                first_bad = format!(
                    "x={x0:.6} y={y0:.6}: symbolic d/d{} = {dv:.8}, finite-diff = {fd:.8}",
                    wrt.name()
                );
            }
        }
    }

    if comparable >= 4 && disagree >= 2 && disagree * 2 > comparable {
        return Some(format!(
            "[finite-diff] d/d{} {expr_text}\n  => {d}\n  {disagree}/{comparable} points disagree; e.g. {first_bad}",
            wrt.name()
        ));
    }
    None
}

/// Property 2: `reparse(render(d))` computes the same value as `d` (immune to
/// benign `*`/`/` reassociation; catches a value-changing paren-drop).
fn fidelity_failure(rng: &mut Rng, expr_text: &str, d: &Expr, wrt: Var) -> Option<String> {
    const RTOL: f64 = 1e-9;
    const ATOL: f64 = 1e-11;
    let rendered = d.to_string();
    if rendered.contains("--") {
        return Some(format!(
            "[render] emitted a `--` comment: d/d{} {expr_text} => {rendered}",
            wrt.name()
        ));
    }
    let reparsed = match try_parse(&rendered) {
        Ok(rp) => rp,
        Err(e) => {
            return Some(format!(
                "[render] engine emitted unparseable SQL: d/d{} {expr_text} => {rendered} ({e})",
                wrt.name()
            ))
        }
    };
    let mut compared = 0u32;
    for _ in 0..40 {
        if compared >= 6 {
            break;
        }
        let x0 = rng.range(0.2, 1.8);
        let y0 = rng.range(0.2, 1.8);
        let (Some(va), Some(vb)) = (eval(d, x0, y0), eval(&reparsed, x0, y0)) else {
            continue;
        };
        if !va.is_finite() || !vb.is_finite() || va.abs() > 1e6 {
            continue;
        }
        compared += 1;
        if (va - vb).abs() > ATOL + RTOL * va.abs() {
            return Some(format!(
                "[render] render changed the value: d/d{} {expr_text}\n  rendered = {rendered}\n  at x={x0:.4} y={y0:.4}: AST = {va:.10}, reparsed = {vb:.10}",
                wrt.name()
            ));
        }
    }
    None
}

/// Property 3: the engine re-consumes its own text output for up to 4 rounds of
/// higher-order differentiation without panicking, erroring unexpectedly, or
/// emitting unparseable SQL.
fn self_consumption_failure(ddx: &Ddx, wrt: &ColRef, original: &str) -> Option<String> {
    let mut current = original.to_string();
    for round in 0..4 {
        let parsed = match try_parse(&current) {
            Ok(p) => p,
            Err(e) => {
                return Some(format!(
                    "[self-consumption] round {round}: engine's own output did not reparse: `{current}` ({e}) [from {original}]"
                ))
            }
        };
        match ddx.differentiate(&parsed, wrt) {
            Ok(d) => {
                let rendered = d.to_string();
                if rendered.contains("--") {
                    return Some(format!(
                        "[self-consumption] round {round}: emitted `--` comment: `{rendered}` [from {original}]"
                    ));
                }
                current = rendered;
            }
            // Re-differentiating can legitimately reach a non-finite constant
            // (e.g. an overflowing exponent) — a *typed* error by design.
            Err(DiffError::NotImplemented(_)) => break,
            Err(e) => {
                return Some(format!(
                    "[self-consumption] round {round}: unexpected error re-differentiating `{current}`: {e} [from {original}]"
                ))
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Property 4: `rewrite_sql` splice fidelity (design.md §3.2, G3/F5).
// ---------------------------------------------------------------------------
//
// The three properties above drive `differentiate` on bare expressions; none of
// them exercise `rewrite_sql` — the parse-free pre-gate, the UTF-8-aware source
// span → byte-offset splice, multiple/nested markers, or the marker-free
// identity guarantee. That subsystem is exactly where bug #52 lived. The
// invariant here is *structural* (byte-level), not numeric: rewriting a marker
// statement must replace **only** each marker's span with `(derivative)` and
// leave every other byte identical.

/// Valid SELECT-list prefixes to place before a marker. Several carry multibyte
/// characters (in string literals / comments) *before* the marker, so the
/// marker's character-column no longer equals its byte offset — the case the
/// `locate` char→byte conversion (G3) must get right.
const STMT_PREFIXES: &[&str] = &[
    "SELECT ",
    "SELECT x, ",
    "SELECT 'héllo', ",
    "SELECT 'naïve café ☕' AS greeting, ",
    "SELECT /* café ☕ */ ",
    "SELECT   ",
    "SELECT y AS why, ",
];

/// Valid statement tails (ASCII identifiers only — unicode kept to string
/// literals/comments, since unquoted-identifier unicode support is dialect-
/// dependent and not what this fuzz is testing).
const STMT_SUFFIXES: &[&str] = &[
    " FROM t",
    " AS d FROM t",
    " FROM data",
    " AS d FROM t WHERE label <> 'niño'",
    "",
];

/// Valid separators between two markers — as sibling select items (`, `) or
/// inside one arithmetic select item (` + `, ` * `).
const STMT_MIDS: &[&str] = &[", ", " + ", " * ", ", z, "];

/// Whether the splice fuzz generates `jvp` markers. Now `true`: #57 is fixed —
/// `rewrite_sql` splices to the call's matching close paren (found over the
/// token stream) rather than the under-reported `Expr::span()` end, so a `jvp`
/// with a compound/casted tangent (last-arg tail a `CAST`/`Nested`) splices
/// correctly. The minimal repro is `splice_handles_marker_with_cast_or_nested_tail`.
const SPLICE_FUZZ_INCLUDES_JVP: bool = true;

/// Build one marker call and the exact text `rewrite_sql` must splice in its
/// place (`(derivative)`), or `None` if the marker's derivative is undefined
/// (in which case `rewrite_sql` must error on the whole statement).
fn gen_marker_segment(rng: &mut Rng, ddx: &Ddx) -> (String, Option<String>) {
    let wrt = if rng.below(2) == 0 { "x" } else { "y" };
    let wrt_col = ColRef::bare(wrt);
    let depth = 2 + rng.below(2) as u32;
    let expr_text = gen_expr(rng, depth);
    let expr = parse_expr(&expr_text);

    match rng.below(3) {
        // Nested higher-order: grad(grad(expr, wrt), wrt).
        0 => {
            let marker = format!("grad(grad({expr_text}, {wrt}), {wrt})");
            let repl = ddx
                .differentiate(&expr, &wrt_col)
                .and_then(|d1| ddx.differentiate(&d1, &wrt_col))
                .ok()
                .map(|dd| format!("({dd})"));
            (marker, repl)
        }
        // jvp(expr, wrt, tangent) — only when enabled (bug #57).
        1 if SPLICE_FUZZ_INCLUDES_JVP => {
            let tan_depth = 1 + rng.below(2) as u32;
            let tan_text = gen_expr(rng, tan_depth);
            let tan = parse_expr(&tan_text);
            let marker = format!("jvp({expr_text}, {wrt}, {tan_text})");
            match ddx.jvp(&expr, &[(wrt_col, tan)]) {
                Ok(v) => (marker, Some(format!("({v})"))),
                Err(_) => (marker, None),
            }
        }
        // grad(expr, wrt)
        _ => {
            let marker = format!("grad({expr_text}, {wrt})");
            match ddx.differentiate(&expr, &wrt_col) {
                Ok(d) => (marker, Some(format!("({d})"))),
                Err(_) => (marker, None),
            }
        }
    }
}

/// Property 4a: assemble a statement with 1–3 markers wrapped in random
/// (Unicode-bearing) scaffolding and assert `rewrite_sql` splices each marker
/// exactly, byte-for-byte, leaving all surrounding text untouched. If any
/// marker's derivative is undefined, the whole rewrite must error instead.
fn splice_failure(rng: &mut Rng, ddx: &Ddx) -> Option<String> {
    let n = 1 + rng.below(3) as usize;
    let prefix = STMT_PREFIXES[rng.below(STMT_PREFIXES.len() as u64) as usize];
    let suffix = STMT_SUFFIXES[rng.below(STMT_SUFFIXES.len() as u64) as usize];

    let mut input = String::from(prefix);
    let mut expected = String::from(prefix);
    let mut any_undefined = false;
    for i in 0..n {
        if i > 0 {
            let mid = STMT_MIDS[rng.below(STMT_MIDS.len() as u64) as usize];
            input.push_str(mid);
            expected.push_str(mid);
        }
        let (marker, repl) = gen_marker_segment(rng, ddx);
        input.push_str(&marker);
        match repl {
            Some(r) => expected.push_str(&r),
            None => any_undefined = true,
        }
    }
    input.push_str(suffix);
    expected.push_str(suffix);

    let got = ddx.rewrite_sql(&input, &GenericDialect {});
    if any_undefined {
        // At least one marker's derivative is undefined → the whole rewrite must
        // fail loud, never partially rewrite.
        return match got {
            Err(_) => None,
            Ok(o) => Some(format!(
                "[splice] expected an error (a marker derivative is undefined) but got Ok:\n  input  = {input}\n  output = {o}"
            )),
        };
    }
    match got {
        Ok(o) if o == expected => None,
        Ok(o) => Some(format!(
            "[splice] rewrite_sql splice mismatch:\n  input    = {input}\n  expected = {expected}\n  actual   = {o}"
        )),
        Err(e) => Some(format!(
            "[splice] rewrite_sql errored on a valid marker statement:\n  input = {input}\n  error = {e}"
        )),
    }
}

/// A marker-free statement — some deliberately containing a `grad(`/`jvp(`
/// substring inside a string literal, a comment, or a *qualified* call, so the
/// pre-gate's substring filter hits and the statement is parsed but no real
/// marker is found. Every one must come back byte-identical.
fn gen_marker_free_stmt(rng: &mut Rng) -> String {
    match rng.below(6) {
        0 => {
            let depth = 2 + rng.below(2) as u32;
            format!("SELECT {} FROM t", gen_expr(rng, depth))
        }
        1 => "SELECT 'grad(x, x)' AS s FROM t".to_string(),
        2 => "SELECT x /* grad(y, y) */ FROM t".to_string(),
        3 => "SELECT myschema.grad(x, x) AS d FROM t".to_string(),
        4 => "SELECT 'jvp(sin(x), x, dx)' AS label, x FROM t".to_string(),
        _ => format!(
            "SELECT {} AS val FROM t WHERE label <> 'grad('",
            gen_expr(rng, 2)
        ),
    }
}

/// Property 4b: a marker-free statement is returned byte-identical.
fn marker_free_failure(rng: &mut Rng, ddx: &Ddx) -> Option<String> {
    let s = gen_marker_free_stmt(rng);
    match ddx.rewrite_sql(&s, &GenericDialect {}) {
        Ok(o) if o == s => None,
        Ok(o) => Some(format!(
            "[identity] marker-free statement was modified:\n  input  = {s}\n  output = {o}"
        )),
        Err(e) => Some(format!(
            "[identity] marker-free statement errored:\n  input = {s}\n  error = {e}"
        )),
    }
}

/// Parse, differentiate, and run every property on one generated expression.
/// Returns each failure report (empty ⇒ all properties held).
fn run_all_checks(rng: &mut Rng, ddx: &Ddx, text: &str, wrt: Var) -> Vec<String> {
    let mut out = Vec::new();
    let parsed = match try_parse(text) {
        Ok(p) => p,
        Err(e) => {
            out.push(format!(
                "[generator] produced unparseable text `{text}` ({e})"
            ));
            return out;
        }
    };
    let wrt_col = ColRef::bare(wrt.name());
    let d = match ddx.differentiate(&parsed, &wrt_col) {
        Ok(d) => d,
        Err(DiffError::NotImplemented(_)) => return out, // outside surface; skip
        Err(e) => {
            out.push(format!("[differentiate] unexpected error on `{text}`: {e}"));
            return out;
        }
    };
    if let Some(f) = fd_failure(rng, text, &d, wrt) {
        out.push(f);
    }
    if let Some(f) = fidelity_failure(rng, text, &d, wrt) {
        out.push(f);
    }
    if let Some(f) = self_consumption_failure(ddx, &wrt_col, text) {
        out.push(f);
    }
    // Statement-level rewrite_sql properties (self-generating; the `text`/`wrt`
    // above are for the expression-level checks).
    if let Some(f) = splice_failure(rng, ddx) {
        out.push(f);
    }
    if let Some(f) = marker_free_failure(rng, ddx) {
        out.push(f);
    }
    out
}

// ---------------------------------------------------------------------------
// Bounded tests (run every `cargo test`).
// ---------------------------------------------------------------------------

#[test]
fn finite_difference_agreement_over_random_expressions() {
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();
    let mut tested = 0u32;

    for seed in 0..4000u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
        let depth = 2 + (seed % 3) as u32;
        let text = gen_expr(&mut rng, depth);
        let parsed = parse_expr(&text);
        let d = match ddx.differentiate(&parsed, &wrt) {
            Ok(d) => d,
            Err(DiffError::NotImplemented(_)) => continue,
            Err(e) => {
                failures.push(format!("UNEXPECTED ERROR on `{text}`: {e}"));
                continue;
            }
        };
        tested += 1;
        if let Some(report) = fd_failure(&mut rng, &text, &d, Var::X) {
            failures.push(report);
        }
    }

    assert!(
        tested > 500,
        "generator produced too few derivable cases: {tested}"
    );
    assert!(
        failures.is_empty(),
        "finite-difference oracle found {} disagreement(s) out of {} tested:\n\n{}",
        failures.len(),
        tested,
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn render_reparse_is_value_preserving() {
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..5000u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xDEAD_BEEF);
        let depth = 2 + (seed % 4) as u32;
        let text = gen_expr(&mut rng, depth);
        let parsed = parse_expr(&text);
        let d = match ddx.differentiate(&parsed, &wrt) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if let Some(report) = fidelity_failure(&mut rng, &text, &d, Var::X) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "render-fidelity fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn higher_order_self_consumption_is_stable() {
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..2000u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678);
        let depth = 2 + (seed % 3) as u32;
        let original = gen_expr(&mut rng, depth);
        if let Some(report) = self_consumption_failure(&ddx, &wrt, &original) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "self-consumption fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn rewrite_sql_splice_is_byte_faithful() {
    // Statement-level fuzz of `rewrite_sql`: markers wrapped in random
    // (Unicode-bearing) scaffolding must be spliced exactly, leaving every other
    // byte identical (design.md §3.2, G3/F5).
    let ddx = Ddx::new();
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..4000u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x5719_C0DE);
        if let Some(report) = splice_failure(&mut rng, &ddx) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "splice-fidelity fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn splice_handles_marker_with_cast_or_nested_tail() {
    // The splice must cover the *whole* marker call. When the last argument's
    // tail is a CAST (span excludes ` AS <type>`) or a Nested `( … )` (span
    // excludes the closing `)`), rewrite_sql currently stops early and leaves
    // trailing bytes behind, producing unbalanced/corrupt SQL (#57).
    let ddx = Ddx::new();
    // jvp(sin(x), x, CAST(y AS DOUBLE)) — tangent tail is a CAST.
    assert_eq!(
        ddx.rewrite_sql(
            "SELECT jvp(sin(x), x, CAST(y AS DOUBLE)) FROM t",
            &GenericDialect {}
        )
        .unwrap(),
        "SELECT (cos(x) * CAST(y AS DOUBLE)) FROM t"
    );
    // jvp(x, x, (y + z)) — tangent tail is a Nested `( … )`.
    assert_eq!(
        ddx.rewrite_sql("SELECT jvp(x, x, (y + z)) FROM t", &GenericDialect {})
            .unwrap(),
        "SELECT ((y + z)) FROM t"
    );
}

#[test]
fn marker_free_statements_are_byte_identical() {
    // The pre-gate / no-marker guarantee: a statement with no real marker —
    // including one whose text carries a `grad(`/`jvp(` substring in a string,
    // comment, or qualified call — comes back byte-identical (design.md §3.2).
    let ddx = Ddx::new();
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..2000u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1DE0_7175);
        if let Some(report) = marker_free_failure(&mut rng, &ddx) {
            failures.push(report);
        }
    }

    assert!(
        failures.is_empty(),
        "marker-free identity fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

// ---------------------------------------------------------------------------
// Soak test — long-running, #[ignore]-d, driven by env vars (see module docs).
// ---------------------------------------------------------------------------

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "soak: long-running continuous fuzz; run explicitly with DDX_SOAK_SECS set"]
fn soak_continuous_property_fuzz() {
    use std::time::Instant;

    let budget_secs = env_u64("DDX_SOAK_SECS", 15);
    let base = env_u64("DDX_SOAK_BASE", 0);
    let log_path = std::env::var("DDX_SOAK_LOG").ok();

    let mut log = log_path.as_ref().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap_or_else(|e| panic!("cannot open DDX_SOAK_LOG `{p}`: {e}"))
    });
    let mut logline = |s: &str| {
        eprintln!("{s}");
        if let Some(f) = log.as_mut() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
    };

    let ddx = Ddx::new();
    let start = Instant::now();
    let deadline = budget_secs;
    let mut iters: u64 = 0;
    let mut failures: u64 = 0;
    let mut last_beat = 0u64;

    logline(&format!(
        "SOAK start: budget={budget_secs}s base={base} log={:?}",
        log_path
    ));

    loop {
        let elapsed = start.elapsed().as_secs();
        if elapsed >= deadline {
            break;
        }

        // A fresh, reproducible seed for this iteration.
        let seed = base.wrapping_add(iters);
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xA5A5_5A5A);
        // Deeper trees than the bounded tests, to reach rarer shapes.
        let depth = 2 + (rng.below(5) as u32); // 2..=6
        let wrt = if rng.below(2) == 0 { Var::X } else { Var::Y };
        let text = gen_expr(&mut rng, depth);

        let reports = run_all_checks(&mut rng, &ddx, &text, wrt);
        if reports.is_empty() {
            // A skip (outside-surface) vs a real pass are indistinguishable
            // here; count both as progress.
        } else {
            for r in &reports {
                failures += 1;
                logline(&format!(
                    "\nFAILURE (seed={seed}, base={base}, depth={depth}, wrt={}):\n{r}",
                    wrt.name()
                ));
            }
        }

        iters += 1;

        // Heartbeat ~once a second.
        if elapsed != last_beat {
            last_beat = elapsed;
            logline(&format!(
                "HEARTBEAT elapsed={elapsed}s iters={iters} failures={failures} rate={}/s",
                iters / elapsed.max(1)
            ));
        }
    }

    let mut summary = String::new();
    let _ = write!(
        summary,
        "SOAK done: elapsed={}s iters={iters} failures={failures} base={base} next_base={}",
        start.elapsed().as_secs(),
        base.wrapping_add(iters)
    );
    logline(&summary);

    assert_eq!(
        failures, 0,
        "soak found {failures} property failure(s) — see the FAILURE lines above (each has a reproducing seed)"
    );
}
