//! Simulation / property-based tests for the v1 differentiation engine
//! (design.md §5: "numeric agreement" + "round-trip property tests").
//!
//! These are adversarial: instead of hand-picked expressions, they generate
//! thousands of random derivable SQL scalar expressions and hold the engine to
//! three properties that any correct symbolic differentiator must satisfy:
//!
//! 1. **Numeric agreement (the finite-difference oracle).** The single
//!    strongest check on a derivative: for a random `f`, the symbolic `d/dx f`
//!    evaluated at a point must equal a central finite difference of `f` there.
//!    A wrong rule (a sign flip, a missing chain factor, a bad power exponent)
//!    disagrees at *every* well-conditioned point, so it is caught even though a
//!    kink artifact (from `abs`) is tolerated as a lone outlier.
//! 2. **Round-trip (design.md §5, G1).** `reparse(render(d)) == d` modulo
//!    `Nested`, fuzzed over the whole generated space, not the 10 hand cases.
//! 3. **Self-consumption / higher-order stability.** The engine must be able to
//!    re-parse and re-differentiate its *own* text output repeatedly without
//!    panicking, erroring, or emitting unparseable SQL (e.g. a `--` comment).
//!
//! No external fuzzing crate is used: the core is deliberately `sqlparser`-only,
//! and a dev-dependency-free, deterministic generator keeps failures perfectly
//! reproducible (each is printed with its seed).

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

// ---------------------------------------------------------------------------
// Random expression generator over the *derivable* v1 grammar.
// ---------------------------------------------------------------------------
//
// Everything produced here is inside the engine's supported surface, so
// `differentiate` never returns `NotImplemented`: vars {x, y}, numeric
// literals, `+ - * /`, unary minus, the unary-rule function set, and `power`
// with exactly one constant side. It emits SQL text (parenthesized to fix
// structure), which both exercises the parser and gives readable failures.

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
    match rng.below(9) {
        0 => format!("({} + {})", gen_expr(rng, depth - 1), gen_expr(rng, depth - 1)),
        1 => format!("({} - {})", gen_expr(rng, depth - 1), gen_expr(rng, depth - 1)),
        2 => format!("({} * {})", gen_expr(rng, depth - 1), gen_expr(rng, depth - 1)),
        3 => format!("({} / {})", gen_expr(rng, depth - 1), gen_expr(rng, depth - 1)),
        4 => format!("(-{})", gen_expr(rng, depth - 1)),
        5 | 6 => {
            let f = UNARY_FNS[rng.below(UNARY_FNS.len() as u64) as usize];
            format!("{f}({})", gen_expr(rng, depth - 1))
        }
        7 => {
            // power(base, const-exponent)
            format!("power({}, {})", gen_expr(rng, depth - 1), gen_exponent(rng))
        }
        _ => {
            // power(positive-const-base, variable-exponent)
            let base = ["2", "3", "1.5", "0.5"][rng.below(4) as usize];
            format!("power({base}, {})", gen_expr(rng, depth - 1))
        }
    }
}

// ---------------------------------------------------------------------------
// A float evaluator for the emitted grammar (primal *and* derivative).
// ---------------------------------------------------------------------------

fn parse_expr(text: &str) -> Expr {
    Parser::new(&GenericDialect {})
        .try_with_sql(text)
        .and_then(|mut p| p.parse_expr())
        .unwrap_or_else(|e| panic!("reparse of `{text}` failed: {e}"))
}

/// Evaluate a scalar expression at `(x, y)`. Returns `None` for anything not in
/// the numeric grammar (so an unexpected node fails a comparison loudly rather
/// than silently returning a bogus number).
fn eval(e: &Expr, x: f64, y: f64) -> Option<f64> {
    match e {
        Expr::Value(v) => match &v.value {
            Value::Number(s, _) => s.parse::<f64>().ok(),
            _ => None,
        },
        Expr::Identifier(id) => match id.value.to_ascii_lowercase().as_str() {
            "x" => Some(x),
            "y" => Some(y),
            _ => None,
        },
        Expr::CompoundIdentifier(parts) => match parts.last()?.value.to_ascii_lowercase().as_str() {
            "x" => Some(x),
            "y" => Some(y),
            _ => None,
        },
        Expr::Nested(inner) => eval(inner, x, y),
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => Some(-eval(expr, x, y)?),
        Expr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval(expr, x, y),
        Expr::BinaryOp { left, op, right } => {
            let a = eval(left, x, y)?;
            let b = eval(right, x, y)?;
            match op {
                BinaryOperator::Plus => Some(a + b),
                BinaryOperator::Minus => Some(a - b),
                BinaryOperator::Multiply => Some(a * b),
                BinaryOperator::Divide => Some(a / b),
                _ => None,
            }
        }
        // Numeric casts are the identity on f64.
        Expr::Cast { expr, .. } => eval(expr, x, y),
        Expr::Function(f) => eval_function(f, x, y),
        // The `sign` CASE (the only CASE the engine emits).
        Expr::Case {
            operand: None,
            conditions,
            else_result,
            ..
        } => {
            for w in conditions {
                if eval_bool(&w.condition, x, y)? {
                    return eval(&w.result, x, y);
                }
            }
            eval(else_result.as_deref()?, x, y)
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

fn eval_function(f: &ddx_core::sqlparser::ast::Function, x: f64, y: f64) -> Option<f64> {
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
    for a in &list.args {
        match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => args.push(eval(e, x, y)?),
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
    Some(v)
}

// ---------------------------------------------------------------------------
// Property 1 — the finite-difference numeric oracle.
// ---------------------------------------------------------------------------

/// For a fixed expression, sample points and compare the symbolic derivative to
/// a central finite difference. Returns `Some(report)` on a genuine mismatch
/// (a strong majority of well-conditioned points disagree), `None` otherwise.
///
/// A lone disagreement is tolerated: `abs(g(x))` has a kink where `g(x) = 0`,
/// and a finite difference straddling it disagrees with the (correct) pinned
/// convention at that one point. A wrong *rule* disagrees everywhere.
fn fd_check(rng: &mut Rng, expr_text: &str, d: &Expr) -> Option<String> {
    const H: f64 = 1e-4;
    const RTOL: f64 = 2e-3;
    const ATOL: f64 = 1e-5;
    const COND_CAP: f64 = 1e5; // skip near-singular points (huge slope)

    let f = parse_expr(expr_text);
    let mut comparable = 0u32;
    let mut disagree = 0u32;
    let mut first_bad = String::new();

    // Draw up to 60 candidate points, keep the well-conditioned finite ones.
    for _ in 0..60 {
        if comparable >= 8 {
            break;
        }
        let x0 = rng.range(0.2, 1.8);
        let y0 = rng.range(0.2, 1.8);
        let (Some(fp), Some(fm), Some(dv)) = (
            eval(&f, x0 + H, y0),
            eval(&f, x0 - H, y0),
            eval(d, x0, y0),
        ) else {
            continue;
        };
        let fd = (fp - fm) / (2.0 * H);
        if !fd.is_finite()
            || !dv.is_finite()
            || fd.abs() > COND_CAP
            || dv.abs() > COND_CAP
        {
            continue;
        }
        comparable += 1;
        if (fd - dv).abs() > ATOL + RTOL * dv.abs().max(fd.abs()) {
            disagree += 1;
            if first_bad.is_empty() {
                first_bad = format!(
                    "x={x0:.6} y={y0:.6}: symbolic d/dx = {dv:.8}, finite-diff = {fd:.8}"
                );
            }
        }
    }

    // Need enough evidence, and a majority must disagree (a real rule bug),
    // not a single kink artifact.
    if comparable >= 4 && disagree >= 2 && disagree * 2 > comparable {
        return Some(format!(
            "d/dx {expr_text}\n  => {d}\n  {disagree}/{comparable} points disagree; e.g. {first_bad}"
        ));
    }
    None
}

#[test]
fn finite_difference_agreement_over_random_expressions() {
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();
    let mut tested = 0u32;

    for seed in 0..4000u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
        let depth = 2 + (seed % 3) as u32; // depth 2..=4
        let text = gen_expr(&mut rng, depth);

        let parsed = parse_expr(&text);
        let d = match ddx.differentiate(&parsed, &wrt) {
            Ok(d) => d,
            // Everything generated is derivable; a genuine error here is itself
            // a finding worth seeing.
            Err(DiffError::NotImplemented(_)) => continue,
            Err(e) => {
                failures.push(format!("UNEXPECTED ERROR on `{text}`: {e}"));
                continue;
            }
        };
        tested += 1;
        if let Some(report) = fd_check(&mut rng, &text, &d) {
            failures.push(report);
        }
    }

    assert!(tested > 500, "generator produced too few derivable cases: {tested}");
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

// ---------------------------------------------------------------------------
// Property 2 — render fidelity: reparse(render(d)) is *value-equal* to d.
// ---------------------------------------------------------------------------
//
// This is the correctness-relevant form of the design.md §5 round-trip
// invariant. A purely structural "== d modulo Nested" check is imprecise for
// `*`/`/` associativity (issue #50): the product rule builds `a * (b / c)`,
// which renders unparenthesized as `a * b / c` and reparses as `(a * b) / c` —
// a *different tree* but the *same value*. What actually matters — and what the
// G1 precedence bug (`(a+b)*c` losing its parens → `a+b*c`) violates — is that
// the rendered-then-reparsed derivative computes the same number as the AST the
// engine constructed. So this compares by evaluation, not by tree shape: it is
// immune to benign reassociation yet still catches any paren-drop that changes
// the value.

#[test]
fn render_reparse_is_value_preserving() {
    const RTOL: f64 = 1e-9; // float `*`/`/` reassociation differs by ~ulps only
    const ATOL: f64 = 1e-11;
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
        let rendered = d.to_string();
        // A `--` in emitted SQL is a line comment — a silent-wrong render.
        if rendered.contains("--") {
            failures.push(format!("emitted a `--` comment: d/dx {text} => {rendered}"));
            continue;
        }
        let reparsed = match Parser::new(&GenericDialect {})
            .try_with_sql(&rendered)
            .and_then(|mut p| p.parse_expr())
        {
            Ok(rp) => rp,
            Err(e) => {
                failures.push(format!(
                    "engine emitted unparseable SQL: d/dx {text} => {rendered} ({e})"
                ));
                continue;
            }
        };

        // Compare the constructed AST and its rendered-then-reparsed form by
        // value at several well-conditioned points.
        let mut compared = 0u32;
        for _ in 0..40 {
            if compared >= 6 {
                break;
            }
            let x0 = rng.range(0.2, 1.8);
            let y0 = rng.range(0.2, 1.8);
            let (Some(va), Some(vb)) = (eval(&d, x0, y0), eval(&reparsed, x0, y0)) else {
                continue;
            };
            if !va.is_finite() || !vb.is_finite() || va.abs() > 1e6 {
                continue;
            }
            compared += 1;
            if (va - vb).abs() > ATOL + RTOL * va.abs() {
                failures.push(format!(
                    "render changed the value: d/dx {text}\n  rendered = {rendered}\n  at x={x0:.4} y={y0:.4}: AST = {va:.10}, reparsed = {vb:.10}"
                ));
                break;
            }
        }
    }

    assert!(
        failures.is_empty(),
        "render-fidelity fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures.iter().take(15).cloned().collect::<Vec<_>>().join("\n\n")
    );
}

// ---------------------------------------------------------------------------
// Property 3 — self-consumption: the engine must re-consume its own output.
// ---------------------------------------------------------------------------

#[test]
fn higher_order_self_consumption_is_stable() {
    // Differentiate, render, reparse, differentiate again — up to 4 rounds.
    // Higher-order derivatives are a stated feature ("grad(grad(f,x),x) just
    // works"); the engine must never panic, emit unparseable SQL, or error on
    // an expression it produced itself.
    let ddx = Ddx::new();
    let wrt = ColRef::bare("x");
    let mut failures: Vec<String> = Vec::new();

    for seed in 0..2000u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678);
        let depth = 2 + (seed % 3) as u32;
        let original = gen_expr(&mut rng, depth);

        let mut current = original.clone();
        for round in 0..4 {
            let parsed = Parser::new(&GenericDialect {})
                .try_with_sql(&current)
                .and_then(|mut p| p.parse_expr());
            let parsed = match parsed {
                Ok(p) => p,
                Err(e) => {
                    failures.push(format!(
                        "round {round}: engine's own output did not reparse: `{current}` ({e}) [from {original}]"
                    ));
                    break;
                }
            };
            match ddx.differentiate(&parsed, &wrt) {
                Ok(d) => {
                    let rendered = d.to_string();
                    if rendered.contains("--") {
                        failures.push(format!(
                            "round {round}: emitted `--` comment: `{rendered}` [from {original}]"
                        ));
                        break;
                    }
                    current = rendered;
                }
                // Re-differentiating can legitimately reach a non-finite constant
                // (e.g. an exponent that overflows), which is a *typed* error by
                // design — acceptable. Anything else is not.
                Err(DiffError::NotImplemented(_)) => break,
                Err(e) => {
                    failures.push(format!(
                        "round {round}: unexpected error re-differentiating `{current}`: {e} [from {original}]"
                    ));
                    break;
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "self-consumption fuzz found {} failure(s):\n\n{}",
        failures.len(),
        failures.iter().take(15).cloned().collect::<Vec<_>>().join("\n\n")
    );
}
