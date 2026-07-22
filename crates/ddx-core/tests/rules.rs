//! The prototype's rule unit tests, ported to `ddx-core` (design.md §3.2, M0:
//! "port the prototype's 15 rule tests — they pin the math unchanged").
//!
//! The *math* is identical to xarray-sql#192's `src/autograd.rs`; the rendered
//! output differs in two documented, deliberate ways introduced by the design:
//! `div()` casts its numerator to `DOUBLE` (F4), and composite operands are
//! `Nested`-wrapped for precedence safety (G1). Both are asserted here.

use ddx_core::sqlparser::dialect::GenericDialect;
use ddx_core::Ddx;

fn d(expr: &str, wrt: &str) -> String {
    Ddx::new()
        .differentiate_sql(expr, wrt, &GenericDialect {})
        .unwrap_or_else(|e| panic!("differentiate_sql({expr}, {wrt}) failed: {e}"))
}

#[test]
fn constant_has_zero_derivative() {
    assert_eq!(d("3.0", "x"), "0.0");
}

#[test]
fn variable_has_unit_derivative() {
    assert_eq!(d("x", "x"), "1.0");
}

#[test]
fn other_variable_has_zero_derivative() {
    assert_eq!(d("y", "x"), "0.0");
}

#[test]
fn sum_rule_folds_constants() {
    // d/dx (x + y) = 1 + 0 = 1
    assert_eq!(d("x + y", "x"), "1.0");
}

#[test]
fn product_rule() {
    // d/dx (x * x) = 1*x + x*1 = x + x
    assert_eq!(d("x * x", "x"), "x + x");
}

#[test]
fn quotient_rule() {
    // d/dx (x / y) = (1*y - x*0) / (y*y) = y / (y*y), numerator cast to DOUBLE.
    assert_eq!(d("x / y", "x"), "CAST(y AS DOUBLE) / (y * y)");
}

#[test]
fn chain_rule_sin() {
    // d/dx sin(x) = cos(x) * 1 = cos(x)
    assert_eq!(d("sin(x)", "x"), "cos(x)");
}

#[test]
fn composite_sin_times_x() {
    // d/dx (sin(x) * x) = cos(x)*x + sin(x)
    assert_eq!(d("sin(x) * x", "x"), "cos(x) * x + sin(x)");
}

#[test]
fn power_constant_exponent() {
    // d/dx power(x, 2) = 2 * power(x, 1) * 1 = 2 * power(x, 1)
    assert_eq!(d("power(x, 2)", "x"), "2.0 * power(x, 1.0)");
}

#[test]
fn higher_order_derivative() {
    // Differentiation composes: d2/dx2 sin(x) = -sin(x).
    let d1 = d("sin(x)", "x");
    assert_eq!(d(&d1, "x"), "-sin(x)");
}

#[test]
fn unsupported_operator_errors() {
    assert!(Ddx::new()
        .differentiate_sql("x % y", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn unsupported_function_errors() {
    // atan2 is binary and has no rule yet.
    assert!(Ddx::new()
        .differentiate_sql("atan2(x, y)", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn general_power_uv_errors() {
    // power(x, x): both base and exponent vary — not supported yet.
    assert!(Ddx::new()
        .differentiate_sql("power(x, x)", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn double_negation_does_not_render_a_comment() {
    // d/dx(-cos(x)) = sin(x). The intermediate is neg(neg(sin(x))); without
    // folding it renders as `--sin(x)`, a SQL line comment (a silently-wrong
    // result). It must fold to `sin(x)`.
    let out = d("-cos(x)", "x");
    assert!(!out.contains("--"), "rendered a `--` comment: {out}");
    assert_eq!(out, "sin(x)");
    // And nested inside a larger expression, still no stray `--`.
    let nested = d("sin(-cos(x))", "x");
    assert!(!nested.contains("--"), "rendered a `--` comment: {nested}");
    assert_eq!(nested, "cos(-cos(x)) * sin(x)");
}

#[test]
fn cast_to_numeric_type_is_differentiable() {
    // A numeric cast is locally linear: d/dx CAST(x*x AS DOUBLE) = CAST(x+x AS DOUBLE).
    assert_eq!(d("CAST(x * x AS DOUBLE)", "x"), "CAST(x + x AS DOUBLE)");
}

#[test]
fn cast_to_non_numeric_type_errors() {
    // Differentiating through CAST(... AS VARCHAR) would emit a nonsensical
    // CAST(1.0 AS VARCHAR); it must be a typed error instead.
    assert!(Ddx::new()
        .differentiate_sql("CAST(x AS VARCHAR)", "x", &GenericDialect {})
        .is_err());
}

#[test]
fn abs_derivative_is_portable_and_pins_the_kink_at_zero() {
    // d/du |u| = sign(u), emitted as a portable CASE (no engine-specific
    // signum/sign builtin) that pins abs'(0) = 0 on every engine (review #44).
    let out = d("abs(x)", "x");
    assert!(
        !out.to_lowercase().contains("signum"),
        "must not emit the non-portable signum builtin: {out}"
    );
    assert_eq!(
        out,
        "CASE WHEN x > 0.0 THEN 1.0 WHEN x < 0.0 THEN -1.0 ELSE 0.0 END"
    );
    // Chain rule still applies: d/dx |x*y| = sign(x*y) * y.
    let chained = d("abs(x * y)", "x");
    assert!(chained.contains("* y"), "chain rule missing: {chained}");
    assert!(!chained.to_lowercase().contains("signum"));
}

#[test]
fn jvp_seeds_a_tangent_on_one_input() {
    // jvp(x*y, x, dx) = product rule with tangent(x)=dx, tangent(y)=0 = dx*y
    let out = Ddx::new()
        .rewrite_sql("SELECT jvp(x * y, x, dx) AS t FROM g", &GenericDialect {})
        .unwrap();
    assert_eq!(out, "SELECT (dx * y) AS t FROM g");
}

#[test]
fn jvp_with_unit_seed_matches_grad() {
    // A one-hot tangent reproduces the partial derivative.
    let jvp = Ddx::new()
        .rewrite_sql("SELECT jvp(sin(x), x, 1.0) AS t FROM g", &GenericDialect {})
        .unwrap();
    let grad = Ddx::new()
        .rewrite_sql("SELECT grad(sin(x), x) AS t FROM g", &GenericDialect {})
        .unwrap();
    assert_eq!(jvp, grad);
}
