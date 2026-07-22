//! Source-to-source SQL rewriting: find every `grad`/`jvp` marker, replace it
//! with derivative SQL, leave everything else byte-identical.
//!
//! This is Path A (design.md §3.3), the universal path every target relies on.
//! It is a real subsystem, not a one-liner (design.md §3.2):
//!
//! * a **parse-free pre-gate** so a marker-free statement is never parsed, and
//!   so a `sqlparser` coverage gap can only ever bound a statement that
//!   *actually contains* a marker (F5);
//! * **splice by source span**, so everything outside a marker stays
//!   byte-identical — which requires a UTF-8-aware character-column→byte-offset
//!   conversion, because `sqlparser` spans are 1-based *characters*, not bytes
//!   (G3);
//! * **multiple and nested markers** — spliced in reverse source order, nested
//!   ones differentiated bottom-up (`grad(grad(f,x),x)` just works);
//! * a safe **fallback** to whole-statement reprinting on the empty spans the
//!   API documents as possible.
//!
//! Two guards run here, both catching a *silently-wrong* derivative and turning
//! it into a typed error: the ambiguity guard lives in the engine (F2), and the
//! CTE-computed-alias guard (F3/G4) lives in [`projection_guard`].

use std::collections::HashSet;
use std::ops::ControlFlow;

use sqlparser::ast::Spanned;
use sqlparser::ast::{
    Expr, Function, ObjectNamePart, Query, Select, SelectItem, SetExpr, Statement, TableFactor,
    Visit, VisitMut, Visitor, VisitorMut,
};
use sqlparser::dialect::Dialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Location, Span};

use crate::colref::{ColRef, IdentCasing};
use crate::engine::{differentiate, jvp, positional_args, RuleRegistry};
use crate::error::{DiffError, Result};

/// Which marker a function call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerKind {
    Grad,
    Jvp,
}

/// Classify a function call as a marker — but only an **unqualified**,
/// case-folded `grad`/`jvp` (design.md §3.2, F8). `myschema.grad(...)` and a
/// user's own multi-part function are left alone.
fn marker_kind(f: &Function) -> Option<MarkerKind> {
    if f.name.0.len() != 1 {
        return None;
    }
    let ObjectNamePart::Identifier(id) = &f.name.0[0] else {
        return None;
    };
    match id.value.to_ascii_lowercase().as_str() {
        "grad" => Some(MarkerKind::Grad),
        "jvp" => Some(MarkerKind::Jvp),
        _ => None,
    }
}

fn is_marker_expr(e: &Expr) -> bool {
    matches!(e, Expr::Function(f) if marker_kind(f).is_some())
}

/// The parse-free pre-gate: a case-insensitive scan for an *unqualified*
/// `grad(`/`jvp(` — the equivalent of `(?i)(?:^|[^A-Za-z0-9_.])(grad|jvp)\s*\(`,
/// hand-rolled so the core depends on `sqlparser` only (design.md §3.2/§6). A
/// statement that doesn't hit is returned verbatim, never parsed (F5). It is a
/// best-effort filter: a false positive (e.g. `grad(` inside a string literal)
/// only costs a parse that then finds no marker, never a wrong rewrite.
fn pre_gate_hit(sql: &str) -> bool {
    // ASCII-lowercasing preserves byte length and offsets, so indices found in
    // `lower` are valid char boundaries in `sql`.
    let lower = sql.to_ascii_lowercase();
    for kw in ["grad", "jvp"] {
        let mut from = 0;
        while let Some(rel) = lower[from..].find(kw) {
            let idx = from + rel;
            from = idx + 1;

            // Preceding character must not be part of a longer identifier or a
            // qualifier (`.`), so `mygrad(` and `schema.grad(` don't match.
            let ok_prev = idx == 0
                || sql[..idx].chars().next_back().is_some_and(|prev| {
                    !(prev.is_ascii_alphanumeric() || prev == '_' || prev == '.')
                });
            if !ok_prev {
                continue;
            }

            // The next non-whitespace character must be `(`.
            if sql[idx + kw.len()..]
                .chars()
                .find(|c| !c.is_whitespace())
                == Some('(')
            {
                return true;
            }
        }
    }
    false
}

/// The public entry point behind [`crate::Ddx::rewrite_sql`].
pub(crate) fn rewrite_sql(
    sql: &str,
    dialect: &dyn Dialect,
    casing: IdentCasing,
    reg: &RuleRegistry,
) -> Result<String> {
    // 1. Parse-free pre-gate: no marker syntax, no parse, byte-identical out.
    if !pre_gate_hit(sql) {
        return Ok(sql.to_string());
    }

    // 2. The statement (or one of them) looks like it carries a marker; parse.
    let statements = Parser::parse_sql(dialect, sql)
        .map_err(|e| DiffError::Parse(format!("failed to parse SQL: {e}")))?;

    // 3. Statement-level context for the projection-boundary guard (F3/G4):
    //    the names of every *computed* select-list alias of a CTE/derived table.
    let mut aliases = HashSet::new();
    for stmt in &statements {
        collect_computed_aliases(stmt, &mut aliases);
    }

    // 4. Locate the outermost markers (with their source spans). Nested markers
    //    are handled when their enclosing outermost marker is differentiated.
    let mut collector = MarkerCollector::default();
    for stmt in &statements {
        let _ = Visit::visit(stmt, &mut collector);
    }
    // Pre-gate false positive (e.g. only qualified markers, or `grad(` inside a
    // string literal): nothing to rewrite, return verbatim.
    if collector.found.is_empty() {
        return Ok(sql.to_string());
    }

    // 5. Empty spans are documented as possible; fall back to a correct (if not
    //    byte-identical) whole-statement reprint if any marker lacks a span.
    if collector.found.iter().any(|(span, _)| is_empty_span(span)) {
        return reprint_fallback(statements, casing, reg, &aliases);
    }

    // 6. Compute each replacement, then splice by byte range in reverse source
    //    order so earlier offsets stay valid.
    let mut repls: Vec<(usize, usize, String)> = Vec::with_capacity(collector.found.len());
    for (span, marker_expr) in &collector.found {
        let text = differentiate_marker_tree(marker_expr, casing, reg, &aliases)?;
        let start = locate(sql, span.start, false)
            .ok_or_else(|| DiffError::Internal("marker span start out of range".into()))?;
        // sqlparser span ends are *inclusive* of the last character (verified
        // against 0.62.0), so the exclusive byte end is one character past it.
        let end = locate(sql, span.end, true)
            .ok_or_else(|| DiffError::Internal("marker span end out of range".into()))?;
        repls.push((start, end, text));
    }
    repls.sort_by(|a, b| b.0.cmp(&a.0));

    let mut out = sql.to_string();
    for (start, end, text) in repls {
        out.replace_range(start..end, &text);
    }
    Ok(out)
}

/// Differentiate one (possibly nested) marker subtree, returning the derivative
/// rendered to SQL text, parenthesized so it keeps the call's precedence.
fn differentiate_marker_tree(
    marker_expr: &Expr,
    casing: IdentCasing,
    reg: &RuleRegistry,
    aliases: &HashSet<String>,
) -> Result<String> {
    let mut clone = marker_expr.clone();
    let mut rw = MarkerRewriter {
        casing,
        reg,
        aliases,
    };
    if let ControlFlow::Break(err) = VisitMut::visit(&mut clone, &mut rw) {
        return Err(err);
    }
    Ok(clone.to_string())
}

/// The whole-statement reprint fallback (empty-span case).
fn reprint_fallback(
    mut statements: Vec<Statement>,
    casing: IdentCasing,
    reg: &RuleRegistry,
    aliases: &HashSet<String>,
) -> Result<String> {
    for stmt in &mut statements {
        let mut rw = MarkerRewriter {
            casing,
            reg,
            aliases,
        };
        if let ControlFlow::Break(err) = VisitMut::visit(stmt, &mut rw) {
            return Err(err);
        }
    }
    Ok(statements
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; "))
}

// ---------------------------------------------------------------------------
// Differentiating a single marker (args assumed already marker-free)
// ---------------------------------------------------------------------------

/// Differentiate a single marker call whose arguments are already free of
/// nested markers (guaranteed by the bottom-up post-order walk).
fn differentiate_marker(f: &Function, casing: IdentCasing, reg: &RuleRegistry) -> Result<Expr> {
    let kind = marker_kind(f).ok_or_else(|| DiffError::Internal("not a marker".into()))?;
    let args = positional_args(f).ok_or_else(|| {
        DiffError::InvalidMarker("marker call has non-positional arguments".into())
    })?;
    match kind {
        MarkerKind::Grad => {
            if args.len() != 2 {
                return Err(DiffError::InvalidMarker(format!(
                    "grad(expr, column) expects 2 arguments, got {}",
                    args.len()
                )));
            }
            let wrt = ColRef::from_wrt_arg("grad", args[1])?;
            differentiate(args[0], &wrt, casing, reg)
        }
        MarkerKind::Jvp => {
            if args.len() != 3 {
                return Err(DiffError::InvalidMarker(format!(
                    "jvp(expr, column, tangent) expects 3 arguments, got {}",
                    args.len()
                )));
            }
            let wrt = ColRef::from_wrt_arg("jvp", args[1])?;
            let seeds = vec![(wrt, args[2].clone())];
            jvp(args[0], &seeds, casing, reg)
        }
    }
}

/// The projection-boundary guard (design.md §3.5, F3/G4).
///
/// Errors if a marker argument references an identifier that is a *computed*
/// select-list alias of a CTE/derived table in the same statement, used as a
/// *non-`wrt`* term — differentiating it would silently treat an upstream
/// expression as a constant and drop gradient terms. The carve-out (G4): when
/// the alias *is* the `wrt` itself, every occurrence is the differentiation
/// leaf, so no term can be dropped and the guard stays quiet.
fn projection_guard(f: &Function, aliases: &HashSet<String>) -> Result<()> {
    if aliases.is_empty() {
        return Ok(());
    }
    let Some(args) = positional_args(f) else {
        return Ok(());
    };
    let Some(expr_arg) = args.first() else {
        return Ok(());
    };
    let wrt_name = args
        .get(1)
        .and_then(|a| ColRef::from_expr(a))
        .map(|c| c.name.value.to_ascii_lowercase());

    let mut cols = ColumnCollector::default();
    let _ = Visit::visit(*expr_arg, &mut cols);
    for c in cols.cols {
        let lname = c.name.value.to_ascii_lowercase();
        // Carve-out: the wrt itself is always a leaf; never an error.
        if Some(&lname) == wrt_name.as_ref() {
            continue;
        }
        if aliases.contains(&lname) {
            return Err(DiffError::ProjectionBoundary(format!(
                "`{}` is a computed select-list alias of a CTE/derived table used \
                 as a non-differentiation term; grad does not see through the \
                 projection boundary — differentiate inside that CTE instead",
                c.display()
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Visitors
// ---------------------------------------------------------------------------

/// Collects the outermost marker expressions (with their spans), skipping
/// markers nested inside another marker's arguments (handled bottom-up later).
#[derive(Default)]
struct MarkerCollector {
    depth: usize,
    found: Vec<(Span, Expr)>,
}

impl Visitor for MarkerCollector {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if is_marker_expr(expr) {
            if self.depth == 0 {
                self.found.push((expr.span(), expr.clone()));
            }
            self.depth += 1;
        }
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if is_marker_expr(expr) {
            self.depth -= 1;
        }
        ControlFlow::Continue(())
    }
}

/// Replaces each marker with `Nested(derivative)`, bottom-up (post-order), so a
/// nested marker's own arguments are already marker-free when it is reached.
struct MarkerRewriter<'a> {
    casing: IdentCasing,
    reg: &'a RuleRegistry,
    aliases: &'a HashSet<String>,
}

impl VisitorMut for MarkerRewriter<'_> {
    type Break = DiffError;

    fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<DiffError> {
        let replacement = match expr {
            Expr::Function(f) if marker_kind(f).is_some() => {
                if let Err(err) = projection_guard(f, self.aliases) {
                    return ControlFlow::Break(err);
                }
                match differentiate_marker(f, self.casing, self.reg) {
                    Ok(d) => Some(d),
                    Err(err) => return ControlFlow::Break(err),
                }
            }
            _ => None,
        };
        if let Some(d) = replacement {
            *expr = Expr::Nested(Box::new(d));
        }
        ControlFlow::Continue(())
    }
}

/// Collects the column references directly appearing in an expression tree.
#[derive(Default)]
struct ColumnCollector {
    cols: Vec<ColRef>,
}

impl Visitor for ColumnCollector {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        match expr {
            Expr::Identifier(_) | Expr::CompoundIdentifier(_) => {
                if let Some(cr) = ColRef::from_expr(expr) {
                    self.cols.push(cr);
                }
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// Computed-alias collection for the projection-boundary guard
// ---------------------------------------------------------------------------

fn collect_computed_aliases(stmt: &Statement, out: &mut HashSet<String>) {
    match stmt {
        Statement::Query(q) => walk_query(q, out),
        Statement::Insert(insert) => {
            if let Some(source) = &insert.source {
                walk_query(source, out);
            }
        }
        _ => {}
    }
}

fn walk_query(q: &Query, out: &mut HashSet<String>) {
    if let Some(with) = &q.with {
        for cte in &with.cte_tables {
            walk_query(&cte.query, out);
        }
    }
    walk_set_expr(&q.body, out);
}

fn walk_set_expr(body: &SetExpr, out: &mut HashSet<String>) {
    match body {
        SetExpr::Select(select) => walk_select(select, out),
        SetExpr::Query(q) => walk_query(q, out),
        SetExpr::SetOperation { left, right, .. } => {
            walk_set_expr(left, out);
            walk_set_expr(right, out);
        }
        _ => {}
    }
}

fn walk_select(select: &Select, out: &mut HashSet<String>) {
    for item in &select.projection {
        if let SelectItem::ExprWithAlias { expr, alias } = item {
            if !is_bare_column(expr) {
                out.insert(alias.value.to_ascii_lowercase());
            }
        }
    }
    // Recurse into derived tables (FROM subqueries), which are projection
    // boundaries too.
    for twj in &select.from {
        walk_table_factor(&twj.relation, out);
        for join in &twj.joins {
            walk_table_factor(&join.relation, out);
        }
    }
}

fn walk_table_factor(tf: &TableFactor, out: &mut HashSet<String>) {
    if let TableFactor::Derived { subquery, .. } = tf {
        walk_query(subquery, out);
    }
}

fn is_bare_column(e: &Expr) -> bool {
    matches!(e, Expr::Identifier(_) | Expr::CompoundIdentifier(_))
        || matches!(e, Expr::Nested(inner) if is_bare_column(inner))
}

// ---------------------------------------------------------------------------
// Span (1-based character line/column) → byte offset conversion (G3)
// ---------------------------------------------------------------------------

/// `sqlparser` uses `line: 0`/`column: 0` for an empty/unknown location.
fn is_empty_span(span: &Span) -> bool {
    span.start.line == 0 || span.start.column == 0 || span.end.line == 0 || span.end.column == 0
}

/// Convert a 1-based (line, character-column) [`Location`] to a byte offset in
/// `sql`. Character-column, not byte-column: a multibyte character before the
/// target shifts the byte offset past the column number (G3).
///
/// With `past = false` the returned offset is the *start* byte of the character
/// at `loc`; with `past = true` it is the byte *one past* that character — used
/// for an inclusive span end, so the whole marker (its closing `)` included) is
/// covered by `start..end`.
fn locate(sql: &str, loc: Location, past: bool) -> Option<usize> {
    let mut line: u64 = 1;
    let mut col: u64 = 1;
    for (byte_idx, ch) in sql.char_indices() {
        if line == loc.line && col == loc.column {
            return Some(if past {
                byte_idx + ch.len_utf8()
            } else {
                byte_idx
            });
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    // A location just past the final character maps to the end of the string.
    if line == loc.line && col == loc.column {
        return Some(sql.len());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_gate_matches_unqualified_markers() {
        assert!(pre_gate_hit("SELECT grad(x, x) FROM t"));
        assert!(pre_gate_hit("SELECT jvp(x, x, dx) FROM t"));
        assert!(pre_gate_hit("grad(x,x)")); // at start of input
        assert!(pre_gate_hit("SELECT GRAD (x, x) FROM t")); // case + whitespace
        assert!(pre_gate_hit("SELECT AVG(grad(x, x)) FROM t")); // after `(`
    }

    #[test]
    fn pre_gate_rejects_non_markers() {
        assert!(!pre_gate_hit("SELECT a + b FROM t")); // no marker
        assert!(!pre_gate_hit("SELECT mygrad(x) FROM t")); // longer identifier
        assert!(!pre_gate_hit("SELECT schema.grad(x, x) FROM t")); // qualified
        assert!(!pre_gate_hit("SELECT grad AS g FROM t")); // no open paren
        assert!(!pre_gate_hit("SELECT upgrade(x) FROM t")); // 'grad' inside a word
    }
}
