use clippy_utils::source::snippet;
use rustc_hir::def::Res;
use rustc_hir::{BinOpKind, Expr, ExprKind, LitKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, Symbol};

use std::collections::HashMap;

/// A Lisp-like representation of mathematical expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum LispExpr {
    Binary(BinOpKind, Box<LispExpr>, Box<LispExpr>),
    Fun(String, Vec<LispExpr>),
    Ident(u64),
    Lit(f64),
    Unary(UnOp, Box<LispExpr>),
}

impl LispExpr {
    /// Convert this expression to a Lisp S-expression string.
    pub fn to_lisp(&self, placeholder: &str) -> String {
        match self {
            LispExpr::Binary(op, lhs, rhs) => {
                let op_str = match op {
                    BinOpKind::Add => "+",
                    BinOpKind::Sub => "-",
                    BinOpKind::Mul => "*",
                    BinOpKind::Div => "/",
                    _ => unreachable!("unsupported binary operator"),
                };
                format!(
                    "({} {} {})",
                    op_str,
                    lhs.to_lisp(placeholder),
                    rhs.to_lisp(placeholder)
                )
            }
            LispExpr::Fun(name, args) => {
                let args_str = args
                    .iter()
                    .map(|a| a.to_lisp(placeholder))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("({} {})", herbie_name(name), args_str)
            }
            LispExpr::Ident(id) => format!("{}{}", placeholder, id),
            LispExpr::Lit(val) => {
                if val.fract() == 0.0 && val.abs() < 1e15 {
                    format!("{}", *val as i64)
                } else {
                    format!("{}", val)
                }
            }
            LispExpr::Unary(UnOp::Neg, inner) => {
                format!("(- {})", inner.to_lisp(placeholder))
            }
            LispExpr::Unary(_, _) => unreachable!("unsupported unary operator"),
        }
    }

    /// Calculate the depth of this expression tree.
    pub fn depth(&self) -> usize {
        match self {
            LispExpr::Binary(_, lhs, rhs) => 1 + lhs.depth().max(rhs.depth()),
            LispExpr::Fun(_, args) => 1 + args.iter().map(|a| a.depth()).max().unwrap_or(0),
            LispExpr::Ident(_) | LispExpr::Lit(_) => 1,
            LispExpr::Unary(_, inner) => 1 + inner.depth(),
        }
    }

    /// Convert a Rust HIR expression to a LispExpr.
    /// Returns None if the expression contains unsupported constructs.
    /// Returns (expr, num_identifiers, bindings) on success.
    pub fn from_expr<'tcx>(
        cx: &LateContext<'tcx>,
        expr: &'tcx Expr<'tcx>,
    ) -> Option<(LispExpr, u64, MatchBindings<'tcx>)> {
        let mut bindings = MatchBindings::new();
        let mut next_id = 0u64;
        let lisp = from_expr_inner(cx, expr, &mut bindings, &mut next_id)?;
        Some((lisp, next_id, bindings))
    }

    /// Try to match this expression against a pattern, returning bindings on success.
    pub fn match_expr<'tcx>(
        cx: &LateContext<'tcx>,
        expr: &'tcx Expr<'tcx>,
        pattern: &LispExpr,
    ) -> Option<MatchBindings<'tcx>> {
        let ty = cx.typeck_results().expr_ty(expr);
        if !is_f64(ty) {
            return None;
        }

        let mut bindings = MatchBindings::new();
        if match_expr_inner(cx, expr, pattern, &mut bindings) {
            Some(bindings)
        } else {
            None
        }
    }

    /// Convert this LispExpr back to Rust source code.
    pub fn to_rust<'tcx>(&self, cx: &LateContext<'tcx>, bindings: &MatchBindings<'tcx>) -> String {
        match self {
            LispExpr::Binary(op, lhs, rhs) => {
                let op_str = match op {
                    BinOpKind::Add => "+",
                    BinOpKind::Sub => "-",
                    BinOpKind::Mul => "*",
                    BinOpKind::Div => "/",
                    _ => unreachable!("unsupported binary operator"),
                };
                format!(
                    "({} {} {})",
                    lhs.to_rust(cx, bindings),
                    op_str,
                    rhs.to_rust(cx, bindings)
                )
            }
            LispExpr::Fun(name, args) => {
                if args.is_empty() {
                    panic!("function with no arguments");
                }
                let receiver = args[0].to_rust(cx, bindings);
                let rest_args = args[1..]
                    .iter()
                    .map(|a| a.to_rust(cx, bindings))
                    .collect::<Vec<_>>()
                    .join(", ");
                if rest_args.is_empty() {
                    format!("{}.{}()", receiver, rust_name(name))
                } else {
                    format!("{}.{}({})", receiver, rust_name(name), rest_args)
                }
            }
            LispExpr::Ident(id) => bindings
                .get(*id)
                .map_or_else(|| format!("herbie{}", id), |binding| binding.to_rust(cx)),
            LispExpr::Lit(val) => {
                if val.fract() == 0.0 && val.abs() < 1e15 {
                    format!("{}.0", *val as i64)
                } else {
                    format!("{}", val)
                }
            }
            LispExpr::Unary(UnOp::Neg, inner) => {
                format!("(-{})", inner.to_rust(cx, bindings))
            }
            LispExpr::Unary(_, _) => unreachable!("unsupported unary operator"),
        }
    }
}

fn is_f64(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Float(ty::FloatTy::F64))
}

/// Map Rust function names to Herbie names.
fn herbie_name(rust_name: &str) -> &str {
    match rust_name {
        "ln" => "log",
        "ln_1p" => "log1p",
        "exp_m1" => "expm1",
        "powf" => "expt",
        _ => rust_name,
    }
}

/// Map Herbie function names to Rust method names.
fn rust_name(herbie_name: &str) -> &str {
    match herbie_name {
        "log" => "ln",
        "log1p" => "ln_1p",
        "expm1" => "exp_m1",
        "expt" => "powf",
        _ => herbie_name,
    }
}

/// Supported mathematical functions with their arities.
fn function_arity(name: &str) -> Option<usize> {
    match name {
        "abs" | "acos" | "asin" | "atan" | "cos" | "cosh" | "exp" | "exp_m1" | "ln" | "ln_1p"
        | "sin" | "sinh" | "sqrt" | "tan" | "tanh" => Some(1),
        "atan2" | "hypot" | "powf" => Some(2),
        _ => None,
    }
}

/// Convert HIR expression to LispExpr (internal recursive helper).
fn from_expr_inner<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    bindings: &mut MatchBindings<'tcx>,
    next_id: &mut u64,
) -> Option<LispExpr> {
    let ty = cx.typeck_results().expr_ty(expr);
    if !is_f64(ty) {
        return None;
    }

    match &expr.kind {
        ExprKind::Binary(op, lhs, rhs) => {
            let bin_op = match op.node {
                BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Div => op.node,
                _ => return None,
            };
            let lhs = from_expr_inner(cx, lhs, bindings, next_id)?;
            let rhs = from_expr_inner(cx, rhs, bindings, next_id)?;
            Some(LispExpr::Binary(bin_op, Box::new(lhs), Box::new(rhs)))
        }

        ExprKind::Unary(UnOp::Neg, inner) => {
            let inner = from_expr_inner(cx, inner, bindings, next_id)?;
            Some(LispExpr::Unary(UnOp::Neg, Box::new(inner)))
        }

        ExprKind::MethodCall(path, receiver, args, _) => {
            let method_name = path.ident.name.as_str();
            let arity = function_arity(method_name)?;
            if args.len() + 1 != arity {
                return None;
            }

            let mut lisp_args = vec![from_expr_inner(cx, receiver, bindings, next_id)?];
            for arg in args.iter() {
                lisp_args.push(from_expr_inner(cx, arg, bindings, next_id)?);
            }
            Some(LispExpr::Fun(method_name.to_string(), lisp_args))
        }

        ExprKind::Lit(lit) => match lit.node {
            LitKind::Float(sym, _) => {
                let val: f64 = sym.as_str().parse().ok()?;
                Some(LispExpr::Lit(val))
            }
            LitKind::Int(n, _) => Some(LispExpr::Lit(n.get() as f64)),
            _ => None,
        },

        ExprKind::Path(qpath) => {
            let id = *next_id;
            *next_id += 1;
            bindings.insert(id, MatchBinding::Path(qpath.clone(), expr.span));
            Some(LispExpr::Ident(id))
        }

        ExprKind::Field(base, field) => {
            let id = *next_id;
            *next_id += 1;
            bindings.insert(id, MatchBinding::Field(base, *field, expr.span));
            Some(LispExpr::Ident(id))
        }

        _ => None,
    }
}

/// Match an expression against a pattern (internal recursive helper).
fn match_expr_inner<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    pattern: &LispExpr,
    bindings: &mut MatchBindings<'tcx>,
) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    if !is_f64(ty) {
        return false;
    }

    match (pattern, &expr.kind) {
        (LispExpr::Ident(id), _) => {
            // Check if this identifier is already bound
            if let Some(existing) = bindings.get(*id) {
                // Must match the same expression
                existing.matches_expr(cx, expr)
            } else {
                // Bind this expression to the identifier
                bindings.insert(*id, MatchBinding::from_expr(expr));
                true
            }
        }

        (LispExpr::Lit(val), ExprKind::Lit(lit)) => match lit.node {
            LitKind::Float(sym, _) => {
                if let Ok(lit_val) = sym.as_str().parse::<f64>() {
                    (lit_val - val).abs() < f64::EPSILON
                } else {
                    false
                }
            }
            LitKind::Int(n, _) => (n.get() as f64 - val).abs() < f64::EPSILON,
            _ => false,
        },

        (LispExpr::Binary(pat_op, pat_lhs, pat_rhs), ExprKind::Binary(expr_op, lhs, rhs)) => {
            *pat_op == expr_op.node
                && match_expr_inner(cx, lhs, pat_lhs, bindings)
                && match_expr_inner(cx, rhs, pat_rhs, bindings)
        }

        (LispExpr::Unary(pat_op, pat_inner), ExprKind::Unary(expr_op, inner)) => {
            *pat_op == *expr_op && match_expr_inner(cx, inner, pat_inner, bindings)
        }

        (LispExpr::Fun(pat_name, pat_args), ExprKind::MethodCall(path, receiver, args, _)) => {
            let method_name = path.ident.name.as_str();
            if method_name != pat_name {
                return false;
            }
            if pat_args.len() != args.len() + 1 {
                return false;
            }
            if !match_expr_inner(cx, receiver, &pat_args[0], bindings) {
                return false;
            }
            for (pat_arg, arg) in pat_args[1..].iter().zip(args.iter()) {
                if !match_expr_inner(cx, arg, pat_arg, bindings) {
                    return false;
                }
            }
            true
        }

        _ => false,
    }
}

/// Tracks variable bindings during pattern matching.
#[derive(Debug, Default)]
pub struct MatchBindings<'tcx> {
    bindings: HashMap<u64, MatchBinding<'tcx>>,
}

impl<'tcx> MatchBindings<'tcx> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: u64, binding: MatchBinding<'tcx>) {
        self.bindings.insert(id, binding);
    }

    pub fn get(&self, id: u64) -> Option<&MatchBinding<'tcx>> {
        self.bindings.get(&id)
    }
}

/// A binding from a pattern variable to a matched expression.
#[derive(Debug, Clone)]
pub enum MatchBinding<'tcx> {
    Path(QPath<'tcx>, Span),
    Field(&'tcx Expr<'tcx>, Symbol, Span),
    Lit(f64, Span),
    Other(Span),
}

impl<'tcx> MatchBinding<'tcx> {
    fn from_expr(expr: &'tcx Expr<'tcx>) -> Self {
        match &expr.kind {
            ExprKind::Path(qpath) => MatchBinding::Path(qpath.clone(), expr.span),
            ExprKind::Field(base, field) => MatchBinding::Field(base, *field, expr.span),
            ExprKind::Lit(lit) => match lit.node {
                LitKind::Float(sym, _) => {
                    if let Ok(val) = sym.as_str().parse::<f64>() {
                        MatchBinding::Lit(val, expr.span)
                    } else {
                        MatchBinding::Other(expr.span)
                    }
                }
                LitKind::Int(n, _) => MatchBinding::Lit(n.get() as f64, expr.span),
                _ => MatchBinding::Other(expr.span),
            },
            _ => MatchBinding::Other(expr.span),
        }
    }

    fn matches_expr(&self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> bool {
        // Compare by getting the source snippets
        let self_snippet = self.to_rust(cx);
        let expr_snippet = snippet(cx, expr.span, "").to_string();
        self_snippet == expr_snippet
    }

    pub fn to_rust(&self, cx: &LateContext<'tcx>) -> String {
        match self {
            MatchBinding::Path(_, span)
            | MatchBinding::Field(_, _, span)
            | MatchBinding::Other(span) => snippet(cx, *span, "").to_string(),
            MatchBinding::Lit(val, _) => {
                if val.fract() == 0.0 && val.abs() < 1e15 {
                    format!("{}.0", *val as i64)
                } else {
                    format!("{}", val)
                }
            }
        }
    }
}

/// Parser for Lisp S-expressions from the Herbie database.
pub struct Parser {
    pos: usize,
    input: Vec<char>,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            pos: 0,
            input: Vec::new(),
        }
    }

    pub fn parse(&mut self, input: &str) -> Result<LispExpr, ParseError> {
        self.pos = 0;
        self.input = input.chars().collect();
        self.skip_whitespace();
        let result = self.parse_expr()?;
        self.skip_whitespace();
        if self.pos < self.input.len() {
            return Err(ParseError::TrailingInput);
        }
        Ok(result)
    }

    fn parse_expr(&mut self) -> Result<LispExpr, ParseError> {
        self.skip_whitespace();
        if self.peek() == Some('(') {
            self.parse_list()
        } else {
            self.parse_atom()
        }
    }

    fn parse_list(&mut self) -> Result<LispExpr, ParseError> {
        self.expect('(')?;
        self.skip_whitespace();

        let op = self.parse_symbol()?;
        self.skip_whitespace();

        match op.as_str() {
            "+" | "-" | "*" | "/" => {
                let args = self.parse_args()?;
                self.expect(')')?;

                if op == "-" && args.len() == 1 {
                    // Unary negation
                    return Ok(LispExpr::Unary(
                        UnOp::Neg,
                        Box::new(args.into_iter().next().unwrap()),
                    ));
                }

                if args.len() != 2 {
                    return Err(ParseError::WrongArity);
                }
                let mut iter = args.into_iter();
                let lhs = iter.next().unwrap();
                let rhs = iter.next().unwrap();
                let bin_op = match op.as_str() {
                    "+" => BinOpKind::Add,
                    "-" => BinOpKind::Sub,
                    "*" => BinOpKind::Mul,
                    "/" => BinOpKind::Div,
                    _ => unreachable!(),
                };
                Ok(LispExpr::Binary(bin_op, Box::new(lhs), Box::new(rhs)))
            }
            "lambda" => {
                // Skip the parameter list and parse the body
                self.skip_whitespace();
                self.expect('(')?;
                while self.peek() != Some(')') {
                    self.parse_symbol()?;
                    self.skip_whitespace();
                }
                self.expect(')')?;
                self.skip_whitespace();
                let body = self.parse_expr()?;
                self.expect(')')?;
                Ok(body)
            }
            _ => {
                // Function call
                let rust_name = rust_name(&op);
                let args = self.parse_args()?;
                self.expect(')')?;
                Ok(LispExpr::Fun(rust_name.to_string(), args))
            }
        }
    }

    fn parse_args(&mut self) -> Result<Vec<LispExpr>, ParseError> {
        let mut args = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() == Some(')') {
                break;
            }
            args.push(self.parse_expr()?);
        }
        Ok(args)
    }

    fn parse_atom(&mut self) -> Result<LispExpr, ParseError> {
        let sym = self.parse_symbol()?;

        // Try to parse as a number
        if let Ok(val) = sym.parse::<f64>() {
            return Ok(LispExpr::Lit(val));
        }

        // Try to parse as a herbie identifier (e.g., "herbie0")
        if let Some(rest) = sym.strip_prefix("herbie") {
            if let Ok(id) = rest.parse::<u64>() {
                return Ok(LispExpr::Ident(id));
            }
        }

        Err(ParseError::UnknownAtom(sym))
    }

    fn parse_symbol(&mut self) -> Result<String, ParseError> {
        let mut sym = String::new();
        while let Some(c) = self.peek() {
            if c.is_whitespace() || c == '(' || c == ')' {
                break;
            }
            sym.push(c);
            self.pos += 1;
        }
        if sym.is_empty() {
            Err(ParseError::UnexpectedEof)
        } else {
            Ok(sym)
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn expect(&mut self, expected: char) -> Result<(), ParseError> {
        match self.peek() {
            Some(c) if c == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(c) => Err(ParseError::UnexpectedChar(c, expected)),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
}

#[derive(Debug)]
pub enum ParseError {
    UnexpectedEof,
    UnexpectedChar(char, char),
    TrailingInput,
    WrongArity,
    UnknownAtom(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedEof => write!(f, "unexpected end of input"),
            ParseError::UnexpectedChar(got, expected) => {
                write!(f, "expected '{}', got '{}'", expected, got)
            }
            ParseError::TrailingInput => write!(f, "trailing input after expression"),
            ParseError::WrongArity => write!(f, "wrong number of arguments"),
            ParseError::UnknownAtom(s) => write!(f, "unknown atom: {}", s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_literal() {
        let mut parser = Parser::new();
        assert_eq!(parser.parse("0.").unwrap(), LispExpr::Lit(0.0));
        assert_eq!(parser.parse("42.5").unwrap(), LispExpr::Lit(42.5));
    }

    #[test]
    fn test_parse_ident() {
        let mut parser = Parser::new();
        assert_eq!(parser.parse("herbie0").unwrap(), LispExpr::Ident(0));
        assert_eq!(parser.parse("herbie42").unwrap(), LispExpr::Ident(42));
    }

    #[test]
    fn test_parse_binary() {
        let mut parser = Parser::new();
        let result = parser.parse("(+ herbie0 herbie1)").unwrap();
        assert_eq!(
            result,
            LispExpr::Binary(
                BinOpKind::Add,
                Box::new(LispExpr::Ident(0)),
                Box::new(LispExpr::Ident(1))
            )
        );
    }

    #[test]
    fn test_parse_function() {
        let mut parser = Parser::new();
        let result = parser.parse("(cos herbie0)").unwrap();
        assert_eq!(
            result,
            LispExpr::Fun("cos".into(), vec![LispExpr::Ident(0)])
        );
    }

    #[test]
    fn test_parse_unary_neg() {
        let mut parser = Parser::new();
        let result = parser.parse("(- herbie0)").unwrap();
        assert_eq!(
            result,
            LispExpr::Unary(UnOp::Neg, Box::new(LispExpr::Ident(0)))
        );
    }

    #[test]
    fn test_parse_lambda() {
        let mut parser = Parser::new();
        let result = parser
            .parse("(lambda (herbie0 herbie1) (+ herbie0 herbie1))")
            .unwrap();
        assert_eq!(
            result,
            LispExpr::Binary(
                BinOpKind::Add,
                Box::new(LispExpr::Ident(0)),
                Box::new(LispExpr::Ident(1))
            )
        );
    }

    #[test]
    fn test_roundtrip() {
        let cases = [
            "(+ herbie0 herbie1)",
            "(- herbie0)",
            "(cos 1)",
            "(log1p (cos herbie0))",
            "(* (+ (/ herbie0 herbie1) herbie2) herbie1)",
        ];
        for case in cases {
            let mut parser = Parser::new();
            let expr = parser.parse(case).unwrap();
            assert_eq!(expr.to_lisp("herbie"), case);
        }
    }

    #[test]
    fn test_depth() {
        let mut parser = Parser::new();
        assert_eq!(parser.parse("herbie0").unwrap().depth(), 1);
        assert_eq!(parser.parse("(+ herbie0 herbie1)").unwrap().depth(), 2);
        assert_eq!(
            parser
                .parse("(+ (+ herbie0 herbie1) herbie2)")
                .unwrap()
                .depth(),
            3
        );
    }
}
