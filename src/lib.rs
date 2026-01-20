#![feature(rustc_private)]
#![feature(once_cell_try)]
#![warn(unused_extern_crates)]

extern crate rustc_ast;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_span;

mod conf;
mod lisp;

use clippy_utils::diagnostics::span_lint_and_sugg;
use rusqlite as sql;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_span::Symbol;

use std::borrow::Cow;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;
use wait_timeout::ChildExt;

use conf::{Conf, ConfError, UseHerbieConf};
use lisp::{LispExpr, MatchBindings, Parser};

/// The embedded Herbie database.
const EMBEDDED_DB: &[u8] = include_bytes!("../db/Herbie.db");

/// Returns the path to the embedded database, extracting it to a temp file if needed.
fn embedded_db_path() -> Result<PathBuf, std::io::Error> {
    static DB_PATH: OnceLock<PathBuf> = OnceLock::new();

    DB_PATH
        .get_or_try_init(|| {
            let path = std::env::temp_dir().join("herbie-lint-embedded.db");
            // Always write - the file might be stale from a previous version
            std::fs::write(&path, EMBEDDED_DB)?;
            Ok(path)
        })
        .cloned()
}

dylint_linting::impl_late_lint! {
    /// Detects numerically unstable floating-point expressions and suggests
    /// more stable alternatives based on the Herbie database.
    pub HERBIE,
    Warn,
    "checks for numerical instability in floating-point expressions",
    Herbie::new()
}

#[derive(Default)]
pub struct Herbie {
    conf: Option<Conf>,
    initialised: bool,
    subs: Vec<(LispExpr, LispExpr)>,
}

impl Herbie {
    pub fn new() -> Self {
        Self::default()
    }

    fn init(&mut self) -> Result<(), InitError> {
        if self.initialised {
            return Ok(());
        }
        self.initialised = true;

        let conf = conf::read_conf()?;

        // Use user-specified db_path if provided, otherwise use embedded database
        let db_path: Cow<'_, str> = match &conf.db_path {
            Some(path) => Cow::Borrowed(path.as_str()),
            None => Cow::Owned(
                embedded_db_path()
                    .map_err(InitError::Io)?
                    .to_string_lossy()
                    .into_owned(),
            ),
        };

        let connection = sql::Connection::open_with_flags(
            db_path.as_ref(),
            sql::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;

        let mut query = connection.prepare("SELECT * FROM HerbieResults")?;
        let mut parser = Parser::new();

        self.subs = query
            .query_map([], |row| {
                let cmdin: String = row.get(1)?;
                let cmdout: String = row.get(2)?;
                let errin: f64 = row.get(4).unwrap_or(0.0);
                let errout: f64 = row.get(5).unwrap_or(0.0);
                Ok((cmdin, cmdout, errin, errout))
            })?
            .filter_map(|row| {
                let (cmdin, cmdout, errin, errout) = row.ok()?;

                // Skip if no improvement
                if cmdin == cmdout || errin <= errout {
                    return None;
                }

                // Skip entries with conditional output (if statements)
                // as these aren't directly translatable to simple Rust
                if cmdout.contains("if") {
                    return None;
                }

                let cmdin = parser.parse(&cmdin).ok()?;
                let cmdout = parser.parse(&cmdout).ok()?;
                Some((cmdin, cmdout))
            })
            .collect();

        self.conf = Some(conf);
        Ok(())
    }
}

impl<'tcx> LateLintPass<'tcx> for Herbie {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        // Check for #[herbie_ignore] on parent item
        if has_herbie_ignore_attr(cx, expr) {
            return;
        }

        // Only check f64 expressions
        let ty = cx.typeck_results().expr_ty(expr);
        if !is_f64(ty) {
            return;
        }

        // Debug: log all f64 expressions containing "scale" or "shift"
        if std::env::var("HERBIE_LINT_DEBUG_ALL").is_ok() {
            let snippet = clippy_utils::source::snippet(cx, expr.span, "<expr>");
            if snippet.contains("scale") || snippet.contains("shift") || snippet.contains("bounds")
            {
                eprintln!("[herbie-lint] f64 expr: {}", snippet);
            }
        }

        // Initialise database connection
        if let Err(err) = self.init() {
            // Report initialisation error once
            cx.tcx
                .dcx()
                .warn(format!("Could not initialise Herbie-Lint: {}", err));
            return;
        }

        // Try to match against known patterns
        let mut got_match = false;
        for (cmdin, cmdout) in &self.subs {
            if let Some(bindings) = LispExpr::match_expr(cx, expr, cmdin) {
                if std::env::var("HERBIE_LINT_DEBUG").is_ok() {
                    let snippet = clippy_utils::source::snippet(cx, expr.span, "<expr>");
                    eprintln!(
                        "[herbie-lint] Database match: {} -> {}",
                        snippet,
                        cmdout.to_lisp("x")
                    );
                }
                report(cx, expr, cmdout, &bindings);
                got_match = true;
            }
        }

        // Optionally try calling Herbie for unknown expressions
        let conf = self
            .conf
            .as_ref()
            .expect("Configuration should be read by now");
        if !got_match && conf.use_herbie != UseHerbieConf::No {
            if std::env::var("HERBIE_LINT_DEBUG_ALL").is_ok() {
                let snippet = clippy_utils::source::snippet(cx, expr.span, "<expr>");
                if snippet.contains("scale") || snippet.contains("shift") {
                    eprintln!("[herbie-lint] Calling try_with_herbie for: {}", snippet);
                }
            }
            if let Err(err) = try_with_herbie(cx, expr, conf) {
                cx.tcx.dcx().span_warn(expr.span, err);
            }
        }
    }
}

fn is_f64(ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Float(ty::FloatTy::F64) => true,
        ty::Float(ty::FloatTy::F16 | ty::FloatTy::F32 | ty::FloatTy::F128) => false,
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Adt(_, _)
        | ty::Foreign(_)
        | ty::Str
        | ty::Array(_, _)
        | ty::Pat(_, _)
        | ty::Slice(_)
        | ty::RawPtr(_, _)
        | ty::Ref(_, _, _)
        | ty::FnDef(_, _)
        | ty::FnPtr(_, _)
        | ty::Dynamic(_, _, _)
        | ty::Closure(_, _)
        | ty::CoroutineClosure(_, _)
        | ty::Coroutine(_, _)
        | ty::CoroutineWitness(_, _)
        | ty::Never
        | ty::Tuple(_)
        | ty::Alias(_, _)
        | ty::Param(_)
        | ty::Bound(_, _)
        | ty::Placeholder(_)
        | ty::Infer(_)
        | ty::Error(_)
        | ty::UnsafeBinder(_) => false,
    }
}

fn has_herbie_ignore_attr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let parent_id = cx.tcx.parent_hir_id(expr.hir_id);
    let attrs = cx.tcx.hir_attrs(parent_id);
    let herbie_ignore = Symbol::intern("herbie_ignore");
    attrs.iter().any(|attr| attr.has_name(herbie_ignore))
}

fn report<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    cmdout: &LispExpr,
    bindings: &MatchBindings<'tcx>,
) {
    let suggestion = cmdout.to_rust(cx, bindings);
    span_lint_and_sugg(
        cx,
        HERBIE,
        expr.span,
        "numerically unstable expression",
        "try this",
        suggestion,
        Applicability::MachineApplicable,
    );
}

fn try_with_herbie<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    conf: &Conf,
) -> Result<(), Cow<'static, str>> {
    // Debug: entering try_with_herbie
    let debug_snippet = if std::env::var("HERBIE_LINT_DEBUG").is_ok() {
        Some(clippy_utils::source::snippet(cx, expr.span, "<expr>").to_string())
    } else {
        None
    };

    // Convert expression to Lisp
    let (lisp_expr, nb_ids, bindings) = match LispExpr::from_expr(cx, expr) {
        Some(r) => {
            if let Some(ref s) = debug_snippet
                && s.contains("scale")
                && s.contains("1.")
            {
                eprintln!(
                    "[herbie-lint] from_expr succeeded for: {} (depth={})",
                    s,
                    r.0.depth()
                );
            }
            r
        }
        None => {
            // Debug: log expressions that couldn't be converted
            if let Some(ref s) = debug_snippet {
                eprintln!("[herbie-lint] Skipping (unsupported construct): {}", s);
            }
            return Ok(()); // Expression contains unsupported constructs
        }
    };

    // Skip trivial expressions
    if lisp_expr.depth() <= 2 {
        if std::env::var("HERBIE_LINT_DEBUG").is_ok() {
            let snippet = clippy_utils::source::snippet(cx, expr.span, "<expr>");
            eprintln!(
                "[herbie-lint] Skipping (too simple, depth={}): {}",
                lisp_expr.depth(),
                snippet
            );
        }
        return Ok(());
    }

    // Spawn Herbie shell process (Herbie 2.x)
    let herbie_cmd = conf.herbie_path.as_deref().unwrap_or("herbie");
    let mut command = Command::new(herbie_cmd);
    command
        .arg("shell")
        .arg("--seed")
        .arg(conf.herbie_seed.as_ref())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            return if conf.use_herbie == UseHerbieConf::Yes {
                Err(format!("Could not call Herbie: {}", err).into())
            } else {
                Ok(())
            };
        }
    };

    // Send expression to Herbie in FPCore format
    let params = (0..nb_ids)
        .map(|id| format!("herbie{}", id))
        .collect::<Vec<_>>()
        .join(" ");
    let cmdin = lisp_expr.to_lisp("herbie");
    let fpcore_input = format!("(FPCore ({}) {})\n", params, cmdin);

    // Debug: log FPCore being sent
    if std::env::var("HERBIE_LINT_DEBUG").is_ok() {
        eprintln!("[herbie-lint] Sending to Herbie: {}", fpcore_input.trim());
    }

    child
        .stdin
        .as_mut()
        .expect("stdin captured")
        .write_all(fpcore_input.as_bytes())
        .expect("write to stdin");

    // Close stdin to signal end of input
    drop(child.stdin.take());

    // Wait with timeout
    let status = match conf.timeout {
        Some(timeout) => match child.wait_timeout(Duration::from_secs(timeout as u64)) {
            Ok(Some(status)) => status,
            Ok(None) => {
                // Timeout - kill the child process
                let _ = child.kill();
                return Ok(());
            }
            Err(err) => return Err(format!("herbie shell error: {}", err).into()),
        },
        None => child
            .wait()
            .map_err(|e| format!("herbie shell error: {}", e))?,
    };

    if !status.success() {
        return Err(format!("herbie shell exited with: {}", status).into());
    }

    // Parse FPCore output
    let mut stdout = child.stdout.ok_or("cannot capture stdout")?;
    let mut output = String::new();
    stdout
        .read_to_string(&mut output)
        .map_err(|e| format!("cannot read output: {}", e))?;

    // Debug: log raw Herbie response
    if std::env::var("HERBIE_LINT_DEBUG").is_ok() {
        eprintln!("[herbie-lint] Received from Herbie:\n{}", output);
    }

    let (errin, errout, cmdout_str) = match parse_fpcore_output(&output)? {
        HerbieResult::Improvement(errin, errout, body) => (errin, errout, body),
        HerbieResult::NoResult => {
            // Herbie couldn't process the expression - emit a note
            cx.tcx.dcx().span_note(
                expr.span,
                "Herbie couldn't analyse this expression (no valid sample points)",
            );
            return Ok(());
        }
    };

    // Check if there is improvement
    if errin <= errout {
        return Ok(());
    }

    // Parse and report
    let mut parser = Parser::new();
    let cmdout = parser
        .parse(&cmdout_str)
        .map_err(|_| "could not parse herbie output")?;

    report(cx, expr, &cmdout, &bindings);

    // Optionally save to database
    save_to_db(conf, &cmdin, &cmdout, errin, errout).ok();

    Ok(())
}

/// Parse FPCore output from Herbie 2.x.
///
/// Herbie shell output includes banners and prompts:
/// ```text
/// Herbie 2.1 with seed 857054429
/// Find help on https://herbie.uwplse.org/, exit with Ctrl-D
/// herbie> (FPCore
///  (params)
///  :herbie-error-input ((256 err1) (8000 err2))
///  :herbie-error-output ((256 err1) (8000 err2))
///  body-expr)
/// herbie>
/// ```
/// Result of parsing Herbie output.
enum HerbieResult {
    /// Herbie found an improvement: (input_error, output_error, improved_expression)
    Improvement(f64, f64, String),
    /// Herbie couldn't process the expression (e.g., "No valid values")
    NoResult,
}

fn parse_fpcore_output(output: &str) -> Result<HerbieResult, Cow<'static, str>> {
    // Find the FPCore expression in the output (skip banners and prompts)
    // If there's no FPCore, Herbie likely output a warning (e.g., "No valid values")
    let Some(fpcore_start) = output.find("(FPCore") else {
        return Ok(HerbieResult::NoResult);
    };
    let fpcore_section = &output[fpcore_start..];

    // Find the matching closing paren
    let fpcore = extract_balanced_list_inclusive(fpcore_section).ok_or("malformed FPCore")?;

    // Extract :herbie-error-input value
    // If missing, Herbie may have encountered an issue
    let Some(errin) = extract_herbie_error(&fpcore, ":herbie-error-input") else {
        return Ok(HerbieResult::NoResult);
    };

    // Extract :herbie-error-output value
    let Some(errout) = extract_herbie_error(&fpcore, ":herbie-error-output") else {
        return Ok(HerbieResult::NoResult);
    };

    // Extract body expression - it's the last balanced expression before the final )
    let body = extract_body_expression(&fpcore).ok_or("missing body expression")?;

    Ok(HerbieResult::Improvement(errin, errout, body))
}

/// Extract the average error value from a Herbie error property.
/// Format: :herbie-error-input ((bits1 err1) (bits2 err2) ...)
fn extract_herbie_error(output: &str, property: &str) -> Option<f64> {
    let start = output.find(property)? + property.len();
    let rest = output[start..].trim_start();

    // Find the balanced list
    if !rest.starts_with('(') {
        return None;
    }

    let list_content = extract_balanced_list(rest)?;

    // Parse error values from ((bits err) ...) format
    let mut total = 0.0;
    let mut count = 0;

    // Simple parsing: find all (bits err) pairs
    let mut pos = 0;
    let chars: Vec<char> = list_content.chars().collect();
    while pos < chars.len() {
        if chars[pos] == '(' {
            // Skip opening paren
            pos += 1;
            // Skip leading whitespace
            while pos < chars.len() && chars[pos].is_whitespace() {
                pos += 1;
            }
            // Skip bits value (first number)
            while pos < chars.len() && chars[pos].is_ascii_digit() {
                pos += 1;
            }
            // Skip whitespace between bits and error value
            while pos < chars.len() && chars[pos].is_whitespace() {
                pos += 1;
            }
            // Now we should be at the error value - collect it
            let mut err_str = String::new();
            while pos < chars.len() && chars[pos] != ')' && !chars[pos].is_whitespace() {
                err_str.push(chars[pos]);
                pos += 1;
            }
            if let Ok(err) = err_str.parse::<f64>() {
                total += err;
                count += 1;
            }
        }
        pos += 1;
    }

    if count > 0 {
        Some(total / count as f64)
    } else {
        None
    }
}

/// Extract a balanced parenthesised list from the start of a string (without outer parens).
fn extract_balanced_list(s: &str) -> Option<String> {
    let full = extract_balanced_list_inclusive(s)?;
    // Remove outer parens
    Some(full[1..full.len() - 1].to_string())
}

/// Extract a balanced parenthesised list from the start of a string (including outer parens).
fn extract_balanced_list_inclusive(s: &str) -> Option<String> {
    if !s.starts_with('(') {
        return None;
    }

    let mut depth = 0;
    let mut end = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    if depth == 0 && end > 0 {
        Some(s[..end].to_string())
    } else {
        None
    }
}

/// Extract the body expression from FPCore output.
/// The body is the last expression before the closing paren.
fn extract_body_expression(output: &str) -> Option<String> {
    // Find the last balanced expression by scanning from end
    let output = output.trim();
    if !output.ends_with(')') {
        return None;
    }

    // Remove the final closing paren of FPCore
    let inner = output[..output.len() - 1].trim_end();

    // Scan backwards to find where the body expression starts
    let chars: Vec<char> = inner.chars().collect();
    let mut end = chars.len();
    let mut depth = 0;

    // Find the start of the last expression
    for i in (0..chars.len()).rev() {
        match chars[i] {
            ')' => depth += 1,
            '(' => {
                if depth == 0 {
                    // This is the start of the body expression
                    let body = inner[i..end].trim().to_string();
                    return Some(body);
                }
                depth -= 1;
            }
            c if c.is_whitespace() && depth == 0 => {
                // This might be a simple atom expression
                let body = inner[i + 1..end].trim();
                if !body.is_empty() && !body.starts_with(':') {
                    return Some(body.to_string());
                }
                end = i;
            }
            _ => {}
        }
    }

    // Check if there's remaining content
    let body = inner[0..end].trim();
    if !body.is_empty() && !body.starts_with(':') {
        Some(body.to_string())
    } else {
        None
    }
}

fn save_to_db(
    conf: &Conf,
    cmdin: &str,
    cmdout: &LispExpr,
    errin: f64,
    errout: f64,
) -> Result<(), sql::Error> {
    // Can only save to user-specified database, not the embedded one
    let Some(db_path) = &conf.db_path else {
        return Ok(());
    };

    let connection =
        sql::Connection::open_with_flags(db_path, sql::OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    connection.execute(
        "INSERT INTO HerbieResults (cmdin, cmdout, opts, errin, errout) VALUES (?1, ?2, ?3, ?4, ?5)",
        (cmdin, cmdout.to_lisp("herbie"), "", errin, errout),
    )?;
    Ok(())
}

#[derive(Debug)]
pub enum InitError {
    Conf(ConfError),
    Io(std::io::Error),
    Sql(sql::Error),
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::Conf(e) => write!(f, "Configuration error: {}", e),
            InitError::Io(e) => write!(f, "IO error: {}", e),
            InitError::Sql(e) => write!(f, "SQL error: {}", e),
        }
    }
}

impl std::error::Error for InitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            InitError::Conf(e) => Some(e),
            InitError::Io(e) => Some(e),
            InitError::Sql(e) => Some(e),
        }
    }
}

impl From<ConfError> for InitError {
    fn from(e: ConfError) -> Self {
        InitError::Conf(e)
    }
}

impl From<sql::Error> for InitError {
    fn from(e: sql::Error) -> Self {
        InitError::Sql(e)
    }
}

#[test]
fn ui() {
    dylint_testing::ui_test(env!("CARGO_PKG_NAME"), "ui");
}

#[cfg(test)]
mod fpcore_tests {
    use super::*;

    #[test]
    fn test_extract_herbie_error_basic() {
        let output = r#"(FPCore (x) :herbie-error-input ((256 0) (8000 0.5)) body)"#;
        let result = extract_herbie_error(output, ":herbie-error-input");
        assert_eq!(result, Some(0.25)); // (0 + 0.5) / 2
    }

    #[test]
    fn test_extract_herbie_error_with_decimal() {
        let output = r#"(FPCore (x) :herbie-error-input ((256 0) (8000 0.009375)) body)"#;
        let result = extract_herbie_error(output, ":herbie-error-input");
        assert_eq!(result, Some(0.0046875)); // (0 + 0.009375) / 2
    }

    #[test]
    fn test_extract_herbie_error_multiline() {
        let output = r#"(FPCore
 (herbie0 herbie1 herbie2)
 :herbie-status ex-start
 :herbie-time 1477.225830078125
 :herbie-error-input
 ((256 0) (8000 0.009375))
 :herbie-error-output
 ((256 0) (8000 0.009375))
 :name "test"
 :precision binary64
 (/ (- herbie0 herbie1) herbie2))"#;
        let errin = extract_herbie_error(output, ":herbie-error-input");
        let errout = extract_herbie_error(output, ":herbie-error-output");
        assert_eq!(errin, Some(0.0046875));
        assert_eq!(errout, Some(0.0046875));
    }

    #[test]
    fn test_extract_herbie_error_improvement() {
        let output = r#"(FPCore
 (herbie0 herbie1)
 :herbie-error-input ((256 0) (8000 0.0016009193652572005))
 :herbie-error-output ((256 0) (8000 0))
 :name "(- (* (- herbie0) herbie1) 1)"
 :precision binary64
 (fma (- herbie1) herbie0 -1.0))"#;
        let errin = extract_herbie_error(output, ":herbie-error-input");
        let errout = extract_herbie_error(output, ":herbie-error-output");
        // Input error: (0 + 0.0016...) / 2 ≈ 0.0008
        // Output error: (0 + 0) / 2 = 0
        assert!(errin.unwrap() > 0.0);
        assert_eq!(errout, Some(0.0));
    }

    #[test]
    fn test_parse_fpcore_output_improvement() {
        let output = r#"herbie> (FPCore
 (herbie0 herbie1)
 :herbie-error-input ((256 0) (8000 0.5))
 :herbie-error-output ((256 0) (8000 0))
 :name "test"
 :precision binary64
 (fma (- herbie1) herbie0 -1.0))
herbie> "#;
        let result = parse_fpcore_output(output).unwrap();
        match result {
            HerbieResult::Improvement(errin, errout, body) => {
                assert_eq!(errin, 0.25);
                assert_eq!(errout, 0.0);
                assert!(body.contains("fma"));
            }
            HerbieResult::NoResult => panic!("Expected Improvement, got NoResult"),
        }
    }

    #[test]
    fn test_extract_body_expression() {
        let fpcore = "(FPCore (x) :name \"test\" (+ x 1))";
        let body = extract_body_expression(fpcore);
        assert_eq!(body, Some("(+ x 1)".to_string()));
    }

    #[test]
    fn test_extract_body_expression_fma() {
        let fpcore =
            "(FPCore (herbie0 herbie1) :precision binary64 (fma (- herbie1) herbie0 -1.0))";
        let body = extract_body_expression(fpcore);
        assert_eq!(body, Some("(fma (- herbie1) herbie0 -1.0)".to_string()));
    }
}
