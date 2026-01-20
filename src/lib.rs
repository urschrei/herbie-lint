#![feature(rustc_private)]
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
use std::process::{Command, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

use conf::{Conf, ConfError, UseHerbieConf};
use lisp::{LispExpr, MatchBindings, Parser};

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
        let connection = sql::Connection::open_with_flags(
            conf.db_path.as_ref(),
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
            if let Err(err) = try_with_herbie(cx, expr, conf) {
                cx.tcx.dcx().span_warn(expr.span, err);
            }
        }
    }
}

fn is_f64(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Float(ty::FloatTy::F64))
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
    // Convert expression to Lisp
    let (lisp_expr, nb_ids, bindings) = match LispExpr::from_expr(cx, expr) {
        Some(r) => r,
        None => return Ok(()), // Expression contains unsupported constructs
    };

    // Skip trivial expressions
    if lisp_expr.depth() <= 2 {
        return Ok(());
    }

    // Spawn Herbie process
    let mut command = Command::new("herbie-inout");
    command
        .arg("--seed")
        .arg(conf.herbie_seed.as_ref())
        .arg("-o")
        .arg("rules:numerics")
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

    // Send expression to Herbie
    let params = (0..nb_ids)
        .map(|id| format!("herbie{}", id))
        .collect::<Vec<_>>()
        .join(" ");
    let cmdin = lisp_expr.to_lisp("herbie");
    let lisp_input = format!("(lambda ({}) {})\n", params, cmdin);

    child
        .stdin
        .as_mut()
        .expect("stdin captured")
        .write_all(lisp_input.as_bytes())
        .expect("write to stdin");

    // Wait with timeout
    let status = match conf.timeout {
        Some(timeout) => match child.wait_timeout(Duration::from_secs(timeout as u64)) {
            Ok(Some(status)) => status,
            Ok(None) => return Ok(()), // Timeout
            Err(err) => return Err(format!("herbie-inout error: {}", err).into()),
        },
        None => child
            .wait()
            .map_err(|e| format!("herbie-inout error: {}", e))?,
    };

    if !status.success() {
        return Err(format!("herbie-inout exited with: {}", status).into());
    }

    // Parse output
    let mut stdout = child.stdout.ok_or("cannot capture stdout")?;
    let mut output = String::new();
    stdout
        .read_to_string(&mut output)
        .map_err(|e| format!("cannot read output: {}", e))?;

    let mut lines = output.lines();
    let errin = parse_error_line(lines.next())?;
    let errout = parse_error_line(lines.next())?;
    let cmdout_str = lines.next().ok_or("missing output expression")?;

    // Check if there is improvement
    if errin <= errout {
        return Ok(());
    }

    // Parse and report
    let mut parser = Parser::new();
    let cmdout = parser
        .parse(cmdout_str)
        .map_err(|_| "could not parse herbie output")?;

    report(cx, expr, &cmdout, &bindings);

    // Optionally save to database
    save_to_db(conf, &cmdin, &cmdout, errin, errout).ok();

    Ok(())
}

fn parse_error_line(line: Option<&str>) -> Result<f64, Cow<'static, str>> {
    line.and_then(|s| s.split_whitespace().last())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "could not parse error value".into())
}

fn save_to_db(
    conf: &Conf,
    cmdin: &str,
    cmdout: &LispExpr,
    errin: f64,
    errout: f64,
) -> Result<(), sql::Error> {
    let connection = sql::Connection::open_with_flags(
        conf.db_path.as_ref(),
        sql::OpenFlags::SQLITE_OPEN_READ_WRITE,
    )?;
    connection.execute(
        "INSERT INTO HerbieResults (cmdin, cmdout, opts, errin, errout) VALUES (?1, ?2, ?3, ?4, ?5)",
        (cmdin, cmdout.to_lisp("herbie"), "", errin, errout),
    )?;
    Ok(())
}

#[derive(Debug)]
pub enum InitError {
    Conf(ConfError),
    Sql(sql::Error),
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::Conf(e) => write!(f, "Configuration error: {}", e),
            InitError::Sql(e) => write!(f, "SQL error: {}", e),
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
