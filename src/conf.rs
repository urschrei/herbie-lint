use serde::Deserialize;
use std::borrow::Cow;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const DEFAULT_HERBIE_SEED: &str =
    "#(1461197085 2376054483 1553562171 1611329376 2497620867 2308122621)";
const DEFAULT_TIMEOUT: u32 = 120;

/// Returns the directory to search for Herbie.toml and resolve relative paths.
/// Uses CARGO_MANIFEST_DIR if set (during cargo build), otherwise falls back to
/// the current working directory.
fn config_base_dir() -> PathBuf {
    env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().unwrap_or_default())
}

#[derive(Debug, Deserialize)]
struct UxConf {
    db_path: Option<String>,
    herbie_seed: Option<String>,
    timeout: Option<u32>,
    use_herbie: Option<bool>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum UseHerbieConf {
    Default,
    No,
    Yes,
}

#[derive(Debug)]
pub struct Conf {
    /// Path to the Herbie database. None means use the embedded database.
    pub db_path: Option<String>,
    pub herbie_seed: Cow<'static, str>,
    pub timeout: Option<u32>,
    pub use_herbie: UseHerbieConf,
}

impl Default for Conf {
    fn default() -> Self {
        Self {
            db_path: None, // Use embedded database
            herbie_seed: DEFAULT_HERBIE_SEED.into(),
            timeout: Some(DEFAULT_TIMEOUT),
            use_herbie: UseHerbieConf::Default,
        }
    }
}

impl UxConf {
    /// Convert user config to internal config, resolving relative paths against base_dir.
    fn into_conf(self, base_dir: &Path) -> Conf {
        Conf {
            db_path: self.db_path.map(|p| resolve_path(base_dir, &p)),
            herbie_seed: self
                .herbie_seed
                .map_or(DEFAULT_HERBIE_SEED.into(), Into::into),
            timeout: self.timeout.map_or(Some(DEFAULT_TIMEOUT), |t| {
                if t == 0 { None } else { Some(t) }
            }),
            use_herbie: self.use_herbie.map_or(UseHerbieConf::Default, |u| {
                if u {
                    UseHerbieConf::Yes
                } else {
                    UseHerbieConf::No
                }
            }),
        }
    }
}

/// Resolve a path relative to base_dir if it's not absolute.
fn resolve_path(base_dir: &Path, path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        base_dir.join(path).to_string_lossy().into_owned()
    }
}

#[derive(Debug)]
pub enum ConfError {
    Io(io::Error),
    Parse(toml::de::Error),
}

impl std::fmt::Display for ConfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfError::Io(e) => write!(f, "Error reading Herbie.toml: {}", e),
            ConfError::Parse(e) => write!(f, "Syntax error in Herbie.toml: {}", e),
        }
    }
}

impl std::error::Error for ConfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfError::Io(e) => Some(e),
            ConfError::Parse(e) => Some(e),
        }
    }
}

pub fn read_conf() -> Result<Conf, ConfError> {
    let base_dir = config_base_dir();
    let config_path = base_dir.join("Herbie.toml");

    match fs::read_to_string(&config_path) {
        Ok(content) => {
            let ux: UxConf = toml::from_str(&content).map_err(ConfError::Parse)?;
            Ok(ux.into_conf(&base_dir))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Conf::default()),
        Err(e) => Err(ConfError::Io(e)),
    }
}
