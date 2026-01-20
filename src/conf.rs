use serde::Deserialize;
use std::borrow::Cow;
use std::fs;
use std::io;

const DEFAULT_HERBIE_SEED: &str =
    "#(1461197085 2376054483 1553562171 1611329376 2497620867 2308122621)";
const DEFAULT_DB_PATH: &str = "Herbie.db";
const DEFAULT_TIMEOUT: u32 = 120;

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
    pub db_path: Cow<'static, str>,
    pub herbie_seed: Cow<'static, str>,
    pub timeout: Option<u32>,
    pub use_herbie: UseHerbieConf,
}

impl Default for Conf {
    fn default() -> Self {
        Self {
            db_path: DEFAULT_DB_PATH.into(),
            herbie_seed: DEFAULT_HERBIE_SEED.into(),
            timeout: Some(DEFAULT_TIMEOUT),
            use_herbie: UseHerbieConf::Default,
        }
    }
}

impl From<UxConf> for Conf {
    fn from(ux: UxConf) -> Self {
        Self {
            db_path: ux.db_path.map_or(DEFAULT_DB_PATH.into(), Into::into),
            herbie_seed: ux
                .herbie_seed
                .map_or(DEFAULT_HERBIE_SEED.into(), Into::into),
            timeout: ux.timeout.map_or(
                Some(DEFAULT_TIMEOUT),
                |t| {
                    if t == 0 { None } else { Some(t) }
                },
            ),
            use_herbie: ux.use_herbie.map_or(UseHerbieConf::Default, |u| {
                if u {
                    UseHerbieConf::Yes
                } else {
                    UseHerbieConf::No
                }
            }),
        }
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

pub fn read_conf() -> Result<Conf, ConfError> {
    match fs::read_to_string("Herbie.toml") {
        Ok(content) => {
            let ux: UxConf = toml::from_str(&content).map_err(ConfError::Parse)?;
            Ok(ux.into())
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Conf::default()),
        Err(e) => Err(ConfError::Io(e)),
    }
}
