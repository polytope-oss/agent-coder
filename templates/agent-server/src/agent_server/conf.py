from pydantic import BaseModel
from .utils import env, log
from .utils.env import EnvVarSpec

logger = log.get_logger(__name__)

# Set to False if you don't want to use PostgreSQL
# When False, all database functionality will be disabled
USE_POSTGRES = False

#### Types ####


class HttpServerConf(BaseModel):
    host: str
    port: int
    autoreload: bool


#### Env Vars ####

## Logging ##
LOG_LEVEL = EnvVarSpec(id="LOG_LEVEL", default="INFO")

## HTTP ##
HTTP_HOST = EnvVarSpec(id="HTTP_HOST", default="0.0.0.0")
HTTP_PORT = EnvVarSpec(id="HTTP_PORT", default="3030")
HTTP_AUTORELOAD = EnvVarSpec(
    id="HTTP_AUTORELOAD",
    parse=lambda x: x.lower() == "true",
    default="false",
    type=(bool, ...),
)
HTTP_DEBUG = EnvVarSpec(
    id="HTTP_DEBUG",
    parse=lambda x: x.lower() == "true",
    default="false",
    type=(bool, ...),
)

## Opper API ##
OPPER_API_KEY = EnvVarSpec(
    id="OPPER_API_KEY",
    is_secret=True,
    is_optional=False,  # Required for Opper agents
)

## PostgreSQL ##
POSTGRES_DB = EnvVarSpec(id="POSTGRES_DB", default="postgres")

POSTGRES_USER = EnvVarSpec(id="POSTGRES_USER", default="postgres")

POSTGRES_PASSWORD = EnvVarSpec(
    id="POSTGRES_PASSWORD", default="postgres", is_secret=True
)

POSTGRES_HOST = EnvVarSpec(id="POSTGRES_HOST", default="postgres")

POSTGRES_PORT = EnvVarSpec(
    id="POSTGRES_PORT", parse=int, default="5432", type=(int, ...)
)

POSTGRES_POOL_MIN = EnvVarSpec(
    id="POSTGRES_POOL_MIN", parse=int, default="1", type=(int, ...)
)

POSTGRES_POOL_MAX = EnvVarSpec(
    id="POSTGRES_POOL_MAX", parse=int, default="10", type=(int, ...)
)

#### Validation ####


def validate() -> bool:
    """Validate required environment variables."""
    env_vars = [
        LOG_LEVEL,
        HTTP_HOST,
        HTTP_PORT,
        HTTP_AUTORELOAD,
        HTTP_DEBUG,
        OPPER_API_KEY,
    ]

    # Only validate PostgreSQL vars if USE_POSTGRES is True
    if USE_POSTGRES:
        env_vars.extend(
            [
                POSTGRES_DB,
                POSTGRES_USER,
                POSTGRES_PASSWORD,
                POSTGRES_HOST,
                POSTGRES_PORT,
                POSTGRES_POOL_MIN,
                POSTGRES_POOL_MAX,
            ]
        )

    is_valid = env.validate(env_vars)

    if not is_valid:
        logger.error("Environment validation failed. Make sure OPPER_API_KEY is set.")

    return is_valid


#### Getters ####


def get_log_level() -> str:
    """Get the log level."""
    return env.parse(LOG_LEVEL)


def get_http_conf() -> HttpServerConf:
    """Get HTTP server configuration."""
    return HttpServerConf(
        host=env.parse(HTTP_HOST),
        port=env.parse(HTTP_PORT),
        autoreload=env.parse(HTTP_AUTORELOAD),
    )


def get_debug_mode() -> bool:
    """Get debug mode flag."""
    return env.parse(HTTP_DEBUG)


def get_opper_api_key() -> str:
    """Get Opper API key."""
    return env.parse(OPPER_API_KEY)


def get_postgres_conf():
    """Get PostgreSQL connection configuration."""
    # Import here to avoid circular dependency
    from .clients.postgres import PostgresConf

    return PostgresConf(
        database=env.parse(POSTGRES_DB),
        user=env.parse(POSTGRES_USER),
        password=env.parse(POSTGRES_PASSWORD),
        host=env.parse(POSTGRES_HOST),
        port=env.parse(POSTGRES_PORT),
    )


def get_postgres_pool_conf():
    """Get PostgreSQL connection pool configuration."""
    # Import here to avoid circular dependency
    from .clients.postgres import PostgresPoolConf

    return PostgresPoolConf(
        min_size=env.parse(POSTGRES_POOL_MIN),
        max_size=env.parse(POSTGRES_POOL_MAX),
    )
