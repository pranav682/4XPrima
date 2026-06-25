"""Runtime configuration, loaded from environment / ``.env``.

Settings classes here are pydantic-settings models. They are constructed once
at process startup and then passed around explicitly — we do NOT import a
module-level singleton. Reasons:

- Tests instantiate fresh settings per test with their own env stub.
- The "right place" for credentials is the env, not a Python module — anyone
  reading the code can see the field names without seeing the values.

All `_API_TOKEN` fields are typed :class:`pydantic.SecretStr`, which keeps
the value out of accidental ``repr()`` / log output. Use ``.get_secret_value()``
only at the point of HTTP construction.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OandaEnv(StrEnum):
    PRACTICE = "practice"
    LIVE = "live"


# Verified against developer.oanda.com (2026-06-24). Practice is the demo
# environment; live carries real money. Switching must be an explicit human
# action — there is no code path that promotes practice → live.
_BASE_URLS: dict[OandaEnv, str] = {
    OandaEnv.PRACTICE: "https://api-fxpractice.oanda.com",
    OandaEnv.LIVE: "https://api-fxtrade.oanda.com",
}


class OandaSettings(BaseSettings):
    """OANDA REST API credentials and environment.

    Loaded from environment variables prefixed by `OANDA_`; the project's
    `.env` file is consulted as a fallback. The token is wrapped in
    :class:`SecretStr` so it stays out of stringified objects and tracebacks.
    """

    # `extra="ignore"` so other env vars in .env (TRADING_MODE, Anthropic
    # keys, etc.) don't trigger validation errors when only OandaSettings is
    # instantiated.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OANDA_",
        extra="ignore",
        case_sensitive=False,
    )

    api_token: SecretStr = Field(
        ...,
        description="OANDA personal API token. Stays in SecretStr to keep it out of logs.",
    )
    account_id: str = Field(
        ...,
        description="OANDA account ID, format 'NNN-NNN-NNNNNNNN-NNN'.",
    )
    env: OandaEnv = Field(
        default=OandaEnv.PRACTICE,
        description="Which OANDA environment to talk to.",
    )

    @property
    def base_url(self) -> str:
        return _BASE_URLS[self.env]


class FredSettings(BaseSettings):
    """FRED (Federal Reserve Economic Data) API credentials.

    Free key from https://fredaccount.stlouisfed.org/apikey. Loaded from
    `FRED_API_KEY` in env / .env. Wrapped in :class:`SecretStr` so it
    stays out of repr / log output.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FRED_",
        extra="ignore",
        case_sensitive=False,
    )

    api_key: SecretStr = Field(
        ...,
        description="FRED API key. SecretStr so it never lands in repr or logs.",
    )

    @property
    def base_url(self) -> str:
        return "https://api.stlouisfed.org"
