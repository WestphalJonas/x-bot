"""Auth-related environment settings for the web application."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    """Environment-backed settings for dashboard authentication."""

    auth_username: str = Field(..., description="Dashboard login username")
    auth_password_hash: str = Field(
        ...,
        description="Argon2id hash (PHC string) of the dashboard password",
    )
    auth_secret_key: str = Field(
        ...,
        description="Secret key for signing session cookies",
    )
    session_cookie_name: str = Field(
        default="xb_session",
        description="Cookie name for the dashboard session",
    )
    session_max_age_seconds: int = Field(
        default=60 * 60 * 8,
        ge=60,
        description="Session lifetime in seconds",
    )
    session_https_only: bool = Field(
        default=False,
        description="Set Secure flag on session cookie (enable in HTTPS environments)",
    )
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _validate_hash(self) -> "AuthSettings":
        """Ensure password hash looks valid."""
        normalized = self.auth_password_hash.lower()
        if not (
            normalized.startswith("argon2id$") or normalized.startswith("$argon2id$")
        ):
            raise ValueError(
                "AUTH_PASSWORD_HASH must be an argon2id PHC string (e.g., '$argon2id$...')"
            )
        return self


@lru_cache
def get_auth_settings() -> AuthSettings:
    """Cached settings loader."""
    return AuthSettings()
