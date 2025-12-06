"""Authentication helpers for the web dashboard."""

import hmac
import logging
from datetime import datetime, timezone
from typing import Any

from argon2 import PasswordHasher
from argon2 import exceptions as argon_exc
from fastapi import HTTPException, Request, status
from starlette.responses import RedirectResponse

from src.web.settings import AuthSettings, get_auth_settings

logger = logging.getLogger(__name__)
_password_hasher = PasswordHasher()


def verify_credentials(
    username: str, password: str, settings: AuthSettings | None = None
) -> bool:
    """Verify provided credentials against settings."""
    auth_settings = settings or get_auth_settings()

    if not hmac.compare_digest(username, auth_settings.auth_username):
        logger.warning("auth_invalid_username", extra={"username": username})
        return False

    try:
        _password_hasher.verify(auth_settings.auth_password_hash, password)
        if _password_hasher.check_needs_rehash(auth_settings.auth_password_hash):
            logger.info("auth_password_rehash_recommended")
        return True
    except argon_exc.VerifyMismatchError:
        logger.warning("auth_invalid_password", extra={"username": username})
        return False
    except argon_exc.InvalidHash as exc:
        logger.error("auth_hash_error", extra={"error": str(exc)})
        return False


def create_session(request: Request, username: str) -> None:
    """Persist session data on the request."""
    request.session["user"] = username
    request.session["created_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("auth_session_created", extra={"username": username})


def clear_session(request: Request) -> None:
    """Clear session data and log the event."""
    request.session.clear()
    logger.info("auth_session_cleared")


async def require_session(request: Request) -> None:
    """Dependency that enforces an active session."""
    session_user: Any = request.session.get("user")
    settings = get_auth_settings()

    if not session_user or not hmac.compare_digest(str(session_user), settings.auth_username):
        logger.warning("auth_missing_or_invalid_session")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )


async def require_session_or_redirect(request: Request) -> None:
    """Dependency that redirects unauthenticated HTML views to login."""
    session_user: Any = request.session.get("user")
    settings = get_auth_settings()

    if session_user and hmac.compare_digest(str(session_user), settings.auth_username):
        return

    # Redirect only for non-API HTML requests; APIs should keep using require_session
    accept = request.headers.get("accept", "")
    if "text/html" in accept or "application/xhtml+xml" in accept:
        target = f"/login?next={request.url.path}"
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": target},
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )

