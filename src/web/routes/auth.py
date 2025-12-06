"""Authentication routes for the web dashboard."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from starlette.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.web.auth import (
    clear_session,
    create_session,
    require_session,
    verify_credentials,
)
from src.web.settings import AuthSettings, get_auth_settings

router = APIRouter(tags=["Auth"])
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    """Login payload."""

    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


SettingsDep = Annotated[AuthSettings, Depends(get_auth_settings)]


@router.get("/login")
async def login_form(request: Request) -> dict[str, object]:
    """Render login page."""
    templates = request.app.state.templates
    error = request.query_params.get("error") == "1"
    next_path = request.query_params.get("next") or "/"
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": error,
            "next_path": next_path,
        },
    )


@router.post("/login")
async def login(
    request: Request,
    settings: SettingsDep,
    payload: LoginRequest | None = None,
    username: str | None = Form(None),
    password: str | None = Form(None),
    next_path: str | None = Form(None),
) -> dict[str, str]:
    """Authenticate user and create a session.

    Supports both JSON body and form-encoded submissions.
    """
    incoming_username = username or (payload.username if payload else None)
    incoming_password = password or (payload.password if payload else None)
    target = next_path or request.query_params.get("next") or "/"

    if not incoming_username or not incoming_password:
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(
                url=f"/login?error=1&next={target}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required",
        )

    if not verify_credentials(incoming_username, incoming_password, settings):
        logger.warning("auth_login_failed", extra={"username": incoming_username})
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(
                url=f"/login?error=1&next={target}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    create_session(request, settings.auth_username)
    logger.info("auth_login_success", extra={"username": incoming_username})

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(
            url=target or "/", status_code=status.HTTP_303_SEE_OTHER
        )

    return {"status": "ok", "message": "Logged in"}


@router.post("/logout")
async def logout(
    request: Request, _: None = Depends(require_session)
) -> dict[str, str]:
    """Clear session and log the user out."""
    clear_session(request)
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return {"status": "ok", "message": "Logged out"}
