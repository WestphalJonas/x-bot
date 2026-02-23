"""FastAPI application for X bot web dashboard."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from src.web.auth import require_session, require_session_or_redirect
from src.web.deps import get_chroma_memory, get_config
from src.web.settings import get_auth_settings

WEB_DIR = Path(__file__).parent
PROJECT_ROOT = WEB_DIR.parent.parent
ENV_FILE = PROJECT_ROOT / "config" / ".env"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Load environment before dependency evaluation and warm caches to avoid first-request latency
    load_dotenv(dotenv_path=ENV_FILE)

    get_config()
    get_chroma_memory()

    yield

    # Shutdown: close cached memory client and clear dependency caches
    memory = get_chroma_memory()
    if memory is not None and hasattr(memory, "close"):
        try:
            await memory.close()  # type: ignore[func-returns-value]
        except Exception:
            pass
    get_config.cache_clear()
    get_chroma_memory.cache_clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Ensure environment variables are loaded before settings are evaluated
    load_dotenv(dotenv_path=ENV_FILE)
    auth_settings = get_auth_settings()

    app = FastAPI(
        title="X Bot Dashboard",
        description="Monitoring dashboard for the autonomous X/Twitter bot",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        SessionMiddleware,
        secret_key=auth_settings.auth_secret_key,
        session_cookie=auth_settings.session_cookie_name,
        max_age=auth_settings.session_max_age_seconds,
        same_site="lax",
        https_only=auth_settings.session_https_only,
    )

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.state.templates = templates

    from src.web.routes.auth import router as auth_router
    from src.web.routes.api import router as api_router
    from src.web.routes.views import router as views_router

    app.include_router(auth_router, tags=["Auth"])
    app.include_router(
        api_router,
        prefix="/api",
        tags=["API"],
        dependencies=[Depends(require_session)],
    )
    app.include_router(
        views_router,
        tags=["Views"],
        dependencies=[Depends(require_session_or_redirect)],
    )

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
