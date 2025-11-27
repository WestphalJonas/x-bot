"""FastAPI application for X bot web dashboard."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core.config import BotConfig

# ChromaDB is optional - may not be available on all Python versions
try:
    from src.memory.chroma_client import ChromaMemory

    CHROMA_AVAILABLE = True
except ImportError:
    ChromaMemory = None  # type: ignore
    CHROMA_AVAILABLE = False

# Module-level globals for shared resources
_config: BotConfig | None = None
_chroma_memory: "ChromaMemory | None" = None

# Paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def get_config() -> BotConfig:
    """Get the loaded bot configuration."""
    global _config
    if _config is None:
        _config = BotConfig.load()
    return _config


def get_chroma_memory():
    """Get the ChromaDB memory client if available."""
    global _chroma_memory
    if not CHROMA_AVAILABLE:
        return None
    if _chroma_memory is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and ChromaMemory is not None:
            _chroma_memory = ChromaMemory(
                config=get_config(),
                openai_api_key=openai_api_key,
            )
    return _chroma_memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Load configuration and initialize resources
    from dotenv import load_dotenv

    load_dotenv()

    # Pre-load config
    get_config()

    # Try to initialize ChromaDB (optional - may not have API key)
    get_chroma_memory()

    yield

    # Shutdown: cleanup if needed
    pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="X Bot Dashboard",
        description="Monitoring dashboard for the autonomous X/Twitter bot",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Mount static files
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Setup templates
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Store templates in app state for access in routes
    app.state.templates = templates

    # Import and include routers
    from src.web.routes.api import router as api_router
    from src.web.routes.views import router as views_router

    app.include_router(api_router, prefix="/api", tags=["API"])
    app.include_router(views_router, tags=["Views"])

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
