"""FastAPI application for X bot web dashboard."""

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
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

# Paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


@lru_cache
def get_config() -> BotConfig:
    """Get the loaded bot configuration.

    Uses lru_cache to ensure only one config instance is created.
    This is the proper FastAPI dependency injection pattern.
    """
    return BotConfig.load()


@lru_cache
def get_chroma_memory() -> "ChromaMemory | None":
    """Get the ChromaDB memory client if available.

    Uses lru_cache to ensure only one memory client instance is created.
    Returns None if ChromaDB is not available or no API key is configured.
    """
    if not CHROMA_AVAILABLE or ChromaMemory is None:
        return None

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return None

    try:
        return ChromaMemory(
            config=get_config(),
            openai_api_key=openai_api_key,
        )
    except Exception:
        return None


# Type aliases for dependency injection
ConfigDep = Annotated[BotConfig, Depends(get_config)]
ChromaMemoryDep = Annotated["ChromaMemory | None", Depends(get_chroma_memory)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Load environment variables
    load_dotenv()

    # Pre-load config and chroma memory to warm up the cache
    get_config()
    get_chroma_memory()

    yield

    # Shutdown: Close cached resources and clear caches
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
