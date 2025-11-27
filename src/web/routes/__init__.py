"""Web routes for the dashboard."""

from src.web.routes.api import router as api_router
from src.web.routes.views import router as views_router

__all__ = ["api_router", "views_router"]
