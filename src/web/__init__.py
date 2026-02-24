"""Web dashboard module for X bot monitoring."""

def create_app():
    """Lazy import to avoid web app initialization side effects on package import."""
    from src.web.app import create_app as _create_app

    return _create_app()

__all__ = ["create_app"]
