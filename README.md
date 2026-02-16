# X Bot

Autonomous Twitter/X bot built with Python, FastAPI, Selenium, and multi-provider LLM support.

## Requirements

- Python `3.13+`
- `uv`
- Google Chrome installed

## Setup

```powershell
uv sync
Copy-Item config\env.template .env
```

Fill `.env` with:

- `TWITTER_USERNAME`
- `TWITTER_PASSWORD`
- At least one LLM key:
  - `OPENAI_API_KEY` or
  - `OPENROUTER_API_KEY` or
  - `GOOGLE_API_KEY` or
  - `ANTHROPIC_API_KEY`

## Run

Run bot (scheduler mode):

```bash
uv run python main.py
```

Run dashboard only:

```bash
uv run python -m src.web.app
```

Open dashboard:

- `http://localhost:8000`

## Tests

```bash
uv run pytest
```

