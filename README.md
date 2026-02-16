# X Bot

Autonomous Twitter/X bot built with Python, Selenium, LangChain/LangGraph, ChromaDB memory, and a FastAPI dashboard.

## What It Does

- Generates autonomous tweets from a configurable personality
- Reads X timeline posts and scores interest with LLM evaluation
- Creates inspiration-based posts from queued interesting content
- Checks notifications and queues replies/mentions for processing
- Prevents duplicate content with ChromaDB similarity checks
- Exposes a web dashboard for status, analytics, logs, and settings

## Prerequisites

- Python `3.13+`
- `uv` package manager
- Google Chrome installed

## Setup (Bash)

```bash
uv sync
cp config/env.template .env
```

### Required `.env` values

```env
TWITTER_USERNAME=your_x_username
TWITTER_PASSWORD=your_x_password

# At least one is required
OPENAI_API_KEY=
OPENROUTER_API_KEY=
GOOGLE_API_KEY=
ANTHROPIC_API_KEY=
```

### Optional `.env` values

```env
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=
LOG_LEVEL=INFO
ENVIRONMENT=dev
```

### Dashboard auth values (required for web dashboard login)

```env
AUTH_USERNAME=admin
AUTH_PASSWORD_HASH=$argon2id$...
AUTH_SECRET_KEY=change-this-to-a-long-random-secret
SESSION_COOKIE_NAME=xb_session
SESSION_MAX_AGE_SECONDS=28800
SESSION_HTTPS_ONLY=false
```

Generate an Argon2 hash:

```bash
uv run python -c "from argon2 import PasswordHasher; print(PasswordHasher().hash('your-password'))"
```

Use the generated hash as `AUTH_PASSWORD_HASH`.

## Configuration

Runtime behavior is controlled in `config/config.yaml`:

- `llm`: provider/model/fallback/temperature/embedding settings
- `personality`: tone/style/topics/tweet length bounds
- `scheduler`: posting/reading/notification/inspiration intervals
- `rate_limits`: max posts/replies and UTC reset time
- `selenium`: delays/headless/user-agent behavior

## Run

### Start bot scheduler mode

```bash
uv run python main.py
```

### Start web dashboard only

```bash
uv run python -m src.web.app
```

Dashboard URL:

- `http://localhost:8000`

## Tests

Run all tests:

```bash
uv run pytest
```

Run a specific test file:

```bash
uv run pytest tests/test_evaluation.py -v
```

## Data and Runtime Files

- `data/state.json`: runtime state and queues
- `data/bot.db`: analytics database
- `data/chroma/`: vector memory store
- `logs/`: runtime logs
- `config/cookie.json`: persisted X login cookies

## Roadmap

Current status and planned work are documented in `ROADMAP.md`.
