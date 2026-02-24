# X Bot

Autonomous Twitter/X bot built with Python, Selenium, LangChain/LangGraph, ChromaDB memory, and a FastAPI dashboard (`Jinja2 + HTMX + Alpine.js + Tailwind CSS`).

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
- Node.js + npm (for Tailwind CSS build)

## Setup (Bash)

```bash
uv sync
cp config/env.template config/.env
npm install
npm run tailwind:build
```

Note: The web dashboard loads environment variables from `config/.env`.

### Required `config/.env` values

```env
TWITTER_USERNAME=your_x_username
TWITTER_PASSWORD=your_x_password

# At least one is required
OPENAI_API_KEY=
OPENROUTER_API_KEY=
GOOGLE_API_KEY=
ANTHROPIC_API_KEY=
```

### Optional `config/.env` values

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

Use the generated hash as `AUTH_PASSWORD_HASH` in `config/.env`.

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

## Docker / Docker Compose

The project can run in Docker with two services:

- `bot`: scheduler + posting/reading/reply jobs
- `web`: FastAPI dashboard on port `8000`

### Prerequisites

- Docker
- Docker Compose (Docker Desktop includes it)
- `config/.env` configured (same file as local setup)

### Start with Compose

```bash
cp config/env.template config/.env  # if not created yet
docker compose up --build
```

This starts:

- Bot scheduler in background mode
- Web dashboard at `http://localhost:8000`

Persistent directories are mounted from the host:

- `./config` (includes `cookie.json`)
- `./data`
- `./logs`

### Start only one service

```bash
docker compose up --build bot
docker compose up --build web
```

### Notes for Selenium / Chrome in Docker

- Set `selenium.headless: true` in `config/config.yaml` for container usage.
- `shm_size: 1gb` is configured to reduce Chrome crashes in containers.
- First start may take longer because dependencies and Chrome are installed during image build.

## Frontend UI (Tailwind)

The dashboard UI is styled via Tailwind CSS and built to:

- `src/web/static/build.css`

Useful commands:

```bash
npm run tailwind:build
npm run tailwind:watch
```

Notes:

- The dashboard stylesheet is generated from `src/web/static/tailwind/input.css`.
- After changing templates or `src/web/static/tailwind/input.css`, rebuild Tailwind (or run watch mode).

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
