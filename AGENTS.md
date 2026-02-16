# X Bot - Agent Documentation

## Project Overview

X Bot is an autonomous Twitter/X posting bot built in Python. It uses LLMs (OpenAI, OpenRouter, Google, Anthropic) to generate tweets, reads the X frontpage for inspiration, checks notifications, and maintains a web dashboard for monitoring. The bot operates on a scheduled basis using APScheduler.

### Core Capabilities

- **Autonomous Tweet Generation**: Generates tweets based on configured personality and topics
- **Frontpage Reading**: Reads X frontpage posts, evaluates interest, and stores for inspiration
- **Inspiration-Based Posting**: Creates tweets inspired by interesting posts found on the timeline
- **Notification Checking**: Monitors replies and mentions, queues them for processing
- **Memory & Duplicate Detection**: Uses ChromaDB vector store to avoid duplicate content
- **Web Dashboard**: FastAPI-based dashboard with HTMX for real-time monitoring
- **Multi-Provider LLM**: Supports multiple LLM providers with automatic fallback
- **Rate Limiting**: Enforces daily posting limits with UTC reset

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13+ |
| Web Framework | FastAPI + Jinja2 + HTMX |
| Browser Automation | Selenium + undetected-chromedriver + selenium-stealth |
| LLM Framework | LangChain + LangGraph |
| Vector Database | ChromaDB |
| Scheduler | APScheduler (BackgroundScheduler) |
| Database | SQLite (aiosqlite) |
| State Storage | JSON (atomic writes) |
| Configuration | YAML + Pydantic |
| Testing | pytest + pytest-asyncio |
| Package Manager | uv |

## Project Structure

```
├── main.py                 # Application entry point
├── pyproject.toml          # Project dependencies (uv-based)
├── config/
│   ├── config.yaml         # Bot configuration (YAML)
│   ├── agent_prompt.txt    # System prompt template
│   ├── cookie.json         # Persisted X cookies
│   └── env.template        # Environment variables template
├── src/
│   ├── core/               # Core business logic
│   │   ├── config.py       # Configuration models (Pydantic)
│   │   ├── llm.py          # High-level LLM client
│   │   ├── langchain_clients.py  # LangChain LLM implementation
│   │   ├── prompts.py      # LLM prompt templates
│   │   ├── evaluation.py   # Tweet re-evaluation logic
│   │   ├── interest.py     # Post interest detection
│   │   └── graph/          # LangGraph workflow definitions
│   │       ├── tweet_generation.py
│   │       ├── reading.py
│   │       ├── notifications.py
│   │       └── replies.py
│   ├── memory/             # Vector memory (ChromaDB)
│   │   └── chroma_client.py
│   ├── scheduler/          # Job scheduling
│   │   ├── bot_scheduler.py    # APScheduler wrapper
│   │   ├── control_server.py   # Cross-process control
│   │   └── jobs/               # Job implementations
│   │       ├── posting.py
│   │       ├── reading.py
│   │       ├── notifications.py
│   │       ├── replies.py
│   │       └── inspiration.py
│   ├── state/              # State management
│   │   ├── models.py       # Pydantic models
│   │   ├── manager.py      # State load/save (JSON)
│   │   └── database.py     # SQLite analytics database
│   ├── web/                # Web dashboard
│   │   ├── app.py          # FastAPI application
│   │   ├── auth.py         # Authentication
│   │   ├── data_tracker.py # Analytics logging
│   │   ├── settings.py     # Web settings
│   │   ├── routes/         # Route handlers
│   │   │   ├── auth.py
│   │   │   ├── api.py
│   │   │   └── views.py
│   │   └── templates/      # Jinja2 templates
│   │       ├── base.html
│   │       ├── dashboard.html
│   │       ├── posts.html
│   │       ├── analytics.html
│   │       ├── settings.html
│   │       ├── chat.html
│   │       ├── login.html
│   │       └── partials/
│   ├── x/                  # X/Twitter automation
│   │   ├── driver.py       # Browser driver creation
│   │   ├── auth.py         # Cookie-based authentication
│   │   ├── posting.py      # Tweet posting
│   │   ├── reading.py      # Frontpage reading
│   │   ├── notifications.py # Notification extraction
│   │   ├── replies.py      # Reply handling
│   │   ├── parser.py       # HTML parsing utilities
│   │   └── session.py      # Async session management
│   └── monitoring/         # Logging and monitoring
│       ├── logging_config.py   # Structured logging setup
│       └── token_logging.py    # Token usage tracking
├── tests/                  # Test suite
│   ├── integration/
│   └── *.py
├── data/                   # Data storage
│   ├── state.json          # Runtime state
│   ├── bot.db              # SQLite analytics
│   └── chroma/             # Vector database
└── logs/                   # Log files
    └── bot.log
```

## Build and Run Commands

### Setup

```bash
# Install dependencies (using uv)
uv sync

# Or install with dev dependencies
uv sync --group dev

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/Mac
```

### Configuration

```bash
# Copy environment template and fill in values
copy config\env.template .env  # Windows
cp config/env.template .env    # Unix/Mac

# Required environment variables:
# - TWITTER_USERNAME
# - TWITTER_PASSWORD
# - At least one LLM API key (OPENAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)
```

### Running the Bot

```bash
# Run the main bot (scheduler mode)
python main.py

# Run the web dashboard only
python -m src.web.app

# Or with uv
uv run python main.py
```

### Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_evaluation.py

# Run with coverage
pytest --cov=src
```

## Code Style Guidelines

### Python Standards

- **Python Version**: 3.13+ with modern syntax (`|` union syntax)
- **Type Hints**: Required on ALL function signatures
- **Formatter**: Black with 100 character line length
- **Models**: Pydantic v2 for all data structures
- **Async**: Use `async/await` for ALL I/O operations
- **Logging**: Structured logging with context (never f-strings)

### Key Patterns

```python
# Type hints with | syntax (not typing.Union)
def process(data: str | None) -> dict[str, Any]: ...

# Pydantic models for data
class AgentState(BaseModel):
    counters: dict[str, int] = Field(default_factory=dict)
    last_post_time: datetime | None = Field(default=None)

# Structured logging (not f-strings)
logger.info("tweet_posted", tweet_id=tweet_id, length=len(text))

# Async I/O
async def fetch_data() -> Data:
    async with aiohttp.ClientSession() as session:
        ...
```

### What NOT to do

```python
# ❌ Don't use typing.Union
def bad(x: Union[str, int]): ...

# ❌ Don't skip type hints
def bad_func(x): ...

# ❌ Don't use f-strings in logs
logger.info(f"Tweet {id} posted")  # Bad!

# ❌ Don't use sync I/O when async available
```

## Architecture Deep Dive

### Configuration System

Configuration is split into two tiers:

1. **Environment Variables** (`.env`): Secrets and API keys
2. **YAML Config** (`config/config.yaml`): Runtime settings

**Key Config Sections:**
- `rate_limits`: Daily posting limits, reset time
- `selenium`: Browser automation settings
- `llm`: Provider, model, fallback, embeddings
- `scheduler`: Job intervals and jitter
- `personality`: Tone, style, topics, length limits

### State Management

Dual-storage approach:

1. **JSON State** (`data/state.json`): Runtime state, counters, queues
   - Atomic writes (temp file → rename)
   - Async lock-protected access
   - Auto-reset counters at UTC midnight

2. **SQLite Database** (`data/bot.db`): Historical data, analytics
   - Tables: `read_posts`, `written_tweets`, `rejected_tweets`, `token_usage`
   - Used for dashboard analytics

### LLM Integration

Multi-provider with automatic fallback:

```python
# Provider priority: config.llm.provider -> fallback_providers
providers = ["openrouter", "openai", "google", "anthropic"]

# Usage
llm_client = LLMClient(config, env_settings)
result = await llm_client.generate_tweet(system_prompt)
```

**Embedding Strategy:**
- Uses configured `embedding_provider` (default: OpenAI)
- ChromaDB caches embeddings by content hash
- Similarity threshold for duplicate detection (default: 0.85)

### Scheduler System

APScheduler BackgroundScheduler with:

- **Posting Job**: Runs every `post_interval_hours` ± `post_jitter_hours`
- **Reading Job**: Runs every `reading_check_minutes`
- **Notifications Job**: Runs every `mention_check_minutes`
- **Inspiration Job**: Runs every `inspiration_check_minutes`
- **Replies Job**: Runs every `mention_check_minutes`

**Job Execution:**
- Global lock prevents parallel execution
- Jobs queue when lock is busy
- Config reload without restart via control server

### Browser Automation

Uses undetected-chromedriver with stealth:

```python
# Key features:
- Cookie persistence (config/cookie.json)
- Random delays between actions
- User-Agent rotation (optional)
- Human-like mouse movements
```

**Important:** Never use standard Selenium - always use `undetected_chromedriver`.

### LangGraph Workflows

Tweet generation uses LangGraph state machines:

1. **Generate**: Create initial tweet
2. **Validate**: Check length, quality
3. **Re-evaluate**: LLM quality gate
4. **Check Duplicate**: ChromaDB similarity search
5. **Post**: Selenium automation

## Testing Instructions

### Test Structure

```
tests/
├── test_evaluation.py      # Tweet evaluation tests
├── test_read_post.py       # Post reading tests
├── test_chat_api.py        # Chat API tests
└── integration/
    └── test_auth.py        # Authentication integration tests
```

### Writing Tests

```python
import pytest

# Use pytest-asyncio for async tests
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is True

# Use fixtures for dependencies
@pytest.fixture
def config():
    return BotConfig()

# Fake/Mock pattern for LLM tests
class FakeLLMClient:
    def __init__(self, responses):
        self._responses = responses
    
    async def chat(self, **kwargs) -> ChatResult:
        return ChatResult(content=self._responses["openai"])
```

### Running Tests

```bash
# All tests
pytest

# With async auto mode (configured in pyproject.toml)
pytest tests/test_evaluation.py

# Specific test
pytest tests/test_evaluation.py::test_re_evaluate_tweet_approves
```

## Security Considerations

### Environment Variables

**NEVER commit `.env` file.** Required secrets:

- `TWITTER_USERNAME` / `TWITTER_PASSWORD`: X credentials
- `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY`: At least one required
- `AUTH_SECRET_KEY`: For web dashboard sessions
- `AUTH_PASSWORD_HASH`: Argon2 hash for dashboard login

### Dashboard Authentication

Dashboard uses session-based auth:
- Argon2 password hashing
- Signed session cookies
- Configurable session timeout

### Rate Limiting

- Daily post/reply limits enforced in code
- Counter reset at configurable UTC time
- Rate limit checks before every action

### Compliance

- Bot disclosure in bio required
- No auto-posting of trending topics
- Opt-out mechanism for replies
- Content filtering for toxicity

## Common Development Tasks

### Adding a New Job

1. Create job function in `src/scheduler/jobs/`
2. Add setup method in `src/scheduler/bot_scheduler.py`
3. Register in `main.py` with wrapper

### Adding a New LLM Provider

1. Add to `src/core/langchain_clients.py`
2. Update `LLMConfig` allowed values
3. Add provider setup in `LangChainLLM._create_client()`

### Adding Dashboard Endpoint

1. Add API route in `src/web/routes/api.py`
2. Add view route in `src/web/routes/views.py` (if needed)
3. Create template in `src/web/templates/`
4. Add HTMX partial in `src/web/templates/partials/` (if needed)

### Modifying State Model

1. Update `src/state/models.py`
2. Pydantic handles migration (default values for new fields)
3. Update state manager if logic changes needed

## Troubleshooting

### Browser Issues

- Chrome/Edge must be installed
- Check `config.selenium.headless` setting
- Clear `config/cookie.json` if auth issues

### LLM Failures

- Check API keys in `.env`
- Verify provider status
- Fallback should activate automatically

### Database Lock Errors

- SQLite doesn't support concurrent writes well
- Ensure only one bot instance running
- Check for zombie Python processes

### Scheduler Not Running

- Check logs in `logs/bot.log`
- Verify config intervals are reasonable
- Ensure `SELENIUM_HEADLESS` setting for server environments

## Resources

- **Roadmap**: `ROADMAP.md` - Development progress and future plans
- **Coding Rules**: `.cursor/rules/project-rules.mdc` - Detailed coding standards
- **Config Template**: `config/env.template` - Environment variables reference
