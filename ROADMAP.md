# X Bot - Development Roadmap

**Current Phase:** MVP Development - Phase 3 (Notification Checking Implemented)

## 📊 Current Status Overview

### ✅ Implemented (Phase 0 - Foundation)

#### Core Infrastructure
- ✅ **Configuration System** (`src/core/config.py`)
  - YAML-based configuration (`config/config.yaml`) with Pydantic validation
  - Rate limits, LLM, Scheduler, Personality configs
  - Environment variable support

- ✅ **LLM Integration** (`src/core/llm.py`)
  - Multi-provider support (OpenAI, OpenRouter)
  - Automatic fallback mechanism
  - Tweet generation and validation
  - Brand alignment checking
  - Note: Google and Anthropic providers not yet implemented (only OpenAI and OpenRouter)

- ✅ **State Management** (`src/state/`)
  - JSON-based state persistence (`data/state.json`)
  - Pydantic models for validation (Post, Notification)
  - Atomic writes (temp file → rename)
  - Counters for rate limiting
  - Interesting posts queue (max 50 posts)
  - Notifications queue (max 50 notifications)
  - Processed notification IDs tracking (max 100 IDs)

#### Twitter/X Automation
- ✅ **Browser Driver** (`src/x/driver.py`)
  - undetected-chromedriver with stealth mode
  - User-Agent rotation
  - Human-like delays

- ✅ **Authentication** (`src/x/auth.py`)
  - Cookie-based login persistence
  - Automatic login fallback
  - Cookie save/load functionality

- ✅ **Posting** (`src/x/posting.py`)
  - Tweet posting via Selenium
  - Multiple selector fallbacks
  - Human-like typing simulation

- ✅ **Reading** (`src/x/reading.py`)
  - Frontpage post reading and extraction
  - Post content, author info (username/display_name), engagement metrics
  - Timestamp and URL extraction
  - Dynamic content loading with scrolling
  - Duplicate post filtering
  - Post type detection (text-only, media-only, retweets, quoted tweets)
  - Skip non-text posts (media-only, retweets without text)

- ✅ **Notification Checking** (`src/x/notifications.py`)
  - Notification page navigation and extraction
  - Reply and mention detection
  - Notification content and metadata extraction
  - Duplicate filtering by notification ID
  - Notification queue management (max 50 notifications)
  - Processed notification IDs tracking

- ✅ **Interest Detection** (`src/core/interest.py`)
  - LLM-based post evaluation against bot personality
  - Match/No Match decision with structured logging
  - Retry logic with tenacity for robustness
  - Token usage tracking for analytics
  - Comprehensive test coverage

#### Main Loop
- ✅ **Basic Bot Flow** (`main.py`)
  - Tweet generation → validation → posting
  - Rate limit checks
  - State updates

- ✅ **Scheduler System** (`src/scheduler/`)
  - APScheduler BackgroundScheduler integration
  - Automated periodic job execution
  - Job failure handling and logging
  - Concurrent execution prevention (`max_instances=1`)
  - Comprehensive test coverage

---

## 🎯 MVP Goals (Phase 1)

### High Priority - Core Features

#### 1. Scheduler System (`src/scheduler/`) ✅ **COMPLETED**
**Status:** ✅ Completed  
**Priority:** Critical  
**Estimated Effort:** 2-3 days (Completed)

**Tasks:**
- [x] Create `src/scheduler/bot_scheduler.py`
- [x] Implement APScheduler BackgroundScheduler
- [x] Add job: `post_autonomous_tweet()` (every 8h with jitter)
- [x] Add job: `read_frontpage_posts()` (every 30min) - Stub implemented
- [x] Add job: `check_notifications()` (every 30min) - Stub implemented
- [x] Implement graceful job failure handling
- [x] Add structured logging for scheduled tasks
- [x] Add `max_instances=1` to prevent concurrent execution
- [x] Add `coalesce=True` for pending job combination
- [x] Create comprehensive test suite (`tests/test_scheduler.py`, `tests/test_jobs.py`)

**Dependencies:**
- ✅ `apscheduler` package
- ✅ Existing `main.py` logic refactoring

---

#### 2. Frontpage Reading (`src/x/reading.py`) ✅ **COMPLETED**
**Status:** ✅ Completed  
**Priority:** Critical  
**Estimated Effort:** 2-3 days (Completed)

**Tasks:**
- [x] Create `src/x/reading.py`
- [x] Implement `read_frontpage_posts(driver, count=10)` function
- [x] Extract post content (text, author username/display_name, engagement metrics)
- [x] Handle dynamic content loading (scroll, wait for elements)
- [x] Create `Post` Pydantic model for structured data
- [x] Add error handling and retry logic
- [x] Extract post timestamps from time elements
- [x] Extract post URLs (filtering out analytics links)
- [x] Implement duplicate post filtering by post ID
- [x] Add comprehensive test coverage
- [x] Detect different post types (text-only, media-only, retweets, quoted tweets)
- [x] Skip non-text posts (media-only, retweets without text) during reading
- [ ] Extract media information (images, videos) for future use
- [ ] Handle quoted tweets and retweets properly

**Dependencies:**
- ✅ Selenium driver
- ✅ Post data model

---

#### 3. Interest Detection (`src/core/interest.py`) ✅ **COMPLETED**
**Status:** ✅ Completed  
**Priority:** Critical  
**Estimated Effort:** 1-2 days (Completed)

**Tasks:**
- [x] Create `src/core/interest.py`
- [x] Implement `check_interest(post: Post, config: BotConfig, llm: LLMClient) -> bool`
- [x] Use LLM to evaluate if post matches personality/topics
- [x] Return Match/No Match decision
- [x] Add retry logic with tenacity
- [x] Add structured logging with token usage
- [x] Add comprehensive test coverage (`tests/test_interest.py`)
- [x] Integrate into scheduler reading job
- [x] Add `is_interesting` flag to Post model
- [x] Create interesting posts queue in AgentState
- [x] Add structured logging for dashboard analytics
- [ ] Add confidence scoring (optional, future enhancement)

**Dependencies:**
- LLM Client
- Post model
- Personality config

---

#### 4. Inspiration-based Posting (`src/scheduler/jobs.py`) ✅ **COMPLETED**
**Status:** ✅ Completed  
**Priority:** Critical
**Estimated Effort:** 2-3 days (Completed)

**Tasks:**
- [x] Create `INSPIRATION_TWEET_PROMPT` in `src/core/prompts.py`
- [x] Implement `generate_inspiration_tweet` in `src/core/llm.py`
- [x] Create `process_inspiration_queue` job in `src/scheduler/jobs.py`
- [x] Implement threshold logic (default: 10 posts)
- [x] Integrate with main scheduler
- [x] Add configuration options (`inspiration_batch_size`, `inspiration_check_minutes`)
- [x] Add comprehensive test coverage (`tests/test_inspiration.py`)
- [x] Fix function signature mismatch (wrapper function created)
- [x] Fix login handling (proper `env_settings` parameter handling)
- [x] Add rate limit check before posting
- [x] Add system prompt to inspiration tweet generation

**Dependencies:**
- ✅ Interest detection
- ✅ LLM Client
- ✅ Posting infrastructure

---

#### 5. Notification Checking (`src/x/notifications.py`) ✅ **COMPLETED**
**Status:** ✅ Completed  
**Priority:** High  
**Estimated Effort:** 2-3 days (Completed)

**Tasks:**
- [x] Create `src/x/notifications.py`
- [x] Implement `check_notifications(driver, config, count=20) -> list[Notification]`
- [x] Navigate to notifications page (`https://x.com/notifications`)
- [x] Extract reply content and metadata
- [x] Create `Notification` Pydantic model in `src/state/models.py`
- [x] Filter for replies vs. other notifications (replies and mentions only)
- [x] Create `NotificationParser` class in `src/x/parser.py`
- [x] Implement duplicate filtering by notification ID
- [x] Integrate into scheduler job (`_check_notifications_async`)
- [x] Store notifications in state queue (`notifications_queue`)
- [x] Track processed notification IDs to avoid duplicates
- [x] Update `last_notification_check_time` in state

**Dependencies:**
- ✅ Selenium driver
- ✅ Notification data model

---

#### 6. Positive Intent Detection (`src/core/intent.py`) 🟡 **MEDIUM**
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 1-2 days

**Tasks:**
- [ ] Create `src/core/intent.py`
- [ ] Implement `check_positive_intent(reply: Reply, llm: LLMClient) -> bool`
- [ ] Use LLM to classify reply sentiment/intent
- [ ] Filter negative/toxicity
- [ ] Return True for positive/constructive replies

**Dependencies:**
- LLM Client
- Reply/Notification models

---

### Medium Priority - Enhancement Features

#### 7. Memory/Chroma Integration (`src/memory/`) 🟡 **MEDIUM**
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 3-4 days

**Tasks:**
- [ ] Create `src/memory/chroma_client.py`
- [ ] Initialize ChromaDB with persistent storage
- [ ] Implement embedding storage for all posts/replies
- [ ] Implement `check_duplicate(text: str, config: BotConfig) -> bool`
- [ ] Use similarity threshold from config
- [ ] Store metadata (timestamp, engagement, topics)
- [ ] Create topic-based collections

**Dependencies:**
- ChromaDB package
- LLM embedding provider
- Config similarity threshold

---

#### 8. Trends Checking (`src/ingest/trends.py`) 🟢 **LOW**
**Status:** ❌ Not Started  
**Priority:** Low (Post-MVP)  
**Estimated Effort:** 2-3 days

**Tasks:**
- [ ] Create `src/ingest/trends.py`
- [ ] Implement trend detection from Twitter/X
- [ ] Filter trending topics (compliance check)
- [ ] Integrate with interest detection flow

**Dependencies:**
- Interest detection
- Compliance checks

---

#### 9. Token Counter & Analytics (`src/core/analytics.py`) 🟡 **MEDIUM**
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 1-2 days

**Tasks:**
- [ ] Create `src/core/analytics.py`
- [ ] Implement token counting for all LLM calls
- [ ] Track token usage per provider (OpenAI, OpenRouter, etc.)
- [ ] Store token metrics in state/database
- [ ] Track costs per provider (if available)
- [ ] Add token usage logging with structured context

**Dependencies:**
- LLM Client
- State management or database

---

#### 10. Tweet Re-Evaluation Before Posting (`src/core/evaluation.py`) 🟡 **MEDIUM**
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 1-2 days

**Tasks:**
- [ ] Create `src/core/evaluation.py`
- [ ] Implement `re_evaluate_tweet(tweet_text: str, config: BotConfig, llm: LLMClient) -> tuple[bool, str]`
- [ ] Use LLM to evaluate tweet against personality/topics
- [ ] Check if tweet is worth posting (quality, relevance, alignment)
- [ ] Return evaluation result (pass/fail) and reasoning
- [ ] Integrate into posting flow before final submission
- [ ] Log failed evaluations for dashboard analytics

**Dependencies:**
- LLM Client
- Personality config
- Posting infrastructure

---

#### 11. Web Dashboard (`src/web/`) 🟡 **MEDIUM**
**Status:** ❌ Not Started  
**Priority:** Medium  
**Estimated Effort:** 3-4 days

**Tasks:**
- [ ] Create `src/web/` directory structure
- [ ] Set up FastAPI application (`src/web/app.py`)
- [ ] Create API endpoints for dashboard data
- [ ] Implement dashboard frontend (HTML/JS or React)
- [ ] Add endpoint: `/api/posts/read` - Last read tweets overview
- [ ] Add endpoint: `/api/posts/written` - Last written/posted tweets
- [ ] Add endpoint: `/api/posts/rejected` - Tweets that didn't pass evaluation
- [ ] Add endpoint: `/api/analytics/tokens` - Token usage analytics
- [ ] Add endpoint: `/api/state` - Current bot state
- [ ] Integrate with existing state management
- [ ] Add authentication/security for dashboard access

**Dependencies:**
- FastAPI package
- Token counter/analytics
- Tweet re-evaluation (for rejected tweets tracking)
- State management

---

## 📅 Implementation Phases

### Phase 1: Scheduler & Reading (Week 1)
**Goal:** Enable automated periodic tasks

1. ✅ Scheduler System
2. ✅ Frontpage Reading
3. ✅ Basic integration with main loop

**Deliverable:** Bot can read posts on schedule ✅ Complete

---

### Phase 2: Interest & Reactions (Week 1-2)
**Goal:** Enable intelligent post reactions

1. ✅ Interest Detection (module complete, integrated into reading job)
2. ✅ Inspiration-based Posting (queue processing implemented)
3. ✅ Integration with scheduler (reading and inspiration jobs integrated)

**Deliverable:** Bot can generate inspired content from interesting posts automatically

---

### Phase 3: Notifications & Intent (Week 2)
**Goal:** Enable reply handling

1. ✅ Notification Checking (implemented, queue ready for processing)
2. ❌ Positive Intent Detection
3. ❌ Reply Generation & Posting

**Deliverable:** Bot can check notifications and queue them for processing ✅ Partial

---

### Phase 4: Memory & Enhancement (Week 2-3)
**Goal:** Add memory and duplicate detection

1. ✅ ChromaDB Integration
2. ✅ Embedding storage
3. ✅ Duplicate detection

**Deliverable:** Bot remembers past interactions, avoids duplicates

---

## 🔄 Flowchart Implementation Status

| Flowchart Step | Status | Implementation |
|----------------|--------|----------------|
| **Start** | ✅ Done | `main.py` |
| **Random Read X posts** | ✅ Done | `src/x/reading.py` |
| **Interest Check** | ✅ Done | `src/core/interest.py` |
| **Inspiration Posting** | ✅ Done | `src/scheduler/jobs.py` (Batch processing) |
| **Check Replies/Notifications** | ✅ Done | `src/x/notifications.py` |
| **Positive Intent Check** | ❌ Missing | `src/core/intent.py` |
| **Reply based on Personality** | ❌ Missing | `src/x/reactions.py` |
| **Personality Integration** | ⚠️ Partial | Config exists, not in reactions |
| **Check Trends** | ❌ Missing | `src/ingest/trends.py` |
| **Scheduler** | ✅ Done | `src/scheduler/bot_scheduler.py` |

---

## 📝 Notes

### Current Limitations
- ✅ Automated scheduling implemented (scheduler system complete)
- ✅ Post reading/scanning capability implemented
- ✅ Post type detection implemented (skips non-text posts during reading)
- ✅ Interest detection integrated into reading job
- ✅ Interesting posts queue implemented (max 50 posts)
- ✅ Structured logging for dashboard analytics
- ✅ Reading job has dedicated `reading_check_minutes` config field
- ✅ Notification checking implemented (replies and mentions queued for processing)
- ✅ Notifications queue implemented (max 50 notifications)
- ✅ Processed notification IDs tracking (max 100 IDs)
- No reaction/reply functionality (notifications queue ready for processing)
- No positive intent detection (notifications queue ready for processing)
- No memory/duplicate detection
- Media-only posts are skipped during reading (no text to evaluate) - working as intended
- Retweets without text are skipped during reading - working as intended

### Technical Debt
- ✅ Main loop refactored for scheduler integration
- ✅ Inspiration queue processing bugs fixed (wrapper function and login handling)
- LLM client needs embedding support (currently only generation)
- State management needs counter reset logic (midnight UTC)
- Error handling improved with graceful job failure handling
- Token counting not yet implemented (needed for analytics)
- Tweet re-evaluation before posting not implemented

### Future Enhancements (Post-MVP)
- RSS feed ingestion (`src/ingest/rss.py`)
- Event detection (`src/ingest/events.py`)
- Advanced monitoring (`src/monitoring/`)
- SQLite/PostgreSQL migration for state
- Web dashboard for monitoring (FastAPI)
- Multi-account support

---

## 🎯 Success Criteria for MVP

- [x] Bot runs continuously with scheduler
- [x] Bot reads frontpage posts periodically
- [x] Bot identifies interesting posts (interest detection integrated, queue implemented)
- [x] Bot generates inspired content from interesting posts (batch processing implemented)
- [x] Bot checks for notifications periodically (notifications checking implemented, queue ready for processing)
- [ ] Bot replies to positive notifications (notifications queue ready, needs intent detection)
- [ ] Bot avoids duplicate posts (memory integration)
- [x] All rate limits enforced
- [x] All compliance checks in place

---

## 📚 Related Documentation

- See `.cursor/rules/project-rules.mdc` for coding standards
- See `config/config.yaml` for configuration options (YAML format)
- See `README.md` for setup instructions

