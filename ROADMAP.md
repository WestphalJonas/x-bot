# X Bot - Development Roadmap

**Last Updated:** 2025-01-27  
**Current Phase:** MVP Development - Phase 1 (Scheduler Complete)

## 📊 Current Status Overview

### ✅ Implemented (Phase 0 - Foundation)

#### Core Infrastructure
- ✅ **Configuration System** (`src/core/config.py`)
  - YAML-based configuration with Pydantic validation
  - Rate limits, LLM, Scheduler, Personality configs
  - Environment variable support

- ✅ **LLM Integration** (`src/core/llm.py`)
  - Multi-provider support (OpenAI, OpenRouter)
  - Automatic fallback mechanism
  - Tweet generation and validation
  - Brand alignment checking

- ✅ **State Management** (`src/state/`)
  - JSON-based state persistence
  - Pydantic models for validation
  - Atomic writes (temp file → rename)
  - Counters for rate limiting

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

#### 2. Frontpage Reading (`src/x/reading.py`) 🔴 **HIGH**
**Status:** ❌ Not Started  
**Priority:** Critical  
**Estimated Effort:** 2-3 days

**Tasks:**
- [ ] Create `src/x/reading.py`
- [ ] Implement `read_frontpage_posts(driver, count=10)` function
- [ ] Extract post content (text, author, engagement metrics)
- [ ] Handle dynamic content loading (scroll, wait for elements)
- [ ] Create `Post` Pydantic model for structured data
- [ ] Add error handling and retry logic

**Dependencies:**
- Selenium driver
- Post data model

---

#### 3. Interest Detection (`src/core/interest.py`) 🔴 **HIGH**
**Status:** ❌ Not Started  
**Priority:** Critical  
**Estimated Effort:** 1-2 days

**Tasks:**
- [ ] Create `src/core/interest.py`
- [ ] Implement `check_interest(post: Post, config: BotConfig, llm: LLMClient) -> bool`
- [ ] Use LLM to evaluate if post matches personality/topics
- [ ] Return Match/No Match decision
- [ ] Add confidence scoring (optional)

**Dependencies:**
- LLM Client
- Post model
- Personality config

---

#### 4. Reaction Writing (`src/x/reactions.py`) 🔴 **HIGH**
**Status:** ❌ Not Started  
**Priority:** Critical  
**Estimated Effort:** 2-3 days

**Tasks:**
- [ ] Create `src/x/reactions.py`
- [ ] Implement `write_reaction(post: Post, config: BotConfig, llm: LLMClient) -> str`
- [ ] Generate reply text based on personality
- [ ] Implement `post_reply(driver, post_id: str, reply_text: str) -> bool`
- [ ] Handle reply UI elements (click reply button, type, submit)
- [ ] Add rate limiting checks

**Dependencies:**
- Interest detection
- LLM Client
- Posting infrastructure

---

#### 5. Notification Checking (`src/x/notifications.py`) 🟡 **MEDIUM**
**Status:** ❌ Not Started  
**Priority:** High  
**Estimated Effort:** 2-3 days

**Tasks:**
- [ ] Create `src/x/notifications.py`
- [ ] Implement `check_notifications(driver) -> list[Notification]`
- [ ] Navigate to notifications page
- [ ] Extract reply content and metadata
- [ ] Create `Notification` Pydantic model
- [ ] Filter for replies vs. other notifications

**Dependencies:**
- Selenium driver
- Notification data model

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

## 📅 Implementation Phases

### Phase 1: Scheduler & Reading (Week 1)
**Goal:** Enable automated periodic tasks

1. ✅ Scheduler System
2. ❌ Frontpage Reading
3. ✅ Basic integration with main loop

**Deliverable:** Bot can read posts on schedule (Scheduler ready, reading pending)

---

### Phase 2: Interest & Reactions (Week 1-2)
**Goal:** Enable intelligent post reactions

1. ✅ Interest Detection
2. ✅ Reaction Writing
3. ✅ Integration with scheduler

**Deliverable:** Bot can react to interesting posts automatically

---

### Phase 3: Notifications & Intent (Week 2)
**Goal:** Enable reply handling

1. ✅ Notification Checking
2. ✅ Positive Intent Detection
3. ✅ Reply Generation & Posting

**Deliverable:** Bot can handle and reply to notifications

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
| **Random Read X posts** | ❌ Missing | `src/x/reading.py` |
| **Interest Check** | ❌ Missing | `src/core/interest.py` |
| **Write Reaction** | ❌ Missing | `src/x/reactions.py` |
| **Check Replies/Notifications** | ❌ Missing | `src/x/notifications.py` |
| **Positive Intent Check** | ❌ Missing | `src/core/intent.py` |
| **Reply based on Personality** | ❌ Missing | `src/x/reactions.py` |
| **Personality Integration** | ⚠️ Partial | Config exists, not in reactions |
| **Check Trends** | ❌ Missing | `src/ingest/trends.py` |
| **Scheduler** | ✅ Done | `src/scheduler/bot_scheduler.py` |

---

## 📝 Notes

### Current Limitations
- ✅ Automated scheduling implemented (scheduler system complete)
- No post reading/scanning capability (stub only)
- No interest-based filtering
- No reaction/reply functionality
- No notification handling (stub only)
- No memory/duplicate detection

### Technical Debt
- ✅ Main loop refactored for scheduler integration
- LLM client needs embedding support (currently only generation)
- State management needs counter reset logic (midnight UTC)
- Error handling improved with graceful job failure handling

### Future Enhancements (Post-MVP)
- RSS feed ingestion (`src/ingest/rss.py`)
- Event detection (`src/ingest/events.py`)
- Advanced monitoring (`src/monitoring/`)
- SQLite/PostgreSQL migration for state
- Web dashboard for monitoring
- Multi-account support

---

## 🎯 Success Criteria for MVP

- [x] Bot runs continuously with scheduler
- [ ] Bot reads frontpage posts periodically (stub implemented)
- [ ] Bot identifies interesting posts (interest detection)
- [ ] Bot reacts to interesting posts automatically
- [ ] Bot checks for notifications periodically (stub implemented)
- [ ] Bot replies to positive notifications
- [ ] Bot avoids duplicate posts (memory integration)
- [x] All rate limits enforced
- [x] All compliance checks in place

---

## 📚 Related Documentation

- See `.cursor/rules/project-rules.mdc` for coding standards
- See `config/config.yaml` for configuration options
- See `README.md` for setup instructions

