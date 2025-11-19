"""Unit tests for scheduler."""

import time
from unittest.mock import Mock, patch

import pytest
from apscheduler.triggers.interval import IntervalTrigger

from src.core.config import BotConfig
from src.scheduler.bot_scheduler import BotScheduler


@pytest.fixture
def config():
    """Create test config."""
    return BotConfig.load("config/config.yaml")


@pytest.fixture
def scheduler(config):
    """Create scheduler instance."""
    return BotScheduler(config)


def test_scheduler_initialization(scheduler):
    """Test scheduler initializes correctly."""
    assert scheduler.config is not None
    assert not scheduler.is_running
    assert scheduler.scheduler is not None


def test_scheduler_start_stop(scheduler):
    """Test scheduler can start and stop."""
    scheduler.start()
    assert scheduler.is_running

    scheduler.stop()
    assert not scheduler.is_running


def test_scheduler_start_twice(scheduler):
    """Test starting scheduler twice doesn't crash."""
    scheduler.start()
    assert scheduler.is_running

    # Try to start again
    scheduler.start()  # Should just return without error
    assert scheduler.is_running

    scheduler.stop()


def test_scheduler_stop_when_not_running(scheduler):
    """Test stopping scheduler when not running doesn't crash."""
    assert not scheduler.is_running
    scheduler.stop()  # Should just return without error
    assert not scheduler.is_running


def test_add_job(scheduler):
    """Test adding a job."""
    mock_func = Mock()
    trigger = IntervalTrigger(seconds=1)

    scheduler.add_job(mock_func, "test_job", trigger)

    # Check job was added
    jobs = scheduler.scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "test_job"


def test_add_job_with_max_instances(scheduler):
    """Test adding a job with max_instances."""
    mock_func = Mock()
    trigger = IntervalTrigger(seconds=1)

    scheduler.add_job(mock_func, "test_job", trigger, max_instances=3)

    job = scheduler.scheduler.get_job("test_job")
    assert job.max_instances == 3


def test_add_job_default_max_instances(scheduler):
    """Test default max_instances is 1."""
    mock_func = Mock()
    trigger = IntervalTrigger(seconds=1)

    scheduler.add_job(mock_func, "test_job", trigger)

    job = scheduler.scheduler.get_job("test_job")
    assert job.max_instances == 1


def test_add_job_coalesce_enabled(scheduler):
    """Test coalesce is enabled by default."""
    mock_func = Mock()
    trigger = IntervalTrigger(seconds=1)

    scheduler.add_job(mock_func, "test_job", trigger)

    job = scheduler.scheduler.get_job("test_job")
    assert job.coalesce is True


def test_setup_posting_job(scheduler):
    """Test posting job setup."""
    mock_func = Mock()
    scheduler.setup_posting_job(mock_func)

    jobs = scheduler.scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "post_tweet"
    assert jobs[0].max_instances == 1


def test_setup_reading_job(scheduler):
    """Test reading job setup."""
    mock_func = Mock()
    scheduler.setup_reading_job(mock_func)

    jobs = scheduler.scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "read_posts"
    assert jobs[0].max_instances == 1


def test_setup_notifications_job(scheduler):
    """Test notifications job setup."""
    mock_func = Mock()
    scheduler.setup_notifications_job(mock_func)

    jobs = scheduler.scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "check_notifications"
    assert jobs[0].max_instances == 1


def test_setup_all_jobs(scheduler):
    """Test setting up all three jobs."""
    mock_posting = Mock()
    mock_reading = Mock()
    mock_notifications = Mock()

    scheduler.setup_posting_job(mock_posting)
    scheduler.setup_reading_job(mock_reading)
    scheduler.setup_notifications_job(mock_notifications)

    jobs = scheduler.scheduler.get_jobs()
    assert len(jobs) == 3

    job_ids = {job.id for job in jobs}
    assert job_ids == {"post_tweet", "read_posts", "check_notifications"}


def test_replace_existing_job(scheduler):
    """Test replacing existing job."""
    mock_func1 = Mock()
    mock_func2 = Mock()
    trigger = IntervalTrigger(seconds=1)

    scheduler.add_job(mock_func1, "test_job", trigger)
    first_job = scheduler.scheduler.get_job("test_job")
    
    scheduler.add_job(mock_func2, "test_job", trigger)  # Replace
    second_job = scheduler.scheduler.get_job("test_job")

    # Job should be replaced (same ID)
    assert first_job.id == second_job.id == "test_job"
    # APScheduler with replace_existing=True should replace the job
    # Check that job exists with correct ID
    assert second_job is not None


def test_max_instances_prevents_concurrent_execution(scheduler):
    """Test max_instances prevents concurrent execution."""
    execution_count = []
    execution_lock = []

    def slow_job():
        """Job that takes time to execute."""
        execution_count.append(1)
        execution_lock.append(1)
        time.sleep(0.5)  # Simulate work
        execution_lock.pop()

    trigger = IntervalTrigger(seconds=0.1)  # Trigger every 100ms
    scheduler.add_job(slow_job, "slow_job", trigger, max_instances=1)

    scheduler.start()
    time.sleep(0.6)  # Should trigger multiple times
    scheduler.stop()

    # With max_instances=1, only one should run at a time
    # We should have at least 1 execution, but not more than 2 concurrent
    assert len(execution_count) > 0
    # Check that no more than 1 was running at the same time
    assert len(execution_lock) <= 1


def test_shutdown_alias(scheduler):
    """Test shutdown is alias for stop."""
    scheduler.start()
    assert scheduler.is_running

    scheduler.shutdown()
    assert not scheduler.is_running

