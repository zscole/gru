"""Tests for scheduler."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gru.db import Database
from gru.scheduler import QueuedTask, Scheduler


@pytest.fixture
async def db():
    """Create temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database = Database(db_path)
        await database.connect()
        yield database
        await database.close()


@pytest.fixture
def scheduler(db):
    """Create scheduler."""
    return Scheduler(db, max_concurrent=3, starvation_threshold=5)


@pytest.mark.asyncio
async def test_enqueue(scheduler):
    """Test enqueueing a task."""
    await scheduler.enqueue("task1", "agent1", "normal")

    assert scheduler.queue_length == 1


@pytest.mark.asyncio
async def test_dequeue(scheduler):
    """Test dequeuing a task."""
    await scheduler.enqueue("task1", "agent1", "normal")

    task = await scheduler.dequeue()

    assert task is not None
    assert task.task_id == "task1"
    assert task.agent_id == "agent1"
    assert scheduler.queue_length == 0


@pytest.mark.asyncio
async def test_dequeue_empty(scheduler):
    """Test dequeue from empty queue."""
    task = await scheduler.dequeue()
    assert task is None


@pytest.mark.asyncio
async def test_priority_ordering(scheduler):
    """Test high priority tasks dequeue first."""
    await scheduler.enqueue("low_task", "agent1", "low")
    await scheduler.enqueue("high_task", "agent2", "high")
    await scheduler.enqueue("normal_task", "agent3", "normal")

    task1 = await scheduler.dequeue()
    task2 = await scheduler.dequeue()
    task3 = await scheduler.dequeue()

    assert task1.task_id == "high_task"
    assert task2.task_id == "normal_task"
    assert task3.task_id == "low_task"


@pytest.mark.asyncio
async def test_fifo_within_priority(scheduler):
    """Test FIFO ordering within same priority."""
    await scheduler.enqueue("task1", "agent1", "normal")
    await scheduler.enqueue("task2", "agent2", "normal")
    await scheduler.enqueue("task3", "agent3", "normal")

    task1 = await scheduler.dequeue()
    task2 = await scheduler.dequeue()
    task3 = await scheduler.dequeue()

    assert task1.task_id == "task1"
    assert task2.task_id == "task2"
    assert task3.task_id == "task3"


@pytest.mark.asyncio
async def test_cancel(scheduler):
    """Test cancelling a queued task."""
    await scheduler.enqueue("task1", "agent1", "normal")
    await scheduler.enqueue("task2", "agent2", "normal")

    success = await scheduler.cancel("task1")

    assert success
    assert scheduler.queue_length == 1

    task = await scheduler.dequeue()
    assert task.task_id == "task2"


@pytest.mark.asyncio
async def test_cancel_nonexistent(scheduler):
    """Test cancelling nonexistent task."""
    success = await scheduler.cancel("nonexistent")
    assert not success


@pytest.mark.asyncio
async def test_prevent_starvation(scheduler):
    """Test starvation prevention boosts low priority tasks."""
    await scheduler.enqueue("low_task", "agent1", "low")

    # Simulate waiting
    for _ in range(10):
        await scheduler.prevent_starvation()

    # Low task should have boosted priority now
    task = scheduler._queue[0]
    assert task.wait_cycles >= 10
    # Priority score should be more negative (higher priority)
    assert task.priority_score < -100


@pytest.mark.asyncio
async def test_starvation_max_boost(scheduler):
    """Test starvation boost is capped."""
    await scheduler.enqueue("low_task", "agent1", "low")

    # Many cycles
    for _ in range(100):
        await scheduler.prevent_starvation()

    task = scheduler._queue[0]
    # Should not exceed max boost
    assert task.priority_score >= -(scheduler.MAX_BOOST + 100 + 10)


@pytest.mark.asyncio
async def test_register_running(scheduler):
    """Test registering a running task."""
    mock_task = MagicMock()
    scheduler.register_running("task1", mock_task)

    assert scheduler.running_count == 1
    assert scheduler.get_running_task("task1") == mock_task


@pytest.mark.asyncio
async def test_unregister_running(scheduler):
    """Test unregistering a completed task."""
    mock_task = MagicMock()
    scheduler.register_running("task1", mock_task)
    scheduler.unregister_running("task1")

    assert scheduler.running_count == 0
    assert scheduler.get_running_task("task1") is None


@pytest.mark.asyncio
async def test_can_run_more(scheduler):
    """Test concurrent limit check."""
    assert scheduler.can_run_more()

    # Fill up to max
    for i in range(3):
        scheduler.register_running(f"task{i}", MagicMock())

    assert not scheduler.can_run_more()

    # Free one slot
    scheduler.unregister_running("task0")
    assert scheduler.can_run_more()


@pytest.mark.asyncio
async def test_get_status(scheduler):
    """Test getting scheduler status."""
    await scheduler.enqueue("task1", "agent1", "high")
    await scheduler.enqueue("task2", "agent2", "normal")
    scheduler.register_running("task3", MagicMock())

    status = await scheduler.get_status()

    assert status["running"] == 1
    assert status["queued"] == 2
    assert status["max_concurrent"] == 3
    assert "task3" in status["running_tasks"]
    assert len(status["queued_tasks"]) == 2


@pytest.mark.asyncio
async def test_concurrent_enqueue(scheduler):
    """Test concurrent enqueue operations."""

    async def enqueue_task(task_id):
        await scheduler.enqueue(task_id, f"agent_{task_id}", "normal")

    # Enqueue concurrently
    await asyncio.gather(*[enqueue_task(f"task{i}") for i in range(10)])

    assert scheduler.queue_length == 10


@pytest.mark.asyncio
async def test_concurrent_dequeue(scheduler):
    """Test concurrent dequeue operations."""
    for i in range(10):
        await scheduler.enqueue(f"task{i}", f"agent{i}", "normal")

    async def dequeue_task():
        return await scheduler.dequeue()

    # Dequeue concurrently
    results = await asyncio.gather(*[dequeue_task() for _ in range(10)])

    # All should succeed, no duplicates
    task_ids = [r.task_id for r in results if r]
    assert len(task_ids) == 10
    assert len(set(task_ids)) == 10


def test_queued_task_ordering():
    """Test QueuedTask comparison for sorting."""
    from datetime import datetime

    t1 = QueuedTask(
        priority_score=-100,
        queued_at=datetime(2024, 1, 1, 12, 0, 0),
        task_id="task1",
        agent_id="agent1",
    )
    t2 = QueuedTask(
        priority_score=-50,
        queued_at=datetime(2024, 1, 1, 12, 0, 0),
        task_id="task2",
        agent_id="agent2",
    )
    t3 = QueuedTask(
        priority_score=-100,
        queued_at=datetime(2024, 1, 1, 12, 0, 1),
        task_id="task3",
        agent_id="agent3",
    )

    # Lower score = higher priority
    assert t1 < t2

    # Same score, earlier time = higher priority
    assert t1 < t3

    # Sorted order
    tasks = sorted([t2, t3, t1])
    assert tasks[0].task_id == "task1"
    assert tasks[1].task_id == "task3"
    assert tasks[2].task_id == "task2"
