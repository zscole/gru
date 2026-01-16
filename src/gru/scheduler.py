"""Priority queue scheduler with starvation prevention."""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gru.db import Database


@dataclass(order=True)
class QueuedTask:
    """Task in the priority queue."""

    priority_score: int = field(compare=True)
    queued_at: datetime = field(compare=True)
    task_id: str = field(compare=False)
    agent_id: str = field(compare=False)
    wait_cycles: int = field(default=0, compare=False)


class Scheduler:
    """Priority queue scheduler with starvation prevention."""

    PRIORITY_SCORES = {"high": 100, "normal": 50, "low": 0}
    STARVATION_BOOST = 10  # Points added per cycle for waiting tasks
    MAX_BOOST = 100  # Maximum boost from waiting

    def __init__(
        self,
        db: Database,
        max_concurrent: int = 10,
        starvation_threshold: int = 10,
    ) -> None:
        self.db = db
        self.max_concurrent = max_concurrent
        self.starvation_threshold = starvation_threshold
        self._running: dict[str, asyncio.Task] = {}
        self._queue: list[QueuedTask] = []
        self._lock = asyncio.Lock()

    @property
    def running_count(self) -> int:
        """Number of currently running tasks."""
        return len(self._running)

    @property
    def queue_length(self) -> int:
        """Number of queued tasks."""
        return len(self._queue)

    async def enqueue(
        self,
        task_id: str,
        agent_id: str,
        priority: str = "normal",
    ) -> None:
        """Add a task to the queue."""
        async with self._lock:
            score = self.PRIORITY_SCORES.get(priority, 50)
            queued_task = QueuedTask(
                priority_score=-score,  # Negative for min-heap behavior
                queued_at=datetime.now(),
                task_id=task_id,
                agent_id=agent_id,
            )
            heapq.heappush(self._queue, queued_task)

    async def dequeue(self) -> QueuedTask | None:
        """Get the highest priority task."""
        async with self._lock:
            if not self._queue:
                return None
            return heapq.heappop(self._queue)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a queued task."""
        async with self._lock:
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    return True
            return False

    async def prevent_starvation(self) -> None:
        """Boost priority of long-waiting tasks."""
        async with self._lock:
            needs_reheapify = False
            for task in self._queue:
                task.wait_cycles += 1
                if task.wait_cycles >= self.starvation_threshold:
                    boost = min(
                        task.wait_cycles * self.STARVATION_BOOST,
                        self.MAX_BOOST,
                    )
                    new_score = -boost - 100
                    if new_score < task.priority_score:
                        task.priority_score = new_score
                        needs_reheapify = True
            if needs_reheapify:
                heapq.heapify(self._queue)

    def register_running(self, task_id: str, task: asyncio.Task) -> None:
        """Register a task as running."""
        self._running[task_id] = task

    def unregister_running(self, task_id: str) -> None:
        """Unregister a completed task."""
        self._running.pop(task_id, None)

    def get_running_task(self, task_id: str) -> asyncio.Task | None:
        """Get a running asyncio task by ID."""
        return self._running.get(task_id)

    def can_run_more(self) -> bool:
        """Check if more tasks can be run."""
        return len(self._running) < self.max_concurrent

    async def get_status(self) -> dict:
        """Get scheduler status."""
        async with self._lock:
            return {
                "running": len(self._running),
                "queued": len(self._queue),
                "max_concurrent": self.max_concurrent,
                "running_tasks": list(self._running.keys()),
                "queued_tasks": [
                    {"task_id": t.task_id, "agent_id": t.agent_id, "wait_cycles": t.wait_cycles} for t in self._queue
                ],
            }
