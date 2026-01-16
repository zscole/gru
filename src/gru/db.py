"""Database layer using aiosqlite."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open database connection and initialize schema."""
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        # Enable WAL mode and foreign keys
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.execute("PRAGMA foreign_keys = ON")

        # Initialize schema
        schema_path = Path(__file__).parent / "schema.sql"
        schema = schema_path.read_text()

        # Execute schema statements (skip PRAGMA lines, already set)
        for statement in schema.split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("PRAGMA"):
                await self._conn.execute(statement)

        # Migrate existing databases: add worktree columns if missing
        await self._migrate_worktree_columns()

        await self._conn.commit()

    async def _migrate_worktree_columns(self) -> None:
        """Add worktree columns to existing databases."""
        if not self._conn:
            return

        # Check if columns exist by querying table info
        cursor = await self._conn.execute("PRAGMA table_info(agents)")
        columns = {row[1] for row in await cursor.fetchall()}

        migrations = [
            ("worktree_path", "ALTER TABLE agents ADD COLUMN worktree_path TEXT"),
            ("worktree_branch", "ALTER TABLE agents ADD COLUMN worktree_branch TEXT"),
            ("base_repo", "ALTER TABLE agents ADD COLUMN base_repo TEXT"),
        ]

        for col_name, sql in migrations:
            if col_name not in columns:
                with contextlib.suppress(Exception):
                    await self._conn.execute(sql)

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for database transactions."""
        if not self._conn:
            raise RuntimeError("Database not connected")
        try:
            yield
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    async def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> aiosqlite.Cursor:
        """Execute a SQL statement."""
        if not self._conn:
            raise RuntimeError("Database not connected")
        return await self._conn.execute(sql, params)

    async def executemany(self, sql: str, params_seq: list[tuple[Any, ...]]) -> aiosqlite.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        if not self._conn:
            raise RuntimeError("Database not connected")
        return await self._conn.executemany(sql, params_seq)

    async def fetchone(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> dict[str, Any] | None:
        """Execute and fetch one row as dict."""
        cursor = await self.execute(sql, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def fetchall(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        """Execute and fetch all rows as dicts."""
        cursor = await self.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def commit(self) -> None:
        """Commit current transaction."""
        if self._conn:
            await self._conn.commit()

    # Agent operations
    async def create_agent(
        self,
        agent_id: str,
        task: str,
        model: str,
        name: str | None = None,
        system_prompt: str | None = None,
        supervised: bool = True,
        timeout_mode: str = "block",
        priority: str = "normal",
        memory_limit: str | None = None,
        cpu_quota: int | None = None,
        workdir: str | None = None,
        worktree_path: str | None = None,
        worktree_branch: str | None = None,
        base_repo: str | None = None,
    ) -> dict[str, Any]:
        """Create a new agent."""
        await self.execute(
            """
            INSERT INTO agents (id, name, task, model, system_prompt, supervised,
                              timeout_mode, priority, memory_limit, cpu_quota, workdir,
                              worktree_path, worktree_branch, base_repo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                name,
                task,
                model,
                system_prompt,
                1 if supervised else 0,
                timeout_mode,
                priority,
                memory_limit,
                cpu_quota,
                workdir,
                worktree_path,
                worktree_branch,
                base_repo,
            ),
        )
        await self.commit()
        return await self.get_agent(agent_id)  # type: ignore

    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Get agent by ID."""
        return await self.fetchone("SELECT * FROM agents WHERE id = ?", (agent_id,))

    async def get_agents(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get agents, optionally filtered by status."""
        if status:
            return await self.fetchall(
                "SELECT * FROM agents WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        return await self.fetchall("SELECT * FROM agents ORDER BY created_at DESC LIMIT ?", (limit,))

    async def update_agent(self, agent_id: str, **fields: Any) -> None:
        """Update agent fields."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [agent_id]
        await self.execute(f"UPDATE agents SET {set_clause} WHERE id = ?", tuple(values))
        await self.commit()

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        cursor = await self.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        await self.commit()
        return cursor.rowcount > 0

    # Task operations
    async def create_task(
        self,
        task_id: str,
        agent_id: str,
        priority: str = "normal",
        deadline: str | None = None,
        parent_task_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new task."""
        priority_scores = {"high": 100, "normal": 50, "low": 0}
        await self.execute(
            """
            INSERT INTO tasks (id, agent_id, priority, priority_score, deadline, parent_task_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task_id, agent_id, priority, priority_scores.get(priority, 50), deadline, parent_task_id),
        )
        await self.commit()
        return await self.get_task(task_id)  # type: ignore

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task by ID."""
        return await self.fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))

    async def get_queued_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get queued tasks ordered by priority."""
        return await self.fetchall(
            """
            SELECT t.*, a.supervised, a.model
            FROM tasks t
            JOIN agents a ON t.agent_id = a.id
            WHERE t.status = 'queued'
            ORDER BY t.priority_score DESC, t.queued_at ASC
            LIMIT ?
            """,
            (limit,),
        )

    async def update_task(self, task_id: str, **fields: Any) -> None:
        """Update task fields."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [task_id]
        await self.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", tuple(values))
        await self.commit()

    # Conversation operations
    async def add_message(
        self,
        agent_id: str,
        role: str,
        content: str,
        tool_use: Any | None = None,
        tool_result: Any | None = None,
    ) -> int:
        """Add a message to conversation history."""
        cursor = await self.execute(
            """
            INSERT INTO conversations (agent_id, role, content, tool_use, tool_result)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                role,
                content,
                json.dumps(tool_use) if tool_use else None,
                json.dumps(tool_result) if tool_result else None,
            ),
        )
        await self.commit()
        return cursor.lastrowid or 0

    async def get_conversation(self, agent_id: str) -> list[dict[str, Any]]:
        """Get conversation history for an agent."""
        rows = await self.fetchall(
            "SELECT * FROM conversations WHERE agent_id = ? ORDER BY id ASC",
            (agent_id,),
        )
        for row in rows:
            if row.get("tool_use"):
                row["tool_use"] = json.loads(row["tool_use"])
            if row.get("tool_result"):
                row["tool_result"] = json.loads(row["tool_result"])
        return rows

    # Approval operations
    async def create_approval(
        self,
        approval_id: str,
        agent_id: str,
        action_type: str,
        action_details: dict[str, Any],
        timeout_at: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a pending approval request."""
        await self.execute(
            """
            INSERT INTO approvals (id, agent_id, task_id, action_type, action_details, timeout_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (approval_id, agent_id, task_id, action_type, json.dumps(action_details), timeout_at),
        )
        await self.commit()
        return await self.get_approval(approval_id)  # type: ignore

    async def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        """Get approval by ID."""
        row = await self.fetchone("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        if row and row.get("action_details"):
            row["action_details"] = json.loads(row["action_details"])
        return row

    async def get_pending_approvals(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        """Get pending approvals."""
        if agent_id:
            rows = await self.fetchall(
                "SELECT * FROM approvals WHERE status = 'pending' AND agent_id = ? ORDER BY created_at ASC",
                (agent_id,),
            )
        else:
            rows = await self.fetchall("SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at ASC")
        for row in rows:
            if row.get("action_details"):
                row["action_details"] = json.loads(row["action_details"])
        return rows

    async def resolve_approval(self, approval_id: str, status: str, resolved_by: str) -> None:
        """Resolve an approval request."""
        await self.execute(
            """
            UPDATE approvals
            SET status = ?, resolved_at = datetime('now'), resolved_by = ?
            WHERE id = ?
            """,
            (status, resolved_by, approval_id),
        )
        await self.commit()

    # Secret operations
    async def store_secret(self, key: str, encrypted_value: bytes, nonce: bytes) -> None:
        """Store an encrypted secret."""
        await self.execute(
            """
            INSERT INTO secrets (key, encrypted_value, nonce)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                encrypted_value = excluded.encrypted_value,
                nonce = excluded.nonce,
                updated_at = datetime('now')
            """,
            (key, encrypted_value, nonce),
        )
        await self.commit()

    async def get_secret(self, key: str) -> tuple[bytes, bytes] | None:
        """Get encrypted secret and nonce."""
        row = await self.fetchone("SELECT encrypted_value, nonce FROM secrets WHERE key = ?", (key,))
        if row:
            return row["encrypted_value"], row["nonce"]
        return None

    async def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        cursor = await self.execute("DELETE FROM secrets WHERE key = ?", (key,))
        await self.commit()
        return cursor.rowcount > 0

    async def list_secrets(self) -> list[str]:
        """List all secret keys."""
        rows = await self.fetchall("SELECT key FROM secrets ORDER BY key")
        return [row["key"] for row in rows]

    # Agent message operations (coordination)
    async def send_agent_message(
        self,
        message_id: str,
        from_agent: str | None,
        to_agent: str | None,
        task_id: str | None,
        message_type: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Send a message between agents."""
        await self.execute(
            """
            INSERT INTO agent_messages (id, from_agent, to_agent, task_id, message_type, content, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                from_agent,
                to_agent,
                task_id,
                message_type,
                content,
                json.dumps(metadata) if metadata else None,
            ),
        )
        await self.commit()

    async def get_agent_messages(self, agent_id: str, unread_only: bool = False) -> list[dict[str, Any]]:
        """Get messages for an agent."""
        if unread_only:
            rows = await self.fetchall(
                "SELECT * FROM agent_messages WHERE to_agent = ? AND read = 0 ORDER BY created_at ASC",
                (agent_id,),
            )
        else:
            rows = await self.fetchall(
                "SELECT * FROM agent_messages WHERE to_agent = ? ORDER BY created_at ASC",
                (agent_id,),
            )
        for row in rows:
            if row.get("metadata"):
                row["metadata"] = json.loads(row["metadata"])
        return rows

    async def mark_message_read(self, message_id: str) -> None:
        """Mark a message as read."""
        await self.execute("UPDATE agent_messages SET read = 1 WHERE id = ?", (message_id,))
        await self.commit()

    # Shared context operations
    async def set_shared_context(self, task_id: str, key: str, value: Any, updated_by: str) -> None:
        """Set a shared context value."""
        await self.execute(
            """
            INSERT INTO shared_context (task_id, key, value, updated_by)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(task_id, key) DO UPDATE SET
                value = excluded.value,
                updated_by = excluded.updated_by,
                updated_at = datetime('now')
            """,
            (task_id, key, json.dumps(value), updated_by),
        )
        await self.commit()

    async def get_shared_context(self, task_id: str) -> dict[str, Any]:
        """Get all shared context for a task."""
        rows = await self.fetchall("SELECT key, value FROM shared_context WHERE task_id = ?", (task_id,))
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # Template operations
    async def save_template(
        self,
        name: str,
        task: str,
        model: str | None = None,
        system_prompt: str | None = None,
        supervised: bool | None = None,
        timeout_mode: str | None = None,
        priority: str | None = None,
    ) -> None:
        """Save an agent template."""
        await self.execute(
            """
            INSERT INTO templates (name, task, model, system_prompt, supervised, timeout_mode, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                task = excluded.task,
                model = excluded.model,
                system_prompt = excluded.system_prompt,
                supervised = excluded.supervised,
                timeout_mode = excluded.timeout_mode,
                priority = excluded.priority,
                updated_at = datetime('now')
            """,
            (
                name,
                task,
                model,
                system_prompt,
                1 if supervised else (0 if supervised is False else None),
                timeout_mode,
                priority,
            ),
        )
        await self.commit()

    async def get_template(self, name: str) -> dict[str, Any] | None:
        """Get a template by name."""
        return await self.fetchone("SELECT * FROM templates WHERE name = ?", (name,))

    async def list_templates(self) -> list[dict[str, Any]]:
        """List all templates."""
        return await self.fetchall("SELECT * FROM templates ORDER BY name")

    async def delete_template(self, name: str) -> bool:
        """Delete a template."""
        cursor = await self.execute("DELETE FROM templates WHERE name = ?", (name,))
        await self.commit()
        return cursor.rowcount > 0
