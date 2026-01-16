-- Gru database schema
-- SQLite with WAL mode and foreign key enforcement

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT,
    status TEXT NOT NULL CHECK(status IN ('idle', 'running', 'paused', 'completed', 'failed', 'terminated')) DEFAULT 'idle',
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT,
    supervised INTEGER NOT NULL DEFAULT 1,
    timeout_mode TEXT NOT NULL CHECK(timeout_mode IN ('block', 'pause', 'auto')) DEFAULT 'block',
    priority TEXT NOT NULL CHECK(priority IN ('high', 'normal', 'low')) DEFAULT 'normal',
    memory_limit TEXT,
    cpu_quota INTEGER,
    workdir TEXT,
    worktree_path TEXT,
    worktree_branch TEXT,
    base_repo TEXT,
    pid INTEGER,
    cgroup_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    error TEXT
);

-- Tasks table (individual task executions)
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    parent_task_id TEXT REFERENCES tasks(id) ON DELETE SET NULL,
    status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'waiting_approval', 'completed', 'failed', 'cancelled')) DEFAULT 'queued',
    priority TEXT NOT NULL CHECK(priority IN ('high', 'normal', 'low')) DEFAULT 'normal',
    priority_score INTEGER NOT NULL DEFAULT 0,
    deadline TEXT,
    queued_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    result TEXT,
    error TEXT
);

-- Conversations table (message history)
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tool_use JSON,
    tool_result JSON,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Approvals table (human-in-the-loop)
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    action_details JSON NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected', 'timeout')) DEFAULT 'pending',
    timeout_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,
    resolved_by TEXT
);

-- Secrets table (encrypted)
CREATE TABLE IF NOT EXISTS secrets (
    key TEXT PRIMARY KEY,
    encrypted_value BLOB NOT NULL,
    nonce BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Agent messages table (coordination)
CREATE TABLE IF NOT EXISTS agent_messages (
    id TEXT PRIMARY KEY,
    from_agent TEXT REFERENCES agents(id) ON DELETE CASCADE,
    to_agent TEXT REFERENCES agents(id) ON DELETE SET NULL,
    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
    message_type TEXT NOT NULL CHECK(message_type IN ('request', 'response', 'info', 'handoff')),
    content TEXT NOT NULL,
    metadata JSON,
    read INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Shared context table (coordination)
CREATE TABLE IF NOT EXISTS shared_context (
    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value JSON NOT NULL,
    updated_by TEXT REFERENCES agents(id),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (task_id, key)
);

-- Templates table
CREATE TABLE IF NOT EXISTS templates (
    name TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    model TEXT,
    system_prompt TEXT,
    supervised INTEGER,
    timeout_mode TEXT,
    priority TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
CREATE INDEX IF NOT EXISTS idx_approvals_agent ON approvals(agent_id);
CREATE INDEX IF NOT EXISTS idx_conversations_agent ON conversations(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_messages_to ON agent_messages(to_agent, read);
CREATE INDEX IF NOT EXISTS idx_agent_messages_task ON agent_messages(task_id);
