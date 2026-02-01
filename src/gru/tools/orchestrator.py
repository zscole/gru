"""Orchestrator tools - spawn and manage autonomous agents.

This module exposes the Orchestrator's agent management to the chat agent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gru.tools.base import register_tool

if TYPE_CHECKING:
    from gru.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_orchestrator: Orchestrator | None = None


def set_orchestrator(orch: Orchestrator) -> None:
    """Set the orchestrator instance."""
    global _orchestrator
    _orchestrator = orch


async def spawn_agent(
    task: str,
    name: str | None = None,
    supervised: bool = False,
    live_output: bool = False,
    workdir: str | None = None,
) -> dict[str, Any]:
    """Spawn an autonomous agent to work on a task."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}

    try:
        agent = await _orchestrator.spawn_agent(
            task=task,
            name=name,
            supervised=supervised,
            live_output=live_output,
            workdir=workdir,
        )
        return {
            "status": "spawned",
            "agent_id": agent["id"],
            "name": agent.get("name"),
            "task": task[:100],
            "supervised": supervised,
            "workdir": workdir or "(default)",
        }
    except Exception as e:
        logger.error(f"Failed to spawn agent: {e}")
        return {"error": str(e)}


async def list_agents(status: str | None = None) -> dict[str, Any]:
    """List all agents, optionally filtered by status."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}

    try:
        agents = await _orchestrator.list_agents(status)
        return {
            "agents": [
                {
                    "id": a["id"],
                    "name": a.get("name"),
                    "status": a["status"],
                    "task": a["task"][:50] + "..." if len(a["task"]) > 50 else a["task"],
                }
                for a in agents
            ],
            "count": len(agents),
        }
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return {"error": str(e)}


async def get_agent_status(agent_id: str) -> dict[str, Any]:
    """Get detailed status of a specific agent."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}

    try:
        agent = await _orchestrator.get_agent(agent_id)
        if not agent:
            return {"error": f"Agent {agent_id} not found"}

        # Get cost info if available
        cost_info = await _orchestrator.get_agent_cost_from_db(agent_id)

        result = {
            "id": agent["id"],
            "name": agent.get("name"),
            "status": agent["status"],
            "task": agent["task"],
            "model": agent.get("model"),
            "created_at": agent.get("created_at"),
            "started_at": agent.get("started_at"),
            "completed_at": agent.get("completed_at"),
        }

        if cost_info:
            input_tokens, output_tokens, cost = cost_info
            result["tokens"] = {"input": input_tokens, "output": output_tokens}
            result["cost_usd"] = cost

        if agent.get("error"):
            result["error"] = agent["error"]

        return result
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        return {"error": str(e)}


async def terminate_agent(agent_id: str) -> dict[str, Any]:
    """Terminate a running agent."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}

    try:
        success = await _orchestrator.terminate_agent(agent_id)
        if success:
            return {"status": "terminated", "agent_id": agent_id}
        return {"error": f"Could not terminate agent {agent_id}"}
    except Exception as e:
        logger.error(f"Failed to terminate agent: {e}")
        return {"error": str(e)}


async def nudge_agent(agent_id: str, message: str) -> dict[str, Any]:
    """Send a message/instruction to a running agent."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}

    try:
        success = await _orchestrator.nudge_agent(agent_id, message)
        if success:
            return {"status": "sent", "agent_id": agent_id, "message": message[:50]}
        return {"error": f"Agent {agent_id} not found or not running"}
    except Exception as e:
        logger.error(f"Failed to nudge agent: {e}")
        return {"error": str(e)}


# Dynamically determine source directory
GRU_SOURCE_DIR = str(Path(__file__).parent.parent.parent.parent)

GRU_ARCHITECTURE = """
# Gru Architecture

Gru is a personal AI assistant platform.

## Directory Structure
- src/gru/           - Main source code
  - main.py          - Entry point, starts bots and orchestrator
  - orchestrator.py  - Agent lifecycle management, spawns autonomous agents
  - agent.py         - ChatAgent for conversational tool use
  - claude.py        - Claude API client
  - telegram_bot.py  - Telegram interface
  - discord_bot.py   - Discord interface
  - slack_bot.py     - Slack interface
  - proactive.py     - ProactiveEngine for reminders, triggers, scheduled tasks
  - memory.py        - MemoryStore for user preferences and facts
  - knowledge_graph.py - Personal knowledge graph
  - self_heal.py     - Self-diagnostics and auto-recovery
  - tools/           - Tool modules exposed to the agent
  - connectors/      - External service connectors
  - actions/         - Autonomous action handlers

## Key Patterns
- Tools are registered via register_tool() in each module
- Tools are async functions returning dict
- ChatAgent uses Claude tool_use for natural language -> tool selection
- Orchestrator spawns autonomous agents for complex tasks
- ProactiveEngine handles time-based triggers and reminders

## Adding a New Tool
1. Create function: async def my_tool(param: str) -> dict
2. Register in register_X_tools(): register_tool(name, description, parameters, handler)
3. Import and call register function in tools/__init__.py

## Testing
- Run: pytest tests/ -v
- Or specific: pytest tests/test_X.py

## Restarting Gru
After code changes, restart with: pkill -f gru && python -m gru.main
"""


async def get_gru_architecture() -> dict[str, Any]:
    """Get information about Gru's codebase structure."""
    return {
        "source_dir": GRU_SOURCE_DIR,
        "architecture": GRU_ARCHITECTURE,
    }


async def improve_gru(feature_description: str) -> dict[str, Any]:
    """Spawn an agent to implement a new feature in Gru itself."""
    if not _orchestrator:
        return {"error": "Orchestrator not initialized"}

    task = f"""You are improving Gru, the AI assistant platform.

SOURCE DIRECTORY: {GRU_SOURCE_DIR}

{GRU_ARCHITECTURE}

YOUR TASK:
{feature_description}

INSTRUCTIONS:
1. First, read relevant existing code to understand the patterns
2. Implement the feature following existing patterns
3. Test if possible
4. Report what you changed and what needs to happen next (e.g., restart)

Be careful not to break existing functionality. Make minimal, focused changes.
"""

    try:
        agent = await _orchestrator.spawn_agent(
            task=task,
            name="gru-improver",
            supervised=False,
            live_output=False,
            workdir=GRU_SOURCE_DIR,
        )
        return {
            "status": "spawned",
            "agent_id": agent["id"],
            "message": f"Agent spawned to implement: {feature_description[:100]}",
            "note": "After completion, Gru will need to be restarted to apply changes",
        }
    except Exception as e:
        logger.error(f"Failed to spawn improvement agent: {e}")
        return {"error": str(e)}


async def restart_gru() -> dict[str, Any]:
    """Restart Gru to apply code changes."""
    import subprocess

    try:
        # Start new instance before killing current one
        # This is done via a detached subprocess
        restart_script = f"""
cd {GRU_SOURCE_DIR}
sleep 2
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')
from gru.main import main
main()
" > /tmp/gru.log 2>&1 &
"""
        # Write restart script
        script_path = "/tmp/gru_restart.sh"
        with open(script_path, "w") as f:
            f.write(restart_script)
        os.chmod(script_path, 0o755)

        # Execute restart script in background
        subprocess.Popen(
            ["bash", script_path],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return {
            "status": "restarting",
            "message": "Gru is restarting. New instance will start in ~2 seconds.",
        }
    except Exception as e:
        logger.error(f"Failed to restart: {e}")
        return {"error": str(e)}


def register_orchestrator_tools() -> None:
    """Register orchestrator/agent management tools."""
    register_tool(
        name="spawn_agent",
        description="Spawn an autonomous agent to work on a complex task. Use for tasks that require multiple steps, research, coding, or long-running work. The agent works independently and reports back.",
        parameters={
            "task": {
                "type": "string",
                "description": "Detailed description of what the agent should accomplish",
            },
            "name": {
                "type": "string",
                "description": "Optional name for the agent",
                "optional": True,
            },
            "supervised": {
                "type": "boolean",
                "description": "If true, agent asks for approval before destructive actions (default: false)",
                "optional": True,
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for the agent (default: config default)",
                "optional": True,
            },
        },
        handler=spawn_agent,
    )

    register_tool(
        name="list_agents",
        description="List all agents and their status. Can filter by status: running, completed, failed, paused.",
        parameters={
            "status": {
                "type": "string",
                "description": "Filter by status (running, completed, failed, paused)",
                "optional": True,
            },
        },
        handler=list_agents,
    )

    register_tool(
        name="get_agent_status",
        description="Get detailed status of a specific agent including task, progress, and cost.",
        parameters={
            "agent_id": {
                "type": "string",
                "description": "The agent ID to check",
            },
        },
        handler=get_agent_status,
    )

    register_tool(
        name="terminate_agent",
        description="Stop a running agent.",
        parameters={
            "agent_id": {
                "type": "string",
                "description": "The agent ID to terminate",
            },
        },
        handler=terminate_agent,
    )

    register_tool(
        name="nudge_agent",
        description="Send a message or instruction to a running agent to guide its work.",
        parameters={
            "agent_id": {
                "type": "string",
                "description": "The agent ID to message",
            },
            "message": {
                "type": "string",
                "description": "The message or instruction to send",
            },
        },
        handler=nudge_agent,
    )

    # Self-improvement tools
    register_tool(
        name="get_gru_architecture",
        description="Get information about Gru's codebase structure and how to modify it.",
        parameters={},
        handler=get_gru_architecture,
    )

    register_tool(
        name="improve_gru",
        description="Spawn an agent to implement a new feature in Gru itself. Use this when asked to add capabilities, fix bugs, or improve Gru. The agent will modify Gru's source code.",
        parameters={
            "feature_description": {
                "type": "string",
                "description": "Detailed description of the feature to implement",
            },
        },
        handler=improve_gru,
    )

    register_tool(
        name="restart_gru",
        description="Restart Gru to apply code changes. Use after improve_gru has made changes.",
        parameters={},
        handler=restart_gru,
    )
