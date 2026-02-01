"""CLI interface for Gru using Click."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click

from gru.config import Config
from gru.connectors.google import GoogleConnector, setup_google_triggers
from gru.crypto import CryptoManager, SecretStore
from gru.db import Database
from gru.memory import MemoryStore
from gru.orchestrator import Orchestrator
from gru.proactive import ProactiveEngine, TriggerType
from gru.session import PERSONAS, get_available_personas
from gru.actions.executor import ActionExecutor
from gru.actions.registry import get_registry
from gru.setup import (
    ConfigManager,
    SetupWizard,
    detect_key_type,
    detect_multiple_keys,
    get_config_manager,
    get_setup_wizard,
    KeyType,
    KEY_TYPE_NAMES,
)


def get_orchestrator(ctx: click.Context) -> Orchestrator:
    """Get orchestrator from context."""
    return ctx.obj["orchestrator"]


def run_async(coro):
    """Run async function synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


@click.group()
@click.option("--data-dir", type=click.Path(), help="Data directory path")
@click.pass_context
def cli(ctx: click.Context, data_dir: str | None) -> None:
    """Gru - AI Agent Orchestration CLI."""
    ctx.ensure_object(dict)

    # Load config
    config = Config.from_env()
    if data_dir:
        config.data_dir = Path(data_dir)
        config.db_path = config.data_dir / "gru.db"

    # Initialize components
    db = Database(config.db_path)
    run_async(db.connect())

    # Register cleanup on exit
    ctx.call_on_close(lambda: run_async(db.close()))

    crypto = CryptoManager(config.data_dir)
    # Auto-initialize with env var if available
    master_pass = os.getenv("GRU_MASTER_PASSWORD")
    if master_pass:
        crypto.initialize(master_pass)

    secrets = SecretStore(db, crypto)
    orchestrator = Orchestrator(config, db, secrets)

    ctx.obj["config"] = config
    ctx.obj["db"] = db
    ctx.obj["crypto"] = crypto
    ctx.obj["secrets"] = secrets
    ctx.obj["orchestrator"] = orchestrator


@cli.command()
@click.argument("task")
@click.option("--name", "-n", help="Agent name")
@click.option("--supervised/--unsupervised", default=True, help="Supervised mode")
@click.option("--priority", type=click.Choice(["high", "normal", "low"]), default="normal")
@click.option("--model", "-m", help="Model to use")
@click.option("--deadline", help="Deadline (e.g., '2h', '30m')")
@click.pass_context
def spawn(
    ctx: click.Context,
    task: str,
    name: str | None,
    supervised: bool,
    priority: str,
    model: str | None,
    deadline: str | None,
) -> None:
    """Start a new agent with the given task."""
    orchestrator = get_orchestrator(ctx)

    agent = run_async(
        orchestrator.spawn_agent(
            task=task,
            name=name,
            model=model,
            supervised=supervised,
            priority=priority,
            deadline=deadline,
        )
    )

    click.echo(f"Agent spawned: {agent['id']}")
    click.echo(f"Task: {task}")
    click.echo(f"Supervised: {supervised}")
    click.echo(f"Priority: {priority}")


@cli.command()
@click.argument("agent_id", required=False)
@click.pass_context
def status(ctx: click.Context, agent_id: str | None) -> None:
    """Show status of orchestrator or specific agent."""
    orchestrator = get_orchestrator(ctx)

    if agent_id:
        agent = run_async(orchestrator.get_agent(agent_id))
        if not agent:
            click.echo(f"Agent not found: {agent_id}", err=True)
            sys.exit(1)

        click.echo(f"Agent: {agent['id']}")
        click.echo(f"Status: {agent['status']}")
        click.echo(f"Task: {agent['task']}")
        click.echo(f"Model: {agent['model']}")
        click.echo(f"Supervised: {bool(agent['supervised'])}")
        click.echo(f"Created: {agent['created_at']}")
        if agent.get("started_at"):
            click.echo(f"Started: {agent['started_at']}")
        if agent.get("completed_at"):
            click.echo(f"Completed: {agent['completed_at']}")
        if agent.get("error"):
            click.echo(f"Error: {agent['error']}")
    else:
        status_data = run_async(orchestrator.get_status())

        click.echo("Orchestrator Status")
        click.echo(f"Running: {status_data['running']}")
        click.echo(f"Agents: {status_data['agents']['total']} total")
        click.echo(f"  Running: {status_data['agents']['running']}")
        click.echo(f"  Paused: {status_data['agents']['paused']}")
        click.echo(f"  Completed: {status_data['agents']['completed']}")
        click.echo(f"  Failed: {status_data['agents']['failed']}")
        click.echo(f"Queue: {status_data['scheduler']['queued']} queued")


@cli.command("list")
@click.option("--status", "-s", "status_filter", help="Filter by status")
@click.option("--limit", "-l", default=20, help="Max results")
@click.pass_context
def list_agents(ctx: click.Context, status_filter: str | None, limit: int) -> None:
    """List agents."""
    orchestrator = get_orchestrator(ctx)
    agents = run_async(orchestrator.list_agents(status_filter))

    if not agents:
        click.echo("No agents found")
        return

    for agent in agents[:limit]:
        task_preview = agent["task"][:50] + "..." if len(agent["task"]) > 50 else agent["task"]
        click.echo(f"{agent['id']} [{agent['status']}] {task_preview}")


@cli.command()
@click.argument("agent_id")
@click.pass_context
def pause(ctx: click.Context, agent_id: str) -> None:
    """Pause an agent."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.pause_agent(agent_id))

    if success:
        click.echo(f"Agent {agent_id} paused")
    else:
        click.echo(f"Could not pause agent {agent_id}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("agent_id")
@click.pass_context
def resume(ctx: click.Context, agent_id: str) -> None:
    """Resume a paused agent."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.resume_agent(agent_id))

    if success:
        click.echo(f"Agent {agent_id} resumed")
    else:
        click.echo(f"Could not resume agent {agent_id}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("agent_id")
@click.pass_context
def terminate(ctx: click.Context, agent_id: str) -> None:
    """Terminate an agent."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.terminate_agent(agent_id))

    if success:
        click.echo(f"Agent {agent_id} terminated")
    else:
        click.echo(f"Could not terminate agent {agent_id}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("agent_id")
@click.argument("message")
@click.pass_context
def nudge(ctx: click.Context, agent_id: str, message: str) -> None:
    """Send a nudge message to an agent."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.nudge_agent(agent_id, message))

    if success:
        click.echo(f"Nudge sent to agent {agent_id}")
    else:
        click.echo(f"Could not nudge agent {agent_id}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("task")
@click.option("--max-iterations", "-i", type=int, default=20, help="Maximum iterations")
@click.option("--completion-promise", "-c", help="String to detect completion")
@click.option("--name", "-n", help="Agent name")
@click.option("--model", "-m", help="Model to use")
@click.option("--priority", type=click.Choice(["high", "normal", "low"]), default="normal")
@click.pass_context
def ralph(
    ctx: click.Context,
    task: str,
    max_iterations: int,
    completion_promise: str | None,
    name: str | None,
    model: str | None,
    priority: str,
) -> None:
    """Start a Ralph Wiggum iterative development loop.

    Ralph is an AI development methodology that creates self-referential
    feedback loops where an agent iteratively improves work through
    continuous iterations until completion or max iterations reached.
    """
    orchestrator = get_orchestrator(ctx)

    agent = run_async(
        orchestrator.spawn_ralph_loop(
            task=task,
            max_iterations=max_iterations,
            completion_promise=completion_promise,
            name=name,
            model=model,
            priority=priority,
        )
    )

    click.echo(f"Ralph loop started: {agent['id']}")
    click.echo(f"Task: {task}")
    click.echo(f"Max iterations: {max_iterations}")
    if completion_promise:
        click.echo(f"Completion promise: {completion_promise}")
    click.echo(f"Priority: {priority}")
    click.echo("\nUse 'gru status <agent_id>' to monitor progress")
    click.echo("Use 'gru cancel-ralph <agent_id>' to stop the loop")


@cli.command("cancel-ralph")
@click.argument("agent_id")
@click.pass_context
def cancel_ralph(ctx: click.Context, agent_id: str) -> None:
    """Cancel an active Ralph loop."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.cancel_ralph_loop(agent_id))

    if success:
        click.echo(f"Ralph loop {agent_id} cancelled")
    else:
        click.echo(f"Could not cancel Ralph loop {agent_id}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("agent_id")
@click.option("--tail", "-t", default=20, help="Number of messages to show")
@click.pass_context
def logs(ctx: click.Context, agent_id: str, tail: int) -> None:
    """Show agent conversation logs."""
    db = ctx.obj["db"]
    conversation = run_async(db.get_conversation(agent_id))

    if not conversation:
        click.echo(f"No logs found for agent {agent_id}")
        return

    for msg in conversation[-tail:]:
        content = msg["content"]
        if isinstance(content, list):
            content = json.dumps(content, indent=2)
        click.echo(f"\n[{msg['role'].upper()}]")
        click.echo(content[:1000])


@cli.command()
@click.pass_context
def pending(ctx: click.Context) -> None:
    """Show pending approvals."""
    orchestrator = get_orchestrator(ctx)
    approvals = run_async(orchestrator.get_pending_approvals())

    if not approvals:
        click.echo("No pending approvals")
        return

    for p in approvals:
        click.echo(f"\n{p['id']}")
        click.echo(f"  Agent: {p['agent_id']}")
        click.echo(f"  Action: {p['action_type']}")
        click.echo(f"  Details: {json.dumps(p['action_details'])[:100]}...")
        click.echo(f"  Created: {p['created_at']}")


@cli.command()
@click.argument("approval_id")
@click.pass_context
def approve(ctx: click.Context, approval_id: str) -> None:
    """Approve a pending action."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.approve(approval_id, approved=True))

    if success:
        click.echo(f"Approved: {approval_id}")
    else:
        click.echo(f"Approval not found or already resolved: {approval_id}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("approval_id")
@click.pass_context
def reject(ctx: click.Context, approval_id: str) -> None:
    """Reject a pending action."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.approve(approval_id, approved=False))

    if success:
        click.echo(f"Rejected: {approval_id}")
    else:
        click.echo(f"Approval not found or already resolved: {approval_id}", err=True)
        sys.exit(1)


# Secret management commands


@cli.group()
def secret():
    """Manage encrypted secrets."""
    pass


@secret.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def secret_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a secret value."""
    secrets = ctx.obj["secrets"]

    if not ctx.obj["crypto"].is_initialized():
        click.echo("Crypto not initialized. Set GRU_MASTER_PASSWORD env var.", err=True)
        sys.exit(1)

    run_async(secrets.set(key, value))
    click.echo(f"Secret '{key}' set")


@secret.command("get")
@click.argument("key")
@click.pass_context
def secret_get(ctx: click.Context, key: str) -> None:
    """Get a secret value."""
    secrets = ctx.obj["secrets"]

    if not ctx.obj["crypto"].is_initialized():
        click.echo("Crypto not initialized. Set GRU_MASTER_PASSWORD env var.", err=True)
        sys.exit(1)

    value = run_async(secrets.get(key))
    if value:
        click.echo(value)
    else:
        click.echo(f"Secret '{key}' not found", err=True)
        sys.exit(1)


@secret.command("list")
@click.pass_context
def secret_list(ctx: click.Context) -> None:
    """List secret keys."""
    secrets = ctx.obj["secrets"]
    keys = run_async(secrets.list_keys())

    if keys:
        for key in keys:
            click.echo(key)
    else:
        click.echo("No secrets stored")


@secret.command("delete")
@click.argument("key")
@click.pass_context
def secret_delete(ctx: click.Context, key: str) -> None:
    """Delete a secret."""
    secrets = ctx.obj["secrets"]
    success = run_async(secrets.delete(key))

    if success:
        click.echo(f"Secret '{key}' deleted")
    else:
        click.echo(f"Secret '{key}' not found", err=True)
        sys.exit(1)


# Template management commands


@cli.group()
def template():
    """Manage agent templates."""
    pass


@template.command("save")
@click.argument("name")
@click.argument("task")
@click.option("--model", "-m", help="Model to use")
@click.option("--supervised/--unsupervised", default=None)
@click.option("--priority", type=click.Choice(["high", "normal", "low"]))
@click.pass_context
def template_save(
    ctx: click.Context,
    name: str,
    task: str,
    model: str | None,
    supervised: bool | None,
    priority: str | None,
) -> None:
    """Save a template."""
    db = ctx.obj["db"]
    run_async(
        db.save_template(
            name=name,
            task=task,
            model=model,
            supervised=supervised,
            priority=priority,
        )
    )
    click.echo(f"Template '{name}' saved")


@template.command("list")
@click.pass_context
def template_list(ctx: click.Context) -> None:
    """List templates."""
    db = ctx.obj["db"]
    templates = run_async(db.list_templates())

    if templates:
        for t in templates:
            task_preview = t["task"][:50] + "..." if len(t["task"]) > 50 else t["task"]
            click.echo(f"{t['name']}: {task_preview}")
    else:
        click.echo("No templates saved")


@template.command("use")
@click.argument("name")
@click.pass_context
def template_use(ctx: click.Context, name: str) -> None:
    """Spawn agent from template."""
    db = ctx.obj["db"]
    orchestrator = get_orchestrator(ctx)

    template_data = run_async(db.get_template(name))
    if not template_data:
        click.echo(f"Template '{name}' not found", err=True)
        sys.exit(1)

    agent = run_async(
        orchestrator.spawn_agent(
            task=template_data["task"],
            model=template_data.get("model"),
            system_prompt=template_data.get("system_prompt"),
            supervised=bool(template_data.get("supervised", 1)),
            priority=template_data.get("priority", "normal"),
        )
    )

    click.echo(f"Agent spawned from template: {agent['id']}")


@template.command("delete")
@click.argument("name")
@click.pass_context
def template_delete(ctx: click.Context, name: str) -> None:
    """Delete a template."""
    db = ctx.obj["db"]
    success = run_async(db.delete_template(name))

    if success:
        click.echo(f"Template '{name}' deleted")
    else:
        click.echo(f"Template '{name}' not found", err=True)
        sys.exit(1)


# Memory management commands


@cli.group()
def memory():
    """Manage persistent memory (facts and preferences)."""
    pass


def get_memory_store(ctx: click.Context) -> MemoryStore:
    """Get or create memory store."""
    if "memory" not in ctx.obj:
        config = ctx.obj["config"]
        db = ctx.obj["db"]
        mem = MemoryStore(db, config.data_dir)
        run_async(mem.initialize())
        ctx.obj["memory"] = mem
    return ctx.obj["memory"]


@memory.command("list")
@click.option("--type", "-t", "fact_type", help="Filter by type (preference, entity, decision, relationship, context)")
@click.option("--limit", "-l", default=20, help="Max results")
@click.pass_context
def memory_list(ctx: click.Context, fact_type: str | None, limit: int) -> None:
    """List stored facts."""
    mem = get_memory_store(ctx)
    facts = run_async(mem.get_facts(fact_type=fact_type, limit=limit))

    if not facts:
        click.echo("No facts stored")
        return

    for fact in facts:
        click.echo(f"[{fact.fact_type}] {fact.subject} {fact.predicate} {fact.object}")
        click.echo(f"  id: {fact.id} | confidence: {fact.confidence}")


@memory.command("search")
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Max results")
@click.pass_context
def memory_search(ctx: click.Context, query: str, limit: int) -> None:
    """Search memory semantically."""
    mem = get_memory_store(ctx)
    results = run_async(mem.search_memory(query, limit=limit))

    if not results:
        click.echo("No matching facts found")
        return

    for fact in results:
        click.echo(f"[{fact['fact_type']}] {fact['subject']} {fact['predicate']} {fact['object']}")


@memory.command("add")
@click.option("--type", "-t", "fact_type", required=True,
              type=click.Choice(["preference", "entity", "decision", "relationship", "context"]))
@click.option("--subject", "-s", required=True, help="Subject of the fact")
@click.option("--predicate", "-p", required=True, help="Predicate/relationship")
@click.option("--object", "-o", "obj", required=True, help="Object of the fact")
@click.option("--confidence", "-c", default=1.0, help="Confidence score (0.0-1.0)")
@click.pass_context
def memory_add(
    ctx: click.Context,
    fact_type: str,
    subject: str,
    predicate: str,
    obj: str,
    confidence: float,
) -> None:
    """Add a fact manually."""
    mem = get_memory_store(ctx)
    fact_id = run_async(
        mem.store_fact(
            fact_type=fact_type,
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
        )
    )
    click.echo(f"Fact stored: {fact_id}")


@memory.command("forget")
@click.argument("fact_id")
@click.pass_context
def memory_forget(ctx: click.Context, fact_id: str) -> None:
    """Forget (deactivate) a fact."""
    mem = get_memory_store(ctx)
    success = run_async(mem.forget_fact(fact_id))

    if success:
        click.echo(f"Fact {fact_id} forgotten")
    else:
        click.echo(f"Fact {fact_id} not found", err=True)
        sys.exit(1)


@memory.command("stats")
@click.pass_context
def memory_stats(ctx: click.Context) -> None:
    """Show memory statistics."""
    mem = get_memory_store(ctx)
    stats = run_async(mem.get_stats())

    click.echo("Memory Statistics")
    click.echo(f"Total facts: {stats['total_facts']}")
    click.echo(f"Total embeddings: {stats['total_embeddings']}")
    click.echo("By type:")
    for fact_type, count in stats.get("by_type", {}).items():
        click.echo(f"  {fact_type}: {count}")


@memory.command("context")
@click.argument("query")
@click.pass_context
def memory_context(ctx: click.Context, query: str) -> None:
    """Preview context that would be injected for a query."""
    mem = get_memory_store(ctx)
    context = run_async(mem.get_relevant_context(query))

    if context:
        click.echo(context)
    else:
        click.echo("No relevant context found")


@memory.command("profile")
@click.pass_context
def memory_profile(ctx: click.Context) -> None:
    """Show user profile built from memory."""
    mem = get_memory_store(ctx)
    profile = run_async(mem.get_user_profile())

    click.echo("User Profile")
    click.echo("=" * 40)

    if profile.get("preferences"):
        click.echo("\nPreferences:")
        for key, value in profile["preferences"].items():
            click.echo(f"  {key}: {value}")

    if profile.get("tools"):
        click.echo(f"\nTools: {', '.join(profile['tools'])}")

    if profile.get("projects"):
        click.echo(f"\nProjects: {', '.join(profile['projects'])}")

    if profile.get("decisions"):
        click.echo("\nPast decisions:")
        for decision in profile["decisions"][:5]:
            click.echo(f"  - {decision}")


@memory.command("feedback")
@click.argument("feedback_text")
@click.pass_context
def memory_feedback(ctx: click.Context, feedback_text: str) -> None:
    """Process feedback to update memory (e.g., 'I actually prefer X')."""
    mem = get_memory_store(ctx)
    orchestrator = get_orchestrator(ctx)

    updates = run_async(mem.process_feedback(feedback_text, orchestrator.claude))
    if updates:
        click.echo(f"Updated {len(updates)} facts based on feedback")
    else:
        click.echo("No updates made from feedback")


@memory.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def memory_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a preference.

    Common preferences:
        gru memory set location "San Francisco, CA"
        gru memory set food "vegetarian"
        gru memory set default_restaurant "Joyland"
        gru memory set budget "moderate"
    """
    mem = get_memory_store(ctx)
    fact_id = run_async(mem.set_preference(key, value))
    click.echo(f"Set {key} = {value} (fact: {fact_id})")


@memory.command("get")
@click.argument("key")
@click.pass_context
def memory_get(ctx: click.Context, key: str) -> None:
    """Get a specific preference value."""
    mem = get_memory_store(ctx)
    value = run_async(mem.get_preference(key))
    if value:
        click.echo(f"{key}: {value}")
    else:
        click.echo(f"No preference set for '{key}'")


# Proactive engine commands


@cli.group()
def proactive():
    """Manage proactive behaviors (triggers, observations)."""
    pass


def get_proactive_engine(ctx: click.Context) -> ProactiveEngine:
    """Get or create proactive engine."""
    if "proactive" not in ctx.obj:
        config = ctx.obj["config"]
        db = ctx.obj["db"]
        mem = get_memory_store(ctx) if config.memory_enabled else None
        engine = ProactiveEngine(config, db, mem)
        run_async(engine.initialize())
        ctx.obj["proactive"] = engine
    return ctx.obj["proactive"]


@proactive.command("triggers")
@click.pass_context
def proactive_triggers(ctx: click.Context) -> None:
    """List all triggers."""
    engine = get_proactive_engine(ctx)
    triggers = run_async(engine.list_triggers())

    if not triggers:
        click.echo("No triggers configured")
        return

    for t in triggers:
        status = "enabled" if t["enabled"] else "disabled"
        fired = f"fired {t['fire_count']}x" if t["fire_count"] else "never fired"
        click.echo(f"{t['id']} [{t['type']}] {t['name']}")
        click.echo(f"  Action: {t['action']}")
        click.echo(f"  Status: {status}, {fired}")


@proactive.command("add-trigger")
@click.option("--name", "-n", required=True, help="Trigger name")
@click.option("--type", "-t", "trigger_type", required=True,
              type=click.Choice(["scheduled", "interval", "condition"]))
@click.option("--action", "-a", required=True, help="Action (e.g., 'notify:Hello' or 'check:daily_summary')")
@click.option("--schedule", "-s", help="Schedule time (HH:MM) for scheduled triggers")
@click.option("--interval", "-i", type=int, help="Interval in minutes for interval triggers")
@click.option("--condition", "-c", help="Condition expression for condition triggers")
@click.pass_context
def proactive_add_trigger(
    ctx: click.Context,
    name: str,
    trigger_type: str,
    action: str,
    schedule: str | None,
    interval: int | None,
    condition: str | None,
) -> None:
    """Add a proactive trigger."""
    engine = get_proactive_engine(ctx)

    type_map = {
        "scheduled": TriggerType.SCHEDULED,
        "interval": TriggerType.INTERVAL,
        "condition": TriggerType.CONDITION,
    }

    trigger_id = run_async(
        engine.add_trigger(
            name=name,
            trigger_type=type_map[trigger_type],
            action=action,
            schedule=schedule,
            interval_minutes=interval or 0,
            condition=condition,
        )
    )
    click.echo(f"Trigger added: {trigger_id}")


@proactive.command("remove-trigger")
@click.argument("trigger_id")
@click.pass_context
def proactive_remove_trigger(ctx: click.Context, trigger_id: str) -> None:
    """Remove a trigger."""
    engine = get_proactive_engine(ctx)
    success = run_async(engine.remove_trigger(trigger_id))

    if success:
        click.echo(f"Trigger {trigger_id} removed")
    else:
        click.echo(f"Trigger {trigger_id} not found", err=True)
        sys.exit(1)


@proactive.command("observations")
@click.pass_context
def proactive_observations(ctx: click.Context) -> None:
    """List pending observations."""
    engine = get_proactive_engine(ctx)
    observations = run_async(engine.get_pending_observations())

    if not observations:
        click.echo("No pending observations")
        return

    for obs in observations:
        priority = "!" if obs.importance >= 0.8 else " "
        click.echo(f"{priority}[{obs.category}] {obs.content}")
        click.echo(f"  id: {obs.id} | importance: {obs.importance:.1f} | source: {obs.source}")


@proactive.command("observe")
@click.argument("content")
@click.option("--category", "-c", default="note",
              type=click.Choice(["reminder", "deadline", "follow_up", "anomaly", "opportunity", "note"]))
@click.option("--importance", "-i", type=float, default=0.5, help="Importance (0.0-1.0)")
@click.option("--expires", "-e", type=int, help="Expires in N hours")
@click.pass_context
def proactive_observe(
    ctx: click.Context,
    content: str,
    category: str,
    importance: float,
    expires: int | None,
) -> None:
    """Add an observation manually."""
    engine = get_proactive_engine(ctx)
    obs_id = run_async(
        engine.add_observation(
            content=content,
            category=category,
            importance=importance,
            source="cli",
            expires_in_hours=expires,
        )
    )
    click.echo(f"Observation added: {obs_id}")


@proactive.command("done")
@click.argument("observation_id")
@click.pass_context
def proactive_done(ctx: click.Context, observation_id: str) -> None:
    """Mark an observation as handled."""
    engine = get_proactive_engine(ctx)
    success = run_async(engine.mark_observation_handled(observation_id))

    if success:
        click.echo(f"Observation {observation_id} marked as done")
    else:
        click.echo(f"Observation {observation_id} not found", err=True)
        sys.exit(1)


# Google integration commands


@cli.group()
def google():
    """Manage Google Calendar and Gmail integration."""
    pass


def get_google_connector(ctx: click.Context) -> GoogleConnector:
    """Get or create Google connector."""
    if "google" not in ctx.obj:
        config = ctx.obj["config"]
        connector = GoogleConnector(config.data_dir)
        ctx.obj["google"] = connector
    return ctx.obj["google"]


@google.command("setup")
@click.option("--client-id", prompt="Google OAuth Client ID", help="OAuth 2.0 Client ID")
@click.option("--client-secret", prompt="Google OAuth Client Secret", hide_input=True,
              help="OAuth 2.0 Client Secret")
@click.pass_context
def google_setup(ctx: click.Context, client_id: str, client_secret: str) -> None:
    """Set up Google OAuth credentials.

    You need to create OAuth credentials in the Google Cloud Console:
    1. Go to https://console.cloud.google.com/apis/credentials
    2. Create a new OAuth 2.0 Client ID (Desktop app)
    3. Enable Calendar API and Gmail API
    4. Enter the Client ID and Secret here
    """
    connector = get_google_connector(ctx)
    run_async(connector.setup_credentials(client_id, client_secret))
    click.echo("Google credentials saved. Run 'gru google login' to authenticate.")


@google.command("login")
@click.option("--headless", is_flag=True, help="Use console-based auth flow (for remote/headless)")
@click.pass_context
def google_login(ctx: click.Context, headless: bool) -> None:
    """Authenticate with Google (opens browser for OAuth)."""
    connector = get_google_connector(ctx)

    if not connector.is_configured():
        click.echo("Google not configured. Run 'gru google setup' first.", err=True)
        sys.exit(1)

    if headless:
        click.echo("Opening console-based authentication...")
    else:
        click.echo("Opening browser for authentication...")

    success = connector.authenticate(headless=headless)

    if success:
        click.echo("Successfully authenticated with Google!")

        # Set up auto-sync triggers
        engine = get_proactive_engine(ctx)
        engine.set_google_connector(connector)
        run_async(setup_google_triggers(engine, connector))
        click.echo("Auto-sync triggers configured (calendar: 15min, email: 5min)")
    else:
        click.echo("Authentication failed", err=True)
        sys.exit(1)


@google.command("status")
@click.pass_context
def google_status(ctx: click.Context) -> None:
    """Show Google integration status."""
    connector = get_google_connector(ctx)
    status = connector.get_status()

    click.echo("Google Integration Status")
    click.echo("=" * 40)
    click.echo(f"Configured: {status['configured']}")
    click.echo(f"Authenticated: {status['authenticated']}")

    if status["last_calendar_sync"]:
        click.echo(f"Last calendar sync: {status['last_calendar_sync']}")
    if status["last_email_sync"]:
        click.echo(f"Last email sync: {status['last_email_sync']}")

    click.echo(f"Events seen: {status['seen_events']}")
    click.echo(f"Emails seen: {status['seen_emails']}")


@google.command("sync")
@click.option("--calendar", "sync_calendar", is_flag=True, help="Sync calendar only")
@click.option("--email", "sync_email", is_flag=True, help="Sync email only")
@click.pass_context
def google_sync(ctx: click.Context, sync_calendar: bool, sync_email: bool) -> None:
    """Manually sync Google Calendar and/or Gmail."""
    connector = get_google_connector(ctx)

    if not connector.load_token():
        click.echo("Not authenticated. Run 'gru google login' first.", err=True)
        sys.exit(1)

    engine = get_proactive_engine(ctx)
    engine.set_google_connector(connector)

    # Default: sync both if neither flag specified
    if not sync_calendar and not sync_email:
        sync_calendar = True
        sync_email = True

    results = {"events": 0, "emails": 0}

    if sync_calendar:
        click.echo("Syncing calendar...")
        events = run_async(connector.sync_calendar(engine))
        results["events"] = len(events)
        click.echo(f"  Found {len(events)} new events")

    if sync_email:
        click.echo("Syncing email...")
        emails = run_async(connector.sync_email(engine))
        results["emails"] = len(emails)
        click.echo(f"  Found {len(emails)} new emails")

    # Show any new observations
    pending = run_async(engine.get_pending_observations())
    if pending:
        click.echo(f"\nNew observations ({len(pending)}):")
        for obs in pending[:5]:
            priority = "!" if obs.importance >= 0.8 else " "
            click.echo(f"  {priority}[{obs.category}] {obs.content}")


@google.command("clear-cache")
@click.pass_context
def google_clear_cache(ctx: click.Context) -> None:
    """Clear the seen items cache (allows re-syncing all items)."""
    connector = get_google_connector(ctx)
    connector.clear_seen_cache()
    click.echo("Cleared seen items cache. Next sync will fetch all recent items.")


# Session/Chat commands


@cli.command()
@click.argument("message")
@click.option("--user", "-u", default="default", help="User ID")
@click.option("--channel", "-c", default="cli", help="Channel (cli, telegram, discord, etc.)")
@click.pass_context
def chat(ctx: click.Context, message: str, user: str, channel: str) -> None:
    """Send a message and get a response (conversational mode).

    Unlike 'spawn', this maintains conversation context and handles
    simple requests directly without spawning full agents.

    Examples:
        gru chat "What's on my calendar today?"
        gru chat "Remind me to call Mom tomorrow"
        gru chat "Build me a REST API for users" (will spawn agent)
    """
    orchestrator = get_orchestrator(ctx)

    result = run_async(orchestrator.chat(user, message, channel))

    click.echo(result["response"])

    if result.get("quick_action"):
        action = result["quick_action"]
        click.echo(f"\n[Quick action: {action.get('type', 'unknown')}]")

    if result.get("agent_id"):
        click.echo(f"\nUse 'gru status {result['agent_id']}' to check agent progress")


@cli.group()
def session():
    """Manage conversation sessions."""
    pass


@session.command("status")
@click.option("--user", "-u", default="default", help="User ID")
@click.option("--channel", "-c", default="cli", help="Channel")
@click.pass_context
def session_status(ctx: click.Context, user: str, channel: str) -> None:
    """Show current session info."""
    orchestrator = get_orchestrator(ctx)
    session_info = run_async(orchestrator.get_session(user, channel))

    if not session_info:
        click.echo("No active session")
        return

    click.echo("Session Info")
    click.echo("=" * 40)
    click.echo(f"ID: {session_info['id']}")
    click.echo(f"User: {session_info['user_id']}")
    click.echo(f"Channel: {session_info['channel']}")
    click.echo(f"Persona: {session_info['persona']}")
    click.echo(f"Messages: {session_info['message_count']}")
    click.echo(f"Created: {session_info['created_at']}")
    click.echo(f"Last active: {session_info['last_active']}")


@session.command("reset")
@click.option("--user", "-u", default="default", help="User ID")
@click.option("--channel", "-c", default="cli", help="Channel")
@click.pass_context
def session_reset(ctx: click.Context, user: str, channel: str) -> None:
    """Reset/clear session history."""
    orchestrator = get_orchestrator(ctx)
    success = run_async(orchestrator.reset_session(user, channel))

    if success:
        click.echo("Session reset")
    else:
        click.echo("Could not reset session", err=True)
        sys.exit(1)


@session.command("persona")
@click.argument("persona_name", required=False)
@click.option("--user", "-u", default="default", help="User ID")
@click.pass_context
def session_persona(ctx: click.Context, persona_name: str | None, user: str) -> None:
    """Set or show persona.

    Without arguments, lists available personas.
    With a persona name, sets it as the default for the user.
    """
    orchestrator = get_orchestrator(ctx)

    if not persona_name:
        click.echo("Available personas:")
        for p in get_available_personas():
            click.echo(f"  {p['name']}: {p['description']}")
        return

    if persona_name not in PERSONAS:
        click.echo(f"Unknown persona: {persona_name}", err=True)
        click.echo(f"Available: {', '.join(PERSONAS.keys())}")
        sys.exit(1)

    success = run_async(orchestrator.set_persona(user, persona_name))

    if success:
        persona = PERSONAS[persona_name]
        click.echo(f"Persona set to '{persona_name}': {persona.description}")
    else:
        click.echo(f"Could not set persona", err=True)
        sys.exit(1)


@session.command("stats")
@click.option("--user", "-u", default="default", help="User ID")
@click.pass_context
def session_stats(ctx: click.Context, user: str) -> None:
    """Show session statistics for a user."""
    orchestrator = get_orchestrator(ctx)

    run_async(orchestrator.initialize_sessions())

    if not orchestrator.session_manager:
        click.echo("Session manager not available", err=True)
        sys.exit(1)

    stats = run_async(orchestrator.session_manager.get_session_stats(user))

    click.echo(f"Session Stats for {user}")
    click.echo("=" * 40)
    click.echo(f"Total sessions: {stats['total_sessions']}")

    for s in stats.get("sessions", []):
        click.echo(f"\n  [{s['channel']}] {s['persona']}")
        click.echo(f"    Messages: {s['message_count']}")
        click.echo(f"    Last active: {s['last_active']}")


# Action commands


@cli.group()
def action():
    """Execute browser actions (ordering, searching, etc.)."""
    pass


def get_action_executor(ctx: click.Context) -> ActionExecutor:
    """Get or create action executor."""
    if "action_executor" not in ctx.obj:
        config = ctx.obj["config"]
        mem = get_memory_store(ctx) if config.memory_enabled else None
        executor = ActionExecutor(config, mem)
        ctx.obj["action_executor"] = executor
    return ctx.obj["action_executor"]


@action.command("list")
@click.option("--category", "-c", help="Filter by category")
@click.pass_context
def action_list(ctx: click.Context, category: str | None) -> None:
    """List available actions."""
    registry = get_registry()
    actions = registry.list_actions(category)

    if not actions:
        click.echo("No actions available")
        return

    # Group by category
    by_category = {}
    for a in actions:
        cat = a["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(a)

    for cat, cat_actions in sorted(by_category.items()):
        click.echo(f"\n{cat.upper()}")
        click.echo("-" * 40)
        for a in cat_actions:
            auth = " [auth]" if a["requires_auth"] else ""
            confirm = " [confirm]" if a["requires_confirmation"] else ""
            click.echo(f"  {a['name']}: {a['description']}{auth}{confirm}")


@action.command("run")
@click.argument("action_name")
@click.option("--param", "-p", multiple=True, help="Parameters as key=value")
@click.option("--user", "-u", default="default", help="User ID")
@click.option("--location", "-l", help="Location (address)")
@click.pass_context
def action_run(
    ctx: click.Context,
    action_name: str,
    param: tuple[str, ...],
    user: str,
    location: str | None,
) -> None:
    """Run an action.

    Examples:
        gru action run web_search -p query="best pizza NYC"
        gru action run restaurant_search -p cuisine=burger -l "San Francisco"
        gru action run ubereats_search -p query="cheeseburger"
    """
    executor = get_action_executor(ctx)

    # Parse params
    params = {}
    for p in param:
        if "=" in p:
            key, value = p.split("=", 1)
            params[key] = value
        else:
            click.echo(f"Invalid param format: {p} (use key=value)", err=True)
            sys.exit(1)

    loc = {"address": location} if location else None

    run_async(executor.start())

    try:
        result = run_async(executor.execute(
            action_name,
            user_id=user,
            location=loc,
            **params
        ))

        click.echo(f"\nStatus: {result.status.value}")
        click.echo(f"Message: {result.message}")

        if result.data:
            click.echo("\nData:")
            click.echo(json.dumps(result.data, indent=2, default=str)[:2000])

        if result.error:
            click.echo(f"\nError: {result.error}")

        if result.confirmation_required:
            click.echo("\nConfirmation required:")
            click.echo(json.dumps(result.confirmation_required, indent=2))

    finally:
        run_async(executor.stop())


@action.command("search")
@click.argument("query")
@click.option("--type", "-t", "search_type", default="web",
              type=click.Choice(["web", "local", "restaurant", "food"]))
@click.option("--location", "-l", help="Location for local searches")
@click.pass_context
def action_search(
    ctx: click.Context,
    query: str,
    search_type: str,
    location: str | None,
) -> None:
    """Quick search command.

    Examples:
        gru action search "python tutorials"
        gru action search "best burger" -t restaurant -l "Austin, TX"
        gru action search "pizza" -t food
    """
    executor = get_action_executor(ctx)

    action_map = {
        "web": "web_search",
        "local": "local_search",
        "restaurant": "restaurant_search",
        "food": "ubereats_search",
    }
    action_name = action_map[search_type]

    loc = {"address": location} if location else None

    run_async(executor.start())

    try:
        result = run_async(executor.execute(
            action_name,
            user_id="default",
            location=loc,
            query=query,
        ))

        click.echo(result.message)

        if result.data.get("results"):
            click.echo("\nResults:")
            for i, r in enumerate(result.data["results"][:5], 1):
                name = r.get("name") or r.get("title", "Unknown")
                rating = f" ({r['rating']}*)" if r.get("rating") else ""
                click.echo(f"  {i}. {name}{rating}")

    finally:
        run_async(executor.stop())


@action.command("browser")
@click.option("--status", "show_status", is_flag=True, help="Show browser status")
@click.option("--login", help="Start interactive login for a service")
@click.option("--logout", help="Clear auth for a service")
@click.pass_context
def action_browser(
    ctx: click.Context,
    show_status: bool,
    login: str | None,
    logout: str | None,
) -> None:
    """Manage browser and service authentication.

    Examples:
        gru action browser --status
        gru action browser --login ubereats
        gru action browser --logout ubereats
    """
    executor = get_action_executor(ctx)

    if show_status:
        run_async(executor.start())
        try:
            status = run_async(executor.get_browser_status())
            click.echo("Browser Status")
            click.echo("=" * 40)
            click.echo(f"Running: {status['running']}")
            click.echo(f"Mode: {'headless' if status.get('headless') else 'headed'}")
            click.echo(f"Type: {status.get('browser_type', 'unknown')}")
            if status.get("contexts"):
                click.echo(f"Active contexts: {', '.join(status['contexts'])}")
        finally:
            run_async(executor.stop())
        return

    if login:
        config = ctx.obj["config"]
        if config.browser_mode == "headless":
            click.echo("Interactive login requires headed mode.", err=True)
            click.echo("Set GRU_BROWSER_MODE=headed and run on a machine with display.")
            sys.exit(1)

        from gru.actions.auth import get_auth_manager

        run_async(executor.start())
        try:
            auth_manager = get_auth_manager(config.data_dir)
            success = run_async(auth_manager.login_interactive(executor._browser, login))
            if success:
                click.echo(f"Logged in to {login}")
            else:
                click.echo(f"Login failed for {login}", err=True)
                sys.exit(1)
        finally:
            run_async(executor.stop())
        return

    if logout:
        from gru.actions.auth import get_auth_manager

        run_async(executor.start())
        try:
            auth_manager = get_auth_manager(ctx.obj["config"].data_dir)
            success = run_async(auth_manager.logout(executor._browser, logout))
            if success:
                click.echo(f"Logged out of {logout}")
            else:
                click.echo(f"Logout failed for {logout}", err=True)
        finally:
            run_async(executor.stop())
        return

    # Default: show available services
    from gru.actions.auth import get_auth_manager
    auth_manager = get_auth_manager(ctx.obj["config"].data_dir)
    services = auth_manager.list_services()

    click.echo("Available services:")
    for s in services:
        status = "authenticated" if s["authenticated"] else "not logged in"
        click.echo(f"  {s['service']}: {status}")


# Setup and config commands


@cli.command("setup")
@click.option("--status", "show_status", is_flag=True, help="Show setup status only")
@click.pass_context
def setup_wizard(ctx: click.Context, show_status: bool) -> None:
    """Interactive setup wizard for Gru.

    Guides you through configuring:
    - Anthropic API key
    - Messaging platform (Telegram/Discord/Slack)
    - Google integration (optional)
    - Admin user ID

    You can also paste API keys directly - Gru will auto-detect and configure them.
    """
    config = ctx.obj["config"]
    wizard = get_setup_wizard(config.data_dir)

    if show_status:
        click.echo(wizard.get_setup_instructions())
        return

    status = wizard.get_setup_status()

    click.echo("Gru Setup Wizard")
    click.echo("=" * 40)
    click.echo("")

    # Show current status
    for step_id, step in status["steps"].items():
        icon = "[x]" if step["configured"] else "[ ]"
        required = "(required)" if step["required"] else "(optional)"
        click.echo(f"{icon} {step['name']} {required}")

    click.echo("")

    if status["complete"]:
        click.echo("Setup is complete! Gru is ready to run.")
        click.echo("")
        click.echo("Start Gru with: gru run")
        return

    # Interactive setup for missing items
    cfg_manager = get_config_manager(config.data_dir)

    # Anthropic API
    if not status["steps"]["anthropic"]["configured"]:
        click.echo("")
        click.echo("Step 1: Anthropic API Key")
        click.echo("-" * 40)
        click.echo("Get your API key from: https://console.anthropic.com/")
        click.echo("")

        key = click.prompt("Paste your Anthropic API key", hide_input=True)
        if key:
            result = cfg_manager.set("anthropic-key", key.strip())
            if result.key_type == KeyType.ANTHROPIC:
                click.echo("Anthropic API key configured!")
            else:
                click.echo("Warning: Key format not recognized, but saved anyway.")

    # Messaging platform
    if not status["steps"]["messaging"]["configured"]:
        click.echo("")
        click.echo("Step 2: Messaging Platform")
        click.echo("-" * 40)
        click.echo("Choose a platform to interact with Gru:")
        click.echo("  1. Telegram (easiest)")
        click.echo("  2. Discord")
        click.echo("  3. Slack")
        click.echo("")

        choice = click.prompt("Select platform", type=click.Choice(["1", "2", "3"]), default="1")

        if choice == "1":
            click.echo("")
            click.echo("Telegram Setup:")
            click.echo("1. Message @BotFather on Telegram")
            click.echo("2. Send /newbot and follow prompts")
            click.echo("3. Copy the bot token")
            click.echo("")
            token = click.prompt("Paste your Telegram bot token", hide_input=True)
            if token:
                cfg_manager.set("telegram-token", token.strip())
                click.echo("Telegram token configured!")

            click.echo("")
            click.echo("Now get your Telegram user ID:")
            click.echo("Message @userinfobot on Telegram to get your ID")
            user_id = click.prompt("Your Telegram user ID")
            if user_id:
                cfg_manager.set("admin-id", user_id.strip())
                # Also update .env with the proper format
                cfg_manager._update_env_file("admin-ids", user_id.strip(), KeyType.UNKNOWN)
                click.echo("Admin ID configured!")

        elif choice == "2":
            click.echo("")
            click.echo("Discord Setup:")
            click.echo("1. Go to https://discord.com/developers/applications")
            click.echo("2. Create a New Application")
            click.echo("3. Go to Bot section, click 'Add Bot'")
            click.echo("4. Copy the bot token")
            click.echo("")
            token = click.prompt("Paste your Discord bot token", hide_input=True)
            if token:
                cfg_manager.set("discord-token", token.strip())
                click.echo("Discord token configured!")

            click.echo("")
            click.echo("Enable Developer Mode in Discord (Settings > Advanced)")
            click.echo("Right-click your username and 'Copy ID'")
            user_id = click.prompt("Your Discord user ID")
            if user_id:
                cfg_manager.set("discord-admin-id", user_id.strip())
                click.echo("Admin ID configured!")

        elif choice == "3":
            click.echo("")
            click.echo("Slack Setup:")
            click.echo("1. Go to https://api.slack.com/apps")
            click.echo("2. Create New App > From scratch")
            click.echo("3. Add Bot Token Scopes under OAuth")
            click.echo("4. Install to workspace")
            click.echo("")
            bot_token = click.prompt("Paste your Slack Bot Token (xoxb-...)", hide_input=True)
            if bot_token:
                cfg_manager.set("slack-bot-token", bot_token.strip())

            click.echo("")
            click.echo("Now get the App-Level Token:")
            click.echo("Go to Basic Information > App-Level Tokens > Generate")
            app_token = click.prompt("Paste your Slack App Token (xapp-...)", hide_input=True)
            if app_token:
                cfg_manager.set("slack-app-token", app_token.strip())
                click.echo("Slack tokens configured!")

            click.echo("")
            user_id = click.prompt("Your Slack user ID (find in profile)")
            if user_id:
                cfg_manager.set("slack-admin-id", user_id.strip())
                click.echo("Admin ID configured!")

    # Google integration (optional)
    click.echo("")
    if click.confirm("Set up Google integration? (Calendar, Gmail, Docs)", default=False):
        click.echo("")
        click.echo("Google Setup:")
        click.echo("1. Go to https://console.cloud.google.com/apis/credentials")
        click.echo("2. Create OAuth 2.0 Client ID (Desktop app)")
        click.echo("3. Enable Calendar, Gmail, and Docs APIs")
        click.echo("")

        client_id = click.prompt("Google OAuth Client ID")
        client_secret = click.prompt("Google OAuth Client Secret", hide_input=True)

        if client_id and client_secret:
            cfg_manager.set("google-client-id", client_id.strip())
            cfg_manager.set("google-client-secret", client_secret.strip())

            # Set up credentials file for Google connector
            from gru.connectors.google import GoogleConnector
            connector = GoogleConnector(config.data_dir)
            run_async(connector.setup_credentials(client_id.strip(), client_secret.strip()))

            click.echo("Google credentials saved!")
            click.echo("")
            if click.confirm("Authenticate with Google now?", default=True):
                success = connector.authenticate()
                if success:
                    click.echo("Google authentication successful!")
                else:
                    click.echo("Authentication failed. Run 'gru google login' later.")

    click.echo("")
    click.echo("=" * 40)
    click.echo("Setup complete!")
    click.echo("")
    click.echo("Start Gru with: gru run")
    click.echo("Or chat via CLI: gru chat \"Hello!\"")


@cli.group()
def config():
    """Manage Gru configuration."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value.

    Keys can be set explicitly or auto-detected from the value format.

    Examples:
        gru config set anthropic-key sk-ant-...
        gru config set telegram-token 123456:ABC...
        gru config set admin-id 12345678

    Common keys:
        anthropic-key      Anthropic API key
        telegram-token     Telegram bot token
        discord-token      Discord bot token
        slack-bot-token    Slack bot token
        slack-app-token    Slack app token
        admin-id           Your user ID
        location           Default location for searches
    """
    cfg = get_config_manager(ctx.obj["config"].data_dir)
    result = cfg.set(key, value)

    if result.key_type:
        click.echo(f"Set {key} ({KEY_TYPE_NAMES[result.key_type]})")
    else:
        click.echo(f"Set {key}")

    if result.is_secret:
        click.echo("(saved to .env file)")


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx: click.Context, key: str) -> None:
    """Get a configuration value."""
    cfg = get_config_manager(ctx.obj["config"].data_dir)
    value = cfg.get(key)

    if value:
        # Mask secrets
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            if len(value) > 16:
                value = value[:8] + "..." + value[-4:]
        click.echo(f"{key}: {value}")
    else:
        click.echo(f"{key}: (not set)")


@config.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """List all configuration."""
    cfg = get_config_manager(ctx.obj["config"].data_dir)
    status = cfg.get_status()

    click.echo("Gru Configuration")
    click.echo("=" * 40)

    if status["configured"]:
        click.echo("\nConfigured:")
        for name, masked in status["configured"].items():
            click.echo(f"  {name}: {masked}")

    if status["missing"]:
        click.echo("\nMissing (required):")
        for name in status["missing"]:
            click.echo(f"  {name}")

    click.echo("")
    click.echo("Run 'gru setup' for interactive configuration.")


@config.command("add")
@click.argument("value")
@click.pass_context
def config_add(ctx: click.Context, value: str) -> None:
    """Auto-detect and add a key/token.

    Just paste a key and Gru will figure out what it is.

    Example:
        gru config add sk-ant-api03-xxx
        gru config add 123456789:AABBccDDee...
    """
    cfg = get_config_manager(ctx.obj["config"].data_dir)

    key_type = detect_key_type(value)

    if key_type == KeyType.UNKNOWN:
        click.echo("Could not detect key type. Use 'gru config set <key> <value>' instead.")
        sys.exit(1)

    result = cfg.set_from_detection(value)

    if result:
        click.echo(f"Detected: {KEY_TYPE_NAMES[key_type]}")
        click.echo(f"Configured: {result.key}")
    else:
        click.echo("Failed to configure key.")
        sys.exit(1)


@config.command("delete")
@click.argument("key")
@click.pass_context
def config_delete(ctx: click.Context, key: str) -> None:
    """Delete a configuration value."""
    cfg = get_config_manager(ctx.obj["config"].data_dir)
    success = cfg.delete(key)

    if success:
        click.echo(f"Deleted: {key}")
    else:
        click.echo(f"Key not found: {key}")


@config.command("status")
@click.pass_context
def config_status(ctx: click.Context) -> None:
    """Show setup status and what's needed."""
    wizard = get_setup_wizard(ctx.obj["config"].data_dir)
    click.echo(wizard.get_setup_instructions())


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
