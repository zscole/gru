"""CLI interface for Gru using Click."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import click

from gru.config import Config
from gru.crypto import CryptoManager, SecretStore
from gru.db import Database
from gru.orchestrator import Orchestrator


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


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
