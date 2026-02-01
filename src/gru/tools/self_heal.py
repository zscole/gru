"""Self-healing and diagnostic tools."""

from __future__ import annotations

import logging
from typing import Any

from gru.tools.base import register_tool

logger = logging.getLogger(__name__)


async def run_self_diagnostic() -> dict[str, Any]:
    """Run comprehensive self-diagnostics on Gru."""
    from gru.self_heal import get_self_heal_engine

    engine = get_self_heal_engine()
    result = await engine.run_diagnostics()

    return {
        "status": result.status.value,
        "checks_passed": result.checks_passed,
        "checks_failed": result.checks_failed,
        "issues": [
            {
                "id": i.id,
                "category": i.category.value,
                "severity": i.severity,
                "description": i.description,
                "auto_fixable": i.auto_fixable,
            }
            for i in result.issues
        ],
        "recommendations": result.recommendations,
        "timestamp": result.timestamp.isoformat(),
    }


async def self_heal(issue_id: str | None = None) -> dict[str, Any]:
    """Attempt to automatically fix detected issues."""
    from gru.self_heal import get_self_heal_engine

    engine = get_self_heal_engine()
    return await engine.self_heal(issue_id)


async def get_health_status() -> dict[str, Any]:
    """Get current health status summary."""
    from gru.self_heal import get_self_heal_engine

    engine = get_self_heal_engine()
    return engine.get_status()


async def list_issues(include_fixed: bool = False) -> dict[str, Any]:
    """List detected issues."""
    from gru.self_heal import get_self_heal_engine

    engine = get_self_heal_engine()
    issues = engine.get_issues(include_fixed=include_fixed)
    return {
        "count": len(issues),
        "issues": issues,
    }


async def analyze_error(
    error_message: str, traceback: str | None = None, file_path: str | None = None
) -> dict[str, Any]:
    """Analyze an error and attempt to fix it."""
    from gru.self_heal import IssueCategory, get_self_heal_engine

    engine = get_self_heal_engine()

    # Create a synthetic exception for analysis
    class AnalyzedError(Exception):
        pass

    error = AnalyzedError(error_message)

    # Record the error
    engine.record_error(
        error_type=error_message.split(":")[0] if ":" in error_message else "Unknown",
        error=error,
        context={"traceback": traceback, "file_path": file_path},
    )

    # Create issue and attempt fix
    issue = engine._create_issue(
        category=IssueCategory.CODE_BUG,
        severity="high",
        description=error_message,
        context={
            "traceback": traceback,
            "file_path": file_path,
        },
        auto_fixable=True,
    )

    # Attempt fix
    fix_result = await engine.self_heal(issue.id)

    return {
        "issue_id": issue.id,
        "error_analyzed": error_message,
        "fix_attempted": True,
        "fix_result": fix_result,
    }


def register_self_heal_tools() -> None:
    """Register all self-healing tools."""
    register_tool(
        name="run_self_diagnostic",
        description="Run comprehensive self-diagnostics on Gru. Checks system health including Python environment, dependencies, database, MCP servers, memory, stuck agents, error rates, and disk space. Use this to understand Gru's current health status.",
        parameters={},
        handler=run_self_diagnostic,
    )

    register_tool(
        name="self_heal",
        description="Attempt to automatically fix detected issues. Can fix stuck agents, restart services, install missing packages, and spawn fix agents for code bugs. Run run_self_diagnostic first to detect issues.",
        parameters={
            "issue_id": {
                "type": "string",
                "description": "Specific issue ID to fix. If not provided, fixes all auto-fixable issues.",
                "optional": True,
            },
        },
        handler=self_heal,
    )

    register_tool(
        name="get_health_status",
        description="Get current health status summary including overall status (healthy/degraded/unhealthy/critical), open issues by severity, and error counts.",
        parameters={},
        handler=get_health_status,
    )

    register_tool(
        name="list_issues",
        description="List detected issues that need attention. Shows issue details including severity, category, and whether they can be auto-fixed.",
        parameters={
            "include_fixed": {
                "type": "boolean",
                "description": "Whether to include already-fixed issues.",
                "optional": True,
            },
        },
        handler=list_issues,
    )

    register_tool(
        name="analyze_error",
        description="Analyze an error and attempt to fix it. Use this when you encounter an error to trigger self-healing. Can spawn a fix agent for complex code bugs.",
        parameters={
            "error_message": {
                "type": "string",
                "description": "The error message to analyze.",
            },
            "traceback": {
                "type": "string",
                "description": "Optional full traceback.",
                "optional": True,
            },
            "file_path": {
                "type": "string",
                "description": "Optional file path where error occurred.",
                "optional": True,
            },
        },
        handler=analyze_error,
    )
