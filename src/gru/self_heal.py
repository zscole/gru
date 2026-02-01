"""Self-healing and self-diagnostic capabilities for Gru."""

from __future__ import annotations

import logging
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gru.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of detected issues."""

    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    MEMORY = "memory"
    STUCK_AGENT = "stuck_agent"
    SERVICE_DOWN = "service_down"
    CODE_BUG = "code_bug"
    CONFIGURATION = "configuration"


@dataclass
class Issue:
    """Represents a detected issue."""

    id: str
    category: IssueCategory
    severity: str  # low, medium, high, critical
    description: str
    detected_at: datetime
    context: dict[str, Any] = field(default_factory=dict)
    auto_fixable: bool = False
    fix_attempted: bool = False
    fix_successful: bool | None = None
    fix_details: str | None = None


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""

    status: HealthStatus
    checks_passed: int
    checks_failed: int
    issues: list[Issue]
    recommendations: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


class SelfHealEngine:
    """Engine for self-diagnosis and self-healing."""

    def __init__(self, orchestrator: Orchestrator | None = None) -> None:
        self.orchestrator = orchestrator
        self._error_counts: dict[str, int] = {}
        self._error_timestamps: dict[str, list[datetime]] = {}
        self._performance_samples: list[dict[str, Any]] = []
        self._issues: dict[str, Issue] = {}
        self._fix_history: list[dict[str, Any]] = []
        self._last_diagnostic: DiagnosticResult | None = None

    def record_error(self, error_type: str, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Record an error for pattern detection."""
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

        if error_type not in self._error_timestamps:
            self._error_timestamps[error_type] = []
        self._error_timestamps[error_type].append(datetime.now())

        # Keep only last hour of timestamps
        cutoff = datetime.now() - timedelta(hours=1)
        self._error_timestamps[error_type] = [ts for ts in self._error_timestamps[error_type] if ts > cutoff]

        # Check for error spike
        recent_count = len(self._error_timestamps[error_type])
        if recent_count >= 10:
            self._create_issue(
                category=IssueCategory.ERROR_RATE,
                severity="high",
                description=f"Error spike detected: {error_type} occurred {recent_count} times in last hour",
                context={
                    "error_type": error_type,
                    "count": recent_count,
                    "sample_error": str(error)[:500],
                    "traceback": traceback.format_exc()[:1000] if context else None,
                },
                auto_fixable=True,
            )

    def record_performance(self, operation: str, duration_ms: float, tokens: int = 0) -> None:
        """Record performance metrics."""
        self._performance_samples.append(
            {
                "operation": operation,
                "duration_ms": duration_ms,
                "tokens": tokens,
                "timestamp": datetime.now(),
            }
        )

        # Keep only last 1000 samples
        if len(self._performance_samples) > 1000:
            self._performance_samples = self._performance_samples[-1000:]

        # Detect slow operations
        if duration_ms > 30000:  # 30 seconds
            self._create_issue(
                category=IssueCategory.PERFORMANCE,
                severity="medium",
                description=f"Slow operation detected: {operation} took {duration_ms / 1000:.1f}s",
                context={"operation": operation, "duration_ms": duration_ms},
                auto_fixable=False,
            )

    def _create_issue(
        self,
        category: IssueCategory,
        severity: str,
        description: str,
        context: dict[str, Any] | None = None,
        auto_fixable: bool = False,
    ) -> Issue:
        """Create and track an issue."""
        import uuid

        issue_id = str(uuid.uuid4())[:8]

        issue = Issue(
            id=issue_id,
            category=category,
            severity=severity,
            description=description,
            detected_at=datetime.now(),
            context=context or {},
            auto_fixable=auto_fixable,
        )

        self._issues[issue_id] = issue
        logger.warning(f"Issue detected [{issue_id}]: {description}")

        return issue

    async def run_diagnostics(self) -> DiagnosticResult:
        """Run comprehensive self-diagnostics."""
        issues: list[Issue] = []
        checks_passed = 0
        checks_failed = 0
        recommendations: list[str] = []

        # Check 1: Python environment
        try:
            python_version = sys.version_info
            if python_version < (3, 10):
                issues.append(
                    self._create_issue(
                        IssueCategory.CONFIGURATION,
                        "high",
                        f"Python version {python_version.major}.{python_version.minor} is below recommended 3.10+",
                    )
                )
                checks_failed += 1
            else:
                checks_passed += 1
        except Exception as e:
            logger.error(f"Python version check failed: {e}")
            checks_failed += 1

        # Check 2: Required dependencies
        required_packages = ["anthropic", "aiohttp", "aiosqlite"]
        for package in required_packages:
            try:
                __import__(package)
                checks_passed += 1
            except ImportError:
                issues.append(
                    self._create_issue(
                        IssueCategory.CONFIGURATION,
                        "critical",
                        f"Required package '{package}' not installed",
                        auto_fixable=True,
                    )
                )
                checks_failed += 1

        # Check 3: Database connectivity
        if self.orchestrator and self.orchestrator.db:
            try:
                async with self.orchestrator.db._get_connection() as conn:
                    await conn.execute("SELECT 1")
                checks_passed += 1
            except Exception as e:
                issues.append(
                    self._create_issue(
                        IssueCategory.SERVICE_DOWN,
                        "critical",
                        f"Database connectivity issue: {e}",
                    )
                )
                checks_failed += 1

        # Check 4: MCP server health
        if self.orchestrator and self.orchestrator.mcp:
            try:
                health = await self.orchestrator.mcp.health_check()
                unhealthy = [name for name, status in health.items() if not status]
                if unhealthy:
                    issues.append(
                        self._create_issue(
                            IssueCategory.SERVICE_DOWN,
                            "high",
                            f"Unhealthy MCP servers: {', '.join(unhealthy)}",
                            context={"unhealthy_servers": unhealthy},
                            auto_fixable=True,
                        )
                    )
                    checks_failed += 1
                else:
                    checks_passed += 1
            except Exception as e:
                logger.error(f"MCP health check failed: {e}")
                checks_failed += 1

        # Check 5: Memory usage
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = usage.ru_maxrss / 1024 / 1024  # Convert to MB on macOS
            if sys.platform == "linux":
                memory_mb = usage.ru_maxrss / 1024  # Already in KB on Linux

            if memory_mb > 1024:  # Over 1GB
                issues.append(
                    self._create_issue(
                        IssueCategory.MEMORY,
                        "medium",
                        f"High memory usage: {memory_mb:.0f}MB",
                        context={"memory_mb": memory_mb},
                    )
                )
                recommendations.append("Consider restarting to free memory")
                checks_failed += 1
            else:
                checks_passed += 1
        except Exception:
            checks_passed += 1  # Skip if resource module not available

        # Check 6: Stuck agents
        if self.orchestrator:
            for agent_id, agent in self.orchestrator._agents.items():
                if agent.is_stuck(5):  # 5 turns without tool calls
                    issues.append(
                        self._create_issue(
                            IssueCategory.STUCK_AGENT,
                            "medium",
                            f"Agent {agent_id} appears stuck",
                            context={"agent_id": agent_id, "turns": agent._turns_since_tool},
                            auto_fixable=True,
                        )
                    )
                    checks_failed += 1

        # Check 7: Error rate analysis
        high_error_types = [error_type for error_type, count in self._error_counts.items() if count > 50]
        if high_error_types:
            for error_type in high_error_types:
                issues.append(
                    self._create_issue(
                        IssueCategory.ERROR_RATE,
                        "high",
                        f"High error count for {error_type}: {self._error_counts[error_type]}",
                        auto_fixable=True,
                    )
                )
            checks_failed += 1
        else:
            checks_passed += 1

        # Check 8: Disk space
        try:
            import shutil

            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            if free_gb < 1:
                issues.append(
                    self._create_issue(
                        IssueCategory.CONFIGURATION,
                        "critical",
                        f"Low disk space: {free_gb:.1f}GB free",
                    )
                )
                checks_failed += 1
            else:
                checks_passed += 1
        except Exception:
            checks_passed += 1

        # Determine overall status
        if any(i.severity == "critical" for i in issues):
            status = HealthStatus.CRITICAL
        elif checks_failed > checks_passed:
            status = HealthStatus.UNHEALTHY
        elif checks_failed > 0:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        # Generate recommendations
        if status != HealthStatus.HEALTHY:
            recommendations.append("Run self-heal to attempt automatic fixes")

        auto_fixable = [i for i in issues if i.auto_fixable and not i.fix_attempted]
        if auto_fixable:
            recommendations.append(f"{len(auto_fixable)} issues can be auto-fixed")

        result = DiagnosticResult(
            status=status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            issues=issues,
            recommendations=recommendations,
        )

        self._last_diagnostic = result
        return result

    async def self_heal(self, issue_id: str | None = None) -> dict[str, Any]:
        """Attempt to automatically fix detected issues."""
        if issue_id:
            issues_to_fix = [self._issues[issue_id]] if issue_id in self._issues else []
        else:
            issues_to_fix = [i for i in self._issues.values() if i.auto_fixable and not i.fix_attempted]

        if not issues_to_fix:
            return {"fixed": 0, "failed": 0, "message": "No auto-fixable issues found"}

        fixed = 0
        failed = 0
        details: list[str] = []

        for issue in issues_to_fix:
            issue.fix_attempted = True
            try:
                success, detail = await self._attempt_fix(issue)
                issue.fix_successful = success
                issue.fix_details = detail

                if success:
                    fixed += 1
                    details.append(f"[FIXED] {issue.description}: {detail}")
                    logger.info(f"Fixed issue {issue.id}: {detail}")
                else:
                    failed += 1
                    details.append(f"[FAILED] {issue.description}: {detail}")
                    logger.warning(f"Failed to fix issue {issue.id}: {detail}")

            except Exception as e:
                failed += 1
                issue.fix_successful = False
                issue.fix_details = str(e)
                details.append(f"[ERROR] {issue.description}: {e}")
                logger.error(f"Error fixing issue {issue.id}: {e}")

        self._fix_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "fixed": fixed,
                "failed": failed,
                "details": details,
            }
        )

        return {
            "fixed": fixed,
            "failed": failed,
            "details": details,
            "message": f"Fixed {fixed} issues, {failed} failed",
        }

    async def _attempt_fix(self, issue: Issue) -> tuple[bool, str]:
        """Attempt to fix a specific issue."""

        if issue.category == IssueCategory.SERVICE_DOWN:
            # Try to restart MCP servers
            if "unhealthy_servers" in issue.context and self.orchestrator:
                recovered = await self.orchestrator.mcp.recover_unhealthy()
                if recovered > 0:
                    return True, f"Recovered {recovered} MCP server(s)"
                return False, "Could not recover servers"

        elif issue.category == IssueCategory.STUCK_AGENT:
            # Nudge stuck agent or cancel it
            agent_id = issue.context.get("agent_id")
            if agent_id and self.orchestrator:
                agent = self.orchestrator._agents.get(agent_id)
                if agent:
                    # Try to unstick by adding a nudge message
                    agent.messages.append(
                        {
                            "role": "user",
                            "content": "You appear to be stuck. Please use a tool to make progress, or if the task is complete, provide a final summary.",
                        }
                    )
                    agent._turns_since_tool = 0
                    return True, "Sent nudge message to stuck agent"
                return False, "Agent not found"

        elif issue.category == IssueCategory.ERROR_RATE:
            # Clear error history and log for investigation
            error_type = issue.context.get("error_type")
            if error_type:
                self._error_counts[error_type] = 0
                self._error_timestamps[error_type] = []
                return True, f"Reset error counters for {error_type}"

        elif issue.category == IssueCategory.CONFIGURATION:
            # Try to install missing packages
            if "package" in issue.description.lower():
                package_match = issue.description.split("'")
                if len(package_match) >= 2:
                    package = package_match[1]
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", package],
                            check=True,
                            capture_output=True,
                        )
                        return True, f"Installed {package}"
                    except subprocess.CalledProcessError as e:
                        return False, f"Failed to install {package}: {e}"

        elif issue.category == IssueCategory.CODE_BUG:
            # Spawn a fix agent to investigate and fix
            if self.orchestrator:
                bug_context = issue.context.get("traceback", issue.description)
                fix_task = f"""A bug has been detected in Gru's codebase:

{bug_context}

Please:
1. Analyze the error and identify the root cause
2. Locate the relevant code file(s)
3. Implement a fix
4. Verify the fix resolves the issue

Work in the Gru source directory and make minimal, targeted changes."""

                agent_id = await self.orchestrator.spawn_agent(
                    task=fix_task,
                    supervised=False,
                    priority="high",
                )
                return True, f"Spawned fix agent {agent_id}"

        return False, "No automatic fix available for this issue type"

    async def analyze_and_fix_code(self, error: Exception, file_path: str | None = None) -> dict[str, Any]:
        """Analyze a code error and attempt to fix it."""
        tb = traceback.format_exc()

        # Create a code bug issue
        issue = self._create_issue(
            category=IssueCategory.CODE_BUG,
            severity="high",
            description=f"Code error: {type(error).__name__}: {error}",
            context={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": tb,
                "file_path": file_path,
            },
            auto_fixable=True,
        )

        # Attempt fix
        return await self.self_heal(issue.id)

    def get_status(self) -> dict[str, Any]:
        """Get current health status summary."""
        open_issues = [i for i in self._issues.values() if not i.fix_successful]

        return {
            "status": self._last_diagnostic.status.value if self._last_diagnostic else "unknown",
            "last_diagnostic": self._last_diagnostic.timestamp.isoformat() if self._last_diagnostic else None,
            "open_issues": len(open_issues),
            "issues_by_severity": {
                "critical": len([i for i in open_issues if i.severity == "critical"]),
                "high": len([i for i in open_issues if i.severity == "high"]),
                "medium": len([i for i in open_issues if i.severity == "medium"]),
                "low": len([i for i in open_issues if i.severity == "low"]),
            },
            "total_errors_recorded": sum(self._error_counts.values()),
            "fixes_attempted": len(self._fix_history),
        }

    def get_issues(self, include_fixed: bool = False) -> list[dict[str, Any]]:
        """Get list of issues."""
        issues = self._issues.values()
        if not include_fixed:
            issues = [i for i in issues if not i.fix_successful]

        return [
            {
                "id": i.id,
                "category": i.category.value,
                "severity": i.severity,
                "description": i.description,
                "detected_at": i.detected_at.isoformat(),
                "auto_fixable": i.auto_fixable,
                "fix_attempted": i.fix_attempted,
                "fix_successful": i.fix_successful,
                "fix_details": i.fix_details,
            }
            for i in sorted(issues, key=lambda x: x.detected_at, reverse=True)
        ]


# Global instance
_self_heal_engine: SelfHealEngine | None = None


def get_self_heal_engine() -> SelfHealEngine:
    """Get or create the global self-heal engine."""
    global _self_heal_engine
    if _self_heal_engine is None:
        _self_heal_engine = SelfHealEngine()
    return _self_heal_engine


def init_self_heal(orchestrator: Orchestrator) -> SelfHealEngine:
    """Initialize self-heal engine with orchestrator."""
    global _self_heal_engine
    _self_heal_engine = SelfHealEngine(orchestrator)
    return _self_heal_engine
