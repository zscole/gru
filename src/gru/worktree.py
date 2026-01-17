"""Git worktree management for agent isolation."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorktreeInfo:
    """Information about a created worktree."""

    path: Path
    branch: str
    base_repo: Path


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def get_repo_root(path: Path) -> Path | None:
    """Get the root directory of the git repository containing path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def get_current_branch(repo_path: Path) -> str | None:
    """Get the current branch name of a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def create_worktree(
    repo_path: Path,
    worktree_path: Path,
    branch_name: str,
    base_branch: str | None = None,
) -> WorktreeInfo:
    """Create a git worktree for an agent.

    Args:
        repo_path: Path to the main git repository
        worktree_path: Path where the worktree should be created
        branch_name: Name for the new branch
        base_branch: Branch to base the new branch on (default: current HEAD)

    Returns:
        WorktreeInfo with details about the created worktree

    Raises:
        RuntimeError: If worktree creation fails
    """
    # Ensure worktree parent directory exists
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = ["git", "worktree", "add", "-b", branch_name, str(worktree_path)]
    if base_branch:
        cmd.append(base_branch)

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        return WorktreeInfo(
            path=worktree_path,
            branch=branch_name,
            base_repo=repo_path,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Worktree creation timed out: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"Git not found: {e}") from e


def remove_worktree(repo_path: Path, worktree_path: Path, force: bool = False) -> bool:
    """Remove a git worktree.

    Args:
        repo_path: Path to the main git repository
        worktree_path: Path to the worktree to remove
        force: Force removal even if worktree has modifications

    Returns:
        True if removal succeeded, False otherwise
    """
    cmd = ["git", "worktree", "remove"]
    if force:
        cmd.append("--force")
    cmd.append(str(worktree_path))

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Try force removal if normal removal failed
            if not force:
                return remove_worktree(repo_path, worktree_path, force=True)
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def delete_branch(repo_path: Path, branch_name: str, force: bool = False) -> bool:
    """Delete a git branch.

    Args:
        repo_path: Path to the git repository
        branch_name: Name of the branch to delete
        force: Force deletion even if branch is not merged

    Returns:
        True if deletion succeeded, False otherwise
    """
    flag = "-D" if force else "-d"
    try:
        result = subprocess.run(
            ["git", "branch", flag, branch_name],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def cleanup_worktree(
    repo_path: Path,
    worktree_path: Path,
    branch_name: str,
    delete_branch_after: bool = False,
) -> bool:
    """Clean up a worktree and optionally its branch.

    Args:
        repo_path: Path to the main git repository
        worktree_path: Path to the worktree
        branch_name: Name of the worktree's branch
        delete_branch_after: Whether to delete the branch after removing worktree

    Returns:
        True if cleanup succeeded
    """
    # Remove worktree first
    worktree_removed = remove_worktree(repo_path, worktree_path, force=True)

    # Prune worktree references
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError, OSError):
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_path,
            capture_output=True,
            timeout=10,
        )

    # Optionally delete branch
    if delete_branch_after:
        delete_branch(repo_path, branch_name, force=True)

    # Clean up any remaining directory (fallback)
    if worktree_path.exists():
        with contextlib.suppress(OSError):
            shutil.rmtree(worktree_path)

    return worktree_removed


def has_changes(worktree_path: Path) -> bool:
    """Check if worktree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def has_commits_ahead(worktree_path: Path) -> bool:
    """Check if worktree branch has commits not on origin."""
    try:
        branch = get_current_branch(worktree_path)
        if not branch:
            return False
        result = subprocess.run(
            ["git", "rev-list", f"origin/{branch}..HEAD", "--count"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # Origin branch doesn't exist yet, so we have commits to push
            return True
        return int(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return False


def commit_and_push(worktree_path: Path, message: str) -> tuple[bool, str]:
    """Stage all changes, commit (amend if exists), and force push.

    Args:
        worktree_path: Path to the worktree
        message: Commit message

    Returns:
        Tuple of (success, status_message)
    """
    if not is_git_repo(worktree_path):
        return False, "Not a git repository"

    # Check for changes
    changes = has_changes(worktree_path)
    has_existing_commits = has_commits_ahead(worktree_path)

    if not changes and not has_existing_commits:
        return True, "No changes to push"

    try:
        # Stage all changes
        if changes:
            result = subprocess.run(
                ["git", "add", "-A"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return False, f"Failed to stage changes: {result.stderr}"

            # Commit (amend if we already have commits ahead)
            if has_existing_commits:
                commit_cmd = ["git", "commit", "--amend", "-m", message]
            else:
                commit_cmd = ["git", "commit", "-m", message]

            result = subprocess.run(
                commit_cmd,
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return False, f"Failed to commit: {result.stderr}"

        # Force push
        branch = get_current_branch(worktree_path)
        if not branch:
            return False, "Could not determine branch"

        result = subprocess.run(
            ["git", "push", "--force", "-u", "origin", branch],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, f"Failed to push: {result.stderr}"

        return True, f"Pushed to {branch}"

    except subprocess.TimeoutExpired:
        return False, "Git operation timed out"
    except (FileNotFoundError, OSError) as e:
        return False, f"Git error: {e}"


def list_worktrees(repo_path: Path) -> list[dict[str, str]]:
    """List all worktrees in a repository.

    Args:
        repo_path: Path to the git repository

    Returns:
        List of dicts with 'path', 'head', and 'branch' keys
    """
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        worktrees = []
        current: dict[str, str] = {}

        for line in result.stdout.strip().split("\n"):
            if not line:
                if current:
                    worktrees.append(current)
                    current = {}
            elif line.startswith("worktree "):
                current["path"] = line[9:]
            elif line.startswith("HEAD "):
                current["head"] = line[5:]
            elif line.startswith("branch "):
                current["branch"] = line[7:]

        if current:
            worktrees.append(current)

        return worktrees
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
