"""Tests for git worktree management."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from gru.worktree import (
    cleanup_worktree,
    create_worktree,
    delete_branch,
    get_current_branch,
    get_repo_root,
    is_git_repo,
    list_worktrees,
    remove_worktree,
)


@pytest.fixture
def git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create initial commit (worktrees require at least one commit)
        (repo_path / "README.md").write_text("# Test Repo")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield repo_path


@pytest.fixture
def non_git_dir():
    """Create a temporary non-git directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestIsGitRepo:
    """Tests for is_git_repo function."""

    def test_git_repo_returns_true(self, git_repo):
        """Test that a git repo returns True."""
        assert is_git_repo(git_repo) is True

    def test_non_git_dir_returns_false(self, non_git_dir):
        """Test that a non-git directory returns False."""
        assert is_git_repo(non_git_dir) is False

    def test_nonexistent_path_returns_false(self):
        """Test that a nonexistent path returns False."""
        assert is_git_repo(Path("/nonexistent/path")) is False


class TestGetRepoRoot:
    """Tests for get_repo_root function."""

    def test_returns_repo_root(self, git_repo):
        """Test getting repo root from repo directory."""
        root = get_repo_root(git_repo)
        # Resolve both paths to handle macOS symlinks (/var -> /private/var)
        assert root.resolve() == git_repo.resolve()

    def test_returns_root_from_subdirectory(self, git_repo):
        """Test getting repo root from a subdirectory."""
        subdir = git_repo / "subdir"
        subdir.mkdir()
        root = get_repo_root(subdir)
        # Resolve both paths to handle macOS symlinks (/var -> /private/var)
        assert root.resolve() == git_repo.resolve()

    def test_returns_none_for_non_git(self, non_git_dir):
        """Test returns None for non-git directory."""
        root = get_repo_root(non_git_dir)
        assert root is None


class TestGetCurrentBranch:
    """Tests for get_current_branch function."""

    def test_returns_branch_name(self, git_repo):
        """Test getting current branch name."""
        branch = get_current_branch(git_repo)
        # Git default branch could be 'main' or 'master'
        assert branch in ("main", "master")

    def test_returns_none_for_non_git(self, non_git_dir):
        """Test returns None for non-git directory."""
        branch = get_current_branch(non_git_dir)
        assert branch is None


class TestCreateWorktree:
    """Tests for create_worktree function."""

    def test_creates_worktree(self, git_repo):
        """Test creating a new worktree."""
        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"

        info = create_worktree(git_repo, worktree_path, branch_name)

        assert info.path == worktree_path
        assert info.branch == branch_name
        assert info.base_repo == git_repo
        assert worktree_path.exists()
        assert (worktree_path / "README.md").exists()

        # Cleanup
        remove_worktree(git_repo, worktree_path, force=True)
        delete_branch(git_repo, branch_name, force=True)

    def test_creates_worktree_with_base_branch(self, git_repo):
        """Test creating a worktree based on a specific branch."""
        # First create another branch
        subprocess.run(
            ["git", "branch", "feature-branch"],
            cwd=git_repo,
            capture_output=True,
            check=True,
        )

        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"

        info = create_worktree(git_repo, worktree_path, branch_name, base_branch="feature-branch")

        assert info.path == worktree_path
        assert worktree_path.exists()

        # Cleanup
        remove_worktree(git_repo, worktree_path, force=True)
        delete_branch(git_repo, branch_name, force=True)
        delete_branch(git_repo, "feature-branch", force=True)

    def test_raises_on_duplicate_branch(self, git_repo):
        """Test that creating a worktree with existing branch name fails."""
        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"

        # Create first worktree
        create_worktree(git_repo, worktree_path, branch_name)

        # Try to create another with same branch name
        worktree_path2 = git_repo.parent / "test_worktree2"
        with pytest.raises(RuntimeError):
            create_worktree(git_repo, worktree_path2, branch_name)

        # Cleanup
        remove_worktree(git_repo, worktree_path, force=True)
        delete_branch(git_repo, branch_name, force=True)


class TestRemoveWorktree:
    """Tests for remove_worktree function."""

    def test_removes_worktree(self, git_repo):
        """Test removing a worktree."""
        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"
        create_worktree(git_repo, worktree_path, branch_name)

        result = remove_worktree(git_repo, worktree_path)

        assert result is True
        assert not worktree_path.exists()

        # Cleanup branch
        delete_branch(git_repo, branch_name, force=True)

    def test_force_removes_modified_worktree(self, git_repo):
        """Test force removing a worktree with modifications."""
        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"
        create_worktree(git_repo, worktree_path, branch_name)

        # Make modifications
        (worktree_path / "new_file.txt").write_text("changes")

        result = remove_worktree(git_repo, worktree_path, force=True)

        assert result is True

        # Cleanup branch
        delete_branch(git_repo, branch_name, force=True)


class TestDeleteBranch:
    """Tests for delete_branch function."""

    def test_deletes_branch(self, git_repo):
        """Test deleting a branch."""
        # Create a branch
        subprocess.run(
            ["git", "branch", "test-branch"],
            cwd=git_repo,
            capture_output=True,
            check=True,
        )

        result = delete_branch(git_repo, "test-branch")
        assert result is True

    def test_force_deletes_unmerged_branch(self, git_repo):
        """Test force deleting an unmerged branch."""
        # Create a branch with commits
        subprocess.run(
            ["git", "checkout", "-b", "unmerged-branch"],
            cwd=git_repo,
            capture_output=True,
            check=True,
        )
        (git_repo / "unmerged.txt").write_text("unmerged")
        subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Unmerged commit"],
            cwd=git_repo,
            capture_output=True,
            check=True,
        )
        # Switch back
        subprocess.run(
            ["git", "checkout", "-"],
            cwd=git_repo,
            capture_output=True,
            check=True,
        )

        result = delete_branch(git_repo, "unmerged-branch", force=True)
        assert result is True


class TestCleanupWorktree:
    """Tests for cleanup_worktree function."""

    def test_cleanup_removes_worktree_and_keeps_branch(self, git_repo):
        """Test cleanup removes worktree but keeps branch by default."""
        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"
        create_worktree(git_repo, worktree_path, branch_name)

        result = cleanup_worktree(git_repo, worktree_path, branch_name, delete_branch_after=False)

        assert result is True
        assert not worktree_path.exists()

        # Branch should still exist
        branches = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert branch_name in branches.stdout

        # Final cleanup
        delete_branch(git_repo, branch_name, force=True)

    def test_cleanup_removes_worktree_and_branch(self, git_repo):
        """Test cleanup removes both worktree and branch."""
        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"
        create_worktree(git_repo, worktree_path, branch_name)

        result = cleanup_worktree(git_repo, worktree_path, branch_name, delete_branch_after=True)

        assert result is True
        assert not worktree_path.exists()

        # Branch should not exist
        branches = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert branch_name not in branches.stdout


class TestListWorktrees:
    """Tests for list_worktrees function."""

    def test_lists_worktrees(self, git_repo):
        """Test listing worktrees."""
        worktree_path = git_repo.parent / "test_worktree"
        branch_name = "test-branch"
        create_worktree(git_repo, worktree_path, branch_name)

        worktrees = list_worktrees(git_repo)

        assert len(worktrees) >= 2  # Main repo + worktree
        # Resolve paths to handle macOS symlinks (/var -> /private/var)
        paths = [Path(w.get("path")).resolve() for w in worktrees]
        assert worktree_path.resolve() in paths

        # Cleanup
        remove_worktree(git_repo, worktree_path, force=True)
        delete_branch(git_repo, branch_name, force=True)

    def test_returns_empty_for_non_git(self, non_git_dir):
        """Test returns empty list for non-git directory."""
        worktrees = list_worktrees(non_git_dir)
        assert worktrees == []
