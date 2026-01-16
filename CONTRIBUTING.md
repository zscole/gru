# Contributing to Gru

Thanks for your interest in contributing to Gru.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/gru.git`
3. Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
4. Install dev dependencies: `pip install -e ".[dev]"`
5. Create a branch: `git checkout -b your-feature-name`

## Development

Run tests:
```bash
PYTHONPATH=src pytest tests/ -v
```

Run linter:
```bash
ruff check src/ tests/
```

Run formatter:
```bash
ruff format src/ tests/
```

Run type checker:
```bash
mypy src/
```

## Pull Requests

1. Ensure all tests pass
2. Ensure code is formatted with `ruff format`
3. Ensure no lint errors from `ruff check`
4. Ensure no type errors from `mypy`
5. Write clear commit messages
6. Update documentation if needed

## Code Style

- Follow existing code patterns
- Use type hints
- Keep functions focused and small
- No unnecessary abstractions

## Reporting Issues

- Check existing issues first
- Include steps to reproduce
- Include Python version and OS
- Include relevant logs or error messages

## Security

See [SECURITY.md](SECURITY.md) for reporting security vulnerabilities.
