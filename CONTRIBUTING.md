# Contributing to API Agent

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync --group dev
   ```
3. Run locally:
   ```bash
   OPENAI_API_KEY=your_key uv run api-agent
   ```

By default, local runs read `api-agent.toml` from the repository root. Set
`API_AGENT_CONFIG=/path/to/api-agent.toml` to use another config file.

## Code Style

- Follow PEP 8 (enforced by Ruff)
- Use type hints where possible
- Write tests for new features
- Keep functions focused and small

## Testing

Run the full local check set before submitting a PR:

```bash
uv lock --check
uv run ruff check api_agent/ tests/
uv run ruff format --check api_agent/ tests/
uv run ty check
uv run pytest tests/ -q
```

Write tests for new behavior. Prefer behavior-focused tests over tests that pin
private implementation details.

## Changelog

Update [CHANGELOG.md](CHANGELOG.md) for notable service changes. The changelog is
for service history tracking; it is not tied to package publishing, GitHub
Releases, or tag-based deployment.

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with clear commit messages
3. Update documentation if needed
4. Update [CHANGELOG.md](CHANGELOG.md) for notable service changes
5. Ensure tests pass and linting is clean
6. Submit PR with a concise description of changes

## Reporting Issues

- Use GitHub Issues
- Provide clear reproduction steps
- Include environment details (Python version, OS, etc.)
- Attach relevant logs or error messages

## Questions?

Open a GitHub Issue.
