# Changelog

This file tracks notable service changes. It is not tied to package publishing,
GitHub Releases, or tag-based deployment.

## Current

### Added

- Add generated downstream API descriptions with storage-backed caching.
- Add shared memory/Redis storage for recipes and generated descriptions.
- Add configurable description model and timeout.

### Changed

- Percent-encode substituted REST path parameters.
- Upgrade query and recipe runtime flow around shared GraphQL/REST execution.
- Move app configuration into `api-agent.toml`.
- Make recipe tools return CSV directly.
- Change learned recipe storage to the `public_contract` and `execution_plan`
  format. Older learned recipes are not migrated automatically and should be
  cleared, migrated, or relearned.
- Split recipe contracts, execution, extraction models, prompts, and tooling into
  focused modules.

### Removed

- Remove the legacy `_execute` public tool surface.

## OpenAPI compatibility - 2026-03-05

### Added

- Add Swagger 2.0 schema loading.
- Add structured HTTP error details.

## Documentation update - 2026-02-23

### Changed

- Clarify unsafe REST path and polling header escaping examples.

## Recipe tools - 2026-02-07

### Added

- Add direct `r_{slug}` recipe tools.
- Add recipe execution for cached GraphQL and REST workflows.
- Add CSV response helpers for direct recipe output.

## Recipe learning - 2026-01-27

### Added

- Add learned recipe extraction, caching, and matching.
- Add recipe-aware query flow and recipe store tests.

## Initial baseline - 2026-01-12

### Added

- Initial release of API Agent as an MCP server for natural-language API queries.
- Add GraphQL and REST schema loading, request execution, and DuckDB SQL
  post-processing.
- Add `{prefix}_query` tool with configurable target API headers.
- Add Docker, CI, README, CONTRIBUTING, and test coverage.
