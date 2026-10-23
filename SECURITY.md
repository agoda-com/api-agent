# Security policy

## Supported versions

Security fixes target the latest commit on `main`. Older commits are not
maintained separately.

## Report a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/agoda-com/api-agent/security/advisories/new).
Do not open a public issue or pull request with exploit details. Include the
affected commit, reproduction steps, and impact; never include real credentials
or other users' data.

The maintainer will coordinate validation, a fix, and disclosure with the
reporter. Include your preferred name for credit, or request no attribution.

## Out of scope

Incorrect model answers, issues that require operator misconfiguration (such as
exposing `/mcp` publicly without authentication), and vulnerabilities in target
APIs themselves.

## Deployment boundaries

Agent prompts and API responses can contain untrusted text. Request-scoped data
does not provide operating-system isolation. Run API Agent with minimal file
permissions and credentials, and mount only files it needs.

`/mcp` has no built-in authentication, and every query spends the server's
OpenAI key. Expose it only on localhost or a trusted network, or put it behind
an authenticating proxy. On shared deployments, set
`[server].allowed_target_hosts` so clients cannot point the server at internal
or cloud-metadata addresses.

Untrusted API content can still steer the agent into misleading answers or
extra read calls on the same target. `X-Allow-Unsafe-Paths` lets the model
write with the caller's credentials; allow only narrow paths on trusted APIs.
Keep `cors_allowed_origins` limited to trusted origins.

Recipes in Redis hold question text and query templates; protect Redis with
auth and network isolation. OpenTelemetry tracing records prompts and API data;
export only to a trusted backend.
