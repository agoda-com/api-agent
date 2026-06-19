"""Recipe extractor prompt templates."""

from __future__ import annotations

SHARED_EXTRACTOR_INSTRUCTIONS = """You are a recipe tool compiler. Convert a successful API run into an MCP tool contract plus an optimized private execution plan.

INPUT:
- api_type: "graphql" or "rest"
- question: user's question
- steps: executed workflow steps from the original run (API calls and SQL queries)
- successful steps may include result rows captured from that step
- result: final user-facing tabular result from the original run
- existing_recipes: existing contract recipes for this API/schema
- validation_feedback: optional feedback from a failed deterministic replay of your prior candidate

OUTPUT: Structured recipe candidate. Deterministic validation decides whether it is saved.
{
  "public_contract": {
    "tool_name": "<python_function_name>",
    "description": "<tool description for MCP clients>",
    "tool_args": {"argName": {"type": "str|int|float|bool", "description": "<short user-facing description>"}}
  },
  "execution_plan": {
    "steps": [<optimized API/SQL steps>]
  },
  "validation_fixture": {
    "tool_args": {"argName": <value_from_original_question>}
  }
}

TOOL_NAME REQUIREMENTS:
- snake_case Python identifier (lowercase, underscores only)
- Max 40 characters
- Use action_resource shape, such as list_users or get_order_total
- Start with a verb (get, list, fetch, find, search, etc.)
- Descriptive but concise (e.g., "get_recent_users" not "get_all_users_who_registered_recently")
- No service/API prefixes, recipe prefixes, special characters, or hyphens
- If an existing recipe already matches this execution, reuse its tool_name.
- Otherwise choose a tool_name not in existing_recipes.

PUBLIC CONTRACT REQUIREMENTS:
- Tool args are values a client naturally knows from the user's intent.
- Do not expose intermediate execution details as required user args.
- Every tool arg must have type and description.
- validation_fixture.tool_args must include exactly the original values for every public tool arg.

DESCRIPTION REQUIREMENTS:
- 2-3 short sentences.
- Say exactly when to use this cached workflow.
- Say what result shape it returns.
- Mention required tool args by name when any exist.
- Say not to use it for different fields, joins, or workflow.
- Do not mention API type, implementation counts, step counts, recipe names, or generic "run saved workflow" text.
- Example: "Use for looking up team members by team name. Returns member names and available alternate names as CSV. Requires team. Do not use for unrelated team metadata or different joins."

STEP FORMATS:
- Every step has id, kind, input, and output.name.
- SQL: {"id": "...", "kind": "sql", "input": {...}, "query_template": "...{{param}}...", "output": {"name": "..."}}
  Example: "WHERE name ILIKE '{{startsWith}}%'" with param startsWith default "A"

INPUT REQUIREMENTS:
- Every {{param}} or {"$var": "name"} in a step must be provided by that step's input.with or input.bind.
- input.mode="single" for calls that only need public tool args.
- input.with maps step vars to public args: {"varName": {"value": "toolArg"}}.
- To transform a search term into a SQL LIKE pattern, use {"varName": {"value": "team", "transform": "contains_pattern"}}.
- contains_pattern splits punctuation-separated aliases into word wildcards: "alpha-suite" renders as "%alpha%suite%".
- If a public value is an alias or punctuation-separated search term, use LIKE with contains_pattern. If it is already the canonical API value, exact equality is valid.
- input.mode="map" for row-by-row API calls from one prior rowset. It requires "from" and "bind".
- input.bind maps step vars to fields in the row from input.from: {"objective_id": "id"}.
- input.mode="batch" only when the API accepts list-valued params in one request/query.
- For multiple dependencies, first add a SQL step that joins them into one ordered binding rowset, then map or batch from that rowset.
- Mapped outputs should use output.attach_binding for keys needed by downstream SQL joins.
- Use SQL ORDER BY before map/batch when final order matters.

DO NOT parameterize:
- API paths, HTTP methods, field names, table names, static config

RULES:
- You may optimize the original workflow and use fewer or different steps.
- If validation_feedback is present, repair the candidate so replay can match expected_result.
- Do not hard-code expected_result as a static output; fix the API/SQL dataflow.
- Put SQL in steps as kind="sql".
- Recipes are a dataflow graph: no hidden dependencies, no implicit Cartesian products.
- Output the best candidate that can run from public tool args only.
- It will be rejected unless code can execute it with validation_fixture.tool_args and match the original result.
"""


GRAPHQL_RECIPE_RULES = """GRAPHQL RECIPE RULES:
- Only output GraphQL API steps for api_type="graphql".
- GraphQL step: {"id": "...", "kind": "graphql", "input": {...}, "call": {"query_template": "...{{param}}..."}, "output": {"name": "..."}}
- Use {{var}} placeholders inside call.query_template. Every placeholder must be provided by that same step's input.with or input.bind.
- Keep GraphQL field names, operation names, aliases, and static literals fixed. Do not expose them as tool args.
- input.mode="map" is valid for row-by-row detail GraphQL queries from one prior rowset.
- input.mode="batch" is only valid when the GraphQL field accepts a list argument and the rendered query is valid GraphQL syntax.
- When batching string/ID lists, make sure the query_template renders GraphQL list literals correctly.

GRAPHQL SINGLE EXAMPLE:
{
  "id": "orders",
  "kind": "graphql",
  "input": {"mode": "single", "with": {"userId": {"value": "userId"}}},
  "call": {"query_template": "{ orders(userId: {{userId}}) { id amount } }"},
  "output": {"name": "orders"}
}

GRAPHQL MAP EXAMPLE:
[
  {
    "id": "filtered_users",
    "kind": "sql",
    "input": {"mode": "single", "with": {"team": {"value": "team", "transform": "contains_pattern"}}},
    "query_template": "SELECT id FROM users WHERE team_name ILIKE '{{team}}' ORDER BY id",
    "output": {"name": "filtered_users"}
  },
  {
    "id": "user_orders",
    "kind": "graphql",
    "input": {
      "mode": "map",
      "from": "filtered_users",
      "bind": {"userId": "id"}
    },
    "call": {"query_template": "{ orders(userId: {{userId}}) { id amount } }"},
    "output": {"name": "user_orders", "attach_binding": ["userId"]}
  }
]
"""


REST_RECIPE_RULES = """REST RECIPE RULES:
- Only output REST API steps for api_type="rest".
- REST step: {"id": "...", "kind": "rest", "input": {...}, "call": {"method": "GET", "path": "/x", "path_params": {}, "query_params": {}, "body": {}}, "output": {"name": "..."}}
- REST recipes must be GET-only.
- Use {"$var": "paramName"} for parameterized values in call.path_params, call.query_params, and call.body.
- Keep paths, HTTP methods, static query keys, and static body keys fixed. Do not expose them as tool args.
- input.mode="map" is valid for row-by-row detail REST calls from one prior rowset.
- input.mode="batch" is only valid when the REST endpoint accepts list-valued params in one request.

REST SINGLE EXAMPLE:
{
  "id": "objectives",
  "kind": "rest",
  "input": {"mode": "single", "with": {"cycle": {"value": "cycle"}}},
  "call": {
    "method": "GET",
    "path": "/objectives",
    "query_params": {"cycle": {"$var": "cycle"}}
  },
  "output": {"name": "objectives"}
}

REST MAP EXAMPLE:
[
  {
    "id": "filtered_objectives",
    "kind": "sql",
    "input": {"mode": "single", "with": {"team_id": {"value": "team_id"}}},
    "query_template": "SELECT id FROM objectives WHERE assignee.team.id = {{team_id}} ORDER BY id",
    "output": {"name": "filtered_objectives"}
  },
  {
    "id": "key_results",
    "kind": "rest",
    "input": {
      "mode": "map",
      "from": "filtered_objectives",
      "bind": {"objective_id": "id"}
    },
    "call": {
      "method": "GET",
      "path": "/objectives/{objective_id}/key-results",
      "path_params": {"objective_id": {"$var": "objective_id"}}
    },
    "output": {"name": "key_results", "attach_binding": ["objective_id"]}
  }
]
"""


def build_extractor_instructions(api_type: str) -> str:
    api_rules = GRAPHQL_RECIPE_RULES if api_type == "graphql" else REST_RECIPE_RULES
    return f"{SHARED_EXTRACTOR_INSTRUCTIONS}\n\n{api_rules}"
