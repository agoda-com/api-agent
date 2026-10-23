"""Recipe template rendering helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import duckdb
from graphql.language import Lexer, Source, TokenKind

_PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")
_BARE_SAFE_RE = re.compile(r"-?[A-Za-z0-9_.]+")


def _string_spans(template: str, language: str) -> list[tuple[int, int, str]]:
    """(start, end, delimiter) of string literals, per the GraphQL or DuckDB lexer."""
    spans: list[tuple[int, int, str]] = []
    if language == "graphql":
        lexer = Lexer(Source(template))
        token = lexer.advance()
        while token.kind != TokenKind.EOF:
            if token.kind in (TokenKind.STRING, TokenKind.BLOCK_STRING):
                delim = '"""' if token.kind == TokenKind.BLOCK_STRING else '"'
                spans.append((token.start, token.end, delim))
            token = lexer.advance()
        return spans
    tokens = duckdb.tokenize(template)
    for (start, kind), (end, _) in zip(tokens, [*tokens[1:], (len(template), None)]):
        if kind == duckdb.token_type.string_const:
            spans.append((start, end, "'"))
        elif kind == duckdb.token_type.identifier and template[start] == '"':
            spans.append((start, end, '"'))
    return spans


def render_text_template(
    template: str, params: dict[str, Any], language: Literal["graphql", "sql"]
) -> str:
    """Render {{param}} placeholders as escaped GraphQL or SQL literals.

    Inside a string literal, values are escaped; bare, strings that are not plain
    names/numbers are emitted as quoted string literals.
    """
    quote = '"' if language == "graphql" else "'"
    spans = _string_spans(template, language)

    def escape(s: str, state: str = quote) -> str:
        if state == '"""':
            return s.replace('"""', '\\"""')
        if state == '"':
            return json.dumps(s)[1:-1] if language == "graphql" else s.replace('"', '""')
        return s.replace("'", "''")

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in params:
            raise KeyError(f"missing param: {name}")
        v = params[name]
        if isinstance(v, bool):
            text = "true" if v else "false"
        elif v is None:
            text = "null"
        elif isinstance(v, list) and language == "graphql":
            text = json.dumps(v)
        else:
            text = str(v)
        state = next((d for s, e, d in spans if s < m.start() < e), None)
        if state:
            return escape(text, state)
        if isinstance(v, str) and not _BARE_SAFE_RE.fullmatch(v):
            return f"{quote}{escape(v)}{quote}"
        return text

    return _PLACEHOLDER_RE.sub(repl, template)


def render_param_refs(obj: Any, params: dict[str, Any]) -> Any:
    """Recursively replace {'$var': 'x'} nodes with params['x']."""
    if isinstance(obj, dict):
        if set(obj.keys()) == {"$var"} and isinstance(obj.get("$var"), str):
            pname = obj["$var"]
            if pname not in params:
                raise KeyError(f"missing param: {pname}")
            return params[pname]
        return {k: render_param_refs(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [render_param_refs(v, params) for v in obj]
    return obj
