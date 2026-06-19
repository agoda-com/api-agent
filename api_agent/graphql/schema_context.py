"""GraphQL introspection schema context formatting."""

from __future__ import annotations


def format_type(t: dict | None) -> str:
    """Convert introspection type to compact notation: [User!]!"""
    if not t:
        return "?"
    kind = t.get("kind")
    name = t.get("name")
    inner = t.get("ofType")

    if kind == "NON_NULL":
        return f"{format_type(inner)}!"
    if kind == "LIST":
        return f"[{format_type(inner)}]"
    return name or "?"


def is_required(type_def: dict | None) -> bool:
    """Check if GraphQL type is required (NON_NULL wrapper)."""
    return type_def.get("kind") == "NON_NULL" if type_def else False


def format_arg(arg: dict) -> str:
    """Format argument with optional default value."""
    type_str = format_type(arg["type"])
    default = arg.get("defaultValue")
    if default is not None:
        return f"{arg['name']}: {type_str} = {default}"
    return f"{arg['name']}: {type_str}"


def filter_required_args(args: list[dict]) -> list[dict]:
    """Filter to only required arguments."""
    return [arg for arg in args if is_required(arg.get("type"))]


def format_field(field: dict) -> str:
    """Format a field with optional args."""
    args = field.get("args", [])
    arg_str = "(" + ", ".join(format_arg(arg) for arg in args) + ")" if args else ""
    desc = f" # {field['description']}" if field.get("description") else ""
    return f"  {field['name']}{arg_str}: {format_type(field['type'])}{desc}"


def build_schema_context(schema: dict) -> str:
    """Build compact SDL context from introspection schema."""
    queries = schema.get("queryType", {}).get("fields", [])
    all_types = [t for t in schema.get("types", []) if not t["name"].startswith("__")]

    objects = [
        t
        for t in all_types
        if t["kind"] == "OBJECT" and t["name"] not in ("Query", "Mutation", "Subscription")
    ]
    enums = [t for t in all_types if t["kind"] == "ENUM"]
    inputs = [t for t in all_types if t["kind"] == "INPUT_OBJECT"]
    interfaces = [t for t in all_types if t["kind"] == "INTERFACE"]
    unions = [t for t in all_types if t["kind"] == "UNION"]

    lines = ["<queries>"]
    for field in queries:
        desc = f" # {field['description']}" if field.get("description") else ""
        args = ", ".join(format_arg(arg) for arg in filter_required_args(field.get("args", [])))
        lines.append(f"{field['name']}({args}) -> {format_type(field['type'])}{desc}")

    if interfaces:
        lines.append("\n<interfaces>")
        for typ in interfaces:
            impl = [p["name"] for p in typ.get("possibleTypes", []) or []]
            impl_str = f" # implemented by: {', '.join(impl)}" if impl else ""
            fields = [format_field(field) for field in typ.get("fields", []) or []]
            lines.append(f"{typ['name']} {{{impl_str}\n" + "\n".join(fields) + "\n}")

    if unions:
        lines.append("\n<unions>")
        for typ in unions:
            types = [p["name"] for p in typ.get("possibleTypes", []) or []]
            lines.append(f"{typ['name']}: {' | '.join(types)}")

    lines.append("\n<types>")
    for typ in objects:
        impl = [i["name"] for i in typ.get("interfaces", []) or []]
        impl_str = f" implements {', '.join(impl)}" if impl else ""
        fields = [format_field(field) for field in typ.get("fields", []) or []]
        lines.append(f"{typ['name']}{impl_str} {{\n" + "\n".join(fields) + "\n}")

    lines.append("\n<enums>")
    for enum in enums:
        vals = " | ".join(v["name"] for v in enum.get("enumValues", []))
        lines.append(f"{enum['name']}: {vals}")

    lines.append("\n<inputs>")
    for inp in inputs:
        required_fields = [
            field for field in (inp.get("inputFields", []) or []) if is_required(field.get("type"))
        ]
        fields = ", ".join(
            f"{field['name']}: {format_type(field['type'])}" for field in required_fields
        )
        lines.append(f"{inp['name']} {{ {fields} }}")

    return "\n".join(lines)
