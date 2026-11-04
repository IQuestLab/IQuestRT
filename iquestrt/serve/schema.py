"""Structural checks for tool schemas; generation is not schema-constrained."""


def check_schema(schema):
    pending = [(schema, 0)]
    visited = 0
    types = {"object", "array", "string", "number", "integer", "boolean", "null"}

    def enqueue(values, depth):
        # Count queued and visited nodes before materializing a wide schema.
        if visited + len(pending) + len(values) > 4096:
            raise ValueError("tool schema is too complex")
        pending.extend((child, depth) for child in values)

    while pending:
        node, depth = pending.pop()
        visited += 1
        if depth > 64 or visited > 4096:
            raise ValueError("tool schema is too complex")
        if isinstance(node, bool):
            continue
        if not isinstance(node, dict):
            raise ValueError("tool schema entries must be objects or booleans")
        if "type" in node:
            value = node["type"]
            values = value if isinstance(value, list) else [value]
            if (not values or any(not isinstance(t, str) or t not in types for t in values)
                    or len(set(values)) != len(values)):
                raise ValueError("invalid tool schema type")
        if "required" in node:
            required = node["required"]
            if (not isinstance(required, list) or any(not isinstance(k, str) for k in required)
                    or len(set(required)) != len(required)):
                raise ValueError("tool schema required must be a list of unique strings")
        for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            if key in node:
                if not isinstance(node[key], dict):
                    raise ValueError(f"tool schema {key} must be an object")
                enqueue(node[key].values(), depth + 1)
        for key in ("items", "additionalProperties", "additionalItems", "contains",
                    "not", "if", "then", "else", "propertyNames", "unevaluatedProperties",
                    "unevaluatedItems"):
            if key in node:
                value = node[key]
                enqueue(value if key == "items" and isinstance(value, list) else [value], depth + 1)
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            if key in node:
                if not isinstance(node[key], list) or not node[key]:
                    raise ValueError(f"tool schema {key} must be a nonempty list")
                enqueue(node[key], depth + 1)
        for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties",
                    "minContains", "maxContains"):
            if key in node and (type(node[key]) is not int or node[key] < 0):
                raise ValueError(f"tool schema {key} must be a nonnegative integer")
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
            if key in node:
                value = node[key]
                if (type(value) not in (int, float) or not -float("inf") < value < float("inf")
                        or (key == "multipleOf" and value <= 0)):
                    raise ValueError(f"tool schema {key} must be a finite number")
        for key in ("description", "title", "pattern", "format", "$ref", "$schema", "$id", "$anchor"):
            if key in node and not isinstance(node[key], str):
                raise ValueError(f"tool schema {key} must be a string")
        if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
            raise ValueError("tool schema uniqueItems must be a boolean")
        if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]):
            raise ValueError("tool schema enum must be a nonempty list")
