from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}]+)\}")


class ContractError(ValueError):
    """Raised when contract payload is invalid."""


def read_structured_data(raw: str, source: str = "payload") -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ContractError(f"{source} is empty")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(text)

    if not isinstance(parsed, dict):
        raise ContractError(f"{source} must decode to an object")
    return parsed


def read_structured_file(path: Path) -> dict[str, Any]:
    return read_structured_data(path.read_text(encoding="utf-8"), source=str(path))


def require_mapping(obj: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"{where}.{key} must be an object")
    return value


def require_extra_args(process_step: dict[str, Any]) -> dict[str, Any]:
    if "extra_args" in process_step:
        value = process_step.get("extra_args")
    else:
        value = process_step.get("extra-args")
    if not isinstance(value, dict):
        raise ContractError("process_step.extra_args must be an object")
    return value


def _expand_text(text: str, variables: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            return match.group(0)
        value = variables[key]
        return "" if value is None else str(value)

    return _TEMPLATE_PATTERN.sub(replace, text)


def expand_templates(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _expand_text(value, variables)
    if isinstance(value, list):
        return [expand_templates(item, variables) for item in value]
    if isinstance(value, dict):
        return {str(k): expand_templates(v, variables) for k, v in value.items()}
    return value
