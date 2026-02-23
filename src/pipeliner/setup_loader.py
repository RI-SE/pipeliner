from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import yaml


@dataclass
class ExperimentSetup:
    variation_points: dict[str, Any]
    process_steps: dict[str, dict[str, Any]]


class SetupError(ValueError):
    pass


def load_setup(path: Path) -> ExperimentSetup:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SetupError("Top-level setup must be a mapping")

    vps = raw.get("variation_points", {})
    steps = raw.get("process_steps", {})

    if not isinstance(vps, dict):
        raise SetupError("variation_points must be a mapping")
    if not isinstance(steps, dict):
        raise SetupError("process_steps must be a mapping")

    return ExperimentSetup(variation_points=vps, process_steps=steps)


def flatten_choices(variation_points: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, value in variation_points.items():
        if isinstance(value, list):
            out[key] = [str(v) for v in value]
        elif isinstance(value, dict):
            out[key] = [str(v) for v in value.keys()]
        else:
            out[key] = [str(value)]
    return out


def render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return Template(value).safe_substitute(context)
    if isinstance(value, list):
        return [render_value(v, context) for v in value]
    if isinstance(value, dict):
        return {k: render_value(v, context) for k, v in value.items()}
    return value
