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


def _normalize_named_values_section(section: Any, section_name: str) -> dict[str, Any]:
    if not isinstance(section, dict):
        return {}
    values = section.get("values")
    if not isinstance(values, list):
        return section

    out: dict[str, Any] = {}
    gui = section.get("gui")
    if isinstance(gui, dict):
        out["gui"] = gui

    for item in values:
        if not isinstance(item, dict) or "name" not in item:
            raise SetupError(f"{section_name}.values entries must be mappings with a 'name' field")
        name = str(item["name"])
        if name in out:
            raise SetupError(f"Duplicate '{name}' in {section_name}.values")
        out[name] = {k: v for k, v in item.items() if k != "name"}
    return out


def load_setup(path: Path) -> ExperimentSetup:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SetupError("Top-level setup must be a mapping")

    vps = raw.get("variation_points", {})
    steps = raw.get("process_steps", {})

    if isinstance(vps, dict) and isinstance(vps.get("values"), list):
        vps = _normalize_named_values_section(vps, "variation_points")
    if isinstance(steps, dict) and isinstance(steps.get("values"), list):
        steps = _normalize_named_values_section(steps, "process_steps")

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
            if isinstance(value.get("values"), list):
                out[key] = [str(v) for v in value.get("values", [])]
            elif isinstance(value.get("options"), list):
                names: list[str] = []
                for item in value.get("options", []):
                    if isinstance(item, dict) and item.get("name") is not None:
                        names.append(str(item.get("name")))
                out[key] = names
            else:
                out[key] = [str(v) for v in value.keys() if v != "gui"]
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
