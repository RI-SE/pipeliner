from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import RuntimeInfo, StepContract
from .setup_loader import ExperimentSetup, render_value


def build_extra_args(step_cfg: dict[str, Any]) -> list[str]:
    raw = step_cfg.get("extra-args", step_cfg.get("extra_args", []))
    if raw is None:
        return []
    if isinstance(raw, dict):
        return []
    if not isinstance(raw, list):
        raise ValueError("process_step.extra-args must be a list or object")

    out: list[str] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"process_step.extra-args[{idx}] must be an object")
        name_raw = item.get("name")
        if name_raw is None:
            raise ValueError(f"process_step.extra-args[{idx}] missing required field: name")
        name = str(name_raw).strip()
        if not name:
            raise ValueError(f"process_step.extra-args[{idx}] name must be non-empty")
        if name.startswith("-"):
            flag = name
        else:
            flag = f"--{name.replace('_', '-')}"

        if "value" not in item:
            out.append(flag)
            continue

        value = item.get("value")
        if isinstance(value, list):
            for elem in value:
                out.extend([flag, str(elem)])
            continue
        if value is None:
            out.append(flag)
            continue
        if isinstance(value, bool):
            out.extend([flag, "true" if value else "false"])
            continue
        out.extend([flag, str(value)])
    return out


def _resolve_dataset_root(setup: ExperimentSetup, selection: dict[str, Any]) -> str:
    if "dataset_root" in selection and str(selection.get("dataset_root", "")).strip():
        return str(selection.get("dataset_root", "")).strip()

    dataset_name_raw = selection.get("dataset_name")
    if dataset_name_raw is None:
        return ""
    dataset_name = str(dataset_name_raw).strip()
    if not dataset_name:
        return ""

    ds_cfg = (
        setup.variation_points.get("dataset_name")
        if isinstance(setup.variation_points, dict)
        else None
    )
    if isinstance(ds_cfg, dict):
        options = ds_cfg.get("options", [])
        if isinstance(options, list):
            for item in options:
                if not isinstance(item, dict):
                    continue
                if str(item.get("name", "")).strip() != dataset_name:
                    continue
                root_raw = item.get("root")
                root = str(root_raw).strip() if root_raw is not None else ""
                if root:
                    return root

    return f"input/{dataset_name}"


@dataclass
class StepRun:
    step_name: str
    script: str
    contract: StepContract

    def command(self, python_bin: str = "python3") -> list[str]:
        cmd = [python_bin, self.script, "--contract-json", self.contract.to_json()]
        cmd.extend(build_extra_args(self.contract.process_step))
        return cmd


def build_step_run(
    setup: ExperimentSetup,
    step_name: str,
    variation_selection: dict[str, Any],
    runtime: RuntimeInfo | None = None,
) -> StepRun:
    if step_name not in setup.process_steps:
        raise KeyError(f"Unknown step: {step_name}")

    step_cfg = setup.process_steps[step_name]
    effective_selection = dict(variation_selection)
    dataset_root = _resolve_dataset_root(setup, effective_selection)
    if dataset_root:
        effective_selection.setdefault("dataset_root", dataset_root)

    context = dict(effective_selection)
    context["process_step"] = step_name

    expanded = render_value(step_cfg, context)
    script = str(expanded.get("script", ""))

    resolved: dict[str, Any] = {}
    if "input" in expanded:
        resolved["input"] = expanded["input"]
    if "output" in expanded:
        resolved["output"] = expanded["output"]

    if runtime is None:
        out_dir = Path(str(resolved.get("output", "")))
        runtime = RuntimeInfo(
            log_out=str(out_dir / "log.stdout") if str(out_dir) else "",
            log_err=str(out_dir / "log.stderr") if str(out_dir) else "",
        )

    contract = StepContract(
        variation_points=effective_selection,
        process_step={"name": step_name, **expanded},
        resolved=resolved,
        runtime=runtime,
    )
    return StepRun(step_name=step_name, script=script, contract=contract)
