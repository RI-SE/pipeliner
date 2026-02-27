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
    if not isinstance(raw, list):
        raise ValueError("process_step.extra-args must be a list")

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
        flag = name if name.startswith("-") else f"--{name}"

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
    context = dict(variation_selection)
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
        variation_points=variation_selection,
        process_step={"name": step_name, **expanded},
        resolved=resolved,
        runtime=runtime,
    )
    return StepRun(step_name=step_name, script=script, contract=contract)
