from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import RuntimeInfo, StepContract
from .setup_loader import ExperimentSetup, render_value


@dataclass
class StepRun:
    step_name: str
    script: str
    contract: StepContract

    def command(self, python_bin: str = "python3") -> list[str]:
        return [python_bin, self.script, "--contract-json", self.contract.to_json()]


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
