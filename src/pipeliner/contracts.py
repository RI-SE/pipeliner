from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RuntimeInfo:
    pipeline_id: str = ""
    log_out: str = ""
    log_err: str = ""
    conda_env: str = ""
    runner_python: str = "python3"


@dataclass
class StepContract:
    variation_points: dict[str, Any]
    process_step: dict[str, Any]
    resolved: dict[str, Any]
    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime"] = asdict(self.runtime)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepContract":
        runtime_raw = data.get("runtime", {})
        runtime = RuntimeInfo(**runtime_raw)
        return cls(
            variation_points=data.get("variation_points", {}),
            process_step=data.get("process_step", {}),
            resolved=data.get("resolved", {}),
            runtime=runtime,
        )

    @classmethod
    def from_json(cls, raw: str) -> "StepContract":
        return cls.from_dict(json.loads(raw))


def write_contract(path: Path, contract: StepContract) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contract.to_json() + "\n", encoding="utf-8")


def read_contract(path: Path) -> StepContract:
    return StepContract.from_json(path.read_text(encoding="utf-8"))
