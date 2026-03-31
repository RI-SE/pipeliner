from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from pipeliner.common.contract_helpers import ContractError, read_structured_data

ASSIGNMENTS_FILE = "dataset_builder_assignments.csv"

APP_TITLE = "Anomaly Dataset Builder"
SESSION_ENV_KEYS = ("DATASET_BUILDER_SESSION", "B10_WEBUI_SESSION")


def _restore_preloaded_session(payload: dict[str, Any]) -> dict[str, Any]:
    restored = dict(payload)
    restored.setdefault("config", {})
    restored.setdefault("split_assignments", {})
    restored.setdefault("csv_loaded", False)
    return restored


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset builder (B10)")
    parser.add_argument("--contract-json", default="")
    parser.add_argument("--mode", choices=["web", "headless", "auto"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    return parser.parse_args(argv)


def _load_contract(raw_contract: str) -> dict[str, Any]:
    if not raw_contract.strip():
        raise ContractError("--contract-json is required")
    return read_structured_data(raw_contract, source="--contract-json")


def create_app() -> Any:
    from .core import (
        initialize_split_assignments,
        load_assignment_csv,
        preview_split,
        scan_dataset_items,
    )
    from .web import build_app, load_preloaded_session

    for key in SESSION_ENV_KEYS:
        raw_path = os.environ.get(key, "").strip()
        if not raw_path:
            continue
        session_path = Path(raw_path)
        if session_path.exists():
            session, items = load_preloaded_session(session_path)
            restored = _restore_preloaded_session(session)
            watch_path_raw = str(restored.get("paths", {}).get("watched_config_path", "")).strip()
            watch_path = Path(watch_path_raw) if watch_path_raw else None
            if not items:
                items = scan_dataset_items(
                    restored.get("input_sections", {}),
                    class_labels=restored.get("config", {}).get("class_labels", []),
                )
                load_assignment_csv(restored, items)
                if not restored.get("split_assignments"):
                    initial = initialize_split_assignments(items, restored.get("split", {}))
                    restored.setdefault("split_assignments", {}).update(initial)
                preview_split(
                    items,
                    restored.get("labels", {}),
                    restored.get("split", {}),
                    restored.get("split_assignments", {}),
                )
            return build_app(restored, items, session_path=session_path, watch_path=watch_path)

    raw_contract = os.environ.get("DATASET_BUILDER_CONTRACT_JSON", "").strip()
    if not raw_contract:
        raise RuntimeError("No preloaded session or contract found for web app")
    contract = _load_contract(raw_contract)
    from .core import execute_web_setup

    session, items, _ = execute_web_setup(contract)
    return build_app(session, items)


def main(argv: list[str] | None = None) -> int:
    from .core import execute_headless, execute_web_setup
    from .web import build_app

    args = _parse_args(argv)
    contract = _load_contract(args.contract_json)

    output_dir = Path(str(contract.get("process_step", {}).get("output", "")).strip()).resolve()
    csv_path = output_dir / ASSIGNMENTS_FILE

    mode = args.mode
    if mode == "auto":
        mode = "headless" if csv_path.exists() else "web"

    if mode == "headless":
        execute_headless(contract)
        return 0

    session, items, _ = execute_web_setup(contract)
    app = build_app(session, items)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


__all__ = [
    "APP_TITLE",
    "SESSION_ENV_KEYS",
    "create_app",
    "main",
    "_restore_preloaded_session",
]
