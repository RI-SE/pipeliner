from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .planner import build_step_run
from .setup_loader import flatten_choices, load_setup


def _parse_set(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}', expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_list(args: argparse.Namespace) -> int:
    setup = load_setup(Path(args.setup))
    payload = {
        "variation_points": flatten_choices(setup.variation_points),
        "process_steps": list(setup.process_steps.keys()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    setup = load_setup(Path(args.setup))
    selection = _parse_set(args.set or [])
    step_run = build_step_run(setup, args.step, selection)

    cmd = step_run.command(python_bin=args.python)
    out = {
        "step": step_run.step_name,
        "script": step_run.script,
        "contract": step_run.contract.to_dict(),
        "cmd": cmd,
        "cmd_str": " ".join(shlex.quote(x) for x in cmd),
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeliner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available variation points and process steps")
    p_list.add_argument("--setup", required=True, help="Path to experiment_setup.yaml")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Resolve one step and print command + contract")
    p_show.add_argument("--setup", required=True, help="Path to experiment_setup.yaml")
    p_show.add_argument("--step", required=True, help="Step name")
    p_show.add_argument("--set", action="append", help="Variation selection key=value")
    p_show.add_argument("--python", default="python3", help="Python executable for command preview")
    p_show.set_defaults(func=cmd_show)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
