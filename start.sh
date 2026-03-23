#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_setup = repo_root / "pipeline" / "experiment_setup.yaml"

    parser = argparse.ArgumentParser(
        description="Bootstrap a Conda env for pipeliner tools and launch the selected app."
    )
    parser.add_argument(
        "--app",
        choices=["pipeliner", "iqviewer"],
        default=None,
        help="Which app to launch. Defaults to `pipeliner`, or infers from wrapper script name.",
    )
    parser.add_argument("--config", "--setup", dest="setup", default="")
    parser.add_argument("--env-name", default="pipeliner")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--yes", action="store_true", help="Create/install without prompting.")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not run `pip install -e .[dev]` inside the Conda env.",
    )
    parser.add_argument(
        "--default-config",
        default=str(default_setup),
        help="Default config path offered when --config is omitted.",
    )
    return parser.parse_args()


def infer_app_name(cli_app: str | None) -> str:
    if cli_app:
        return cli_app
    stem = Path(sys.argv[0]).name
    if "iqviewer" in stem:
        return "iqviewer"
    return "pipeliner"


def default_port_for_app(app_name: str) -> int:
    if app_name == "iqviewer":
        return 8780
    return 8080


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        check=True,
        capture_output=capture,
    )


def conda_envs() -> set[str]:
    commands = [
        ["conda", "env", "list", "--json"],
        ["conda", "info", "--envs", "--json"],
    ]
    for cmd in commands:
        try:
            completed = run(cmd, capture=True)
        except Exception:
            continue
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            continue
        envs = payload.get("envs", [])
        if isinstance(envs, list):
            return {Path(str(item)).name for item in envs}
    return set()


def ensure_conda_available() -> None:
    if shutil.which("conda") is None:
        raise RuntimeError("Could not find `conda` on PATH.")


def resolve_setup_path(raw_value: str, default_value: str) -> Path:
    value = raw_value.strip()
    if not value:
        prompt = f"Path to experiment_setup.yaml"
        if default_value.strip():
            prompt += f" [{default_value}]"
        prompt += ": "
        value = input(prompt).strip() or default_value.strip()
    if not value:
        raise RuntimeError("No config path provided.")
    setup_path = Path(value).expanduser().resolve()
    if not setup_path.exists():
        raise FileNotFoundError(f"Config file not found: {setup_path}")
    return setup_path


def ensure_env(env_name: str, python_version: str, *, assume_yes: bool) -> bool:
    known_envs = conda_envs()
    if env_name in known_envs:
        return False
    if not assume_yes and not prompt_yes_no(
        f"Conda env `{env_name}` was not found. Create it with Python {python_version}?"
    ):
        raise RuntimeError(f"Required Conda env `{env_name}` is missing.")
    run(["conda", "create", "-y", "-n", env_name, f"python={python_version}", "pip"])
    return True


def install_pipeliner(env_name: str, package_dir: Path) -> None:
    run(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pip",
            "install",
            "-e",
            ".[dev]",
        ],
        cwd=package_dir,
    )


def launch_app(app_name: str, env_name: str, setup_path: Path, host: str, port: int) -> int:
    executable = "pipeliner-iqviewer" if app_name == "iqviewer" else "pipeliner"
    cmd = [
        "conda",
        "run",
        "--live-stream",
        "-n",
        env_name,
        executable,
        "--host",
        host,
        "--port",
        str(port),
        "--config",
        str(setup_path),
    ]
    proc = subprocess.run(cmd, text=True)
    return proc.returncode


def main() -> int:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    app_name = infer_app_name(args.app)
    port = args.port if args.port is not None else default_port_for_app(app_name)

    ensure_conda_available()
    setup_path = resolve_setup_path(args.setup, args.default_config)
    ensure_env(args.env_name, args.python_version, assume_yes=args.yes)

    if not args.skip_install:
        print(f"Installing/updating pipeliner in Conda env `{args.env_name}` from {package_dir}")
        install_pipeliner(args.env_name, package_dir)

    print(f"Launching {app_name} on http://{args.host}:{port}")
    print(f"Using config: {setup_path}")
    print(f"Using Conda env: {args.env_name}")
    return launch_app(app_name, args.env_name, setup_path, args.host, port)


if __name__ == "__main__":
    raise SystemExit(main())
