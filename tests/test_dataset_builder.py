import importlib.util
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

from pipeliner import dataset_builder


def _load_start_module():
    path = Path(__file__).resolve().parents[1] / "start.sh"
    loader = SourceFileLoader("pipeliner_start", str(path))
    spec = importlib.util.spec_from_loader("pipeliner_start", loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("pipeliner_start", module)
    spec.loader.exec_module(module)
    return module


def test_dataset_builder_metadata() -> None:
    assert dataset_builder.APP_TITLE == "Anomaly Dataset Builder"
    assert dataset_builder.SESSION_ENV_KEYS == ("DATASET_BUILDER_SESSION", "B10_WEBUI_SESSION")


def test_restore_preloaded_session_keeps_config_and_assignments() -> None:
    restored = dataset_builder._restore_preloaded_session(
        {
            "config": {"class_labels": ["good", "defect_type_1"]},
            "split_assignments": {"item-1": "discard"},
            "csv_loaded": True,
        }
    )

    assert restored["config"]["class_labels"] == ["good", "defect_type_1"]
    assert restored["split_assignments"] == {"item-1": "discard"}
    assert restored["csv_loaded"] is True


def test_start_infers_dataset_builder_wrapper() -> None:
    start_module = _load_start_module()
    old_argv = sys.argv
    try:
        sys.argv = [str(Path("/tmp/dataset_builder"))]
        assert start_module.infer_app_name(None) == "dataset_builder"
    finally:
        sys.argv = old_argv


def test_start_uses_dataset_builder_executable(monkeypatch, tmp_path: Path) -> None:
    start_module = _load_start_module()
    recorded: dict[str, object] = {}

    def fake_run(cmd, text=True):  # noqa: ANN001
        recorded["cmd"] = cmd

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(start_module.subprocess, "run", fake_run)
    rc = start_module.launch_app("dataset_builder", "pipeliner", tmp_path / "setup.yaml", "127.0.0.1", 8008)
    assert rc == 0
    assert recorded["cmd"][5] == "pipeliner-dataset-builder"
