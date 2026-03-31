from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from pipeliner.dataset_builder.core import DatasetItem
from pipeliner.dataset_builder.web import build_app


def _session_payload(section: str, item: DatasetItem) -> dict:
    return {
        "paths": {},
        "split": {
            "split_labels": ["train", "val", "test", "discard"],
            "split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
            "split_seed": 0,
        },
        "config": {
            "class_labels": ["good", "defect"],
            "output_tree_structure": {"images": "{split_label}/{class_label}/{image_file}"},
        },
        "input_sections": {section: str(Path(item.image_path).parent)},
        "labels": {},
        "split_assignments": {},
        "csv_loaded": False,
        "items": [
            {
                "item_id": item.item_id,
                "section": section,
                "image_path": item.image_path,
                "display_name": item.display_name,
                "group_key": item.group_key,
            }
        ],
    }


def test_dataset_builder_config_status_and_reload(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake")
    item = DatasetItem(
        item_id="item-1",
        section="real",
        image_path=str(image_path),
        display_name="img.png",
        group_key="img",
    )
    session_path = tmp_path / "dataset_builder_session.yaml"
    session_path.write_text(yaml.safe_dump(_session_payload("real", item), sort_keys=False), encoding="utf-8")

    app = build_app(_session_payload("real", item), [item], session_path=session_path)
    client = TestClient(app)

    status_before = client.get("/api/config-status").json()
    assert status_before["watch_enabled"] is True
    assert status_before["changed"] is False

    session_path.write_text(yaml.safe_dump(_session_payload("repaired", item), sort_keys=False), encoding="utf-8")
    status_changed = client.get("/api/config-status").json()
    assert status_changed["changed"] is True

    reload_payload = client.post("/api/reload-config").json()
    assert reload_payload["status"] == "reloaded"
    assert reload_payload["config_status"]["changed"] is False

    page = client.get("/").text
    assert "repaired (1)" in page


def test_dataset_builder_config_status_uses_watched_setup_path(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake")
    item = DatasetItem(
        item_id="item-2",
        section="real",
        image_path=str(image_path),
        display_name="img.png",
        group_key="img",
    )
    session = _session_payload("real", item)
    session_path = tmp_path / "dataset_builder_session.yaml"
    session_path.write_text(yaml.safe_dump(session, sort_keys=False), encoding="utf-8")
    setup_path = tmp_path / "experiment_setup.yaml"
    setup_path.write_text("process_steps: {}\n", encoding="utf-8")

    app = build_app(session, [item], session_path=session_path, watch_path=setup_path)
    client = TestClient(app)

    before = client.get("/api/config-status").json()
    assert before["changed"] is False

    setup_path.write_text("process_steps:\n  values: []\n", encoding="utf-8")
    after = client.get("/api/config-status").json()
    assert after["changed"] is True
