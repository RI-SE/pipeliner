from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient

from pipeliner.dataset_builder.core import (
    DatasetItem,
    execute_web_setup,
    export_dataset,
)
from pipeliner.dataset_builder.web import build_app


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def _contract(input_dir: Path, output_dir: Path) -> dict:
    return {
        "process_step": {
            "input": {"real": str(input_dir)},
            "output": str(output_dir),
            "extra_args": {
                "split_labels": ["train", "val", "discard"],
                "split_ratios": {"train": 1.0, "val": 0.0},
                "split_seed": 7,
                "class_labels": ["good", "defect"],
                "output_tree_structure": {"images": "{split_label}/{class_label}/{image_file}"},
            },
        },
        "resolved": {},
    }


def _item_by_name(items: list[DatasetItem], name: str) -> DatasetItem:
    return next(item for item in items if item.display_name == name)


def test_execute_web_setup_appends_new_rows_and_marks_only_new_items(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_image(input_dir / "known.png", (10, 20, 30))
    _write_image(input_dir / "fresh.png", (40, 50, 60))

    contract = _contract(input_dir, output_dir)
    session, items, _preview = execute_web_setup(contract)
    known_item = _item_by_name(items, "known.png")
    csv_path = output_dir / "dataset_builder_assignments.csv"

    csv_path.write_text(
        "item_id,section,image_path,display_name,group_key,class_label,split_label,is_new\n"
        f"{known_item.item_id},{known_item.section},{known_item.image_path},{known_item.display_name},{known_item.group_key},defect,val,\n",
        encoding="utf-8",
    )

    session, items, _preview = execute_web_setup(contract)
    known_item = _item_by_name(items, "known.png")
    fresh_item = _item_by_name(items, "fresh.png")

    assert session["labels"][known_item.item_id] == "defect"
    assert session["split_assignments"][known_item.item_id] == "val"
    assert session["split_assignments"][fresh_item.item_id] == "train"
    assert session["new_item_ids"] == [fresh_item.item_id]

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    row_by_name = {row["display_name"]: row for row in rows}
    assert row_by_name["known.png"]["is_new"] == ""
    assert row_by_name["fresh.png"]["is_new"] == "1"


def test_export_dataset_clears_new_flags_after_success(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_image(input_dir / "fresh.png", (10, 20, 30))

    session, items, preview = execute_web_setup(_contract(input_dir, output_dir))
    fresh_item = _item_by_name(items, "fresh.png")
    assert session["new_item_ids"] == [fresh_item.item_id]

    export_dataset(session, items, preview, recreate_dataset=True)

    assert session["new_item_ids"] == []

    csv_path = output_dir / "dataset_builder_assignments.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["is_new"] == ""


def test_web_page_supports_show_new_only_and_marks_new_cards(tmp_path: Path) -> None:
    image_path = tmp_path / "fresh.png"
    image_path.write_bytes(b"fake")
    item = DatasetItem(
        item_id="item-1",
        section="real",
        image_path=str(image_path),
        display_name="fresh.png",
        group_key="fresh",
    )
    session = {
        "paths": {},
        "split": {
            "split_labels": ["train", "val", "discard"],
            "split_ratios": {"train": 1.0, "val": 0.0},
            "split_seed": 0,
        },
        "config": {
            "class_labels": ["good", "defect"],
            "output_tree_structure": {"images": "{split_label}/{class_label}/{image_file}"},
        },
        "input_sections": {"real": str(tmp_path)},
        "labels": {},
        "split_assignments": {"item-1": "train"},
        "new_item_ids": ["item-1"],
        "csv_loaded": True,
    }

    client = TestClient(build_app(session, [item]))
    page = client.get("/").text

    assert "Show new only" in page
    assert "NEW" in page
    assert "data-is-new='1'" in page
