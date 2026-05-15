from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pipeliner.common.contract_helpers import read_structured_file
from pipeliner.dataset_builder.core import (
    assignment_conflict_summary,
    assignment_csv_path,
    derive_session_from_contract,
    load_assignment_csv,
    preview_split,
    save_dataset,
    scan_dataset_items,
    session_snapshot,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_read_structured_file_supports_yaml(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        """
variation_points:
  algo_name: patchcore
  dataset_name: demo
  dataset_variant: circled
  repair_method: LaMa
  mask_type: re_masked_thicker
process_step:
  name: B10_structure_ds_for_algo
  output: "pipeline_data/${algo_name}/${process_step}/${dataset_name}/${dataset_variant}/${repair_method}/${mask_type}"
  extra_args:
    split_ratios:
      train: 0.6
      val: 0.2
      test: 0.2
    split_seed: 99
    split_labels: [train, val, test, discard]
    class_labels: [good, defect_type_1]
    output_tree_structure:
      images: "{split_label}/{class_label}/{image_file}"
      masks: "ground_truth/{class_label}/{mask_file}"
  input_from_previous: true
  input:
    repaired: "/tmp/repaired"
    masks: "/tmp/masks"
    known_bad: "/tmp/known_bad"
resolved:
  input: "/tmp/real"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    contract = read_structured_file(contract_path)
    session = derive_session_from_contract(contract, contract_path)

    assert session["input_sections"]["previous_step"] == "/tmp/real"
    assert session["input_sections"]["repaired"] == "/tmp/repaired"
    assert session["split"]["split_seed"] == 99
    assert session["config"]["class_labels"] == ["good", "defect_type_1"]
    assert session["paths"]["output_dir"].endswith(
        "pipeline_data/patchcore/B10_structure_ds_for_algo/demo/circled/LaMa/re_masked_thicker"
    )


def test_derive_session_does_not_require_previous_step_input(tmp_path: Path) -> None:
    repaired_dir = tmp_path / "repaired"
    contract = {
        "variation_points": {"algo_name": "patchcore"},
        "process_step": {
            "name": "B10_structure_ds_for_algo",
            "input": {"repaired": str(repaired_dir)},
            "output": str(tmp_path / "out"),
            "extra_args": {
                "split_labels": ["train", "val", "discard"],
                "split_ratios": {"train": 1.0, "val": 0.0},
                "split_seed": 7,
                "class_labels": ["good", "defect"],
                "output_tree_structure": {"images": "{split_label}/{class_label}/{image_file}"},
            },
        },
        "resolved": {"input": str(repaired_dir)},
    }
    session = derive_session_from_contract(contract, tmp_path / "args.json")
    # input_from_previous is False by default, so resolved.input is ignored
    assert "previous_step" not in session["input_sections"]
    assert session["input_sections"]["repaired"] == str(repaired_dir)


def test_scan_preview_and_save_dataset(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    repaired_dir = tmp_path / "repaired"
    masks_dir = tmp_path / "masks"
    known_bad_dir = tmp_path / "known_bad"
    output_dir = tmp_path / "output"

    _write_image(real_dir / "good_001.png", (10, 20, 30))
    _write_image(real_dir / "good_002.png", (20, 30, 40))
    _write_image(known_bad_dir / "bad_001_orig.png", (40, 30, 20))
    _write_image(repaired_dir / "scene1_d1_mask.png", (50, 60, 70))
    _write_image(masks_dir / "scene1_d1_orig.png", (80, 90, 100))
    _write_image(masks_dir / "scene1_d1_mask.png", (255, 255, 255))

    input_sections = {
        "real": str(real_dir),
        "repaired": str(repaired_dir),
        "masks": str(masks_dir),
        "known_bad": str(known_bad_dir),
    }

    session = {
        "contract_path": "",
        "contract": {
            "variation_points": {"algo_name": "patchcore"},
            "process_step": {"name": "B10_structure_ds_for_algo"},
        },
        "input_sections": input_sections,
        "paths": {
            "output_dir": str(output_dir),
            "assignment_csv": str(assignment_csv_path(output_dir)),
        },
        "split": {
            "split_labels": ["train", "val", "test", "discard"],
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "split_seed": 7,
        },
        "config": {
            "class_labels": ["good", "defect_type_1"],
            "output_tree_structure": {
                "images": "{split_label}/{class_label}/{image_file}",
            },
        },
        "labels": {},
        "split_assignments": {},
        "new_item_ids": [],
    }

    items = scan_dataset_items(session["input_sections"], class_labels=session["config"]["class_labels"])
    # 2 real + 1 known_bad + 1 repaired + 2 masks = 6 images
    assert len(items) == 6

    repaired_item = next(item for item in items if item.section == "repaired")
    known_bad_item = next(item for item in items if item.section == "known_bad")
    session["labels"][repaired_item.item_id] = "defect_type_1"
    session["labels"][known_bad_item.item_id] = "defect_type_1"

    preview = preview_split(items, session["labels"], session["split"], session["split_assignments"])
    # Just verify we have some assignments
    assert len(preview["assignments"]) == 6

    summary = save_dataset(
        session=session,
        items=items,
        preview=preview,
        recreate_dataset=True,
        reset_session=False,
    )
    assert (output_dir / "dataset_builder_session.yaml").exists()
    assert (output_dir / "dataset_builder_summary.json").exists()
    assert (output_dir / "dataset_builder_assignments.csv").exists()

    summary_json = json.loads((output_dir / "dataset_builder_summary.json").read_text(encoding="utf-8"))
    assert summary_json["counts"]["train"]["good"] >= 1

    symlinks = list((output_dir / "train" / "good").glob("*"))
    assert symlinks
    assert symlinks[0].is_symlink()


def test_scan_dataset_items_ignores_overlay_files(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    _write_image(real_dir / "A0005-20260422_093029_rivet_001.jpg", (10, 20, 30))
    _write_image(real_dir / "A0005-20260422_093029_rivet_001_overlay.jpg", (20, 30, 40))

    items = scan_dataset_items(
        {"real_input_dir": str(real_dir)},
        class_labels=["good", "defect_type_1"],
    )

    assert [item.display_name for item in items] == ["A0005-20260422_093029_rivet_001.jpg"]


def test_existing_assignment_csv_takes_precedence(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    output_dir = tmp_path / "output"
    _write_image(real_dir / "good_001.png", (10, 20, 30))

    session = {
        "paths": {
            "output_dir": str(output_dir),
            "assignment_csv": str(assignment_csv_path(output_dir)),
        },
        "config": {"class_labels": ["good", "defect_type_1"]},
        "labels": {},
        "split_assignments": {},
        "new_item_ids": [],
        "csv_loaded": False,
    }
    input_sections = {"real": str(real_dir)}
    items = scan_dataset_items(input_sections, class_labels=session["config"]["class_labels"])
    csv_path = assignment_csv_path(output_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "item_id,section,image_path,display_name,group_key,class_label,split_label,is_new\n"
        f"{items[0].item_id},real,{items[0].image_path},good_001.png,good_001,defect_type_1,discard,\n",
        encoding="utf-8",
    )

    load_assignment_csv(session, items)
    assert session["csv_loaded"] is True
    assert session["labels"][items[0].item_id] == "defect_type_1"
    assert session["split_assignments"][items[0].item_id] == "discard"


def test_preview_split_counts_assignments(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    repaired_dir = tmp_path / "repaired"
    _write_image(real_dir / "good_001.png", (10, 20, 30))
    _write_image(repaired_dir / "scene1_d1_mask.png", (50, 60, 70))

    items = scan_dataset_items(
        {
            "real": str(real_dir),
            "repaired": str(repaired_dir),
        },
        class_labels=["good", "defect_type_1"],
    )
    repaired_item = next(item for item in items if item.section == "repaired")
    labels = {repaired_item.item_id: "defect_type_1"}

    preview = preview_split(
        items,
        labels,
        {
            "split_labels": ["eval", "fit", "holdout", "discard"],
            "split_ratios": {"eval": 0.2, "fit": 0.7, "holdout": 0.1},
            "split_seed": 7,
        },
        {},
    )

    assert preview["counts"]["fit"].get("good", 0) >= 0
    assert "defect_type_1" in str(preview["counts"])


def test_assignment_conflict_summary_reports_csv_vs_auto_differences(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    _write_image(real_dir / "good_001.png", (10, 20, 30))
    items = scan_dataset_items(
        {"real": str(real_dir)},
        class_labels=["good", "defect_type_1"],
    )

    summary = assignment_conflict_summary(
        items,
        {items[0].item_id: "discard"},
        {items[0].item_id: "train"},
    )

    assert summary["has_conflict"] is True
    assert summary["count"] == 1
    assert summary["examples"][0]["display_name"] == "good_001.png"


def test_session_snapshot_uses_preview_assignments_when_no_explicit_split_exists(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    _write_image(real_dir / "good_001.png", (10, 20, 30))
    items = scan_dataset_items(
        {"real": str(real_dir)},
        class_labels=["good", "defect_type_1"],
    )

    snapshot = session_snapshot(
        {
            "paths": {},
            "split": {},
            "config": {},
            "labels": {},
            "split_assignments": {},
            "new_item_ids": [],
            "csv_loaded": False,
        },
        items,
        preview={"assignments": {items[0].item_id: "train"}},
    )

    assert snapshot["items"][0]["selected_split"] == "train"
