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

    assert session["paths"]["real_input_dir"] == "/tmp/real"
    assert session["paths"]["repaired_dir"] == "/tmp/repaired"
    assert session["split"]["seed"] == 99
    assert session["config"]["class_labels"] == ["good", "defect_type_1"]
    assert session["paths"]["output_dir"].endswith(
        "pipeline_data/patchcore/B10_structure_ds_for_algo/demo/circled/LaMa/re_masked_thicker"
    )


def test_derive_session_does_not_use_repaired_dir_as_real_input(tmp_path: Path) -> None:
    repaired_dir = tmp_path / "repaired"
    contract = {
        "variation_points": {"algo_name": "patchcore"},
        "process_step": {
            "name": "B10_structure_ds_for_algo",
            "input": {"repaired": str(repaired_dir)},
            "output": str(tmp_path / "out"),
        },
        "resolved": {"input": str(repaired_dir)},
    }
    session = derive_session_from_contract(contract, tmp_path / "args.json")
    assert session["paths"]["real_input_dir"] == ""


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

    session = {
        "contract_path": "",
        "contract": {
            "variation_points": {"algo_name": "patchcore"},
            "process_step": {"name": "B10_structure_ds_for_algo"},
        },
        "paths": {
            "real_input_dir": str(real_dir),
            "repaired_dir": str(repaired_dir),
            "masks_dir": str(masks_dir),
            "known_bad_dir": str(known_bad_dir),
            "output_dir": str(output_dir),
            "assignment_csv": str(assignment_csv_path(output_dir)),
        },
        "split": {
            "split_labels": ["train", "val", "test", "discard"],
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "seed": 7,
            "one_class": True,
        },
        "config": {
            "class_labels": ["good", "defect_type_1"],
            "output_tree_structure": {
                "images": "{split_label}/{class_label}/{image_file}",
                "masks": "ground_truth/{class_label}/{mask_file}",
            },
        },
        "meta": {"repair_method": "LaMa", "mask_type": "re_masked_thicker"},
        "labels": {},
        "split_assignments": {},
    }

    items = scan_dataset_items(session["paths"], session["meta"], class_labels=session["config"]["class_labels"])
    assert len(items) == 4

    repaired_item = next(item for item in items if item.section == "repaired")
    known_bad_item = next(item for item in items if item.section == "known_bad")
    session["labels"][repaired_item.item_id] = "defect_type_1"
    session["labels"][known_bad_item.item_id] = "defect_type_1"

    preview = preview_split(items, session["labels"], session["split"], session["split_assignments"])
    assert preview["counts"]["train"].get("defect_type_1", 0) == 0
    assert preview["counts"]["val"].get("defect_type_1", 0) + preview["counts"]["test"].get("defect_type_1", 0) >= 1

    summary = save_dataset(session=session, items=items, preview=preview, recreate_dataset=True, reset_session=False)
    assert (output_dir / "dataset_builder_session.yaml").exists()
    assert (output_dir / "dataset_builder_summary.json").exists()
    assert (output_dir / "dataset_builder_assignments.csv").exists()

    summary_json = json.loads((output_dir / "dataset_builder_summary.json").read_text(encoding="utf-8"))
    assert summary_json["counts"]["train"]["good"] >= 1
    assert summary_json["ground_truth_counts"]["defect_type_1"] >= 1

    symlinks = list((output_dir / "train" / "good").glob("*"))
    assert symlinks
    assert symlinks[0].is_symlink()


def test_existing_assignment_csv_takes_precedence(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    output_dir = tmp_path / "output"
    _write_image(real_dir / "good_001.png", (10, 20, 30))

    session = {
        "paths": {
            "real_input_dir": str(real_dir),
            "repaired_dir": "",
            "masks_dir": "",
            "known_bad_dir": "",
            "output_dir": str(output_dir),
            "assignment_csv": str(assignment_csv_path(output_dir)),
        },
        "config": {"class_labels": ["good", "defect_type_1"]},
        "labels": {},
        "split_assignments": {},
        "csv_loaded": False,
    }
    items = scan_dataset_items(session["paths"], {}, class_labels=session["config"]["class_labels"])
    csv_path = assignment_csv_path(output_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "imagepath,maskpath,class_label,split_label\n"
        f"{items[0].image_path},,defect_type_1,discard\n",
        encoding="utf-8",
    )

    load_assignment_csv(session, items)
    assert session["csv_loaded"] is True
    assert session["labels"][items[0].item_id] == "defect_type_1"
    assert session["split_assignments"][items[0].item_id] == "discard"


def test_preview_split_uses_largest_ratio_split_for_good_only_bucket(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    repaired_dir = tmp_path / "repaired"
    _write_image(real_dir / "good_001.png", (10, 20, 30))
    _write_image(repaired_dir / "scene1_d1_mask.png", (50, 60, 70))

    items = scan_dataset_items(
        {
            "real_input_dir": str(real_dir),
            "repaired_dir": str(repaired_dir),
            "masks_dir": "",
            "known_bad_dir": "",
            "output_dir": str(tmp_path / "output"),
            "assignment_csv": str(assignment_csv_path(tmp_path / "output")),
        },
        {},
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
            "seed": 7,
            "one_class": True,
        },
        {},
    )

    assert preview["normal_only_split"] == "fit"
    assert preview["counts"]["fit"].get("defect_type_1", 0) == 0


def test_assignment_conflict_summary_reports_csv_vs_auto_differences(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    _write_image(real_dir / "good_001.png", (10, 20, 30))
    items = scan_dataset_items(
        {
            "real_input_dir": str(real_dir),
            "repaired_dir": "",
            "masks_dir": "",
            "known_bad_dir": "",
            "output_dir": str(tmp_path / "output"),
            "assignment_csv": str(assignment_csv_path(tmp_path / "output")),
        },
        {},
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
        {
            "real_input_dir": str(real_dir),
            "repaired_dir": "",
            "masks_dir": "",
            "known_bad_dir": "",
            "output_dir": str(tmp_path / "output"),
            "assignment_csv": str(assignment_csv_path(tmp_path / "output")),
        },
        {},
        class_labels=["good", "defect_type_1"],
    )

    snapshot = session_snapshot(
        {
            "paths": {},
            "split": {},
            "config": {},
            "meta": {},
            "labels": {},
            "split_assignments": {},
            "csv_loaded": False,
        },
        items,
        preview={"assignments": {items[0].item_id: "train"}},
    )

    assert snapshot["items"][0]["selected_split"] == "train"
