from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeliner.iqviewer import _auto_tune_quality_thresholds, _load_cutout_tab, _load_quality_tab
from pipeliner.setup_loader import load_setup


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_quality_tab_uses_cutout_metrics_from_previous_step(tmp_path: Path) -> None:
    setup_path = tmp_path / "pipeline" / "experiment_setup.yaml"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(
        """
variation_points:
  dataset_name: [demo]
  dataset_variant: [circled]
process_steps:
  A20_cut_out:
    script: pipeline/A20_cut_out.py
  A25_image_cut_out_quality:
    script: pipeline/image_base_quality/image_base_quality.py
    input_from_previous: true
    previous_step: A20_cut_out
""".strip()
        + "\n",
        encoding="utf-8",
    )
    setup = load_setup(setup_path)

    cutout_run = tmp_path / "pipeline_data" / "A20_cut_out" / "demo" / "circled"
    _write_json(
        cutout_run / "args.json",
        {
            "process_step": {"name": "A20_cut_out"},
            "variation_points": {"dataset_name": "demo", "dataset_variant": "circled"},
        },
    )
    _write_json(
        cutout_run / "a20_summary.json",
        {
            "images": [
                {
                    "image_name": "img.png",
                    "crops": [{"crop_file": "img_rivet_001.png", "output_size": [256, 256]}],
                }
            ]
        },
    )
    with (cutout_run / "a20_cutout_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["crop_file", "source_crop_width", "source_crop_height", "source_crop_box_clipped", "target_width", "target_height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "crop_file": "img_rivet_001.png",
                "source_crop_width": "412",
                "source_crop_height": "318",
                "source_crop_box_clipped": "[10,20,422,338]",
                "target_width": "256",
                "target_height": "256",
            }
        )

    quality_run = tmp_path / "pipeline_data" / "A25_image_cut_out_quality" / "demo" / "circled"
    _write_json(
        quality_run / "args.json",
        {
            "process_step": {"name": "A25_image_cut_out_quality"},
            "variation_points": {"dataset_name": "demo", "dataset_variant": "circled"},
            "resolved": {"output": str(quality_run)},
        },
    )
    with (quality_run / "image_quality_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["imgpath", "final_status"])
        writer.writeheader()
        writer.writerow({"imgpath": str(quality_run / "img_rivet_001.png"), "final_status": "PASS"})

    tab = _load_quality_tab(
        quality_run / "image_quality_metrics.csv",
        "A25_image_cut_out_quality",
        quality_run / "args.json",
        project_root=tmp_path,
        setup=setup,
    )

    assert tab["status"] == "ok"
    assert tab["rows"][0]["image_size"] == [412.0, 318.0]
    assert tab["rows"][0]["source_crop_size"] == [412.0, 318.0]
    assert tab["rows"][0]["source_crop_box_clipped"] == "[10,20,422,338]"
    assert tab["rows"][0]["target_size"] == [256.0, 256.0]


def test_load_cutout_tab_reads_new_rivet_bbox_fields(tmp_path: Path) -> None:
    cutout_run = tmp_path / "pipeline_data" / "A20_cut_out" / "demo" / "circled"
    _write_json(
        cutout_run / "args.json",
        {
            "process_step": {"name": "A20_cut_out"},
            "variation_points": {"dataset_name": "demo", "dataset_variant": "circled"},
        },
    )
    _write_json(
        cutout_run / "a20_summary.json",
        {
            "images": [
                {
                    "image_name": "img.png",
                    "resolved_image": str(tmp_path / "img.png"),
                    "crops": [{"crop_file": "img_rivet_001.png", "rivet_bbox_xyxy": [1, 2, 11, 12], "output_size": [256, 256]}],
                }
            ]
        },
    )
    with (cutout_run / "a20_cutout_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["crop_file", "rivet_bbox", "target_size"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "crop_file": "img_rivet_001.png",
                "rivet_bbox": "[1,2,11,12]",
                "target_size": "[256,256]",
            }
        )

    tab = _load_cutout_tab(tmp_path, "demo", "circled")

    assert tab["status"] == "ok"
    assert tab["cutouts"][0]["bbox"] == "[1,2,11,12]"


def test_load_quality_tab_handles_uncropped_runs_without_cutout_metrics(tmp_path: Path) -> None:
    setup_path = tmp_path / "pipeline" / "experiment_setup.yaml"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(
        """
variation_points:
  dataset_name: [demo]
  dataset_variant: [circled]
process_steps:
  A02_image_base_quality:
    script: pipeline/image_base_quality/image_base_quality.py
    input_from_previous: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    setup = load_setup(setup_path)

    quality_run = tmp_path / "pipeline_data" / "A02_image_base_quality" / "demo" / "circled"
    _write_json(
        quality_run / "args.json",
        {
            "process_step": {"name": "A02_image_base_quality"},
            "variation_points": {"dataset_name": "demo", "dataset_variant": "circled"},
        },
    )
    with (quality_run / "image_quality_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["imgpath", "final_status"])
        writer.writeheader()
        writer.writerow({"imgpath": str(quality_run / "img.png"), "final_status": "PASS"})

    tab = _load_quality_tab(
        quality_run / "image_quality_metrics.csv",
        "A02_image_base_quality",
        quality_run / "args.json",
        project_root=tmp_path,
        setup=setup,
    )

    assert tab["status"] == "ok"
    assert tab["rows"][0]["image_size"] is None
    assert tab["rows"][0]["source_crop_size"] is None
    assert tab["rows"][0]["source_crop_box_clipped"] == ""
    assert tab["rows"][0]["target_size"] is None


def test_auto_tune_quality_thresholds_uses_manual_labels_and_respects_enabled_checks() -> None:
    result = _auto_tune_quality_thresholds(
        [
            {
                "imgpath": "img_a.png",
                "final_status": "FAIL",
                "lap_var": 10,
                "brisque": 10,
                "niqe": 1,
                "mean": 120,
                "std": 30,
                "black_clip": 0.01,
                "white_clip": 0.01,
            },
            {
                "imgpath": "img_b.png",
                "final_status": "FAIL",
                "lap_var": 20,
                "brisque": 12,
                "niqe": 1.5,
                "mean": 125,
                "std": 28,
                "black_clip": 0.02,
                "white_clip": 0.02,
            },
            {
                "imgpath": "img_c.png",
                "final_status": "PASS",
                "lap_var": 80,
                "brisque": 14,
                "niqe": 2,
                "mean": 130,
                "std": 35,
                "black_clip": 0.03,
                "white_clip": 0.03,
            },
            {
                "imgpath": "img_d.png",
                "final_status": "PASS",
                "lap_var": 100,
                "brisque": 16,
                "niqe": 2.5,
                "mean": 135,
                "std": 36,
                "black_clip": 0.04,
                "white_clip": 0.04,
            },
        ],
        {
            "laplacian_pass": True,
            "brisque_pass": False,
            "niqe_pass": False,
            "darkness_pass": False,
            "brightness_pass": False,
            "contrast_pass": False,
            "black_clip_pass": False,
            "white_clip_pass": False,
        },
    )

    assert "lapl_blur_threshold" in result["thresholds"]
    assert "brisque_threshold" not in result["thresholds"]
    assert result["report"]["accuracy"] == 1.0
    assert result["report"]["pass_rate"] == 0.5
