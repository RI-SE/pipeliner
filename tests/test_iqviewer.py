from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeliner.iqviewer import _load_quality_tab
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
  A25_image_base_quality:
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

    quality_run = tmp_path / "pipeline_data" / "A25_image_base_quality" / "demo" / "circled"
    _write_json(
        quality_run / "args.json",
        {
            "process_step": {"name": "A25_image_base_quality"},
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
        "A25_image_base_quality",
        quality_run / "args.json",
        project_root=tmp_path,
        setup=setup,
    )

    assert tab["status"] == "ok"
    assert tab["rows"][0]["image_size"] == [412.0, 318.0]
    assert tab["rows"][0]["source_crop_size"] == [412.0, 318.0]
    assert tab["rows"][0]["source_crop_box_clipped"] == "[10,20,422,338]"
    assert tab["rows"][0]["target_size"] == [256.0, 256.0]


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
