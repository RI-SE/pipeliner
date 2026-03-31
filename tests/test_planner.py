from pathlib import Path

from pipeliner.planner import build_step_run
from pipeliner.setup_loader import load_setup


def test_build_step_run(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.yaml"
    setup_file.write_text(
        """
variation_points:
  dataset_name: [kickoff]
process_steps:
  A40_repair:
    script: pipeline/A40_repair.py
    input: pipeline_data/A30_mask/${dataset_name}
    output: pipeline_data/A40_repair/${dataset_name}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    setup = load_setup(setup_file)
    run = build_step_run(setup, "A40_repair", {"dataset_name": "kickoff"})
    assert run.script == "pipeline/A40_repair.py"
    assert run.contract.resolved["input"] == "pipeline_data/A30_mask/kickoff"
    assert run.contract.resolved["output"] == "pipeline_data/A40_repair/kickoff"


def test_build_step_run_extra_args(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.yaml"
    setup_file.write_text(
        """
variation_points:
  dataset_name: [kickoff]
process_steps:
  A05_segment_rivets:
    script: pipeline/show_instructions.py
    extra-args:
      - name: filepath
        value: pipeline/A05_segment_rivets_instructions.txt
      - name: pause
        value: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    setup = load_setup(setup_file)
    run = build_step_run(setup, "A05_segment_rivets", {"dataset_name": "kickoff"})
    cmd = run.command()
    assert cmd[-4:] == [
        "--filepath",
        "pipeline/A05_segment_rivets_instructions.txt",
        "--pause",
        "true",
    ]


def test_build_step_run_extra_args_normalizes_underscores(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.yaml"
    setup_file.write_text(
        """
variation_points:
  dataset_name: [kickoff]
process_steps:
  A02_image_base_quality:
    script: pipeline/image_base_quality/image_base_quality.py
    extra-args:
      - name: lapl_blur_threshold
        value: 100.0
      - name: create_latex_report
        value: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    setup = load_setup(setup_file)
    run = build_step_run(setup, "A02_image_base_quality", {"dataset_name": "kickoff"})
    cmd = run.command()
    assert cmd[-4:] == [
        "--lapl-blur-threshold",
        "100.0",
        "--create-latex-report",
        "true",
    ]


def test_build_step_run_preserves_structured_extra_args_without_cli_flags(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.yaml"
    setup_file.write_text(
        """
variation_points:
  dataset_name: [kickoff]
process_steps:
  B10_structure_ds_for_algo:
    script: pipeline/dataset_builder.py
    extra_args:
      split_ratios:
        train: 0.7
        val: 0.1
        test: 0.2
      split_seed: 42
      split_labels: [train, val, test, discard]
      class_labels: [good, defect_type_1]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    setup = load_setup(setup_file)
    run = build_step_run(setup, "B10_structure_ds_for_algo", {"dataset_name": "kickoff"})
    cmd = run.command()
    assert cmd[:4] == [
        "python3",
        "pipeline/dataset_builder.py",
        "--contract-json",
        run.contract.to_json(),
    ]
    assert cmd == cmd[:4]
    assert run.contract.process_step["extra_args"]["split_seed"] == 42
    assert run.contract.process_step["extra_args"]["split_labels"] == ["train", "val", "test", "discard"]


def test_build_step_run_dataset_root_from_option(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.yaml"
    setup_file.write_text(
        """
variation_points:
  values:
    - name: dataset_name
      options:
        - name: real_data
          values: [pers_bilder_ophone16]
          root: /Volumes/OneDrive/ImageDatasets
process_steps:
  values:
    - name: A05_segment_rivets
      script: pipeline/show_instructions.py
      input: ${dataset_root}/${dataset_variant}
      output: pipeline_data/${process_step}/${dataset_name}/${dataset_variant}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    setup = load_setup(setup_file)
    run = build_step_run(
        setup,
        "A05_segment_rivets",
        {"dataset_name": "real_data", "dataset_variant": "pers_bilder_ophone16"},
    )
    assert run.contract.resolved["input"] == "/Volumes/OneDrive/ImageDatasets/pers_bilder_ophone16"
    assert run.contract.variation_points["dataset_root"] == "/Volumes/OneDrive/ImageDatasets"


def test_build_step_run_dataset_root_default_fallback(tmp_path: Path) -> None:
    setup_file = tmp_path / "setup.yaml"
    setup_file.write_text(
        """
variation_points:
  values:
    - name: dataset_name
      options:
        - name: kickoff_full
          values: [circled]
process_steps:
  values:
    - name: A05_segment_rivets
      script: pipeline/show_instructions.py
      input: ${dataset_root}/${dataset_variant}
      output: pipeline_data/${process_step}/${dataset_name}/${dataset_variant}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    setup = load_setup(setup_file)
    run = build_step_run(
        setup,
        "A05_segment_rivets",
        {"dataset_name": "kickoff_full", "dataset_variant": "circled"},
    )
    assert run.contract.resolved["input"] == "input/kickoff_full/circled"
    assert run.contract.variation_points["dataset_root"] == "input/kickoff_full"
