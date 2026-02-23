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
