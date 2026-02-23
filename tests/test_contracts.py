from pipeliner.contracts import StepContract


def test_contract_roundtrip() -> None:
    c1 = StepContract(
        variation_points={"dataset_name": "kickoff"},
        process_step={"name": "A40_repair"},
        resolved={"input": "in", "output": "out"},
    )
    c2 = StepContract.from_json(c1.to_json())
    assert c2.variation_points["dataset_name"] == "kickoff"
    assert c2.process_step["name"] == "A40_repair"
    assert c2.resolved["output"] == "out"
