from __future__ import annotations
from pathlib import Path
from pipeliner.dataset_builder.core import DatasetItem, preview_split

def test_preview_split_mutual_exclusion():
    # Setup: One original and one repaired image for the same rivet
    item_orig = DatasetItem(
        item_id="orig1",
        section="A20_cut_out",
        image_path="path/to/orig.png",
        display_name="orig.png",
        group_key="rivet1"
    )
    item_rep = DatasetItem(
        item_id="rep1",
        section="A40_repair",
        image_path="path/to/rep_mask.png",
        display_name="rep_mask.png",
        group_key="rivet1"
    )
    
    items = [item_orig, item_rep]
    labels = {"orig1": "defect"} # Original is marked as defect
    split = {
        "split_labels": ["train", "test", "discard"],
        "split_ratios": {"train": 1.0, "test": 0.0},
        "split_seed": 42
    }
    split_assignments = {"orig1": "test"} # Original is put in test
    
    preview = preview_split(items, labels, split, split_assignments)
    
    # The repaired image should be automatically discarded to avoid leakage
    assert preview["assignments"]["rep1"] == "discard"
    assert preview["assignments"]["orig1"] == "test"

def test_preview_split_no_exclusion_if_not_test():
    item_orig = DatasetItem(
        item_id="orig1",
        section="A20_cut_out",
        image_path="path/to/orig.png",
        display_name="orig.png",
        group_key="rivet1"
    )
    item_rep = DatasetItem(
        item_id="rep1",
        section="A40_repair",
        image_path="path/to/rep_mask.png",
        display_name="rep_mask.png",
        group_key="rivet1"
    )
    
    items = [item_orig, item_rep]
    labels = {} # No defect label
    split = {
        "split_labels": ["train", "test", "discard"],
        "split_ratios": {"train": 1.0, "test": 0.0},
        "split_seed": 42
    }
    split_assignments = {} # Everything defaults to train
    
    preview = preview_split(items, labels, split, split_assignments)
    
    # Repaired stays in train, but original is now AUTO-DISCARDED by default
    assert preview["assignments"]["rep1"] == "train"
    assert preview["assignments"]["orig1"] == "discard"
