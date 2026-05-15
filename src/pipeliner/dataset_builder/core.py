from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pipeliner.common.contract_helpers import (
    ContractError,
    expand_templates,
    require_extra_args,
    require_mapping,
)

ASSIGNMENTS_FILE = "dataset_builder_assignments.csv"
SESSION_FILE = "dataset_builder_session.yaml"
SUMMARY_FILE = "dataset_builder_summary.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetItem:
    item_id: str
    section: str
    image_path: str
    display_name: str
    group_key: str


def assignment_csv_path(output_dir: str | Path) -> Path:
    return Path(output_dir).resolve() / ASSIGNMENTS_FILE


def _csv_flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _require_list_of_strings(extra_args: dict[str, Any], key: str) -> list[str]:
    raw = extra_args.get(key)
    if not isinstance(raw, list) or not raw:
        raise ContractError(f"process_step.extra_args.{key} must be a non-empty list")
    values = [str(v) for v in raw]
    if any(not value for value in values):
        raise ContractError(f"process_step.extra_args.{key} values must be non-empty")
    return values


def _parse_ratio(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"process_step.extra_args.split_ratios.{key} must be numeric") from exc


def _validate_split_config(extra_args: dict[str, Any]) -> dict[str, Any]:
    split_labels = _require_list_of_strings(extra_args, "split_labels")

    split_ratios_raw = extra_args.get("split_ratios")
    if not isinstance(split_ratios_raw, dict) or not split_ratios_raw:
        raise ContractError("process_step.extra_args.split_ratios must be a non-empty object")

    split_ratios: dict[str, float] = {}
    for split_name, ratio_raw in split_ratios_raw.items():
        split_key = str(split_name)
        split_ratios[split_key] = _parse_ratio(ratio_raw, split_key)

    missing = [label for label in split_labels if label != "discard" and label not in split_ratios]
    if missing:
        missing_text = ", ".join(missing)
        raise ContractError(f"split_ratios missing split_labels entries: {missing_text}")

    total = sum(split_ratios.get(label, 0.0) for label in split_labels if label != "discard")
    if not (abs(total - 1.0) < 1e-9 or abs(total - 100.0) < 1e-9):
        raise ContractError("split_ratios must sum to 1 or 100")

    return {
        "split_labels": split_labels,
        "split_ratios": split_ratios,
        "split_seed": int(extra_args.get("split_seed", 0)),
    }


def derive_session_from_contract(contract: dict[str, Any], contract_path: Path | None = None) -> dict[str, Any]:
    process_step = require_mapping(contract, "process_step", "contract")
    inputs = require_mapping(process_step, "input", "process_step")
    extra_args = require_extra_args(process_step)

    for required_key in ("split_labels", "split_ratios", "class_labels", "output_tree_structure"):
        if required_key not in extra_args:
            raise ContractError(f"missing required process_step.extra_args.{required_key}")

    split_config = _validate_split_config(extra_args)
    class_labels = _require_list_of_strings(extra_args, "class_labels")

    output_tree = extra_args.get("output_tree_structure")
    if not isinstance(output_tree, dict) or not output_tree:
        raise ContractError("process_step.extra_args.output_tree_structure must be a non-empty object")
    if "images" not in output_tree or not isinstance(output_tree.get("images"), str):
        raise ContractError("output_tree_structure.images must be a string")

    # Prepare variables for template expansion
    variables = dict(contract.get("variation_points", {}))
    variables.update(process_step)
    if "name" in process_step:
        variables["process_step"] = process_step["name"]
    variables.update(contract.get("resolved", {}))

    output_dir_raw = process_step.get("output", contract.get("resolved", {}).get("output", ""))
    output_dir = str(expand_templates(output_dir_raw, variables)).strip()
    if not output_dir:
        raise ContractError("process_step.output is required")

    input_sections: dict[str, str] = {}
    for section_name, path_value in inputs.items():
        section = str(section_name)
        location = str(path_value).strip()
        if not section:
            raise ContractError("process_step.input keys must be non-empty")
        input_sections[section] = location

    # When input_from_previous is enabled, include resolved.input as an extra section.
    if bool(process_step.get("input_from_previous", False)):
        resolved = contract.get("resolved", {})
        prev_input_raw = resolved.get("input", "") if isinstance(resolved, dict) else ""
        prev_input = str(prev_input_raw).strip()
        if prev_input:
            prev_step_name = str(process_step.get("previous_step", "")).strip() or "previous_step"
            input_sections[prev_step_name] = prev_input

    out_path = Path(output_dir).resolve()
    return {
        "contract_path": str(contract_path.resolve()) if contract_path else "",
        "contract": contract,
        "input_sections": input_sections,
        "paths": {
            "output_dir": str(out_path),
            "assignment_csv": str(assignment_csv_path(out_path)),
        },
        "split": {
            "split_labels": split_config["split_labels"],
            "split_ratios": split_config["split_ratios"],
            "split_seed": split_config["split_seed"],
        },
        "config": {
            "class_labels": class_labels,
            "output_tree_structure": output_tree,
        },
        "labels": {},
        "split_assignments": {},
        "inventory_item_ids": [],
        "new_item_ids": [],
        "csv_loaded": False,
    }


def _iter_images(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "overlay" not in path.stem.lower()
    )


def _build_item_id(section: str, image_path: Path) -> str:
    digest = hashlib.sha1(f"{section}:{image_path}".encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest[:16]


def _normalize_group_key(stem: str) -> str:
    if stem.endswith("_mask"):
        return stem[:-5]
    return stem


def scan_dataset_items(
    input_sections: dict[str, str] | dict[str, Any],
    _meta: dict[str, Any] | None = None,
    class_labels: list[str] | None = None,
) -> list[DatasetItem]:
    section_map = dict(input_sections)
    if "input_sections" in section_map and isinstance(section_map.get("input_sections"), dict):
        section_map = dict(section_map["input_sections"])

    items: list[DatasetItem] = []
    for section, raw_path in section_map.items():
        section_name = str(section)
        root_path = Path(str(raw_path).strip()) if str(raw_path).strip() else None
        if root_path is None:
            continue
        for image_path in _iter_images(root_path):
            rel_name = str(image_path.relative_to(root_path))
            item = DatasetItem(
                item_id=_build_item_id(section_name, image_path.resolve()),
                section=section_name,
                image_path=str(image_path.resolve()),
                display_name=rel_name,
                group_key=_normalize_group_key(Path(rel_name).stem),
            )
            items.append(item)

    del class_labels
    return items


def _normalize_ratios(split: dict[str, Any]) -> dict[str, float]:
    split_labels = [str(v) for v in split.get("split_labels", [])]
    raw = split.get("split_ratios", {})
    ratios = {name: float(raw.get(name, 0.0)) for name in split_labels if name != "discard"}
    total = sum(ratios.values())
    if abs(total - 100.0) < 1e-9:
        return {k: v / 100.0 for k, v in ratios.items()}
    return ratios


def initialize_split_assignments(items: list[DatasetItem], split: dict[str, Any]) -> dict[str, str]:
    split_labels = [str(v) for v in split.get("split_labels", [])]
    export_splits = [name for name in split_labels if name != "discard"]
    if not export_splits:
        return {}

    # If an item belongs to a section that should be discarded by default (e.g. original images)
    # or is auxiliary data (like masks), we assign it to 'discard' immediately.
    assignments: dict[str, str] = {}
    remaining_items: list[DatasetItem] = []
    for item in items:
        section_lower = item.section.lower()
        # Originals and Cut-outs go to discard by default
        if "cut_out" in section_lower or "original" in section_lower:
            assignments[item.item_id] = "discard"
        # Masks and auxiliary files should ALWAYS be discard (they are exported separately)
        elif "mask" in section_lower or "preview" in section_lower:
            assignments[item.item_id] = "discard"
        else:
            remaining_items.append(item)

    if not remaining_items:
        return assignments

    ratios = _normalize_ratios(split)
    for name in export_splits:
        if name not in ratios:
            raise ContractError(f"split_ratios missing {name}")

    seed = int(split.get("split_seed", 0))
    ordered = sorted(remaining_items, key=lambda item: item.item_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total_count = len(ordered)
    raw_counts = {name: total_count * ratios[name] for name in export_splits}
    counts = {name: int(raw_counts[name]) for name in export_splits}
    remainder = total_count - sum(counts.values())
    rank = sorted(export_splits, key=lambda name: (raw_counts[name] - counts[name]), reverse=True)
    for index in range(remainder):
        counts[rank[index % len(rank)]] += 1

    cursor = 0
    for split_name in export_splits:
        target = counts[split_name]
        for item in ordered[cursor : cursor + target]:
            assignments[item.item_id] = split_name
        cursor += target
    return assignments


def load_assignment_csv(session: dict[str, Any], items: list[DatasetItem]) -> bool:
    csv_path = Path(str(session.get("paths", {}).get("assignment_csv", "")))
    if not csv_path.exists():
        session["csv_loaded"] = False
        return False

    index_by_item = {item.item_id: item for item in items}
    index_by_image = {item.image_path: item for item in items}

    labels = session.setdefault("labels", {})
    splits = session.setdefault("split_assignments", {})
    new_item_ids: set[str] = set(session.setdefault("new_item_ids", []))
    inventory_item_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item: DatasetItem | None = None
            item_id = str(row.get("item_id", "")).strip()
            if item_id:
                item = index_by_item.get(item_id)
            if item is None:
                image_value = str(row.get("image_path", row.get("imagepath", ""))).strip()
                if image_value:
                    item = index_by_image.get(str(Path(image_value).resolve())) or index_by_image.get(image_value)
            if item is None:
                continue

            inventory_item_ids.add(item.item_id)
            class_label = str(row.get("class_label", "")).strip()
            split_label = str(row.get("split_label", "")).strip()
            if class_label:
                labels[item.item_id] = class_label
            if split_label:
                splits[item.item_id] = split_label
            if _csv_flag(row.get("is_new", "")):
                new_item_ids.add(item.item_id)

    session["inventory_item_ids"] = sorted(inventory_item_ids)
    session["new_item_ids"] = sorted(new_item_ids)
    session["csv_loaded"] = True
    return True


def merge_assignment_inventory(session: dict[str, Any], items: list[DatasetItem]) -> dict[str, Any]:
    csv_found = load_assignment_csv(session, items)
    explicit_splits = session.setdefault("split_assignments", {})
    known_item_ids = set(str(item_id) for item_id in session.get("inventory_item_ids", []))

    new_items = [item for item in items if item.item_id not in known_item_ids]
    if new_items:
        auto_assignments = initialize_split_assignments(new_items, session.get("split", {}))
        explicit_splits.update(auto_assignments)

    current_new_item_ids = set(session.get("new_item_ids", []))
    current_new_item_ids.update(item.item_id for item in new_items)
    session["new_item_ids"] = sorted(current_new_item_ids)
    session["csv_loaded"] = csv_found
    return {
        "csv_found": csv_found,
        "new_item_ids": list(session["new_item_ids"]),
    }


def save_assignment_csv(session: dict[str, Any], items: list[DatasetItem]) -> Path:
    csv_path = Path(str(session.get("paths", {}).get("assignment_csv", ""))).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    labels = session.get("labels", {})
    splits = session.get("split_assignments", {})
    new_item_ids = set(str(item_id) for item_id in session.get("new_item_ids", []))

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "section",
                "image_path",
                "display_name",
                "group_key",
                "class_label",
                "split_label",
                "is_new",
            ],
        )
        writer.writeheader()
        for item in sorted(items, key=lambda value: value.item_id):
            writer.writerow(
                {
                    "item_id": item.item_id,
                    "section": item.section,
                    "image_path": item.image_path,
                    "display_name": item.display_name,
                    "group_key": item.group_key,
                    "class_label": str(labels.get(item.item_id, "")),
                    "split_label": str(splits.get(item.item_id, "")),
                    "is_new": "1" if item.item_id in new_item_ids else "",
                }
            )
    return csv_path


def preview_split(
    items: list[DatasetItem],
    labels: dict[str, str],
    split: dict[str, Any],
    split_assignments: dict[str, str],
) -> dict[str, Any]:
    auto = initialize_split_assignments(items, split)

    merged: dict[str, str] = {}
    split_labels = [str(v) for v in split.get("split_labels", [])]

    # Identify groups that should be excluded from training (mutual exclusion)
    # If an original/bad image from a group is assigned to a non-discard split,
    # we should probably not use the repaired version for training.
    excluded_groups: set[str] = set()
    for item in items:
        assigned = split_assignments.get(item.item_id, auto.get(item.item_id, ""))
        label = labels.get(item.item_id, "")
        # If explicitly assigned to a test/val split or has a defect label
        if assigned in {"test", "val"} or (label and label != "good"):
            if "repair" not in item.section.lower():
                excluded_groups.add(item.group_key)

    for item in items:
        assigned = split_assignments.get(item.item_id, auto.get(item.item_id, ""))

        # Apply mutual exclusion: if group is excluded and this is a repaired image, discard it.
        if item.group_key in excluded_groups and "repair" in item.section.lower():
            assigned = "discard"

        if assigned:
            merged[item.item_id] = assigned

    class_labels = [str(v) for v in split.get("class_labels", [])]
    del class_labels

    counts: dict[str, dict[str, int]] = {name: {} for name in split_labels if name != "discard"}
    for item in items:
        split_name = merged.get(item.item_id, "")
        if split_name == "discard" or split_name not in counts:
            continue
        label = str(labels.get(item.item_id, "good"))
        bucket = counts[split_name]
        bucket[label] = bucket.get(label, 0) + 1

    return {
        "assignments": merged,
        "counts": counts,
    }


def assignment_conflict_summary(
    items: list[DatasetItem],
    csv_assignments: dict[str, str],
    auto_assignments: dict[str, str],
) -> dict[str, Any]:
    by_id = {item.item_id: item for item in items}
    conflicts: list[dict[str, str]] = []
    for item_id, csv_split in csv_assignments.items():
        auto_split = auto_assignments.get(item_id)
        if auto_split is None or auto_split == csv_split:
            continue
        item = by_id.get(item_id)
        if item is None:
            continue
        conflicts.append(
            {
                "item_id": item_id,
                "display_name": item.display_name,
                "csv_split": csv_split,
                "auto_split": auto_split,
            }
        )

    return {
        "has_conflict": bool(conflicts),
        "count": len(conflicts),
        "examples": conflicts[:10],
    }


def session_snapshot(session: dict[str, Any], items: list[DatasetItem], preview: dict[str, Any]) -> dict[str, Any]:
    assignments = dict(preview.get("assignments", {}))
    explicit = dict(session.get("split_assignments", {}))
    labels = dict(session.get("labels", {}))
    new_item_ids = set(str(item_id) for item_id in session.get("new_item_ids", []))

    payload_items = []
    for item in items:
        selected_split = explicit.get(item.item_id, assignments.get(item.item_id, ""))
        payload_items.append(
            {
                "item_id": item.item_id,
                "section": item.section,
                "image_path": item.image_path,
                "display_name": item.display_name,
                "group_key": item.group_key,
                "selected_split": selected_split,
                "selected_label": labels.get(item.item_id, ""),
                "is_new": item.item_id in new_item_ids,
            }
        )

    return {
        "contract_path": session.get("contract_path", ""),
        "contract": session.get("contract", {}),
        "input_sections": session.get("input_sections", {}),
        "paths": session.get("paths", {}),
        "split": session.get("split", {}),
        "config": session.get("config", {}),
        "labels": labels,
        "split_assignments": explicit,
        "new_item_ids": sorted(new_item_ids),
        "csv_loaded": bool(session.get("csv_loaded", False)),
        "items": payload_items,
    }


def _safe_unlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _render_output_path(template: str, split_label: str, class_label: str, item: DatasetItem) -> str:
    return (
        template.replace("{split_label}", split_label)
        .replace("{class_label}", class_label)
        .replace("{image_file}", Path(item.image_path).name)
        .replace("{mask_file}", Path(item.image_path).name)
    )


def export_dataset(
    session: dict[str, Any],
    items: list[DatasetItem],
    preview: dict[str, Any],
    recreate_dataset: bool = True,
) -> dict[str, Any]:
    output_dir = Path(str(session.get("paths", {}).get("output_dir", ""))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if recreate_dataset:
        for split_name in session.get("split", {}).get("split_labels", []):
            split_text = str(split_name)
            if split_text == "discard":
                continue
            split_path = output_dir / split_text
            if split_path.exists() or split_path.is_symlink():
                _safe_unlink(split_path)
        
        # Also clean up ground_truth dir if it exists
        gt_path = output_dir / "ground_truth"
        if gt_path.exists():
            _safe_unlink(gt_path)

    # Ensure all split/class directory combinations exist (Anomalib requirement)
    split_labels = [str(v) for v in session.get("split", {}).get("split_labels", [])]
    class_labels = [str(v) for v in session.get("config", {}).get("class_labels", [])]
    for split_name in split_labels:
        if split_name == "discard":
            continue
        for class_name in class_labels:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)
    
    # Also ensure ground_truth/defect exists
    (output_dir / "ground_truth" / "defect").mkdir(parents=True, exist_ok=True)

    output_tree = dict(session.get("config", {}).get("output_tree_structure", {}))
    image_template = str(output_tree.get("images", "{split_label}/{class_label}/{image_file}"))
    mask_template = str(output_tree.get("masks", "ground_truth/{class_label}/{mask_file}"))

    labels = session.setdefault("labels", {})
    explicit_splits = session.setdefault("split_assignments", {})
    all_assignments = dict(preview.get("assignments", {}))

    # Index items by group_key and section to find masks easily
    items_by_group: dict[str, dict[str, DatasetItem]] = {}
    for item in items:
        items_by_group.setdefault(item.group_key, {})[item.section] = item

    counts: dict[str, dict[str, int]] = {}
    for item in items:
        split_name = explicit_splits.get(item.item_id, all_assignments.get(item.item_id, ""))
        if not split_name or split_name == "discard":
            continue

        class_label = str(labels.get(item.item_id, "good"))
        target_rel = _render_output_path(image_template, split_name, class_label, item)
        target_path = output_dir / target_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() or target_path.is_symlink():
            _safe_unlink(target_path)
        target_path.symlink_to(Path(item.image_path))

        # Handle mask if this is a defect
        if class_label == "defect":
            # Try to find a mask for this group
            group_sections = items_by_group.get(item.group_key, {})
            # Prefer masks from the 'masks' section, but fallback to any section containing 'mask'
            mask_item = group_sections.get("masks")
            if not mask_item:
                for sec_name, sec_item in group_sections.items():
                    if "mask" in sec_name.lower() and sec_name != item.section:
                        mask_item = sec_item
                        break
            
            if mask_item:
                mask_target_rel = _render_output_path(mask_template, split_name, class_label, item)
                mask_target_path = output_dir / mask_target_rel
                mask_target_path.parent.mkdir(parents=True, exist_ok=True)
                if mask_target_path.exists() or mask_target_path.is_symlink():
                    _safe_unlink(mask_target_path)
                mask_target_path.symlink_to(Path(mask_item.image_path))

        bucket = counts.setdefault(split_name, {})
        bucket[class_label] = bucket.get(class_label, 0) + 1

    session["new_item_ids"] = []
    snapshot = session_snapshot(session, items, preview)
    summary = {
        "counts": counts,
        "total_items": len(items),
        "exported_items": sum(sum(part.values()) for part in counts.values()),
        "assignment_csv": str(assignment_csv_path(output_dir)),
    }

    session_path = output_dir / SESSION_FILE
    summary_path = output_dir / SUMMARY_FILE
    session_path.write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    # Save a copy of the inventory CSV in the output directory
    save_assignment_csv(session, items)
    
    return summary


def save_dataset(
    session: dict[str, Any],
    items: list[DatasetItem],
    preview: dict[str, Any],
    recreate_dataset: bool,
    reset_session: bool,
) -> dict[str, Any]:
    del reset_session
    return export_dataset(session=session, items=items, preview=preview, recreate_dataset=recreate_dataset)


def execute_headless(contract: dict[str, Any], contract_path: Path | None = None) -> dict[str, Any]:
    session = derive_session_from_contract(contract, contract_path)
    items = scan_dataset_items(session["input_sections"], class_labels=session["config"]["class_labels"])

    csv_found = merge_assignment_inventory(session, items)["csv_found"]
    if not csv_found:
        csv_path = session["paths"]["assignment_csv"]
        raise FileNotFoundError(f"headless mode requires existing CSV at {csv_path}")

    preview = preview_split(items, session["labels"], session["split"], session["split_assignments"])
    return export_dataset(session, items, preview, recreate_dataset=True)


def execute_web_setup(contract: dict[str, Any], contract_path: Path | None = None) -> tuple[dict[str, Any], list[DatasetItem], dict[str, Any]]:
    session = derive_session_from_contract(contract, contract_path)
    items = scan_dataset_items(session["input_sections"], class_labels=session["config"]["class_labels"])

    merge_assignment_inventory(session, items)
    preview = preview_split(items, session["labels"], session["split"], session["split_assignments"])
    save_assignment_csv(session, items)
    return session, items, preview


__all__ = [
    "ASSIGNMENTS_FILE",
    "SESSION_FILE",
    "SUMMARY_FILE",
    "DatasetItem",
    "assignment_conflict_summary",
    "assignment_csv_path",
    "derive_session_from_contract",
    "execute_headless",
    "execute_web_setup",
    "export_dataset",
    "initialize_split_assignments",
    "load_assignment_csv",
    "merge_assignment_inventory",
    "preview_split",
    "save_assignment_csv",
    "save_dataset",
    "scan_dataset_items",
    "session_snapshot",
]
