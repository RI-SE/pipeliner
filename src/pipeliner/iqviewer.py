#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import mimetypes
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from .setup_loader import ExperimentSetup, load_setup

try:
    from PIL import Image, ImageColor, ImageDraw
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageColor = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


def _infer_project_root(setup_path: Path) -> Path:
    setup_path = setup_path.resolve()
    if setup_path.parent.name == "pipeline":
        return setup_path.parent.parent
    return setup_path.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple image-quality viewer for pipeline outputs.")
    parser.add_argument("--setup", "--config", dest="setup", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--root", type=Path, default=None)
    return parser.parse_args()


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _error_response(handler: BaseHTTPRequestHandler, message: str, status: HTTPStatus) -> None:
    _json_response(handler, {"error": message}, status=status)


def _dataset_choices(setup: ExperimentSetup) -> dict[str, dict[str, Any]]:
    raw = setup.variation_points.get("dataset_name", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    options = raw.get("options", [])
    if isinstance(options, list):
        for item in options:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            variants = item.get("values", [])
            if not isinstance(variants, list):
                variants = []
            out[name] = {
                "variants": [str(v) for v in variants],
                "root": str(item.get("root", f"input/{name}")),
            }
        return out
    for name, meta in raw.items():
        if name == "gui" or not isinstance(meta, dict):
            continue
        variants = meta.get("values", [])
        if not isinstance(variants, list):
            variants = []
        out[str(name)] = {"variants": [str(v) for v in variants], "root": str(meta.get("root", f"input/{name}"))}
    return out


def _resolve_allowed_roots(project_root: Path, dataset_choices: dict[str, dict[str, Any]]) -> list[Path]:
    roots = [project_root.resolve()]
    seen = {roots[0]}
    for meta in dataset_choices.values():
        raw_root = str(meta.get("root", "")).strip()
        if not raw_root:
            continue
        path = Path(raw_root)
        resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _default_dataset(dataset_map: dict[str, dict[str, Any]]) -> tuple[str, str]:
    if "real_data" in dataset_map:
        variants = dataset_map["real_data"].get("variants", [])
        if variants:
            preferred = "circled" if "circled" in variants else str(variants[0])
            return "real_data", preferred
    if not dataset_map:
        return "", ""
    first_dataset = sorted(dataset_map)[0]
    variants = dataset_map[first_dataset].get("variants", [])
    first_variant = str(variants[0]) if variants else ""
    return first_dataset, first_variant


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _scan_matching_run_dirs(project_root: Path, step_name: str, dataset_name: str, dataset_variant: str) -> list[Path]:
    pipeline_data = project_root / "pipeline_data"
    if not pipeline_data.exists():
        return []
    matches: list[tuple[float, Path]] = []
    for args_path in pipeline_data.rglob("args.json"):
        payload = _read_json(args_path)
        if not payload:
            continue
        process_step = payload.get("process_step", {})
        variation_points = payload.get("variation_points", {})
        if not isinstance(process_step, dict) or not isinstance(variation_points, dict):
            continue
        if str(process_step.get("name", "")).strip() != step_name:
            continue
        if str(variation_points.get("dataset_name", "")).strip() != dataset_name:
            continue
        if str(variation_points.get("dataset_variant", "")).strip() != dataset_variant:
            continue
        try:
            mtime = args_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        matches.append((mtime, args_path.parent.resolve()))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in matches]


def _image_size(path: Path, cache: dict[Path, list[int] | None]) -> list[int] | None:
    if path in cache:
        return cache[path]
    if Image is None or not path.exists():
        cache[path] = None
        return None
    try:
        with Image.open(path) as img:
            cache[path] = [int(img.width), int(img.height)]
    except Exception:  # noqa: BLE001
        cache[path] = None
    return cache[path]


def _encode_img_url(path: str) -> str:
    return f"/img?path={quote(path, safe='')}"


def _encode_overlay_url(
    image_path: str,
    image_name: str,
    annotation_xmls: list[str],
    *,
    max_dim: int | None = None,
) -> str:
    query: list[tuple[str, str]] = [("path", image_path), ("image_name", image_name)]
    for xml_path in annotation_xmls:
        query.append(("xml", xml_path))
    if max_dim is not None:
        query.append(("max_dim", str(max_dim)))
    return "/overlay?" + urlencode(query)


def _build_tooltip(
    original_size: list[int] | None,
    bbox: Any,
    final_size: Any,
    *,
    source_crop_size: list[float] | None = None,
    source_crop_box_clipped: Any = None,
) -> str:
    original_txt = f"{original_size[0]}x{original_size[1]}" if original_size and len(original_size) == 2 else "unknown"
    if isinstance(bbox, str):
        bbox_txt = bbox
    else:
        bbox_txt = json.dumps(bbox)
    if isinstance(final_size, list) and len(final_size) == 2:
        final_txt = f"{final_size[0]}x{final_size[1]}"
    elif isinstance(final_size, str):
        final_txt = final_size
    else:
        final_txt = "unknown"
    if source_crop_size and len(source_crop_size) == 2:
        source_crop_txt = f"{source_crop_size[0]}x{source_crop_size[1]}"
    else:
        source_crop_txt = "unknown"
    if isinstance(source_crop_box_clipped, str):
        clipped_txt = source_crop_box_clipped
    elif source_crop_box_clipped is not None:
        clipped_txt = json.dumps(source_crop_box_clipped)
    else:
        clipped_txt = "unknown"
    return (
        f"Original size: {original_txt}\n"
        f"Rivet bbox: {bbox_txt}\n"
        f"Source crop size: {source_crop_txt}\n"
        f"Source crop box clipped: {clipped_txt}\n"
        f"Final size: {final_txt}"
    )


def _extract_bbox(element: Any) -> tuple[float, float, float, float] | None:
    tag = str(getattr(element, "tag", "")).lower()
    attrib = getattr(element, "attrib", {})
    if tag == "mask":
        try:
            left = float(attrib.get("left", 0.0))
            top = float(attrib.get("top", 0.0))
            width = float(attrib.get("width", 0.0))
            height = float(attrib.get("height", 0.0))
        except (TypeError, ValueError):
            return None
        return left, top, left + width, top + height
    if tag in {"box", "rectangle"}:
        try:
            xtl = float(attrib.get("xtl", 0.0))
            ytl = float(attrib.get("ytl", 0.0))
            xbr = float(attrib.get("xbr", 0.0))
            ybr = float(attrib.get("ybr", 0.0))
        except (TypeError, ValueError):
            return None
        return xtl, ytl, xbr, ybr
    if tag == "polygon":
        raw_points = str(attrib.get("points", "")).strip()
        coords: list[tuple[float, float]] = []
        for pair in raw_points.split(";"):
            pair = pair.strip()
            if not pair or "," not in pair:
                continue
            xs, ys = pair.split(",", 1)
            try:
                coords.append((float(xs), float(ys)))
            except ValueError:
                continue
        if not coords:
            return None
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        return min(xs), min(ys), max(xs), max(ys)
    return None


def _annotation_xml_paths(run_dir: Path) -> list[Path]:
    return sorted(path.resolve() for path in run_dir.glob("*.xml") if path.is_file())


def _parse_target_size_value(raw: Any) -> tuple[int, int] | None:
    text = str(raw).strip().lower()
    if not text:
        return None
    try:
        if "x" in text:
            w_raw, h_raw = text.split("x", 1)
            width = int(w_raw)
            height = int(h_raw)
        else:
            width = int(text)
            height = width
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _step_extra_args_map(step_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = step_cfg.get("extra-args", step_cfg.get("extra_args", []))
    out: dict[str, Any] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("name", "")).strip()
        if not key:
            continue
        out[key] = item.get("value")
    return out


def _a20_simulation_defaults(setup: ExperimentSetup) -> dict[str, Any]:
    step_cfg = setup.process_steps.get("A20_cut_out", {})
    if not isinstance(step_cfg, dict):
        return {}
    extras = _step_extra_args_map(step_cfg)
    target_size = _parse_target_size_value(extras.get("target-size", extras.get("target_size", "")))
    return {
        "target_width": target_size[0] if target_size else 256,
        "target_height": target_size[1] if target_size else 256,
        "k_aoi_scale_factor": _parse_number_or_none(extras.get("k-aoi-scale-factor", extras.get("k_aoi_scale_factor"))),
        "aoi_fill_percentage": _parse_number_or_none(extras.get("aoi-fill-percentage", extras.get("aoi_fill_percentage"))),
        "label": str(step_cfg.get("label", "rivet")).strip() or "rivet",
    }


def _resolve_dataset_input_dir(
    project_root: Path,
    setup: ExperimentSetup,
    dataset_name: str,
    dataset_variant: str,
) -> Path | None:
    dataset_meta = _dataset_choices(setup).get(dataset_name, {})
    raw_root = str(dataset_meta.get("root", f"input/{dataset_name}")).strip()
    if not raw_root:
        return None
    root_path = Path(raw_root)
    dataset_root = root_path.resolve() if root_path.is_absolute() else (project_root / root_path).resolve()
    input_dir = dataset_root / dataset_variant if dataset_variant else dataset_root
    return input_dir if input_dir.exists() else None


def _find_original_image(input_dir: Path, image_name: str) -> Path | None:
    candidate = (input_dir / image_name).resolve()
    try:
        if candidate.is_file() and (candidate == input_dir or input_dir in candidate.parents):
            return candidate
    except OSError:
        return None

    basename = Path(image_name).name
    if not basename:
        return None
    matches = sorted(path.resolve() for path in input_dir.rglob(basename) if path.is_file())
    if len(matches) == 1:
        return matches[0]
    return None


@functools.lru_cache(maxsize=16)
def _load_annotation_index(xml_key: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    import xml.etree.ElementTree as ET

    index: dict[str, dict[str, Any]] = {}
    for raw_path in xml_key:
        xml_path = Path(raw_path)
        if not xml_path.exists():
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:  # noqa: BLE001
            continue
        for image_node in root.findall(".//image"):
            image_name = str(image_node.attrib.get("name", "")).strip()
            if not image_name:
                continue
            image_entry = index.setdefault(
                image_name,
                {
                    "image_name": image_name,
                    "width": int(float(image_node.attrib.get("width", "0") or 0)),
                    "height": int(float(image_node.attrib.get("height", "0") or 0)),
                    "segments": [],
                },
            )
            for shape in list(image_node):
                tag = str(shape.tag).lower()
                if tag not in {"mask", "box", "rectangle", "polygon"}:
                    continue
                bbox = _extract_bbox(shape)
                segment: dict[str, Any] = {
                    "type": tag,
                    "label": str(shape.attrib.get("label", "")).strip(),
                    "bbox": [round(v, 2) for v in bbox] if bbox else [],
                }
                if tag == "mask":
                    segment.update(
                        {
                            "left": int(float(shape.attrib.get("left", "0") or 0)),
                            "top": int(float(shape.attrib.get("top", "0") or 0)),
                            "width": int(float(shape.attrib.get("width", "0") or 0)),
                            "height": int(float(shape.attrib.get("height", "0") or 0)),
                            "rle": str(shape.attrib.get("rle", "")).strip(),
                        }
                    )
                elif tag in {"box", "rectangle"}:
                    segment["box"] = [round(v, 2) for v in bbox] if bbox else []
                elif tag == "polygon":
                    raw_points = str(shape.attrib.get("points", "")).strip()
                    points: list[list[float]] = []
                    for pair in raw_points.split(";"):
                        pair = pair.strip()
                        if not pair or "," not in pair:
                            continue
                        xs, ys = pair.split(",", 1)
                        try:
                            points.append([float(xs), float(ys)])
                        except ValueError:
                            continue
                    segment["points"] = points
                image_entry["segments"].append(segment)
    return index


def _decode_cvat_mask(segment: dict[str, Any]) -> Any | None:
    if Image is None:
        return None
    width = int(segment.get("width", 0) or 0)
    height = int(segment.get("height", 0) or 0)
    raw_rle = str(segment.get("rle", "")).strip()
    if width <= 0 or height <= 0 or not raw_rle:
        return None
    try:
        counts = [int(part.strip()) for part in raw_rle.split(",") if part.strip()]
    except ValueError:
        return None
    total = width * height
    if sum(counts) != total:
        return None
    data = bytearray(total)
    cursor = 0
    fill = 0
    for run in counts:
        if fill:
            data[cursor : cursor + run] = b"\xff" * run
        cursor += run
        fill = 1 - fill
    return Image.frombytes("L", (width, height), bytes(data))


def _draw_segments_on_image(base_image: Any, segments: list[dict[str, Any]]) -> Any:
    if Image is None or ImageDraw is None or ImageColor is None:
        return base_image
    canvas = base_image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    palette = ["#d94f3d", "#f0a202", "#2a9d8f", "#4d6cfa", "#b565d9", "#0081a7"]

    for idx, segment in enumerate(segments):
        color = ImageColor.getrgb(palette[idx % len(palette)])
        fill_rgba = color + (88,)
        outline_rgba = color + (230,)
        seg_type = str(segment.get("type", "")).lower()
        if seg_type == "mask":
            mask = _decode_cvat_mask(segment)
            left = int(segment.get("left", 0) or 0)
            top = int(segment.get("top", 0) or 0)
            if mask is not None:
                solid = Image.new("RGBA", mask.size, fill_rgba)
                overlay.paste(solid, (left, top), mask)
            bbox = segment.get("bbox", [])
            if isinstance(bbox, list) and len(bbox) == 4:
                draw.rectangle(bbox, outline=outline_rgba, width=3)
            continue
        if seg_type in {"box", "rectangle"}:
            box = segment.get("box", segment.get("bbox", []))
            if isinstance(box, list) and len(box) == 4:
                draw.rectangle(box, fill=fill_rgba, outline=outline_rgba, width=3)
            continue
        if seg_type == "polygon":
            points = segment.get("points", [])
            if isinstance(points, list) and len(points) >= 3:
                flat_points = [tuple(point) for point in points if isinstance(point, list) and len(point) == 2]
                if len(flat_points) >= 3:
                    draw.polygon(flat_points, fill=fill_rgba, outline=outline_rgba)
    return Image.alpha_composite(canvas, overlay)


def _compute_simulated_crop_box(
    bbox: list[float],
    *,
    target_width: int,
    target_height: int,
    aoi_fill_percentage: float,
) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None
    if target_width <= 0 or target_height <= 0:
        return None
    if not math.isfinite(aoi_fill_percentage):
        return None
    if aoi_fill_percentage <= 0.0 or aoi_fill_percentage >= 1.0:
        return None
    x0, y0, x1, y1 = bbox
    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    base_side = max(bw, bh)
    target_ar = float(target_width) / float(target_height)
    target_area = (base_side * base_side) / aoi_fill_percentage
    crop_w = math.sqrt(target_area * target_ar)
    crop_h = math.sqrt(target_area / target_ar)
    crop_ar = crop_w / crop_h
    if crop_ar < target_ar:
        crop_w = crop_h * target_ar
    elif crop_ar > target_ar:
        crop_h = crop_w / target_ar
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return (
        cx - crop_w / 2.0,
        cy - crop_h / 2.0,
        cx + crop_w / 2.0,
        cy + crop_h / 2.0,
    )


def _draw_simulated_crop_boxes(
    overlay: Any,
    *,
    segments: list[dict[str, Any]],
    image_size: tuple[int, int],
    target_width: int,
    target_height: int,
    aoi_fill_percentage: float,
    label: str,
) -> Any:
    if ImageDraw is None:
        return overlay
    draw = ImageDraw.Draw(overlay, "RGBA")
    wanted_label = label.strip().casefold()
    image_width, image_height = image_size
    for segment in segments:
        segment_label = str(segment.get("label", "")).strip().casefold()
        if wanted_label and segment_label != wanted_label:
            continue
        bbox = segment.get("bbox", [])
        if not isinstance(bbox, list):
            continue
        crop_box = _compute_simulated_crop_box(
            [float(v) for v in bbox],
            target_width=target_width,
            target_height=target_height,
            aoi_fill_percentage=aoi_fill_percentage,
        )
        if crop_box is None:
            continue
        x0, y0, x1, y1 = crop_box
        clipped = (
            max(0.0, min(float(image_width), x0)),
            max(0.0, min(float(image_height), y0)),
            max(0.0, min(float(image_width), x1)),
            max(0.0, min(float(image_height), y1)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        draw.rectangle(clipped, outline=(255, 214, 10, 255), width=4)
    return overlay


def _render_overlay_bytes(
    image_path: Path,
    image_name: str,
    annotation_xmls: list[Path],
    *,
    max_dim: int | None = None,
    simulate_crop: bool = False,
    target_width: int = 256,
    target_height: int = 256,
    aoi_fill_percentage: float = math.nan,
    label: str = "rivet",
) -> tuple[bytes, str]:
    if Image is None:
        ctype, _ = mimetypes.guess_type(image_path.name)
        return image_path.read_bytes(), ctype or "application/octet-stream"
    xml_key = tuple(sorted(str(path.resolve()) for path in annotation_xmls if path.exists()))
    annotation_index = _load_annotation_index(xml_key)
    segments = annotation_index.get(image_name, {}).get("segments", [])
    with Image.open(image_path) as img:
        rendered = _draw_segments_on_image(img, segments)
        if simulate_crop:
            rendered = _draw_simulated_crop_boxes(
                rendered,
                segments=segments,
                image_size=img.size,
                target_width=target_width,
                target_height=target_height,
                aoi_fill_percentage=aoi_fill_percentage,
                label=label,
            )
        if max_dim is not None and max_dim > 0:
            rendered.thumbnail((max_dim, max_dim))
        from io import BytesIO

        buf = BytesIO()
        rendered.convert("RGB").save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"


def _metrics_by_crop(metrics_csv: Path) -> dict[str, dict[str, str]]:
    if not metrics_csv.exists():
        return {}
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("crop_file", "")).strip(): row
            for row in csv.DictReader(handle)
            if str(row.get("crop_file", "")).strip()
        }


def _parse_number_or_none(value: Any) -> float | None:
    try:
        num = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _cutout_metrics_map(
    project_root: Path,
    dataset_name: str,
    dataset_variant: str,
    cutout_step_name: str,
) -> dict[str, dict[str, Any]]:
    cutout = _load_cutout_tab(project_root, dataset_name, dataset_variant)
    if cutout.get("status") != "ok":
        return {}
    metrics_map = _metrics_by_crop(Path(str(cutout.get("metrics_csv", "")).strip()))
    out: dict[str, dict[str, Any]] = {}
    run_dirs = _scan_matching_run_dirs(project_root, cutout_step_name, dataset_name, dataset_variant)
    summary_path = run_dirs[0] / "a20_summary.json" if run_dirs else None
    summary = _read_json(summary_path) if summary_path is not None and summary_path.exists() else None
    if isinstance(summary, dict):
        for image_row in summary.get("images", []):
            if not isinstance(image_row, dict):
                continue
            for crop_row in image_row.get("crops", []):
                if not isinstance(crop_row, dict):
                    continue
                crop_file = str(crop_row.get("crop_file", "")).strip()
                output_size = crop_row.get("output_size")
                metric_row = metrics_map.get(crop_file, {})
                source_crop_width = _parse_number_or_none(metric_row.get("source_crop_width"))
                source_crop_height = _parse_number_or_none(metric_row.get("source_crop_height"))
                target_width = _parse_number_or_none(metric_row.get("target_width"))
                target_height = _parse_number_or_none(metric_row.get("target_height"))
                image_size = None
                if source_crop_width is not None and source_crop_height is not None:
                    image_size = [source_crop_width, source_crop_height]
                elif isinstance(output_size, list) and len(output_size) == 2:
                    try:
                        image_size = [int(output_size[0]), int(output_size[1])]
                    except (TypeError, ValueError):
                        image_size = None
                out[crop_file] = {
                    "image_size": image_size,
                    "source_crop_size": [source_crop_width, source_crop_height]
                    if source_crop_width is not None and source_crop_height is not None
                    else None,
                    "source_crop_box_clipped": metric_row.get("source_crop_box_clipped") or "",
                    "target_size": [target_width, target_height]
                    if target_width is not None and target_height is not None
                    else None,
                }
    return out


def _quality_cutout_step_name(setup: ExperimentSetup, step_name: str) -> str | None:
    step_cfg = setup.process_steps.get(step_name, {})
    if not isinstance(step_cfg, dict):
        return None
    if not step_cfg.get("input_from_previous"):
        return None
    previous_step = str(step_cfg.get("previous_step", "")).strip()
    return previous_step or None


def _parse_float(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _format_metric_value(value: float, *, digits: int = 2, pct: bool = False) -> str:
    if not math.isfinite(value):
        return "nan"
    if pct:
        return f"{value * 100:.{digits}f}%"
    return f"{value:.{digits}f}"


def _json_safe_float(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(value)


def _quality_thresholds_from_args(args_path: Path) -> dict[str, float]:
    payload = _read_json(args_path) if args_path.exists() else None
    process_step = payload.get("process_step", {}) if isinstance(payload, dict) else {}
    raw = process_step.get("extra-args", process_step.get("extra_args", [])) if isinstance(process_step, dict) else []
    thresholds: dict[str, float] = {}
    if not isinstance(raw, list):
        return thresholds
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("name", "")).strip()
        value = _parse_float(item.get("value"))
        if not key or not math.isfinite(value):
            continue
        thresholds[key] = value
    return thresholds


def _parse_positive_int_or_default(raw: Any, default: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


QUALITY_AUTO_TUNE_SPECS: tuple[dict[str, str], ...] = (
    {"metric": "lap_var", "source": "lap_var", "direction": "low_is_bad", "check": "laplacian_pass", "threshold_key": "lapl_blur_threshold"},
    {"metric": "brisque", "source": "brisque", "direction": "high_is_bad", "check": "brisque_pass", "threshold_key": "brisque_threshold"},
    {"metric": "niqe", "source": "niqe", "direction": "high_is_bad", "check": "niqe_pass", "threshold_key": "niqe_threshold"},
    {"metric": "mean_darkness", "source": "mean", "direction": "low_is_bad", "check": "darkness_pass", "threshold_key": "darkness_threshold"},
    {"metric": "mean_brightness", "source": "mean", "direction": "high_is_bad", "check": "brightness_pass", "threshold_key": "brightness_threshold"},
    {"metric": "std", "source": "std", "direction": "low_is_bad", "check": "contrast_pass", "threshold_key": "contrast_threshold"},
    {"metric": "black_clip", "source": "black_clip", "direction": "high_is_bad", "check": "black_clip_pass", "threshold_key": "black_clip_threshold"},
    {"metric": "white_clip", "source": "white_clip", "direction": "high_is_bad", "check": "white_clip_pass", "threshold_key": "white_clip_threshold"},
)


def _auto_tune_quality_thresholds(
    rows: list[dict[str, Any]],
    enabled_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"thresholds": {}, "report": {"combine": "AND", "pass_rate": 1.0}, "violations": {"per_rule": {}, "overall": []}}

    import importlib.util
    import types

    for optional_module in ("numexpr", "bottleneck"):
        if optional_module not in sys.modules:
            stub = types.ModuleType(optional_module)
            stub.__version__ = "0.0"
            sys.modules[optional_module] = stub

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for quality auto tuning.") from exc

    tuner_path = Path(__file__).resolve().parents[3] / "pipeline" / "image_base_quality" / "img_quality_param_auto_tuning.py"
    spec = importlib.util.spec_from_file_location("iqviewer_quality_auto_tuning", tuner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load quality auto tuning module from {tuner_path}.")
    tuner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner_module)
    fit_quality_thresholds = tuner_module.fit_quality_thresholds

    active_specs = [
        spec
        for spec in QUALITY_AUTO_TUNE_SPECS
        if not isinstance(enabled_checks, dict) or enabled_checks.get(spec["check"], True) is not False
    ]
    if not active_specs:
        return {"thresholds": {}, "report": {"combine": "AND", "pass_rate": 1.0}, "violations": {"per_rule": {}, "overall": []}}

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        record: dict[str, Any] = {
            "imgpath": str(row.get("imgpath") or row.get("path") or row.get("name") or f"row_{idx}").strip() or f"row_{idx}",
            "final_status": str(row.get("final_status", "PASS")).strip().upper() or "PASS",
        }
        for spec in active_specs:
            record[spec["metric"]] = _parse_float(row.get(spec["source"]))
        records.append(record)

    if not records:
        raise ValueError("No valid quality rows were provided for auto tuning.")

    df = pd.DataFrame.from_records(records)
    metrics = [spec["metric"] for spec in active_specs if spec["metric"] in df.columns]
    if not metrics:
        return {"thresholds": {}, "report": {"combine": "AND", "pass_rate": 1.0}, "violations": {"per_rule": {}, "overall": []}}
    directions = {spec["metric"]: spec["direction"] for spec in active_specs if spec["metric"] in metrics}
    thresholds, report, violations = fit_quality_thresholds(
        df,
        metrics=metrics,
        directions=directions,
        combine="AND",
        use_labels=True,
        label_col="final_status",
        label_pass_value="PASS",
        strategy="youden",
        id_col="imgpath",
    )

    threshold_overrides: dict[str, float] = {}
    for spec in active_specs:
        threshold_value = thresholds.get(spec["metric"], {}).get("threshold")
        if not isinstance(threshold_value, (int, float)) or not math.isfinite(threshold_value):
            continue
        threshold_overrides[spec["threshold_key"]] = float(threshold_value)
    return {
        "thresholds": threshold_overrides,
        "report": report,
        "violations": violations,
    }


def _quality_fail_reasons(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    checks = [
        ("blur_fail", "blur"),
        ("lighting_fail", "lighting"),
        ("niqe_fail", "niqe"),
        ("laplacian_pass", "low laplacian"),
        ("brisque_pass", "high brisque"),
        ("darkness_pass", "too dark"),
        ("brightness_pass", "too bright"),
        ("contrast_pass", "low contrast"),
        ("black_clip_pass", "black clipping"),
        ("white_clip_pass", "white clipping"),
    ]
    for key, label in checks:
        if key.endswith("_pass"):
            if not _parse_bool(row.get(key, "")):
                reasons.append(label)
        elif _parse_bool(row.get(key, "")):
            reasons.append(label)
    error = str(row.get("error", "")).strip()
    if error:
        reasons.append(f"error: {error}")
    seen: set[str] = set()
    unique: list[str] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        unique.append(reason)
    return unique


def _load_quality_tab(
    csv_path: Path,
    step_name: str,
    args_path: Path | None,
    *,
    project_root: Path,
    setup: ExperimentSetup,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    fail_count = 0
    pass_count = 0
    error_count = 0
    thresholds = _quality_thresholds_from_args(args_path) if args_path is not None else {}
    cutout_metrics: dict[str, dict[str, Any]] = {}
    payload = _read_json(args_path) if args_path is not None and args_path.exists() else None
    variation_points = payload.get("variation_points", {}) if isinstance(payload, dict) else {}
    dataset_name = str(variation_points.get("dataset_name", "")).strip()
    dataset_variant = str(variation_points.get("dataset_variant", "")).strip()
    if dataset_name and dataset_variant:
        cutout_step_name = _quality_cutout_step_name(setup, step_name)
        if cutout_step_name:
            cutout_metrics = _cutout_metrics_map(project_root, dataset_name, dataset_variant, cutout_step_name)
    csv_path_display = str(csv_path.resolve())
    if payload and isinstance(payload, dict):
        resolved = payload.get("resolved", {})
        output_root = resolved.get("output") if isinstance(resolved, dict) else None
        if isinstance(output_root, str) and output_root.strip():
            try:
                output_dir = Path(output_root).resolve()
                csv_path_display = str(csv_path.resolve().relative_to(output_dir))
            except ValueError:
                csv_path_display = str(csv_path.resolve())
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for raw_row in reader:
                image_path = str(raw_row.get("imgpath", "")).strip()
                if not image_path:
                    continue
                crop_name = Path(image_path).name
                crop_metrics = cutout_metrics.get(crop_name, {})
                image_size = crop_metrics.get("image_size")
                status = str(raw_row.get("final_status", "")).strip().upper() or "UNKNOWN"
                if status == "FAIL":
                    fail_count += 1
                elif status == "PASS":
                    pass_count += 1
                if str(raw_row.get("error", "")).strip():
                    error_count += 1
                lap_var = _parse_float(raw_row.get("lap_var"))
                brisque = _parse_float(raw_row.get("brisque"))
                niqe = _parse_float(raw_row.get("niqe"))
                mean = _parse_float(raw_row.get("mean"))
                std = _parse_float(raw_row.get("std"))
                black_clip = _parse_float(raw_row.get("black_clip"))
                white_clip = _parse_float(raw_row.get("white_clip"))
                fail_reasons = _quality_fail_reasons(raw_row)
                metrics_summary = " | ".join(
                    [
                        f"lap { _format_metric_value(lap_var) }",
                        f"brisque { _format_metric_value(brisque) }",
                        f"mean { _format_metric_value(mean) }",
                        f"std { _format_metric_value(std) }",
                    ]
                )
                rows.append(
                    {
                        "name": Path(image_path).name,
                        "path": image_path,
                        "image_url": _encode_img_url(image_path),
                        "full_url": _encode_img_url(image_path),
                        "title": (
                            f"{Path(image_path).name}\n"
                            f"Status: {status}\n"
                            f"Path: {image_path}\n"
                            f"Source crop size: {crop_metrics.get('source_crop_size') or 'unknown'}\n"
                            f"Source crop box clipped: {crop_metrics.get('source_crop_box_clipped') or 'unknown'}\n"
                            f"Target size: {crop_metrics.get('target_size') or 'unknown'}"
                        ),
                        "status": status,
                        "status_class": status.lower(),
                        "summary": metrics_summary,
                        "image_size": image_size,
                        "source_crop_size": crop_metrics.get("source_crop_size"),
                        "source_crop_box_clipped": crop_metrics.get("source_crop_box_clipped") or "",
                        "target_size": crop_metrics.get("target_size"),
                        "fail_reasons": fail_reasons,
                        "error": str(raw_row.get("error", "")).strip(),
                        "metrics": {
                            "lap_var": _json_safe_float(lap_var),
                            "brisque": _json_safe_float(brisque),
                            "niqe": _json_safe_float(niqe),
                            "mean": _json_safe_float(mean),
                            "std": _json_safe_float(std),
                            "black_clip": _json_safe_float(black_clip),
                            "white_clip": _json_safe_float(white_clip),
                        },
                        "checks": {
                            "laplacian_pass": _parse_bool(raw_row.get("laplacian_pass")),
                            "brisque_pass": _parse_bool(raw_row.get("brisque_pass")),
                            "niqe_pass": _parse_bool(raw_row.get("niqe_pass")),
                            "darkness_pass": _parse_bool(raw_row.get("darkness_pass")),
                            "brightness_pass": _parse_bool(raw_row.get("brightness_pass")),
                            "contrast_pass": _parse_bool(raw_row.get("contrast_pass")),
                            "black_clip_pass": _parse_bool(raw_row.get("black_clip_pass")),
                            "white_clip_pass": _parse_bool(raw_row.get("white_clip_pass")),
                        },
                    }
                )
    except Exception as exc:  # noqa: BLE001
        return {
            "id": f"quality::{step_name}::{csv_path.as_posix()}",
            "label": step_name,
            "tab_type": "quality_metrics",
            "status": "error",
            "csv_path": str(csv_path.resolve()),
            "message": f"Could not read quality CSV: {exc}",
            "rows": [],
            "row_count": 0,
            "columns": [],
            "thresholds": thresholds,
        }

    rows.sort(key=lambda row: (row["status"] == "PASS", row["name"].lower()))
    return {
        "id": f"quality::{step_name}::{csv_path.as_posix()}",
        "label": step_name,
        "tab_type": "quality_metrics",
        "status": "ok",
        "csv_path": str(csv_path.resolve()),
        "csv_path_display": csv_path_display,
        "message": "",
        "rows": rows,
        "row_count": len(rows),
        "columns": columns,
        "thresholds": thresholds,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
    }


def _load_original_segments_tab(
    project_root: Path,
    dataset_name: str,
    dataset_variant: str,
    *,
    setup: ExperimentSetup,
) -> dict[str, Any]:
    cutout = _load_cutout_tab(project_root, dataset_name, dataset_variant)
    a05_run_dirs = _scan_matching_run_dirs(project_root, "A05_segment_rivets", dataset_name, dataset_variant)
    if not a05_run_dirs:
        return {
            "tab_type": "original_segments",
            "status": "missing",
            "message": f"No A05_segment_rivets output found for {dataset_name}/{dataset_variant}.",
            "images": [],
            "annotation_xmls": [],
            "run_dir": "",
        }

    annotation_xmls = _annotation_xml_paths(a05_run_dirs[0])
    if not annotation_xmls:
        return {
            "tab_type": "original_segments",
            "status": "missing",
            "message": f"No annotation XML files found in {a05_run_dirs[0]}",
            "images": [],
            "annotation_xmls": [],
            "run_dir": str(a05_run_dirs[0]),
        }

    xml_key = tuple(str(path) for path in annotation_xmls)
    annotation_index = _load_annotation_index(xml_key)
    images: list[dict[str, Any]] = []
    total_segments = 0
    source_mode = "a20_cut_out"
    message = ""

    if cutout.get("status") == "ok":
        seen_paths: set[str] = set()
        for cutout_item in cutout.get("cutouts", []):
            if not isinstance(cutout_item, dict):
                continue
            original_path = str(cutout_item.get("original_path", "")).strip()
            if not original_path or original_path in seen_paths:
                continue
            seen_paths.add(original_path)
            image_name = Path(original_path).name
            annotation_entry = annotation_index.get(image_name, {})
            segments = annotation_entry.get("segments", [])
            if not isinstance(segments, list) or not segments:
                continue
            total_segments += len(segments)
            images.append(
                {
                    "name": image_name,
                    "image_name": image_name,
                    "image_url": _encode_overlay_url(
                        original_path, image_name, [str(path) for path in annotation_xmls], max_dim=420
                    ),
                    "full_url": _encode_overlay_url(original_path, image_name, [str(path) for path in annotation_xmls]),
                    "title": f"{image_name}\nSegments: {len(segments)}\nOriginal: {original_path}",
                    "path": original_path,
                    "segment_count": len(segments),
                    "summary": f"{len(segments)} segments",
                }
            )
    else:
        source_mode = "dataset_input"
        input_dir = _resolve_dataset_input_dir(project_root, setup, dataset_name, dataset_variant)
        if input_dir is None:
            return {
                "tab_type": "original_segments",
                "status": "missing",
                "message": (
                    f"A20_cut_out is not available yet, and the dataset input folder for "
                    f"{dataset_name}/{dataset_variant} could not be resolved from setup."
                ),
                "images": [],
                "annotation_xmls": [str(path) for path in annotation_xmls],
                "run_dir": str(a05_run_dirs[0]),
            }

        message = "A20_cut_out is not available yet. Showing original images directly from dataset input."
        for image_name, annotation_entry in sorted(annotation_index.items()):
            segments = annotation_entry.get("segments", [])
            if not isinstance(segments, list) or not segments:
                continue
            original_path = _find_original_image(input_dir, image_name)
            if original_path is None:
                continue
            total_segments += len(segments)
            images.append(
                {
                    "name": Path(image_name).name,
                    "image_name": image_name,
                    "image_url": _encode_overlay_url(
                        str(original_path), image_name, [str(path) for path in annotation_xmls], max_dim=420
                    ),
                    "full_url": _encode_overlay_url(str(original_path), image_name, [str(path) for path in annotation_xmls]),
                    "title": f"{Path(image_name).name}\nSegments: {len(segments)}\nOriginal: {original_path}",
                    "path": str(original_path),
                    "segment_count": len(segments),
                    "summary": f"{len(segments)} segments",
                }
            )

    return {
        "tab_type": "original_segments",
        "status": "ok",
        "message": message,
        "run_dir": str(a05_run_dirs[0]),
        "annotation_xmls": [str(path) for path in annotation_xmls],
        "simulation_defaults": _a20_simulation_defaults(setup),
        "source_mode": source_mode,
        "total_images": len(images),
        "total_segments": total_segments,
        "images": images,
    }


def _load_cutout_tab(project_root: Path, dataset_name: str, dataset_variant: str) -> dict[str, Any]:
    run_dirs = _scan_matching_run_dirs(project_root, "A20_cut_out", dataset_name, dataset_variant)
    if not run_dirs:
        return {
            "tab_type": "cutout",
            "status": "missing",
            "message": f"No A20_cut_out output found for {dataset_name}/{dataset_variant}.",
            "cutouts": [],
            "missing_cutouts": [],
            "run_dir": "",
        }

    run_dir = run_dirs[0]
    summary_path = run_dir / "a20_summary.json"
    if not summary_path.exists():
        return {
            "tab_type": "cutout",
            "status": "missing",
            "message": f"Missing a20_summary.json in {run_dir}",
            "cutouts": [],
            "missing_cutouts": [],
            "run_dir": str(run_dir),
        }

    summary = _read_json(summary_path)
    if summary is None:
        return {
            "tab_type": "cutout",
            "status": "error",
            "message": f"Could not parse {summary_path}",
            "cutouts": [],
            "missing_cutouts": [],
            "run_dir": str(run_dir),
        }

    metrics_map = _metrics_by_crop(run_dir / "a20_cutout_metrics.csv")
    size_cache: dict[Path, list[int] | None] = {}
    cutouts: list[dict[str, Any]] = []
    missing_cutouts: list[dict[str, Any]] = []

    for image_row in summary.get("images", []):
        if not isinstance(image_row, dict):
            continue
        image_name = str(image_row.get("image_name", "")).strip()
        original_path_raw = str(image_row.get("resolved_image", "")).strip()
        original_path = Path(original_path_raw) if original_path_raw else None
        original_size = _image_size(original_path, size_cache) if original_path is not None else None
        crops = image_row.get("crops", [])
        if not isinstance(crops, list):
            crops = []
        if not crops:
            if original_path_raw:
                missing_cutouts.append(
                    {
                        "name": image_name or Path(original_path_raw).name,
                        "image_url": _encode_img_url(original_path_raw),
                        "full_url": _encode_img_url(original_path_raw),
                        "title": f"Original size: {original_size[0]}x{original_size[1]}" if original_size else "Original size: unknown",
                        "path": original_path_raw,
                    }
                )
            continue

        for crop_row in crops:
            if not isinstance(crop_row, dict):
                continue
            crop_file = str(crop_row.get("crop_file", "")).strip()
            if not crop_file:
                continue
            crop_path = run_dir / crop_file
            metric_row = metrics_map.get(crop_file, {})
            bbox = metric_row.get("rivet_bbox") or crop_row.get("rivet_bbox_xyxy") or []
            final_size = metric_row.get("target_size") or crop_row.get("output_size") or []
            source_crop_size = None
            source_crop_width = _parse_number_or_none(metric_row.get("source_crop_width"))
            source_crop_height = _parse_number_or_none(metric_row.get("source_crop_height"))
            if source_crop_width is not None and source_crop_height is not None:
                source_crop_size = [source_crop_width, source_crop_height]
            tooltip = _build_tooltip(
                original_size,
                bbox,
                final_size,
                source_crop_size=source_crop_size,
                source_crop_box_clipped=metric_row.get("source_crop_box_clipped"),
            )
            cutouts.append(
                {
                    "name": crop_file,
                    "image_url": _encode_img_url(str(crop_path)),
                    "full_url": _encode_img_url(str(crop_path)),
                    "title": tooltip,
                    "original_path": original_path_raw,
                    "cutout_path": str(crop_path),
                    "bbox": bbox,
                    "original_size": original_size,
                    "final_size": final_size,
                }
            )

    return {
        "tab_type": "cutout",
        "status": "ok",
        "message": "",
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "metrics_csv": str(run_dir / "a20_cutout_metrics.csv") if (run_dir / "a20_cutout_metrics.csv").exists() else "",
        "total_images": len(summary.get("images", [])) if isinstance(summary.get("images", []), list) else 0,
        "total_cutouts": len(cutouts),
        "missing_cutout_images": len(missing_cutouts),
        "cutouts": cutouts,
        "missing_cutouts": missing_cutouts,
    }


def _discover_quality_tabs(
    project_root: Path,
    dataset_name: str,
    dataset_variant: str,
    *,
    setup: ExperimentSetup,
) -> list[dict[str, Any]]:
    pipeline_data = project_root / "pipeline_data"
    if not pipeline_data.exists():
        return []
    tabs: list[dict[str, Any]] = []
    for csv_path in sorted(pipeline_data.rglob("image_quality_metrics.csv")):
        args_path = csv_path.parent / "args.json"
        payload = _read_json(args_path) if args_path.exists() else None
        step_name = csv_path.parent.name
        if payload:
            variation_points = payload.get("variation_points", {})
            process_step = payload.get("process_step", {})
            if isinstance(variation_points, dict):
                if str(variation_points.get("dataset_name", "")).strip() != dataset_name:
                    continue
                if str(variation_points.get("dataset_variant", "")).strip() != dataset_variant:
                    continue
            if isinstance(process_step, dict):
                step_name = str(process_step.get("name", step_name)).strip() or step_name
        else:
            parts = csv_path.parts
            if len(parts) < 4 or parts[-3] != dataset_name or parts[-2] != dataset_variant:
                continue
        tabs.append(
            _load_quality_tab(
                csv_path,
                step_name,
                args_path if args_path.exists() else None,
                project_root=project_root,
                setup=setup,
            )
        )
    return tabs


def _build_selection_payload(
    project_root: Path,
    dataset_name: str,
    dataset_variant: str,
    *,
    setup: ExperimentSetup,
) -> dict[str, Any]:
    original_segments = _load_original_segments_tab(project_root, dataset_name, dataset_variant, setup=setup)
    cutout = _load_cutout_tab(project_root, dataset_name, dataset_variant)
    quality_tabs = _discover_quality_tabs(project_root, dataset_name, dataset_variant, setup=setup)
    return {
        "dataset_name": dataset_name,
        "dataset_variant": dataset_variant,
        "tabs": [
            {
                "id": "original_segments",
                "label": "Originals + Masks",
                "tab_type": "original_segments",
                "status": original_segments.get("status", "missing"),
            },
            {"id": "cutout", "label": "A20_cut_out", "tab_type": "cutout", "status": cutout.get("status", "missing")},
        ]
        + quality_tabs,
        "original_segments": original_segments,
        "cutout": cutout,
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IQ Viewer</title>
  <style>
    :root {
      --bg: #eef4f8;
      --panel: #fbfdff;
      --line: #c9d9e6;
      --text: #14202b;
      --muted: #5b6b79;
      --accent: #0e7490;
      --accent-soft: #d9f1f6;
      --shadow: 0 10px 30px rgba(20, 44, 66, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(14, 116, 144, 0.14), transparent 28%),
        linear-gradient(180deg, #f7fbfe 0%, #edf4f8 100%);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }
    .wrap { max-width: 1480px; margin: 0 auto; padding: 24px; }
    .hero { margin-bottom: 18px; }
    .hero h1 { margin: 0; font-size: 28px; }
    .hero p { margin: 8px 0 0; color: var(--muted); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      align-items: end;
      margin-bottom: 18px;
    }
    label {
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    select, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 14px;
      background: #fff;
      color: var(--text);
    }
    button {
      background: var(--accent);
      color: #fffaf2;
      border-color: var(--accent);
      cursor: pointer;
      font-weight: 600;
    }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0;
    }
    .tab-btn {
      width: auto;
      background: #fff;
      color: var(--text);
      border-color: var(--line);
    }
    .tab-btn.active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--text);
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .stat {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }
    .stat .k { display: block; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .stat .v { display: block; margin-top: 5px; font-size: 20px; font-weight: 700; }
    .meta-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }
    .meta-row .k {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      white-space: nowrap;
    }
    .meta-row .v {
      font-size: 13px;
      color: var(--text);
      word-break: break-all;
    }
    .meta-inline {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
      font-size: 12px;
      color: var(--muted);
    }
    .meta-chip {
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .section-title {
      margin: 20px 0 12px;
      font-size: 18px;
    }
    .thumb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 14px;
    }
    .thumb-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      overflow: hidden;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .thumb-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(20, 44, 66, 0.12);
    }
    .thumb-img-wrap {
      aspect-ratio: 1 / 1;
      background: linear-gradient(135deg, #dcebf3, #f8fbfd);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    .thumb-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .thumb-img-error {
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 12px;
      text-align: center;
      font-size: 12px;
      color: var(--muted);
      background: linear-gradient(135deg, #eef4f8, #f8fbfd);
    }
    .thumb-img-wrap.error .thumb-img-error {
      display: flex;
    }
    .thumb-name {
      padding: 10px 10px 4px;
      font-size: 13px;
      text-align: center;
      word-break: break-word;
    }
    .thumb-meta {
      padding: 0 10px 10px;
      font-size: 12px;
      text-align: center;
      color: var(--muted);
    }
    .thumb-body {
      padding: 10px 12px 12px;
    }
    .thumb-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .thumb-title {
      font-size: 13px;
      font-weight: 600;
      word-break: break-word;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .status-btn {
      border: 0;
      cursor: pointer;
    }
    .status-btn.status-manual {
      box-shadow: inset 0 0 0 1px rgba(16, 24, 40, 0.18);
    }
    .status-pass {
      background: #dff5e8;
      color: #146c43;
    }
    .status-fail {
      background: #fee5e5;
      color: #b42318;
    }
    .status-unknown {
      background: #e8eef3;
      color: #51606d;
    }
    .metric-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 10px;
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }
    .metric-list strong {
      color: var(--text);
      font-weight: 600;
    }
    .metric-ok {
      color: #146c43;
    }
    .metric-bad {
      color: #b42318;
    }
    .metric-neutral {
      color: var(--muted);
    }
    .reason-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }
    .reason-pill {
      border-radius: 999px;
      background: #edf4f8;
      color: #375063;
      padding: 3px 8px;
      font-size: 11px;
    }
    .size-line {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    .filter-btn {
      width: auto;
      padding: 8px 12px;
      border-radius: 999px;
      background: #fff;
      color: var(--text);
      border-color: var(--line);
    }
    .filter-btn.active {
      background: var(--accent-soft);
      border-color: var(--accent);
    }
    .threshold-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .threshold-card {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }
    .threshold-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }
    .threshold-card label {
      margin-bottom: 0;
    }
    .threshold-toggle {
      width: auto;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .threshold-toggle input {
      width: auto;
    }
    .threshold-input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 14px;
      background: #fff;
      color: var(--text);
    }
    .threshold-input:disabled {
      background: #f1f5f8;
      color: #8a98a5;
      cursor: not-allowed;
    }
    .simulate-controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .simulate-card {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
    }
    .simulate-card label {
      margin-bottom: 6px;
    }
    .simulate-input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 14px;
      background: #fff;
      color: var(--text);
    }
    .simulate-row {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .simulate-btn {
      width: auto;
      padding: 8px 14px;
    }
    .threshold-section-title {
      margin: 10px 0 8px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .empty {
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 14px;
      color: var(--muted);
      background: rgba(255,255,255,0.7);
    }
    .path {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }
    .modal {
      position: fixed;
      inset: 0;
      background: rgba(20, 16, 12, 0.82);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }
    .modal.open { display: flex; }
    .modal-shell {
      width: min(92vw, 1400px);
      height: min(90vh, 980px);
      background: #16120f;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 18px;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .modal-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      color: #f6efe3;
      background: rgba(255,255,255,0.05);
    }
    .modal-title {
      font-size: 14px;
      word-break: break-all;
    }
    .modal-actions {
      display: flex;
      gap: 8px;
    }
    .modal-btn {
      width: auto;
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      color: #fff;
    }
    .viewport {
      position: relative;
      overflow: hidden;
      cursor: grab;
    }
    .viewport.dragging { cursor: grabbing; }
    .zoom-img {
      position: absolute;
      top: 50%;
      left: 50%;
      transform-origin: center center;
      user-select: none;
      -webkit-user-drag: none;
      max-width: none;
      max-height: none;
    }
    @media (max-width: 720px) {
      .wrap { padding: 16px; }
      .thumb-grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>IQ Viewer</h1>
      <p>Dataset and variant aware view of original images with segmentation overlays first, then cut-outs and quality tabs.</p>
    </div>
    <div class="panel">
      <div class="controls">
        <div>
          <label for="dataset_name">Dataset</label>
          <select id="dataset_name"></select>
        </div>
        <div>
          <label for="dataset_variant">Variant</label>
          <select id="dataset_variant"></select>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="refresh_btn" type="button">Refresh</button>
        </div>
      </div>
      <div id="tab_bar" class="tabs"></div>
      <div id="content"></div>
    </div>
  </div>

  <div id="modal" class="modal">
    <div class="modal-shell">
      <div class="modal-head">
        <div id="modal_title" class="modal-title"></div>
        <div class="modal-actions">
          <button class="modal-btn" id="zoom_reset" type="button">Reset</button>
          <button class="modal-btn" id="modal_close" type="button">Close</button>
        </div>
      </div>
      <div id="viewport" class="viewport">
        <img id="modal_img" class="zoom-img" alt="" />
      </div>
    </div>
  </div>

  <script>
    let OPTIONS = null;
    let VIEW = null;
    let ACTIVE_TAB = "original_segments";
    let QUALITY_FILTER = "all";
    let QUALITY_THRESHOLD_OVERRIDES = {};
    let QUALITY_MEASURE_OVERRIDES = {};
    let QUALITY_STATUS_OVERRIDES = {};
    let QUALITY_TUNING_REPORTS = {};
    let ORIGINAL_SEGMENT_SIM_OVERRIDES = {};
    let zoomState = { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 };

    function el(id) {
      return document.getElementById(id);
    }

    async function fetchJson(url) {
      const response = await fetch(url, { cache: "no-store" });
      return await response.json();
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `Request failed with status ${response.status}`);
      }
      return data;
    }

    function currentQuery() {
      const p = new URLSearchParams();
      p.set("dataset_name", el("dataset_name").value);
      p.set("dataset_variant", el("dataset_variant").value);
      p.set("_ts", Date.now().toString());
      return p.toString();
    }

    function fillSelect(node, values, selectedValue) {
      node.innerHTML = "";
      values.forEach(value => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        node.appendChild(option);
      });
      if (selectedValue && values.includes(selectedValue)) {
        node.value = selectedValue;
      } else if (values.length) {
        node.value = values[0];
      }
    }

    function syncVariantOptions() {
      const dataset = el("dataset_name").value;
      const datasetMeta = (OPTIONS.dataset_choices || {})[dataset] || {};
      fillSelect(el("dataset_variant"), datasetMeta.variants || [], el("dataset_variant").value);
    }

    function renderTabs() {
      const host = el("tab_bar");
      host.innerHTML = "";
      const tabs = (VIEW && VIEW.tabs) ? VIEW.tabs : [];
      tabs.forEach(tab => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "tab-btn" + (tab.id === ACTIVE_TAB ? " active" : "");
        button.textContent = tab.label;
        button.addEventListener("click", () => {
          ACTIVE_TAB = tab.id;
          renderTabs();
          renderContent();
        });
        host.appendChild(button);
      });
    }

    function createThumbCard(item) {
      const card = document.createElement("div");
      card.className = "thumb-card";
      card.title = item.title || "";
      card.innerHTML = `
        <div class="thumb-img-wrap">
          <img class="thumb-img" loading="lazy" src="${item.image_url}" alt="${item.name}" />
          <div class="thumb-img-error">Image preview could not be loaded.</div>
        </div>
        <div class="thumb-name">${item.name}</div>
        <div class="thumb-meta">${item.summary || ""}</div>
      `;
      const image = card.querySelector(".thumb-img");
      const imageWrap = card.querySelector(".thumb-img-wrap");
      if (image && imageWrap) {
        image.addEventListener("error", () => {
          imageWrap.classList.add("error");
        });
      }
      card.addEventListener("click", () => openModal(item.full_url, item.name));
      return card;
    }

    function currentOriginalSegmentParams(tab) {
      const defaults = (tab && tab.simulation_defaults) ? tab.simulation_defaults : {};
      return {
        target_width: Number(ORIGINAL_SEGMENT_SIM_OVERRIDES.target_width ?? defaults.target_width ?? 256),
        target_height: Number(ORIGINAL_SEGMENT_SIM_OVERRIDES.target_height ?? defaults.target_height ?? 256),
        k_aoi_scale_factor: Number(
          ORIGINAL_SEGMENT_SIM_OVERRIDES.k_aoi_scale_factor ?? defaults.k_aoi_scale_factor ?? 3.8
        ),
        aoi_fill_percentage: Number(
          ORIGINAL_SEGMENT_SIM_OVERRIDES.aoi_fill_percentage ?? defaults.aoi_fill_percentage ?? 0.95
        ),
        label: String(ORIGINAL_SEGMENT_SIM_OVERRIDES.label ?? defaults.label ?? "rivet"),
      };
    }

    function buildOriginalOverlayUrl(item, tab, maxDim) {
      const params = new URLSearchParams();
      params.set("path", item.path);
      params.set("image_name", item.image_name || item.name);
      (tab.annotation_xmls || []).forEach(xmlPath => params.append("xml", xmlPath));
      if (maxDim) params.set("max_dim", String(maxDim));
      const sim = currentOriginalSegmentParams(tab);
      params.set("simulate_crop", "true");
      params.set("target_width", String(sim.target_width));
      params.set("target_height", String(sim.target_height));
      params.set("aoi_fill_percentage", String(sim.aoi_fill_percentage));
      params.set("label", sim.label);
      return "/overlay?" + params.toString();
    }

    function createOriginalSegmentCard(item, tab) {
      const previewUrl = buildOriginalOverlayUrl(item, tab, 420);
      const fullUrl = buildOriginalOverlayUrl(item, tab, 0);
      return createThumbCard({
        ...item,
        image_url: previewUrl,
        full_url: fullUrl,
      });
    }

    function formatMetric(value, digits = 2, pct = false) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "nan";
      }
      const num = Number(value);
      return pct ? `${(num * 100).toFixed(digits)}%` : num.toFixed(digits);
    }

    function formatSize(value) {
      if (!Array.isArray(value) || value.length !== 2) return "unknown";
      const [w, h] = value;
      if (w === null || w === undefined || h === null || h === undefined) return "unknown";
      const width = Number(w);
      const height = Number(h);
      if (!Number.isFinite(width) || !Number.isFinite(height)) return "unknown";
      return `${width} x ${height} px`;
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") return "unknown";
      if (Array.isArray(value) || typeof value === "object") return JSON.stringify(value);
      return String(value);
    }

    function createQualityCard(item, tab) {
      const card = document.createElement("div");
      card.className = "thumb-card";
      card.title = item.title || "";
      const labelStatus = qualityLabelStatus(item, tab);
      const statusClass = labelStatus === "PASS" ? "status-pass" : labelStatus === "FAIL" ? "status-fail" : "status-unknown";
      const reasons = (item.computed.fail_reasons || []).map(reason => `<span class="reason-pill">${reason}</span>`).join("");
      const checks = item.computed.checks || {};
      const computedClass = item.computed.status === "PASS" ? "status-pass" : item.computed.status === "FAIL" ? "status-fail" : "status-unknown";
      const showComputedStatus = labelStatus !== item.computed.status;
      const metricClass = (passFlag) => {
        if (passFlag === null || passFlag === undefined) return "metric-neutral";
        return passFlag ? "metric-ok" : "metric-bad";
      };
      const meanClass = (checks.darkness_pass === false || checks.brightness_pass === false) ? "metric-bad" : "metric-ok";
      const sourceCropSize = formatSize(item.source_crop_size || item.image_size);
      card.innerHTML = `
        <div class="thumb-img-wrap">
          <img class="thumb-img" loading="lazy" src="${item.image_url}" alt="${item.name}" />
          <div class="thumb-img-error">Image preview could not be loaded.</div>
        </div>
        <div class="thumb-body">
          <div class="thumb-header">
            <div class="thumb-title">${item.name}</div>
            <div style="display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end;">
              <button type="button" class="status-pill status-btn ${statusClass}${showComputedStatus ? " status-manual" : ""}">${labelStatus}</button>
              ${showComputedStatus ? `<span class="status-pill ${computedClass}">Sim ${item.computed.status}</span>` : ""}
            </div>
          </div>
          <div class="metric-list">
            <div class="${metricClass(checks.laplacian_pass)}"><strong>Lap</strong> ${formatMetric(item.metrics.lap_var)}</div>
            <div class="${metricClass(checks.brisque_pass)}"><strong>BRISQUE</strong> ${formatMetric(item.metrics.brisque)}</div>
            <div class="${metricClass(checks.niqe_pass)}"><strong>NIQE</strong> ${formatMetric(item.metrics.niqe)}</div>
            <div class="${meanClass}"><strong>Mean</strong> ${formatMetric(item.metrics.mean)}</div>
            <div class="${metricClass(checks.contrast_pass)}"><strong>Std</strong> ${formatMetric(item.metrics.std)}</div>
            <div class="${metricClass(checks.black_clip_pass)}"><strong>Black</strong> ${formatMetric(item.metrics.black_clip, 2, true)}</div>
            <div class="${metricClass(checks.white_clip_pass)}"><strong>White</strong> ${formatMetric(item.metrics.white_clip, 2, true)}</div>
          </div>
          ${reasons ? `<div class="reason-list">${reasons}</div>` : ""}
          <div class="size-line"><strong>Source crop size</strong> ${sourceCropSize}</div>
        </div>
      `;
      const image = card.querySelector(".thumb-img");
      const imageWrap = card.querySelector(".thumb-img-wrap");
      if (image && imageWrap) {
        image.addEventListener("error", () => {
          imageWrap.classList.add("error");
        });
      }
      const statusBtn = card.querySelector(".status-btn");
      if (statusBtn) {
        statusBtn.addEventListener("click", event => {
          event.stopPropagation();
          const key = item.path || item.name;
          const overrides = currentQualityStatusOverrides(tab);
          const currentStatus = qualityLabelStatus(item, tab);
          const nextStatus = currentStatus === "PASS" ? "FAIL" : "PASS";
          if (nextStatus === item.computed.status) {
            delete overrides[key];
          } else {
            overrides[key] = nextStatus;
          }
          renderContent();
        });
      }
      card.addEventListener("click", () => openModal(item.full_url, item.name));
      return card;
    }

    function currentQualityThresholds(tab) {
      if (!tab) return {};
      if (!QUALITY_THRESHOLD_OVERRIDES[tab.id]) {
        QUALITY_THRESHOLD_OVERRIDES[tab.id] = { ...(tab.thresholds || {}) };
      }
      return QUALITY_THRESHOLD_OVERRIDES[tab.id];
    }

    function currentQualityMeasures(tab) {
      if (!tab) return {};
      if (!QUALITY_MEASURE_OVERRIDES[tab.id]) {
        QUALITY_MEASURE_OVERRIDES[tab.id] = {
          laplacian_pass: true,
          brisque_pass: true,
          niqe_pass: true,
          darkness_pass: true,
          brightness_pass: true,
          contrast_pass: true,
          black_clip_pass: true,
          white_clip_pass: true,
        };
      }
      return QUALITY_MEASURE_OVERRIDES[tab.id];
    }

    function currentQualityStatusOverrides(tab) {
      if (!tab) return {};
      if (!QUALITY_STATUS_OVERRIDES[tab.id]) {
        QUALITY_STATUS_OVERRIDES[tab.id] = {};
      }
      return QUALITY_STATUS_OVERRIDES[tab.id];
    }

    function setThresholdInputState(currentTab) {
      const enabledChecks = currentQualityMeasures(currentTab);
      const mapping = [
        ["chk_lap", "thr_lap", "laplacian_pass"],
        ["chk_brisque", "thr_brisque", "brisque_pass"],
        ["chk_niqe", "thr_niqe", "niqe_pass"],
        ["chk_dark", "thr_dark", "darkness_pass"],
        ["chk_bright", "thr_bright", "brightness_pass"],
        ["chk_contrast", "thr_contrast", "contrast_pass"],
        ["chk_black", "thr_black", "black_clip_pass"],
        ["chk_white", "thr_white", "white_clip_pass"],
      ];
      mapping.forEach(([checkId, inputId, key]) => {
        const check = el(checkId);
        const input = el(inputId);
        const enabled = enabledChecks[key] !== false;
        if (check) {
          check.checked = enabled;
        }
        if (input) {
          input.disabled = !enabled;
        }
      });
    }

    function parseThresholdValue(rawValue, fallbackValue) {
      const parsed = Number(rawValue);
      return Number.isFinite(parsed) ? parsed : fallbackValue;
    }

    function classifyQualityRow(row, thresholds, enabledChecks) {
      const metrics = row.metrics || {};
      const checks = {
        laplacian_pass: Number(metrics.lap_var) >= Number(thresholds.lapl_blur_threshold),
        brisque_pass: Number(metrics.brisque) <= Number(thresholds.brisque_threshold),
        niqe_pass: metrics.niqe === null || metrics.niqe === undefined || !Number.isFinite(Number(metrics.niqe))
          ? true
          : Number(metrics.niqe) <= Number(thresholds.niqe_threshold),
        darkness_pass: Number(metrics.mean) >= Number(thresholds.darkness_threshold),
        brightness_pass: Number(metrics.mean) <= Number(thresholds.brightness_threshold),
        contrast_pass: Number(metrics.std) >= Number(thresholds.contrast_threshold),
        black_clip_pass: Number(metrics.black_clip) <= Number(thresholds.black_clip_threshold),
        white_clip_pass: Number(metrics.white_clip) <= Number(thresholds.white_clip_threshold),
      };
      const activeFailures = Object.entries(checks)
        .filter(([key]) => enabledChecks[key] !== false)
        .filter(([, passed]) => !passed)
        .map(([key]) => key);
      const final_fail = activeFailures.length > 0;
      const failReasons = [];
      if (enabledChecks.laplacian_pass !== false && !checks.laplacian_pass) failReasons.push("low laplacian");
      if (enabledChecks.brisque_pass !== false && !checks.brisque_pass) failReasons.push("high brisque");
      if (enabledChecks.niqe_pass !== false && !checks.niqe_pass) failReasons.push("niqe");
      if (enabledChecks.darkness_pass !== false && !checks.darkness_pass) failReasons.push("too dark");
      if (enabledChecks.brightness_pass !== false && !checks.brightness_pass) failReasons.push("too bright");
      if (enabledChecks.contrast_pass !== false && !checks.contrast_pass) failReasons.push("low contrast");
      if (enabledChecks.black_clip_pass !== false && !checks.black_clip_pass) failReasons.push("black clipping");
      if (enabledChecks.white_clip_pass !== false && !checks.white_clip_pass) failReasons.push("white clipping");
      if (row.error) failReasons.push(`error: ${row.error}`);
      return {
        status: final_fail ? "FAIL" : "PASS",
        checks,
        fail_reasons: [...new Set(failReasons)],
      };
    }

    function qualityLabelStatus(row, tab) {
      const key = row.path || row.name;
      const overrides = currentQualityStatusOverrides(tab);
      return overrides[key] || row.computed.status;
    }

    function simulatedQualityRows(tab) {
      const thresholds = currentQualityThresholds(tab);
      const enabledChecks = currentQualityMeasures(tab);
      const overrides = currentQualityStatusOverrides(tab);
      return (tab.rows || []).map(row => ({
        ...row,
        computed: classifyQualityRow(row, thresholds, enabledChecks),
        label_status: overrides[row.path || row.name] || null,
      }));
    }

    function syncQualityControls(currentTab) {
      const thresholds = currentQualityThresholds(currentTab);
      const enabledChecks = currentQualityMeasures(currentTab);
      thresholds.lapl_blur_threshold = parseThresholdValue(el("thr_lap")?.value, thresholds.lapl_blur_threshold);
      thresholds.brisque_threshold = parseThresholdValue(el("thr_brisque")?.value, thresholds.brisque_threshold);
      thresholds.niqe_threshold = parseThresholdValue(el("thr_niqe")?.value, thresholds.niqe_threshold);
      thresholds.darkness_threshold = parseThresholdValue(el("thr_dark")?.value, thresholds.darkness_threshold);
      thresholds.brightness_threshold = parseThresholdValue(el("thr_bright")?.value, thresholds.brightness_threshold);
      thresholds.contrast_threshold = parseThresholdValue(el("thr_contrast")?.value, thresholds.contrast_threshold);
      thresholds.black_clip_threshold = parseThresholdValue(el("thr_black")?.value, thresholds.black_clip_threshold);
      thresholds.white_clip_threshold = parseThresholdValue(el("thr_white")?.value, thresholds.white_clip_threshold);
      enabledChecks.laplacian_pass = !!el("chk_lap")?.checked;
      enabledChecks.brisque_pass = !!el("chk_brisque")?.checked;
      enabledChecks.niqe_pass = !!el("chk_niqe")?.checked;
      enabledChecks.darkness_pass = !!el("chk_dark")?.checked;
      enabledChecks.brightness_pass = !!el("chk_bright")?.checked;
      enabledChecks.contrast_pass = !!el("chk_contrast")?.checked;
      enabledChecks.black_clip_pass = !!el("chk_black")?.checked;
      enabledChecks.white_clip_pass = !!el("chk_white")?.checked;
    }

    function renderOriginalSegmentsTab(tab) {
      if (!tab || tab.status !== "ok") {
        return `<div class="empty">${tab && tab.message ? tab.message : "No original segment data found."}</div>`;
      }

      const sourceLabel = tab.source_mode === "dataset_input" ? "Dataset Input" : "A20 Summary";
      const sim = currentOriginalSegmentParams(tab);
      const stats = `
        ${tab.message ? `<div class="empty" style="margin-bottom:14px;">${tab.message}</div>` : ""}
        <div class="meta">
          <div class="stat"><span class="k">A05 Run Dir</span><span class="v path">${tab.run_dir}</span></div>
          <div class="stat"><span class="k">Image Source</span><span class="v">${sourceLabel}</span></div>
          <div class="stat"><span class="k">Images With Segments</span><span class="v">${tab.total_images}</span></div>
          <div class="stat"><span class="k">Segments Found</span><span class="v">${tab.total_segments}</span></div>
        </div>
      `;

      return `
        ${stats}
        <div class="section-title">A20 Cut Simulation</div>
        <div class="simulate-controls">
          <div class="simulate-card">
            <label for="sim_target_width">Target Width</label>
            <input id="sim_target_width" class="simulate-input" type="number" min="1" step="1" value="${sim.target_width}" />
          </div>
          <div class="simulate-card">
            <label for="sim_target_height">Target Height</label>
            <input id="sim_target_height" class="simulate-input" type="number" min="1" step="1" value="${sim.target_height}" />
          </div>
          <div class="simulate-card">
            <label for="sim_aoi_fill_percentage">AOI Fill Percentage</label>
            <input id="sim_aoi_fill_percentage" class="simulate-input" type="number" min="0.0001" max="0.9999" step="any" value="${sim.aoi_fill_percentage}" />
          </div>
          <div class="simulate-card">
            <label for="sim_label">Label</label>
            <input id="sim_label" class="simulate-input" type="text" value="${sim.label}" />
          </div>
        </div>
        <div class="simulate-row">
          <button type="button" id="simulate_original_btn" class="simulate-btn">Simulate</button>
        </div>
        <div>
          <div class="section-title">Original Images With Segmentation Overlay And Simulated Cut Box</div>
          <div id="original_segments_grid" class="thumb-grid"></div>
        </div>
      `;
    }

    function renderCutoutTab(cutout) {
      if (!cutout || cutout.status !== "ok") {
        return `<div class="empty">${cutout && cutout.message ? cutout.message : "No cutout data found."}</div>`;
      }

      const stats = `
        <div class="meta">
          <div class="stat"><span class="k">Run Dir</span><span class="v path">${cutout.run_dir}</span></div>
          <div class="stat"><span class="k">Source Images</span><span class="v">${cutout.total_images}</span></div>
          <div class="stat"><span class="k">Cut-Outs</span><span class="v">${cutout.total_cutouts}</span></div>
          <div class="stat"><span class="k">No Cut-Outs</span><span class="v">${cutout.missing_cutout_images}</span></div>
        </div>
      `;

      return `
        ${stats}
        <div>
          <div class="section-title">Cut-Out Images</div>
          <div id="cutout_grid" class="thumb-grid"></div>
        </div>
        <div>
          <div class="section-title">Images Without Cut-Outs</div>
          <div id="missing_grid" class="thumb-grid"></div>
        </div>
      `;
    }

    function renderQualityTab(tab) {
      if (!tab || tab.status !== "ok") {
        return `<div class="empty">${tab && tab.message ? tab.message : "No quality data found."}</div>`;
      }
      const thresholds = currentQualityThresholds(tab);
      const enabledChecks = currentQualityMeasures(tab);
      const rows = simulatedQualityRows(tab);
      const passCount = rows.filter(row => qualityLabelStatus(row, tab) === "PASS").length;
      const failCount = rows.filter(row => qualityLabelStatus(row, tab) === "FAIL").length;
      const thresholdBits = [
        thresholds.lapl_blur_threshold !== undefined ? `lap >= ${formatMetric(thresholds.lapl_blur_threshold)}` : "",
        thresholds.brisque_threshold !== undefined ? `brisque <= ${formatMetric(thresholds.brisque_threshold)}` : "",
        thresholds.niqe_threshold !== undefined ? `niqe <= ${formatMetric(thresholds.niqe_threshold)}` : "",
        thresholds.darkness_threshold !== undefined ? `dark >= ${formatMetric(thresholds.darkness_threshold)}` : "",
        thresholds.brightness_threshold !== undefined ? `bright <= ${formatMetric(thresholds.brightness_threshold)}` : "",
        thresholds.contrast_threshold !== undefined ? `contrast >= ${formatMetric(thresholds.contrast_threshold)}` : "",
        thresholds.black_clip_threshold !== undefined ? `black <= ${formatMetric(thresholds.black_clip_threshold, 2, true)}` : "",
        thresholds.white_clip_threshold !== undefined ? `white <= ${formatMetric(thresholds.white_clip_threshold, 2, true)}` : "",
      ].filter(Boolean).join(" | ");
      const tuningReport = QUALITY_TUNING_REPORTS[tab.id];
      return `
        <div class="meta-inline">
          <span class="meta-chip">Rows ${rows.length}</span>
          <span class="meta-chip">PASS ${passCount}</span>
          <span class="meta-chip">FAIL ${failCount}</span>
        </div>
        <div class="threshold-section-title">Blur Thresholds</div>
        <div class="threshold-grid">
          <div class="threshold-card"><div class="threshold-head"><label for="thr_lap">Laplacian Min</label><label class="threshold-toggle"><input id="chk_lap" type="checkbox" ${enabledChecks.laplacian_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_lap" class="threshold-input" type="number" step="any" value="${thresholds.lapl_blur_threshold ?? ""}" /></div>
          <div class="threshold-card"><div class="threshold-head"><label for="thr_brisque">BRISQUE Max</label><label class="threshold-toggle"><input id="chk_brisque" type="checkbox" ${enabledChecks.brisque_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_brisque" class="threshold-input" type="number" step="any" value="${thresholds.brisque_threshold ?? ""}" /></div>
          <div class="threshold-card"><div class="threshold-head"><label for="thr_niqe">NIQE Max</label><label class="threshold-toggle"><input id="chk_niqe" type="checkbox" ${enabledChecks.niqe_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_niqe" class="threshold-input" type="number" step="any" value="${thresholds.niqe_threshold ?? ""}" /></div>
        </div>
        <div class="threshold-section-title">Lighting Thresholds</div>
        <div class="threshold-grid">
          <div class="threshold-card"><div class="threshold-head"><label for="thr_dark">Darkness Min</label><label class="threshold-toggle"><input id="chk_dark" type="checkbox" ${enabledChecks.darkness_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_dark" class="threshold-input" type="number" step="any" value="${thresholds.darkness_threshold ?? ""}" /></div>
          <div class="threshold-card"><div class="threshold-head"><label for="thr_bright">Brightness Max</label><label class="threshold-toggle"><input id="chk_bright" type="checkbox" ${enabledChecks.brightness_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_bright" class="threshold-input" type="number" step="any" value="${thresholds.brightness_threshold ?? ""}" /></div>
          <div class="threshold-card"><div class="threshold-head"><label for="thr_contrast">Contrast Min</label><label class="threshold-toggle"><input id="chk_contrast" type="checkbox" ${enabledChecks.contrast_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_contrast" class="threshold-input" type="number" step="any" value="${thresholds.contrast_threshold ?? ""}" /></div>
          <div class="threshold-card"><div class="threshold-head"><label for="thr_black">Black Clip Max</label><label class="threshold-toggle"><input id="chk_black" type="checkbox" ${enabledChecks.black_clip_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_black" class="threshold-input" type="number" step="any" value="${thresholds.black_clip_threshold ?? ""}" /></div>
          <div class="threshold-card"><div class="threshold-head"><label for="thr_white">White Clip Max</label><label class="threshold-toggle"><input id="chk_white" type="checkbox" ${enabledChecks.white_clip_pass !== false ? "checked" : ""} /> Use</label></div><input id="thr_white" class="threshold-input" type="number" step="any" value="${thresholds.white_clip_threshold ?? ""}" /></div>
        </div>
        <div class="simulate-row">
          <button type="button" id="simulate_btn" class="simulate-btn">Simulate</button>
          <button type="button" id="auto_tune_btn" class="simulate-btn">Auto Tune</button>
        </div>
        <div class="toolbar">
          <button type="button" class="filter-btn" data-quality-filter="all">All</button>
          <button type="button" class="filter-btn" data-quality-filter="fail">Only FAIL</button>
          <button type="button" class="filter-btn" data-quality-filter="pass">Only PASS</button>
        </div>
        ${tuningReport ? `<div class="empty" style="margin-bottom:14px;">Auto tune: pass rate ${formatMetric(tuningReport.pass_rate)}${tuningReport.accuracy !== undefined ? ` | accuracy ${formatMetric(tuningReport.accuracy)}` : ""}</div>` : ""}
        ${thresholdBits ? `<div class="empty" style="margin-bottom:14px;">Thresholds: ${thresholdBits}</div>` : ""}
        <div id="quality_grid" class="thumb-grid"></div>
      `;
    }

    function renderContent() {
      const host = el("content");
      const tabs = (VIEW && VIEW.tabs) ? VIEW.tabs : [];
      const currentTab = tabs.find(tab => tab.id === ACTIVE_TAB) || tabs[0] || null;
      if (!currentTab) {
        host.innerHTML = '<div class="empty">No data available for this selection.</div>';
        return;
      }
      if (currentTab.id === "original_segments") {
        host.innerHTML = renderOriginalSegmentsTab(VIEW.original_segments);
        const grid = el("original_segments_grid");
        const images = (VIEW.original_segments && VIEW.original_segments.images) ? VIEW.original_segments.images : [];
        const simulateBtn = el("simulate_original_btn");
        if (simulateBtn) {
          simulateBtn.addEventListener("click", () => {
            ORIGINAL_SEGMENT_SIM_OVERRIDES = {
              target_width: Number(el("sim_target_width")?.value || 256),
              target_height: Number(el("sim_target_height")?.value || 256),
              aoi_fill_percentage: Number(el("sim_aoi_fill_percentage")?.value || 0.95),
              label: String(el("sim_label")?.value || "rivet"),
            };
            renderContent();
          });
        }
        if (grid) {
          if (images.length) {
            images.forEach(item => grid.appendChild(createOriginalSegmentCard(item, VIEW.original_segments)));
          } else {
            grid.innerHTML = '<div class="empty">No segmented source images found.</div>';
          }
        }
        return;
      }
      if (currentTab.id === "cutout") {
        host.innerHTML = renderCutoutTab(VIEW.cutout);
        const cutoutGrid = el("cutout_grid");
        const missingGrid = el("missing_grid");
        const cutouts = (VIEW.cutout && VIEW.cutout.cutouts) ? VIEW.cutout.cutouts : [];
        const missing = (VIEW.cutout && VIEW.cutout.missing_cutouts) ? VIEW.cutout.missing_cutouts : [];
        if (cutoutGrid) {
          if (cutouts.length) {
            cutouts.forEach(item => cutoutGrid.appendChild(createThumbCard(item)));
          } else {
            cutoutGrid.innerHTML = '<div class="empty">No cut-out images found.</div>';
          }
        }
        if (missingGrid) {
          if (missing.length) {
            missing.forEach(item => missingGrid.appendChild(createThumbCard(item)));
          } else {
            missingGrid.innerHTML = '<div class="empty">Every source image emitted at least one cut-out.</div>';
          }
        }
        return;
      }
      host.innerHTML = renderQualityTab(currentTab);
      const qualityGrid = el("quality_grid");
      if (qualityGrid) {
        const rows = simulatedQualityRows(currentTab);
        const filtered = rows.filter(item => {
          if (QUALITY_FILTER === "fail") return qualityLabelStatus(item, currentTab) === "FAIL";
          if (QUALITY_FILTER === "pass") return qualityLabelStatus(item, currentTab) === "PASS";
          return true;
        });
        if (filtered.length) {
          filtered.forEach(item => qualityGrid.appendChild(createQualityCard(item, currentTab)));
        } else {
          qualityGrid.innerHTML = '<div class="empty">No images match the current filter.</div>';
        }
        document.querySelectorAll("[data-quality-filter]").forEach(button => {
          const isActive = button.getAttribute("data-quality-filter") === QUALITY_FILTER;
          button.classList.toggle("active", isActive);
          button.addEventListener("click", () => {
            QUALITY_FILTER = button.getAttribute("data-quality-filter") || "all";
            renderContent();
          });
        });
        setThresholdInputState(currentTab);
        [
          "chk_lap",
          "chk_brisque",
          "chk_niqe",
          "chk_dark",
          "chk_bright",
          "chk_contrast",
          "chk_black",
          "chk_white",
        ].forEach(checkId => {
          const checkbox = el(checkId);
          if (!checkbox) return;
          checkbox.addEventListener("change", () => {
            const enabledChecks = currentQualityMeasures(currentTab);
            enabledChecks.laplacian_pass = !!el("chk_lap")?.checked;
            enabledChecks.brisque_pass = !!el("chk_brisque")?.checked;
            enabledChecks.niqe_pass = !!el("chk_niqe")?.checked;
            enabledChecks.darkness_pass = !!el("chk_dark")?.checked;
            enabledChecks.brightness_pass = !!el("chk_bright")?.checked;
            enabledChecks.contrast_pass = !!el("chk_contrast")?.checked;
            enabledChecks.black_clip_pass = !!el("chk_black")?.checked;
            enabledChecks.white_clip_pass = !!el("chk_white")?.checked;
            setThresholdInputState(currentTab);
          });
        });
        const simulateBtn = el("simulate_btn");
        if (simulateBtn) {
          simulateBtn.addEventListener("click", () => {
            syncQualityControls(currentTab);
            renderContent();
          });
        }
        const autoTuneBtn = el("auto_tune_btn");
        if (autoTuneBtn) {
          autoTuneBtn.addEventListener("click", async () => {
            syncQualityControls(currentTab);
            const originalLabel = autoTuneBtn.textContent;
            autoTuneBtn.disabled = true;
            autoTuneBtn.textContent = "Tuning...";
            try {
              const rowsForTuning = simulatedQualityRows(currentTab).map(row => ({
                imgpath: row.path || row.name,
                final_status: qualityLabelStatus(row, currentTab),
                ...(row.metrics || {}),
              }));
              const payload = await postJson("/api/quality/auto_tune", {
                rows: rowsForTuning,
                enabled_checks: currentQualityMeasures(currentTab),
              });
              Object.assign(currentQualityThresholds(currentTab), payload.thresholds || {});
              QUALITY_TUNING_REPORTS[currentTab.id] = payload.report || null;
              renderContent();
            } catch (err) {
              window.alert(`Auto tuning failed: ${err}`);
            } finally {
              autoTuneBtn.disabled = false;
              autoTuneBtn.textContent = originalLabel;
            }
          });
        }
      }
    }

    function applyTransform() {
      const img = el("modal_img");
      img.style.transform = `translate(calc(-50% + ${zoomState.x}px), calc(-50% + ${zoomState.y}px)) scale(${zoomState.scale})`;
    }

    function resetZoom() {
      zoomState = { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 };
      applyTransform();
    }

    function openModal(src, title) {
      el("modal_img").src = src;
      el("modal_title").textContent = title;
      el("modal").classList.add("open");
      resetZoom();
    }

    function closeModal() {
      el("modal").classList.remove("open");
      el("modal_img").src = "";
    }

    async function refreshView() {
      VIEW = await fetchJson("/api/view?" + currentQuery());
      const validIds = (VIEW.tabs || []).map(tab => tab.id);
      if (!validIds.includes(ACTIVE_TAB)) {
        ACTIVE_TAB = "original_segments";
      }
      QUALITY_FILTER = "all";
      ORIGINAL_SEGMENT_SIM_OVERRIDES = {};
      QUALITY_STATUS_OVERRIDES = {};
      QUALITY_TUNING_REPORTS = {};
      renderTabs();
      renderContent();
    }

    async function loadOptions() {
      OPTIONS = await fetchJson("/api/options?_ts=" + Date.now());
      const datasetNames = Object.keys(OPTIONS.dataset_choices || {});
      fillSelect(el("dataset_name"), datasetNames, OPTIONS.default_dataset_name);
      syncVariantOptions();
      if (OPTIONS.default_dataset_variant) {
        el("dataset_variant").value = OPTIONS.default_dataset_variant;
      }
      await refreshView();
    }

    el("dataset_name").addEventListener("change", async () => {
      syncVariantOptions();
      await refreshView();
    });
    el("dataset_variant").addEventListener("change", refreshView);
    el("refresh_btn").addEventListener("click", refreshView);
    el("modal_close").addEventListener("click", closeModal);
    el("zoom_reset").addEventListener("click", resetZoom);
    el("modal").addEventListener("click", (event) => {
      if (event.target === el("modal")) closeModal();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeModal();
    });

    const viewport = el("viewport");
    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = event.deltaY < 0 ? 1.1 : 0.9;
      zoomState.scale = Math.max(0.2, Math.min(12, zoomState.scale * delta));
      applyTransform();
    });
    viewport.addEventListener("pointerdown", (event) => {
      zoomState.dragging = true;
      zoomState.startX = event.clientX - zoomState.x;
      zoomState.startY = event.clientY - zoomState.y;
      viewport.classList.add("dragging");
    });
    window.addEventListener("pointermove", (event) => {
      if (!zoomState.dragging) return;
      zoomState.x = event.clientX - zoomState.startX;
      zoomState.y = event.clientY - zoomState.startY;
      applyTransform();
    });
    window.addEventListener("pointerup", () => {
      zoomState.dragging = false;
      viewport.classList.remove("dragging");
    });

    loadOptions();
  </script>
</body>
</html>
"""


@dataclass
class AppState:
    setup_path: Path
    project_root: Path
    setup: ExperimentSetup
    dataset_choices: dict[str, dict[str, Any]]
    default_dataset_name: str
    default_dataset_variant: str
    allowed_roots: list[Path]


class IQViewerHandler(BaseHTTPRequestHandler):
    server_version = "iqviewer/0.1"

    @property
    def app_state(self) -> AppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[iqviewer] " + (format % args) + "\n")

    def _read_json_body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            raise ValueError("Invalid Content-Length header.") from None
        if content_length <= 0:
            raise ValueError("Missing request body.")
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            _text_response(self, HTML)
            return
        if parsed.path == "/api/options":
            _json_response(
                self,
                {
                    "setup_path": str(self.app_state.setup_path),
                    "project_root": str(self.app_state.project_root),
                    "dataset_choices": self.app_state.dataset_choices,
                    "default_dataset_name": self.app_state.default_dataset_name,
                    "default_dataset_variant": self.app_state.default_dataset_variant,
                },
            )
            return
        if parsed.path == "/api/view":
            dataset_name = str(query.get("dataset_name", [self.app_state.default_dataset_name])[0]).strip()
            dataset_variant = str(query.get("dataset_variant", [self.app_state.default_dataset_variant])[0]).strip()
            if dataset_name not in self.app_state.dataset_choices:
                _error_response(self, f"Unknown dataset_name: {dataset_name}", HTTPStatus.BAD_REQUEST)
                return
            variants = self.app_state.dataset_choices[dataset_name].get("variants", [])
            if dataset_variant not in variants:
                _error_response(self, f"Unknown dataset_variant for {dataset_name}: {dataset_variant}", HTTPStatus.BAD_REQUEST)
                return
            _json_response(
                self,
                _build_selection_payload(
                    self.app_state.project_root,
                    dataset_name,
                    dataset_variant,
                    setup=self.app_state.setup,
                ),
            )
            return
        if parsed.path == "/img":
            raw_path = str(query.get("path", [""])[0]).strip()
            if not raw_path:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing image path")
                return
            img_path = Path(unquote(raw_path)).expanduser()
            if not img_path.exists() or not img_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return
            resolved_img_path = img_path.resolve()
            if not any(root == resolved_img_path or root in resolved_img_path.parents for root in self.app_state.allowed_roots):
                self.send_error(HTTPStatus.FORBIDDEN, "Image path outside allowed roots")
                return
            ctype, _ = mimetypes.guess_type(img_path.name)
            data = resolved_img_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/overlay":
            raw_path = str(query.get("path", [""])[0]).strip()
            image_name = str(query.get("image_name", [""])[0]).strip()
            xml_paths = [Path(unquote(value)).expanduser() for value in query.get("xml", []) if str(value).strip()]
            simulate_crop = str(query.get("simulate_crop", ["false"])[0]).strip().lower() in {"1", "true", "yes", "on"}
            target_width = _parse_positive_int_or_default(query.get("target_width", ["256"])[0], 256)
            target_height = _parse_positive_int_or_default(query.get("target_height", ["256"])[0], 256)
            aoi_fill_percentage = _parse_number_or_none(query.get("aoi_fill_percentage", [""])[0])
            label = str(query.get("label", ["rivet"])[0]).strip() or "rivet"
            try:
                max_dim = int(str(query.get("max_dim", ["0"])[0]).strip() or "0")
            except ValueError:
                max_dim = 0
            if not raw_path or not image_name:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing overlay parameters")
                return
            image_path = Path(unquote(raw_path)).expanduser()
            if not image_path.exists() or not image_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return
            resolved_image_path = image_path.resolve()
            if not any(root == resolved_image_path or root in resolved_image_path.parents for root in self.app_state.allowed_roots):
                self.send_error(HTTPStatus.FORBIDDEN, "Image path outside allowed roots")
                return
            resolved_xmls: list[Path] = []
            for xml_path in xml_paths:
                if not xml_path.exists() or not xml_path.is_file():
                    continue
                resolved_xml = xml_path.resolve()
                if not any(root == resolved_xml or root in resolved_xml.parents for root in self.app_state.allowed_roots):
                    self.send_error(HTTPStatus.FORBIDDEN, "Annotation path outside allowed roots")
                    return
                resolved_xmls.append(resolved_xml)
            try:
                data, content_type = _render_overlay_bytes(
                    resolved_image_path,
                    image_name,
                    resolved_xmls,
                    max_dim=max_dim if max_dim > 0 else None,
                    simulate_crop=simulate_crop,
                    target_width=target_width,
                    target_height=target_height,
                    aoi_fill_percentage=aoi_fill_percentage if aoi_fill_percentage is not None else math.nan,
                    label=label,
                )
            except Exception as exc:  # noqa: BLE001
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not render overlay: {exc}")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/quality/auto_tune":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = self._read_json_body()
            rows = payload.get("rows", [])
            enabled_checks = payload.get("enabled_checks", {})
            if not isinstance(rows, list):
                raise ValueError("'rows' must be a list.")
            if enabled_checks is not None and not isinstance(enabled_checks, dict):
                raise ValueError("'enabled_checks' must be an object.")
            result = _auto_tune_quality_thresholds(rows, enabled_checks if isinstance(enabled_checks, dict) else None)
        except ValueError as exc:
            _error_response(self, str(exc), HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:  # noqa: BLE001
            _error_response(self, f"Auto tuning failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        _json_response(self, result)


def build_app_state(setup_path: Path, project_root: Path) -> AppState:
    setup = load_setup(setup_path)
    dataset_choices = _dataset_choices(setup)
    default_dataset_name, default_dataset_variant = _default_dataset(dataset_choices)
    allowed_roots = _resolve_allowed_roots(project_root, dataset_choices)
    return AppState(
        setup_path=setup_path.resolve(),
        project_root=project_root.resolve(),
        setup=setup,
        dataset_choices=dataset_choices,
        default_dataset_name=default_dataset_name,
        default_dataset_variant=default_dataset_variant,
        allowed_roots=allowed_roots,
    )


def main() -> int:
    args = parse_args()
    setup_path = Path(args.setup).expanduser().resolve() if args.setup else (Path.cwd() / "pipeline" / "experiment_setup.yaml").resolve()
    if not setup_path.exists():
        raise FileNotFoundError(f"Setup file not found: {setup_path}")
    project_root = args.root.resolve() if args.root else _infer_project_root(setup_path)
    app_state = build_app_state(setup_path, project_root)

    server = ThreadingHTTPServer((args.host, args.port), IQViewerHandler)
    server.app_state = app_state  # type: ignore[attr-defined]
    print(f"[iqviewer] setup : {setup_path}")
    print(f"[iqviewer] root  : {project_root}")
    print(f"[iqviewer] url   : http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[iqviewer] stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
