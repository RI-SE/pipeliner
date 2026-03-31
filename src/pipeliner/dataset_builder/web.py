from __future__ import annotations

import html
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from .core import DatasetItem, export_dataset, preview_split
from ..planner import build_step_run
from ..setup_loader import load_setup


@dataclass
class WebState:
    session: dict[str, Any]
    items: list[DatasetItem]
    session_path: Path | None
    loaded_session_mtime_ns: int | None
    watch_path: Path | None
    loaded_watch_mtime_ns: int | None


def _group_by_section(items: list[DatasetItem]) -> dict[str, list[DatasetItem]]:
    grouped: dict[str, list[DatasetItem]] = {}
    for item in items:
        grouped.setdefault(item.section, []).append(item)
    for section_name in grouped:
        grouped[section_name] = sorted(grouped[section_name], key=lambda value: value.display_name)
    return grouped


def _selected(value: str, expected: str) -> str:
    return " selected" if value == expected else ""


def _render_item_card(item: DatasetItem, labels: list[str], split_labels: list[str], state: WebState) -> str:
    default_label = labels[0] if labels else ""
    current_label = str(state.session.get("labels", {}).get(item.item_id, default_label))
    current_split = str(state.session.get("split_assignments", {}).get(item.item_id, ""))

    label_options: list[str] = []
    for label in labels:
        option = f"<option value=\"{html.escape(label)}\"{_selected(current_label, label)}>{html.escape(label)}</option>"
        label_options.append(option)

    split_options = [f"<option value=\"\"{_selected(current_split, '')}></option>"]
    for split in split_labels:
        option = f"<option value=\"{html.escape(split)}\"{_selected(current_split, split)}>{html.escape(split)}</option>"
        split_options.append(option)

    return (
        "<article class='card'>"
        f"<img class='thumb' src='/image/{html.escape(item.item_id)}' alt='{html.escape(item.display_name)}' loading='lazy' />"
        "<div class='card-meta'>"
        f"<div class='name'>{html.escape(item.display_name)}</div>"
        f"<div class='group'>group: {html.escape(item.group_key)}</div>"
        "</div>"
        "<form method='get' action='/assign' class='assign-form'>"
        f"<input type='hidden' name='item_id' value='{html.escape(item.item_id)}' />"
        f"<select name='class_label' onchange='assignForm(this.form)'>{''.join(label_options)}</select>"
        f"<select name='split_label' onchange='assignForm(this.form)'>{''.join(split_options)}</select>"
        "</form>"
        "</article>"
    )


def _render_page(state: WebState) -> str:
    grouped = _group_by_section(state.items)
    configured_inputs = state.session.get("input_sections", {})
    configured_sections: list[str] = []
    if isinstance(configured_inputs, dict):
        configured_sections = [str(name) for name in configured_inputs.keys()]
    labels = [str(v) for v in state.session.get("config", {}).get("class_labels", [])]
    split_labels = [str(v) for v in state.session.get("split", {}).get("split_labels", [])]

    preview = preview_split(
        state.items,
        state.session.get("labels", {}),
        state.session.get("split", {}),
        state.session.get("split_assignments", {}),
    )

    section_names = list(configured_sections)
    for section_name in sorted(grouped):
        if section_name not in section_names:
            section_names.append(section_name)
    tab_buttons: list[str] = []
    tab_panels: list[str] = []
    for index, section_name in enumerate(section_names):
        section_items = grouped.get(section_name, [])
        panel_id = f"section-{index}"
        active_class = " active" if index == 0 else ""
        empty_class = " empty" if not section_items else ""
        tab_buttons.append(
            f"<button class='tab-btn{active_class}{empty_class}' type='button' onclick=\"openTab('{panel_id}', this)\">"
            f"{html.escape(section_name)} ({len(section_items)})"
            "</button>"
        )
        input_dir = ""
        if isinstance(configured_inputs, dict):
            input_dir = str(configured_inputs.get(section_name, ""))
        cards = "".join(
            _render_item_card(item, labels=labels, split_labels=split_labels, state=state)
            for item in section_items
        )
        empty_html = (
            "<div class='empty-section'>"
            "<div>Katalogen är tom.</div>"
            f"<div class='path'>{html.escape(input_dir)}</div>"
            "</div>"
            if not section_items
            else ""
        )
        tab_panels.append(
            f"<section id='{panel_id}' class='tab-panel{active_class}'>"
            f"{empty_html}"
            f"<div class='cards'>{cards}</div>"
            "</section>"
        )

    counts = preview.get("counts", {})
    stat_tiles: list[str] = []
    total_images = len(state.items)
    assigned_images = sum(sum(class_map.values()) for class_map in counts.values())
    stat_tiles.append(f"<div class='stat'><span>Total</span><strong>{total_images}</strong></div>")
    stat_tiles.append(f"<div class='stat'><span>Assigned</span><strong>{assigned_images}</strong></div>")
    for split_name, class_map in counts.items():
        split_total = sum(class_map.values())
        details = ", ".join(f"{k}:{v}" for k, v in sorted(class_map.items()))
        details_html = f"<small>{html.escape(details)}</small>" if details else ""
        stat_tiles.append(
            "<div class='stat'>"
            f"<span>{html.escape(split_name)}</span>"
            f"<strong>{split_total}</strong>"
            f"{details_html}"
            "</div>"
        )
    html_body = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Dataset Builder</title>
  <style>
    body {{ font-family: 'Avenir Next', 'Segoe UI', sans-serif; margin: 0; background: #f4f6f8; color: #1a2433; }}
    .page {{ max-width: 1600px; margin: 0 auto; padding: 18px; }}
    form.inline {{ display: inline; }}
    .top {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
    h1 {{ margin: 0; font-size: 34px; }}
    .toolbar button {{ border: none; background: #0d4f8b; color: #fff; padding: 10px 16px; border-radius: 10px; font-weight: 600; cursor: pointer; }}
    .config-alert {{ display: none; margin: 12px 0 10px; padding: 10px 12px; border-radius: 10px; border: 1px solid #f0b429; background: #fff7df; color: #7a4f01; }}
    .config-alert.show {{ display: block; }}
    .config-alert-row {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }}
    .config-alert-text {{ font-weight: 600; }}
    .btn-reload-config {{ border: 1px solid #1f7a3d; background: #1f9d57; color: #fff; padding: 8px 12px; border-radius: 8px; cursor: pointer; font-weight: 700; }}
    .btn-reload-config[disabled] {{ opacity: 0.7; cursor: wait; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 16px 0 18px; }}
    .stat {{ background: #fff; border: 1px solid #dbe2ea; border-radius: 12px; padding: 10px 12px; display: grid; gap: 4px; }}
    .stat span {{ font-size: 12px; color: #48617d; text-transform: uppercase; letter-spacing: .04em; }}
    .stat strong {{ font-size: 24px; line-height: 1; }}
    .stat small {{ color: #5e6f83; font-size: 12px; }}
    .tab-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 12px; }}
    .tab-btn {{ padding: 8px 12px; border: 1px solid #c4d0dd; border-radius: 999px; background: #fff; cursor: pointer; }}
    .tab-btn.active {{ background: #0d4f8b; border-color: #0d4f8b; color: #fff; font-weight: 600; }}
    .tab-btn.empty {{ background: #f2f4f7; border-color: #d3dae3; color: #738398; }}
    .tab-btn.empty.active {{ background: #dbe3ee; border-color: #9fb0c8; color: #2f435b; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #dbe2ea; border-radius: 12px; padding: 8px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .thumb {{ width: 100%; aspect-ratio: 1/1; object-fit: contain; background: #f8fafc; border-radius: 8px; }}
    .card-meta {{ margin: 8px 2px; font-size: 13px; }}
    .name {{ font-weight: 600; word-break: break-all; }}
    .group {{ color: #48617d; }}
    .assign-form {{ display: grid; gap: 6px; }}
    .assign-form select {{ border: 1px solid #c4d0dd; border-radius: 8px; padding: 7px 8px; background: #fff; }}
    .empty {{ margin-top: 16px; color: #900; font-weight: 600; }}
    .empty-section {{ margin: 6px 0 12px; padding: 12px; border: 1px dashed #c4cfdb; border-radius: 10px; background: #f8fafc; color: #43586f; }}
    .empty-section .path {{ margin-top: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #62748a; word-break: break-all; }}
    @media (max-width: 640px) {{
      .page {{ padding: 10px; }}
      .cards {{ grid-template-columns: 1fr; }}
    }}
  </style>
  <script>
    let CONFIG_STATUS_TIMER = null;
    let CONFIG_STATUS = null;

    async function fetchJson(url, options = undefined) {{
      const response = await fetch(url, options);
      const payload = await response.json().catch(() => ({{}}));
      return {{ response, payload }};
    }}

    function renderConfigStatus(status) {{
      const banner = document.getElementById('config_alert');
      const text = document.getElementById('config_alert_text');
      if (!banner || !text) return;
      if (!status || !status.changed) {{
        banner.classList.remove('show');
        text.textContent = 'Config changed on disk.';
        return;
      }}
      const disk = status.disk_mtime || 'unknown';
      const loaded = status.loaded_mtime || 'unknown';
      text.textContent = 'Configuration file changed on disk. Running view uses older copy. loaded=' + loaded + ', disk=' + disk;
      banner.classList.add('show');
    }}

    async function refreshConfigStatus() {{
      const {{ response, payload }} = await fetchJson('/api/config-status');
      if (!response.ok) return;
      CONFIG_STATUS = payload;
      renderConfigStatus(CONFIG_STATUS);
    }}

    async function reloadConfigFromDisk() {{
      const btn = document.getElementById('reload_config_btn');
      if (!btn) return;
      btn.disabled = true;
      const old = btn.textContent;
      btn.textContent = 'Reloading...';
      try {{
        const {{ response, payload }} = await fetchJson('/api/reload-config', {{ method: 'POST' }});
        if (response.ok) {{
          CONFIG_STATUS = payload.config_status || null;
          renderConfigStatus(CONFIG_STATUS);
          window.location.reload();
          return;
        }}
      }} finally {{
        btn.disabled = false;
        btn.textContent = old;
      }}
    }}

    async function assignForm(form) {{
      const params = new URLSearchParams(new FormData(form));
      await fetch('/assign-async?' + params.toString(), {{ method: 'POST' }});
    }}
    function openTab(panelId, btn) {{
      document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      const panel = document.getElementById(panelId);
      if (panel) panel.classList.add('active');
      if (btn) btn.classList.add('active');
    }}

    window.addEventListener('load', async () => {{
      const reloadBtn = document.getElementById('reload_config_btn');
      if (reloadBtn) reloadBtn.addEventListener('click', reloadConfigFromDisk);
      await refreshConfigStatus();
      CONFIG_STATUS_TIMER = window.setInterval(refreshConfigStatus, 3000);
    }});
  </script>
</head>
<body>
  <div class="page">
  <div class="top">
    <h1>Dataset Builder</h1>
    <div class="toolbar">
      <form class=\"inline\" method=\"post\" action=\"/generate\"><button type=\"submit\">Generate Training Structure</button></form>
    </div>
  </div>
  <div id="config_alert" class="config-alert">
    <div class="config-alert-row">
      <div id="config_alert_text" class="config-alert-text">Config changed on disk.</div>
      <button class="btn-reload-config" id="reload_config_btn" type="button">Reload Config</button>
    </div>
  </div>
  <div class="stats">{''.join(stat_tiles)}</div>
  <h2>Inputs</h2>
  {"<div class='empty'>No images found for configured input sections.</div>" if total_images == 0 else ""}
  <div class='tab-bar'>{''.join(tab_buttons)}</div>
  {''.join(tab_panels)}
  </div>
</body>
</html>
"""
    return html_body


def build_app(
    session: dict[str, Any],
    items: list[DatasetItem],
    session_path: Path | None = None,
    watch_path: Path | None = None,
) -> FastAPI:
    resolved_session_path = session_path.resolve() if session_path else None
    loaded_session_mtime_ns = resolved_session_path.stat().st_mtime_ns if resolved_session_path and resolved_session_path.exists() else None
    resolved_watch_path = watch_path.resolve() if watch_path else None
    loaded_watch_mtime_ns = resolved_watch_path.stat().st_mtime_ns if resolved_watch_path and resolved_watch_path.exists() else None
    state = WebState(
        session=session,
        items=items,
        session_path=resolved_session_path,
        loaded_session_mtime_ns=loaded_session_mtime_ns,
        watch_path=resolved_watch_path,
        loaded_watch_mtime_ns=loaded_watch_mtime_ns,
    )
    item_index = {item.item_id: item for item in state.items}
    app = FastAPI(title="Dataset Builder")

    def _iso_for_mtime_ns(mtime_ns: int | None) -> str:
        if mtime_ns is None:
            return ""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime_ns / 1_000_000_000))

    def _current_config_status() -> dict[str, Any]:
        status_path = state.watch_path or state.session_path
        loaded_mtime_ns = state.loaded_watch_mtime_ns if state.watch_path else state.loaded_session_mtime_ns
        if status_path is None:
            return {
                "setup_path": "",
                "loaded_mtime": "",
                "disk_mtime": "",
                "changed": False,
                "watch_enabled": False,
            }
        disk_mtime_ns = status_path.stat().st_mtime_ns if status_path.exists() else None
        return {
            "setup_path": str(status_path),
            "loaded_mtime": _iso_for_mtime_ns(loaded_mtime_ns),
            "disk_mtime": _iso_for_mtime_ns(disk_mtime_ns),
            "changed": disk_mtime_ns != loaded_mtime_ns,
            "watch_enabled": True,
        }

    def _apply_reloaded_payload(payload: dict[str, Any], payload_items: list[DatasetItem]) -> None:
        restored = dict(payload) if isinstance(payload, dict) else {}
        restored.setdefault("config", {})
        restored.setdefault("split", {})
        restored.setdefault("labels", {})
        restored.setdefault("split_assignments", {})
        restored.setdefault("paths", {})
        restored.setdefault("input_sections", {})
        restored.setdefault("csv_loaded", False)

        if payload_items:
            next_items = payload_items
        else:
            from .core import initialize_split_assignments, load_assignment_csv, scan_dataset_items

            next_items = scan_dataset_items(
                restored.get("input_sections", {}),
                class_labels=restored.get("config", {}).get("class_labels", []),
            )
            load_assignment_csv(restored, next_items)
            if not restored.get("split_assignments"):
                initial = initialize_split_assignments(next_items, restored.get("split", {}))
                restored.setdefault("split_assignments", {}).update(initial)
            preview_split(
                next_items,
                restored.get("labels", {}),
                restored.get("split", {}),
                restored.get("split_assignments", {}),
            )

        state.session = restored
        state.items = next_items
        item_index.clear()
        item_index.update({item.item_id: item for item in next_items})

    def _try_rebuild_from_setup() -> tuple[dict[str, Any], list[DatasetItem]] | None:
        if state.watch_path is None or not state.watch_path.exists():
            return None
        if state.watch_path.suffix.lower() not in {".yaml", ".yml"}:
            return None
        contract = state.session.get("contract", {})
        if not isinstance(contract, dict):
            return None
        process_step = contract.get("process_step", {})
        if not isinstance(process_step, dict):
            return None
        step_name = str(process_step.get("name", "")).strip()
        if not step_name:
            return None
        variation_points = contract.get("variation_points", {})
        if not isinstance(variation_points, dict):
            return None
        setup = load_setup(state.watch_path)
        run = build_step_run(setup, step_name, variation_points)
        from .core import derive_session_from_contract, initialize_split_assignments, load_assignment_csv, scan_dataset_items

        rebuilt = derive_session_from_contract(run.contract.to_dict(), state.watch_path)
        rebuilt_paths = rebuilt.setdefault("paths", {})
        rebuilt_paths["watched_config_path"] = str(state.watch_path)
        for key in ("contract_path", "contract", "input_sections"):
            if key in state.session and key not in rebuilt:
                rebuilt[key] = state.session[key]
        next_items = scan_dataset_items(
            rebuilt.get("input_sections", {}),
            class_labels=rebuilt.get("config", {}).get("class_labels", []),
        )
        load_assignment_csv(rebuilt, next_items)
        if not rebuilt.get("split_assignments"):
            initial = initialize_split_assignments(next_items, rebuilt.get("split", {}))
            rebuilt.setdefault("split_assignments", {}).update(initial)
        preview_split(
            next_items,
            rebuilt.get("labels", {}),
            rebuilt.get("split", {}),
            rebuilt.get("split_assignments", {}),
        )
        return rebuilt, next_items

    def _reload_config_from_disk() -> dict[str, Any]:
        rebuilt = None
        try:
            rebuilt = _try_rebuild_from_setup()
        except Exception:  # noqa: BLE001
            rebuilt = None
        if rebuilt is not None:
            payload, payload_items = rebuilt
            _apply_reloaded_payload(payload, payload_items)
        elif state.session_path is not None and state.session_path.exists():
            payload, payload_items = load_preloaded_session(state.session_path)
            _apply_reloaded_payload(payload, payload_items)
        state.loaded_session_mtime_ns = state.session_path.stat().st_mtime_ns if state.session_path and state.session_path.exists() else None
        state.loaded_watch_mtime_ns = state.watch_path.stat().st_mtime_ns if state.watch_path and state.watch_path.exists() else None
        return _current_config_status()

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(_render_page(state))

    @app.get("/api/config-status", response_model=None)
    def config_status() -> JSONResponse:
        return JSONResponse(_current_config_status())

    @app.post("/api/reload-config", response_model=None)
    def reload_config() -> JSONResponse:
        status = _reload_config_from_disk()
        return JSONResponse({"status": "reloaded", "config_status": status})

    @app.get("/assign")
    def assign(item_id: str, class_label: str = "", split_label: str = "") -> RedirectResponse:
        if class_label:
            state.session.setdefault("labels", {})[item_id] = class_label
        elif item_id in state.session.setdefault("labels", {}):
            del state.session["labels"][item_id]

        if split_label:
            state.session.setdefault("split_assignments", {})[item_id] = split_label
        elif item_id in state.session.setdefault("split_assignments", {}):
            del state.session["split_assignments"][item_id]

        return RedirectResponse(url="/", status_code=303)

    @app.post("/assign-async", response_model=None)
    def assign_async(item_id: str, class_label: str = "", split_label: str = "") -> JSONResponse:
        if class_label:
            state.session.setdefault("labels", {})[item_id] = class_label
        elif item_id in state.session.setdefault("labels", {}):
            del state.session["labels"][item_id]

        if split_label:
            state.session.setdefault("split_assignments", {})[item_id] = split_label
        elif item_id in state.session.setdefault("split_assignments", {}):
            del state.session["split_assignments"][item_id]

        return JSONResponse({"ok": True})

    @app.get("/image/{item_id}", response_model=None)
    def image(item_id: str):
        item = item_index.get(item_id)
        if item is None:
            return JSONResponse({"error": "item not found"}, status_code=404)
        path = Path(item.image_path)
        if not path.exists():
            return JSONResponse({"error": "image missing on disk"}, status_code=404)
        return FileResponse(path)

    @app.post("/generate")
    def generate() -> RedirectResponse:
        preview = preview_split(
            state.items,
            state.session.get("labels", {}),
            state.session.get("split", {}),
            state.session.get("split_assignments", {}),
        )
        export_dataset(state.session, state.items, preview, recreate_dataset=True)
        return RedirectResponse(url="/", status_code=303)

    return app


def load_preloaded_session(path: Path) -> tuple[dict[str, Any], list[DatasetItem]]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"session file must contain an object: {path}")

    items_raw = payload.get("items", [])
    items: list[DatasetItem] = []
    if isinstance(items_raw, list):
        for row in items_raw:
            if not isinstance(row, dict):
                continue
            items.append(
                DatasetItem(
                    item_id=str(row.get("item_id", "")),
                    section=str(row.get("section", "")),
                    image_path=str(row.get("image_path", "")),
                    display_name=str(row.get("display_name", "")),
                    group_key=str(row.get("group_key", "")),
                )
            )

    return payload, items
