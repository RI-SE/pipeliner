#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .planner import build_extra_args

PipelineContext = None
PipelineLayout = None


def _load_layout_classes(root: Path):
    layout_path = root / "pipeline" / "pipeline_layout.py"
    if not layout_path.exists():
        raise FileNotFoundError(f"Cannot find pipeline layout module: {layout_path}")

    spec = importlib.util.spec_from_file_location("pipeliner_consumer_pipeline_layout", layout_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {layout_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "PipelineContext") or not hasattr(module, "PipelineLayout"):
        raise ImportError(f"{layout_path} does not define PipelineContext/PipelineLayout")
    return module.PipelineContext, module.PipelineLayout


def _infer_project_root(setup_path: Path) -> Path:
    setup_path = setup_path.resolve()
    if setup_path.parent.name == "pipeline":
        return setup_path.parent.parent
    return setup_path.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web config viewer for pipeline path/script resolution.")
    parser.add_argument("--setup", "--config", "-config", dest="setup", default="")
    parser.add_argument("--mode", choices=["viewer", "runner"], default=None)
    parser.add_argument("--default-tab", choices=["viewer", "runner", "analysis"], default="viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--experiment-name", default="experiment1_")
    parser.add_argument("--runner-python", default="python3")
    parser.add_argument("--lama-python-bin", default="/Users/dfm01/miniconda/envs/lama-inpainting/bin/python")
    return parser.parse_args()


def as_json(handler: BaseHTTPRequestHandler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_tree_text(root: Path, max_depth: int = 3) -> str:
    if not root.exists():
        return f"(missing) {root}"

    lines: list[str] = [str(root)]

    def walk(path: Path, depth: int, prefix: str) -> None:
        if depth > max_depth:
            return
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            marker = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{branch}{entry.name}{marker}")
            if entry.is_dir():
                walk(entry, depth + 1, prefix + ("    " if last else "│   "))

    walk(root, 1, "")
    return "\n".join(lines)


def make_ctx(query: dict[str, list[str]], default_experiment_name: str, layout: PipelineLayout) -> PipelineContext:
    dataset_name = query.get("dataset_name", [""])[0]
    dataset_variant = query.get("dataset_variant", [""])[0]
    detection_algorithm = query.get("detection_algorithm", [""])[0]
    tiling = query.get("tiling", ["whole"])[0]
    experiment_name = query.get("experiment_name", [default_experiment_name])[0]
    repair_method = query.get("repair_method", ["LaMa"])[0]
    mask_type = query.get("mask_type", ["re_masked_thicker"])[0]
    extra_vars: dict[str, str] = {}
    for key in layout.get_extra_variation_points():
        value = query.get(key, [""])[0]
        if value != "":
            extra_vars[key] = str(value)
    for key in layout.get_extra_hier_variation_points():
        parent = query.get(key, [""])[0]
        variant = query.get(f"{key}_variant", [""])[0]
        if parent != "":
            extra_vars[key] = str(parent)
        if variant != "":
            extra_vars[f"{key}_variant"] = str(variant)
    return PipelineContext(
        dataset_name=dataset_name,
        dataset_variant=dataset_variant,
        detection_algorithm=detection_algorithm,
        tiling=tiling,
        experiment_name=experiment_name,
        repair_method=repair_method,
        mask_type=mask_type,
        extra_vars=extra_vars,
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root { --bg:#f6f8fb; --fg:#1e2430; --card:#ffffff; --line:#d6dce8; --accent:#1d6fd6; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color:var(--fg); background:var(--bg); }
    .wrap { width: 100%; max-width: none; margin: 16px 0; padding: 0 16px; box-sizing: border-box; }
    .title { margin: 0 0 14px; font-size: 24px; font-weight: 700; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap: 10px; }
    .card { background:var(--card); border:1px solid var(--line); border-radius: 10px; padding: 12px; }
    label { display:block; font-size: 12px; margin-bottom: 6px; color:#5a6578; }
    select,input { width:100%; box-sizing:border-box; border:1px solid var(--line); border-radius:8px; padding:8px; font-size:14px; }
    .btn { margin-top: 10px; border:1px solid var(--accent); background:var(--accent); color:#fff; padding:8px 12px; border-radius:8px; cursor:pointer; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
    .table-wrap { width: 100%; overflow-x: auto; }
    table { width:100%; border-collapse: collapse; font-size: 13px; min-width: 1400px; }
    th, td { border:1px solid var(--line); padding: 6px 8px; text-align:left; vertical-align: top; }
    th { background:#eef3fc; }
    td.path { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: nowrap; }
    .tabs { display:flex; gap:8px; margin: 0 0 12px; }
    .tab { border:1px solid var(--line); background:#fff; color:#1e2430; padding:8px 12px; border-radius:8px; cursor:pointer; }
    .tab.active { border-color:var(--accent); color:#fff; background:var(--accent); }
    .status-dot { display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:8px; }
    .status-never { background:#9aa3b2; }
    .status-success { background:#1f9d57; }
    .status-failed { background:#d64545; }
    .status-running { background:#2b77d1; }
    .config-alert { margin: 0 0 12px; padding: 14px 16px; border: 2px solid #b42318; border-radius: 12px; background: #fef3f2; color: #7a271a; display:none; }
    .config-alert.show { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .config-alert-text { font-size: 14px; font-weight: 700; }
    .config-alert-sub { margin-top:4px; font-size:12px; font-weight:400; color:#912018; }
    .btn-reload-config { border:1px solid #1f7a3d; background:#1f9d57; color:#fff; padding:10px 14px; border-radius:8px; cursor:pointer; font-weight:700; white-space:nowrap; }
    .btn-reload-config[disabled] { opacity:0.7; cursor:wait; }
    .group-row { background:#f8fbff; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    .kv { display:grid; grid-template-columns: 120px 1fr; gap:6px 10px; font-size:13px; }
    .cmdbox { margin-top:10px; border:1px solid var(--line); border-radius:8px; background:#fbfdff; }
    .cmdhead { display:flex; justify-content:flex-start; align-items:center; gap:8px; padding:8px 10px; border-bottom:1px solid var(--line); font-size:12px; color:#5a6578; }
    .cmdpre { padding:10px; margin:0; white-space: pre-wrap; word-break: break-word; }
    .btn-mini { border:1px solid #9ed9b3; background:#e7f7ee; color:#1f7a3d; border-radius:6px; padding:4px 8px; font-size:12px; cursor:pointer; width:auto; }
    .muted { color:#5a6578; font-size:12px; }
    .step-cell { display:flex; align-items:center; gap:8px; }
    .step-copy { border:1px solid #9ed9b3; background:#e7f7ee; color:#1f7a3d; border-radius:6px; padding:2px 6px; font-size:11px; cursor:pointer; }
    .step-copy-link { cursor: default; }
    .step-copy-mini { margin-right:8px; border:1px solid #9ed9b3; background:#e7f7ee; color:#1f7a3d; border-radius:6px; padding:1px 6px; font-size:11px; cursor:pointer; }
    .btn-danger-mini { border:1px solid #d64545; background:#fff5f5; color:#b23030; border-radius:6px; padding:2px 6px; font-size:11px; cursor:pointer; width:auto; }
    .split-row { display:block; }
    .split-row.runner { display:grid; grid-template-columns: 7fr 3fr; gap:12px; align-items:start; }
    .split-row .card { margin-top:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="tabs">
      <button class="tab" id="tab_viewer" data-tab="viewer">Viewer</button>
      <button class="tab" id="tab_runner" data-tab="runner">Runner</button>
      <button class="tab" id="tab_analysis" data-tab="analysis">Analysis</button>
    </div>
    <div id="config_alert" class="config-alert">
      <div>
        <div id="config_alert_text" class="config-alert-text">Config changed on disk.</div>
        <div id="config_alert_sub" class="config-alert-sub"></div>
      </div>
      <button class="btn-reload-config" id="reload_config_btn" type="button">Reload Config</button>
    </div>
    <div class="card">
      <div class="grid" id="variation_grid"></div>
      <div class="grid" style="margin-top:10px;">
        <div><label>Step</label><select id="step"></select></div>
        <div><label>Experiment Name</label><input id="experiment_name" value="experiment1_" /></div>
        <div><label>Path Display</label><label><input id="hide_root" type="checkbox" /> Hide project root prefix</label></div>
      </div>
      <button class="btn" id="refresh">Refresh</button>
      <button class="btn" id="run_step">Run Selected Step</button>
      <button class="btn" id="run_pipeline">Run Pipeline(s)</button>
      <button class="btn" id="cancel_job">Cancel Job</button>
    </div>
    <div class="card" id="analysis_panel" style="margin-top:12px; display:none;">
      <h3 style="margin:0 0 8px;">Analysis</h3>
      <pre id="analysis_note">Analysis tab scaffold. Add metrics/charts/log summaries here.</pre>
    </div>
    <div class="card" id="runner_plan_panel" style="margin-top:12px; display:none;">
      <h3 style="margin:0 0 8px;">Runner Plan</h3>
      <div class="table-wrap">
        <table id="runner_plan_tbl">
          <thead><tr><th style="width:36px;"></th><th>Pipeline ID</th><th>Pipeline / Combination</th><th>Status</th><th>Last Run</th><th>Logs</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div class="card" id="selected_step_card" style="margin-top:12px;">
      <h3 style="margin:0 0 8px;">Selected Step</h3>
      <div class="cmdbox" style="margin-top:8px;">
        <div class="cmdhead">
          <button class="btn-mini" id="copy_cmd_paste_btn">Copy</button>
          <span>Copy/Paste Command</span>
        </div>
        <pre id="selected_cmd_paste" class="cmdpre mono"></pre>
      </div>
    </div>
    <div class="card" id="all_steps_card" style="margin-top:12px;">
      <h3 style="margin:0 0 8px;">All Steps (Current Selection)</h3>
      <div class="table-wrap">
        <table id="tbl">
          <thead><tr><th>Step</th><th>Script</th><th>Input</th><th>Output</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div id="run_trees_row" class="split-row">
      <div class="card" id="run_log_card">
        <h3 style="margin:0 0 8px;">Run Log</h3>
        <pre id="runlog"></pre>
      </div>
      <div class="card" id="trees_card">
        <h3 style="margin:0 0 8px;">Selected Trees</h3>
        <div class="table-wrap">
          <table>
            <thead><tr><th style="width:50%;">Input Tree</th><th style="width:50%;"><button class="btn-danger-mini" id="purge_output_btn" title="Delete selected output directory contents">☠ purge</button> Output Tree</th></tr></thead>
            <tbody>
              <tr>
                <td><pre id="input_treeview"></pre></td>
                <td><pre id="output_treeview"></pre></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
<script>
let OPTIONS = null;
let LAST_RUN_LOG = null;
let CURRENT_JOB_ID = null;
let CACHE_BUST = Date.now().toString();
let VAR_QUERY_KEYS = [];
let VAR_DISPLAY_NAMES = {};
let VAR_ENUMERATE_FLAGS = {};
let ACTIVE_TAB = "__DEFAULT_TAB__";
let CONFIG_STATUS = null;
let CONFIG_STATUS_TIMER = null;
const ALL_TOKEN = "ALL";

const QUERY_KEY_BY_VARIATION = {
  algo_name: "detection_algorithm",
};

const LABEL_BY_QUERY_KEY = {
  dataset_name: "Dataset",
  dataset_variant: "Variant",
  detection_algorithm: "Algorithm",
  tiling: "Tiling",
  repair_method: "Repair Method",
  mask_type: "Mask Type",
};

function toQueryKey(variationKey) {
  return QUERY_KEY_BY_VARIATION[variationKey] || variationKey;
}

function toLabel(queryKey, variationKey = null) {
  if (variationKey && VAR_DISPLAY_NAMES[variationKey]) return VAR_DISPLAY_NAMES[variationKey];
  if (LABEL_BY_QUERY_KEY[queryKey]) return LABEL_BY_QUERY_KEY[queryKey];
  return queryKey.replaceAll("_", " ");
}

function setTab(tab) {
  ACTIVE_TAB = tab;
  ["viewer", "runner", "analysis"].forEach(t => {
    const b = document.getElementById(`tab_${t}`);
    if (b) b.classList.toggle("active", t === ACTIVE_TAB);
  });
  const inAnalysis = ACTIVE_TAB === "analysis";
  const inRunner = ACTIVE_TAB === "runner";
  document.getElementById("run_pipeline").style.display = inRunner ? "" : "none";
  document.getElementById("run_step").style.display = inAnalysis ? "none" : "";
  document.getElementById("cancel_job").style.display = inAnalysis ? "none" : "";
  document.getElementById("analysis_panel").style.display = inAnalysis ? "" : "none";
  document.getElementById("runner_plan_panel").style.display = inRunner ? "" : "none";
  document.getElementById("selected_step_card").style.display = inRunner ? "none" : "";
  document.getElementById("all_steps_card").style.display = inRunner ? "none" : "";
  document.getElementById("run_log_card").style.display = inAnalysis ? "none" : "";
  document.getElementById("trees_card").style.display = inAnalysis ? "none" : "";
  const split = document.getElementById("run_trees_row");
  if (split) split.classList.toggle("runner", inRunner);
}

function stripRootFromText(value) {
  if (typeof value !== "string") return value;
  const root = OPTIONS && OPTIONS.root_dir ? OPTIONS.root_dir : "";
  const hide = document.getElementById("hide_root").checked;
  if (!hide || !root) return value;
  let out = value.split(root + "/").join("");
  if (out === root) out = ".";
  out = out.split(root).join(".");
  return out;
}

function stripRootInObject(value) {
  if (Array.isArray(value)) return value.map(stripRootInObject);
  if (value && typeof value === "object") {
    const out = {};
    Object.entries(value).forEach(([k, v]) => { out[k] = stripRootInObject(v); });
    return out;
  }
  return stripRootFromText(value);
}

function qs() {
  const p = new URLSearchParams();
  p.set("experiment_name", document.getElementById("experiment_name").value);
  p.set("step", document.getElementById("step").value);
  VAR_QUERY_KEYS.forEach(k => {
    const el = document.getElementById(`vp_${k}`);
    if (el) p.set(k, el.value);
  });
  p.set("_ts", CACHE_BUST);
  return p.toString();
}

function hasAllSelection() {
  return VAR_QUERY_KEYS.some(k => {
    const el = document.getElementById(`vp_${k}`);
    return el && el.value === ALL_TOKEN;
  });
}

function renderVariationPoints() {
  const host = document.getElementById("variation_grid");
  host.innerHTML = "";
  VAR_QUERY_KEYS = [];
  const allowAll = ACTIVE_TAB === "runner";

  const order = (OPTIONS && OPTIONS.variation_order) ? OPTIONS.variation_order : [];
  const pointsFlat = (OPTIONS && OPTIONS.variation_points_flat) ? OPTIONS.variation_points_flat : {};
  const pointsHier = (OPTIONS && OPTIONS.variation_points_hier) ? OPTIONS.variation_points_hier : {};

  order.forEach((variationKey) => {
    const queryKey = toQueryKey(variationKey);

    if (pointsHier[variationKey]) {
      const hier = pointsHier[variationKey];
      const parentMap = hier.values || {};
      const parentNames = Object.keys(parentMap);
      const childKey = hier.child_key;
      const enumerateOptions = !!VAR_ENUMERATE_FLAGS[variationKey];

      const wrapParent = document.createElement("div");
      const labelParent = document.createElement("label");
      labelParent.textContent = toLabel(queryKey, variationKey);
      const selParent = document.createElement("select");
      selParent.id = `vp_${queryKey}`;
      fillSelect(
        selParent,
        allowAll ? [ALL_TOKEN, ...parentNames] : parentNames,
        enumerateOptions,
        allowAll ? 1 : 0
      );
      wrapParent.appendChild(labelParent);
      wrapParent.appendChild(selParent);
      host.appendChild(wrapParent);

      const wrapChild = document.createElement("div");
      const labelChild = document.createElement("label");
      labelChild.textContent = toLabel(childKey);
      const selChild = document.createElement("select");
      selChild.id = `vp_${childKey}`;
      wrapChild.appendChild(labelChild);
      wrapChild.appendChild(selChild);
      host.appendChild(wrapChild);

      const syncChild = () => {
        const parent = selParent.value;
        let values = [];
        if (parent === ALL_TOKEN && allowAll) {
          const uniq = new Set();
          Object.values(parentMap).forEach(vs => (vs || []).forEach(v => uniq.add(v)));
          values = Array.from(uniq);
        } else {
          values = parentMap[parent] || [];
        }
        fillSelect(
          selChild,
          allowAll ? [ALL_TOKEN, ...values] : values,
          enumerateOptions,
          allowAll ? 1 : 0
        );
      };
      syncChild();
      selParent.addEventListener("change", async () => {
        syncChild();
        await refreshAll();
      });
      selChild.addEventListener("change", refreshAll);

      VAR_QUERY_KEYS.push(queryKey);
      VAR_QUERY_KEYS.push(childKey);
      return;
    }

    const values = pointsFlat[variationKey] || [];
    if (!values.length) return;
    const enumerateOptions = !!VAR_ENUMERATE_FLAGS[variationKey];
    const wrap = document.createElement("div");
    const label = document.createElement("label");
    label.textContent = toLabel(queryKey, variationKey);
    const sel = document.createElement("select");
    sel.id = `vp_${queryKey}`;
    fillSelect(
      sel,
      allowAll ? [ALL_TOKEN, ...values] : values,
      enumerateOptions,
      allowAll ? 1 : 0
    );
    sel.addEventListener("change", refreshAll);
    wrap.appendChild(label);
    wrap.appendChild(sel);
    host.appendChild(wrap);
    VAR_QUERY_KEYS.push(queryKey);
  });
}

async function fetchJson(url, options = null) {
  const opts = options || {};
  opts.cache = "no-store";
  const r = await fetch(url, opts);
  const j = await r.json();
  return { r, j };
}

function formatTimestamp(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

function updateConfigBanner(status) {
  CONFIG_STATUS = status || null;
  const banner = document.getElementById("config_alert");
  const text = document.getElementById("config_alert_text");
  const sub = document.getElementById("config_alert_sub");
  if (!banner || !text || !sub) return;
  if (!status || !status.changed) {
    banner.classList.remove("show");
    text.textContent = "Config changed on disk.";
    sub.textContent = "";
    return;
  }
  const setupPath = stripRootFromText(status.setup_path || "");
  const diskMtime = formatTimestamp(status.disk_mtime || "");
  const loadedMtime = formatTimestamp(status.loaded_mtime || "");
  text.textContent = "Configuration file changed on disk. The running server is using an older copy.";
  sub.textContent = `Loaded: ${loadedMtime || "unknown"} | On disk: ${diskMtime || "unknown"} | ${setupPath}`;
  banner.classList.add("show");
}

async function refreshConfigStatus() {
  try {
    const { r, j } = await fetchJson("/api/config-status?_ts=" + Date.now());
    if (!r.ok) return;
    updateConfigBanner(j);
  } catch (_) {
    // ignore transient polling issues
  }
}

async function reloadConfigFromDisk() {
  const btn = document.getElementById("reload_config_btn");
  const runlog = document.getElementById("runlog");
  const guiState = snapshotGuiState();
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Reloading...";
  }
  try {
    const { r, j } = await fetchJson("/api/reload-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!r.ok) {
      if (runlog) runlog.textContent = `HTTP ${r.status}\n\n${formatRunLog(j)}`;
      return;
    }
    await loadOptions();
    restoreGuiState(guiState);
    updateConfigBanner(j.config_status || null);
    CACHE_BUST = Date.now().toString();
    await refreshAll();
    if (runlog) {
      const setupPath = stripRootFromText((j.config_status || {}).setup_path || "");
      runlog.textContent = `Reloaded config from disk: ${setupPath}`;
    }
  } catch (err) {
    if (runlog) runlog.textContent = `Reload failed: ${err}`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Reload Config";
    }
  }
}

function fillSelect(el, values, enumerateOptions = false, defaultIndex = 0) {
  const prev = el.value;
  el.innerHTML = "";
  values.forEach((v, idx) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = enumerateOptions ? `${idx + 1}. ${v}` : v;
    el.appendChild(o);
  });
  if (values.includes(prev)) {
    el.value = prev;
    return;
  }
  if (!values.length) return;
  const idx = Math.max(0, Math.min(defaultIndex, values.length - 1));
  el.value = values[idx];
}

function fillStepSelect(el, values, stepDisplayNames = {}, stepEnumerate = {}) {
  const prev = el.value;
  el.innerHTML = "";
  values.forEach((v, idx) => {
    const o = document.createElement("option");
    o.value = v;
    const base = stepDisplayNames[v] || v;
    o.textContent = stepEnumerate[v] ? `${idx + 1}. ${base}` : base;
    el.appendChild(o);
  });
  if (values.includes(prev)) el.value = prev;
}

async function refreshSelected() {
  if (ACTIVE_TAB === "runner" && hasAllSelection()) {
    document.getElementById("selected_cmd_paste").textContent = "ALL selected: use Runner Plan for expanded combinations.";
    return;
  }
  const { j } = await fetchJson("/api/resolve?" + qs());
  const view = stripRootInObject(j);
  document.getElementById("selected_cmd_paste").textContent = view.cmd_paste || view.cmd_str || "";
}

async function copyTextFrom(elId) {
  const txt = document.getElementById(elId).textContent || "";
  if (!txt.trim()) return;
  await copyText(txt);
}

async function copyText(txt) {
  if (!txt || !txt.trim()) return;
  try {
    await navigator.clipboard.writeText(txt);
  } catch (_) {
    // ignore clipboard failures (manual copy still possible)
  }
}

async function refreshPipeline() {
  if (ACTIVE_TAB === "runner" && hasAllSelection()) {
    const tb = document.querySelector("#tbl tbody");
    tb.innerHTML = "";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4">ALL selected: per-combination steps shown in Runner Plan.</td>`;
    tb.appendChild(tr);
    return;
  }
  const { j } = await fetchJson("/api/pipeline?" + qs());
  const tb = document.querySelector("#tbl tbody");
  tb.innerHTML = "";
  (j.steps || []).forEach(s => {
    const tr = document.createElement("tr");
    const stepTd = document.createElement("td");
    const stepWrap = document.createElement("div");
    stepWrap.className = "step-cell";
    if (s.cmd_str) {
      const copyBtn = document.createElement("button");
      copyBtn.className = "step-copy";
      copyBtn.textContent = "Copy";
      copyBtn.title = "Copy exact run command for this step";
      copyBtn.addEventListener("click", () => copyText(s.cmd_str));
      stepWrap.appendChild(copyBtn);
    }
    const stepName = document.createElement("span");
    stepName.className = "mono";
    stepName.textContent = s.step || "";
    stepWrap.appendChild(stepName);
    stepTd.appendChild(stepWrap);

    const scriptTd = document.createElement("td");
    scriptTd.className = "path";
    scriptTd.textContent = stripRootFromText(s.script || "");
    const inputTd = document.createElement("td");
    inputTd.className = "path";
    inputTd.textContent = stripRootFromText(s.input || "");
    const outputTd = document.createElement("td");
    outputTd.className = "path";
    outputTd.textContent = stripRootFromText(s.output || "");

    tr.appendChild(stepTd);
    tr.appendChild(scriptTd);
    tr.appendChild(inputTd);
    tr.appendChild(outputTd);
    tb.appendChild(tr);
  });
}

async function refreshAll() {
  await refreshConfigStatus();
  await refreshSelected();
  await refreshPipeline();
  await refreshTree();
  await refreshRunnerPlan();
}

function statusClass(s) {
  if (s === "running") return "status-running";
  if (s === "success") return "status-success";
  if (s === "failed") return "status-failed";
  return "status-never";
}

async function refreshRunnerPlan() {
  const tb = document.querySelector("#runner_plan_tbl tbody");
  if (!tb) return;
  tb.innerHTML = "";
  const { j } = await fetchJson("/api/runner-plan?" + qs());
  if (j.error) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6">error: ${j.error}</td>`;
    tb.appendChild(tr);
    return;
  }
  (j.pipelines || []).forEach((p, idx) => {
    const rowId = `pipeline_${idx}`;
    const logs = [];
    if (p.out_log) logs.push(`<a href="${stripRootFromText(p.out_log)}" target="_blank">out</a>`);
    if (p.err_log) logs.push(`<a href="${stripRootFromText(p.err_log)}" target="_blank">err</a>`);
    const g = document.createElement("tr");
    g.className = "group-row";
    g.innerHTML = `
      <td><button type="button" data-row="${rowId}">+</button></td>
      <td>${p.pipeline_id || ""}</td>
      <td><span class="mono">${p.context_label || ""}</span></td>
      <td><span class="status-dot ${statusClass(p.status)}"></span>${p.status}</td>
      <td>${p.last_run_at || p.live_elapsed || ""}</td>
      <td>${logs.join(" ")}</td>
    `;
    tb.appendChild(g);

    (p.steps || []).forEach((s) => {
      const tr = document.createElement("tr");
      tr.dataset.parent = rowId;
      tr.style.display = "none";

      const td0 = document.createElement("td");
      const td1 = document.createElement("td");
      const td2 = document.createElement("td");
      const td3 = document.createElement("td");
      const td4 = document.createElement("td");
      const td5 = document.createElement("td");

      const stepSpan = document.createElement("span");
      stepSpan.className = "mono step-copy-link";
      stepSpan.textContent = `- ${s.step || ""}`;
      if (s.cmd_str) stepSpan.title = s.cmd_str;
      if (s.cmd_str) {
        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "step-copy-mini";
        copyBtn.textContent = "copy";
        copyBtn.title = s.cmd_str;
        copyBtn.setAttribute("data-copy-cmd", encodeURIComponent(s.cmd_str));
        td2.appendChild(copyBtn);
      }
      td2.appendChild(stepSpan);

      td3.innerHTML = `<span class="status-dot ${statusClass(s.status)}"></span>${s.status || "never"}`;
      td4.textContent = s.last_run_at || s.live_elapsed || "";

      tr.appendChild(td0);
      tr.appendChild(td1);
      tr.appendChild(td2);
      tr.appendChild(td3);
      tr.appendChild(td4);
      tr.appendChild(td5);
      tb.appendChild(tr);
    });
  });

  tb.querySelectorAll("button[data-row]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const rowId = btn.dataset.row;
      const rows = tb.querySelectorAll(`tr[data-parent="${rowId}"]`);
      const open = btn.textContent === "-";
      rows.forEach((r) => { r.style.display = open ? "none" : ""; });
      btn.textContent = open ? "+" : "-";
    });
  });
  tb.querySelectorAll("[data-copy-cmd]").forEach((el) => {
    el.addEventListener("click", () => {
      const raw = el.getAttribute("data-copy-cmd") || "";
      copyText(decodeURIComponent(raw));
    });
  });
}

async function runApi(path) {
  const runlog = document.getElementById("runlog");
  await refreshConfigStatus();
  if (CONFIG_STATUS && CONFIG_STATUS.changed) {
    runlog.textContent = "Configuration changed on disk. Reload the config before starting a run.";
    return;
  }
  if (path.includes("run-pipeline") && ACTIVE_TAB !== "runner") {
    runlog.textContent = "Run Pipeline(s) is only enabled in Runner tab.";
    return;
  }
  LAST_RUN_LOG = null;
  runlog.textContent = path.includes("run-pipeline") ? "Running pipeline..." : "Running selected step...";
  try {
    const { r, j } = await fetchJson(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(new URLSearchParams(qs()))),
    });
    if (!r.ok) {
      runlog.textContent = `HTTP ${r.status}\n\n${formatRunLog(j)}`;
      return;
    }
    if (!j.job_id) {
      LAST_RUN_LOG = j;
      runlog.textContent = formatRunLog(LAST_RUN_LOG);
      return;
    }
    CURRENT_JOB_ID = j.job_id;
    await pollJob(j.job_id);
  } catch (err) {
    runlog.textContent = `Request failed: ${err}`;
  }
}

async function purgeOutputTree() {
  const runlog = document.getElementById("runlog");
  const ok = confirm("Delete selected output directory and recreate it empty?");
  if (!ok) return;
  try {
    const { r, j } = await fetchJson("/api/delete-output", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(new URLSearchParams(qs()))),
    });
    if (!r.ok) {
      runlog.textContent = `HTTP ${r.status}\n\n${formatRunLog(j)}`;
      return;
    }
    runlog.textContent = `purged output: ${stripRootFromText(j.output || "")}`;
    CACHE_BUST = Date.now().toString();
    await refreshTree();
  } catch (err) {
    runlog.textContent = `Purge failed: ${err}`;
  }
}

function sleepMs(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function formatJobStatus(job) {
  if (!job) return "";
  const lines = [
    `job_id: ${job.job_id || ""}`,
    `status: ${job.status || ""}`,
    `kind: ${job.kind || ""}`,
  ];
  if (job.step) lines.push(`step: ${job.step}`);
  if (job.cancel_requested) lines.push(`cancel_requested: true`);
  if (job.started_at) lines.push(`started_at: ${job.started_at}`);
  if (job.live_elapsed) lines.push(`running_for: ${job.live_elapsed}`);
  if (job.ended_at) lines.push(`ended_at: ${job.ended_at}`);
  if (job.live_log) lines.push("", "live output:", stripRootFromText(job.live_log));
  if (job.result) lines.push("", "result:", formatRunLog(stripRootInObject(job.result)));
  if (job.error) lines.push("", `error: ${job.error}`);
  return lines.join("\\n");
}

async function cancelCurrentJob() {
  const runlog = document.getElementById("runlog");
  if (!CURRENT_JOB_ID) {
    runlog.textContent = "No running job to cancel.";
    return;
  }
  try {
    const { r, j } = await fetchJson("/api/run-cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: CURRENT_JOB_ID }),
    });
    if (!r.ok) {
      runlog.textContent = `HTTP ${r.status}\n\n${formatRunLog(j)}`;
      return;
    }
    runlog.textContent = formatJobStatus(j);
  } catch (err) {
    runlog.textContent = `Cancel failed: ${err}`;
  }
}

async function pollJob(jobId) {
  const runlog = document.getElementById("runlog");
  while (true) {
    const { r, j } = await fetchJson(`/api/run-status?job_id=${encodeURIComponent(jobId)}&_ts=${Date.now()}`);
    const statusText = formatJobStatus(j);
    runlog.textContent = statusText;
    if (!r.ok || (j.status !== "queued" && j.status !== "running")) {
      CURRENT_JOB_ID = null;
      if (j.result) LAST_RUN_LOG = j.result;
      CACHE_BUST = Date.now().toString();
      await refreshAll();
      if (LAST_RUN_LOG !== null) {
        runlog.textContent = `${statusText}\n\n=== final result ===\n${formatRunLog(LAST_RUN_LOG)}`;
      }
      break;
    }
    await sleepMs(1000);
  }
}

function renderOneRun(x) {
  if (!x) return "";
  const cmd = Array.isArray(x.cmd) ? stripRootFromText(x.cmd.join(" ")) : "";
  const lines = [
    `step: ${x.step || ""}`,
    `pipeline_id: ${x.pipeline_id || ""}`,
    `script: ${stripRootFromText(x.script || "")}`,
    `returncode: ${x.returncode ?? ""}`,
    `cmd: ${cmd}`,
  ];
  if (x.out_log) lines.push(`out_log: ${stripRootFromText(x.out_log)}`);
  if (x.err_log) lines.push(`err_log: ${stripRootFromText(x.err_log)}`);
  lines.push("", "stdout:", stripRootFromText(x.stdout || ""), "", "stderr:", stripRootFromText(x.stderr || ""));
  return lines.join("\\n");
}

function formatRunLog(j) {
  if (!j) return "";
  if (j.error) return `error: ${j.error}`;
  if (Array.isArray(j.runs)) {
    return j.runs.map((run, i) => {
      const head = `=== pipeline ${i + 1} (${run.pipeline_id || ""}) status=${run.status || ""} ===`;
      const ctx = run.context ? JSON.stringify(stripRootInObject(run.context), null, 2) : "";
      const stepLogs = Array.isArray(run.results)
        ? run.results.map((x, k) => `--- step ${k + 1} ---\\n${renderOneRun(x)}`).join("\\n\\n")
        : "";
      return [head, ctx, stepLogs].filter(Boolean).join("\\n\\n");
    }).join("\\n\\n");
  }
  if (Array.isArray(j.results)) {
    const blocks = j.results.map((x, i) => `=== run ${i + 1} ===\\n${renderOneRun(x)}`);
    return blocks.join("\\n\\n");
  }
  return renderOneRun(j);
}

async function refreshTree() {
  if (ACTIVE_TAB === "runner" && hasAllSelection()) {
    document.getElementById("input_treeview").textContent = "ALL selected: input tree is combination-dependent.";
    document.getElementById("output_treeview").textContent = "ALL selected: output tree is combination-dependent.";
    return;
  }
  const { j } = await fetchJson("/api/tree?" + qs());
  if (j.error) {
    document.getElementById("input_treeview").textContent = `error: ${j.error}`;
    document.getElementById("output_treeview").textContent = `error: ${j.error}`;
    return;
  }
  document.getElementById("input_treeview").textContent = stripRootFromText(j.input_tree || "");
  document.getElementById("output_treeview").textContent = stripRootFromText(j.output_tree || "");
}

async function loadOptions() {
  const { j } = await fetchJson("/api/options?_ts=" + Date.now());
  OPTIONS = j;
  VAR_DISPLAY_NAMES = OPTIONS.variation_display_names || {};
  VAR_ENUMERATE_FLAGS = OPTIONS.variation_enumerate || {};
  fillStepSelect(
    document.getElementById("step"),
    OPTIONS.steps || [],
    OPTIONS.step_display_names || {},
    OPTIONS.step_enumerate || {},
  );
}

function snapshotGuiState() {
  const variationValues = {};
  VAR_QUERY_KEYS.forEach((key) => {
    const el = document.getElementById(`vp_${key}`);
    if (el) variationValues[key] = el.value;
  });
  return {
    activeTab: ACTIVE_TAB,
    step: document.getElementById("step").value,
    experimentName: document.getElementById("experiment_name").value,
    variationValues,
  };
}

function restoreGuiState(state) {
  if (!state) return;

  const stepEl = document.getElementById("step");
  if (stepEl && Array.from(stepEl.options).some((opt) => opt.value === state.step)) {
    stepEl.value = state.step;
  }

  const experimentEl = document.getElementById("experiment_name");
  if (experimentEl) experimentEl.value = state.experimentName || "";

  setTab(state.activeTab || ACTIVE_TAB);
  renderVariationPoints();

  Object.entries(state.variationValues || {}).forEach(([key, value]) => {
    const el = document.getElementById(`vp_${key}`);
    if (!el || !Array.from(el.options).some((opt) => opt.value === value)) return;
    el.value = value;
    if (!key.endsWith("_variant")) {
      el.dispatchEvent(new Event("change"));
    }
  });
}

async function init() {
  await loadOptions();
  ["viewer", "runner", "analysis"].forEach(t => {
    const b = document.getElementById(`tab_${t}`);
    if (b) b.addEventListener("click", async () => {
      setTab(t);
      renderVariationPoints();
      if (t !== "analysis") await refreshAll();
    });
  });
  setTab(ACTIVE_TAB);
  renderVariationPoints();
  await refreshAll();

  ["step", "experiment_name"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", refreshAll);
  });
  document.getElementById("hide_root").addEventListener("change", async () => {
    await refreshAll();
    if (LAST_RUN_LOG !== null) {
      document.getElementById("runlog").textContent = formatRunLog(LAST_RUN_LOG);
    }
  });
  document.getElementById("refresh").addEventListener("click", async () => {
    CACHE_BUST = Date.now().toString();
    LAST_RUN_LOG = null;
    document.getElementById("runlog").textContent = "";
    await refreshAll();
  });
  document.getElementById("copy_cmd_paste_btn").addEventListener("click", () => copyTextFrom("selected_cmd_paste"));
  document.getElementById("run_step").addEventListener("click", () => runApi("/api/run-step"));
  document.getElementById("run_pipeline").addEventListener("click", () => runApi("/api/run-pipeline"));
  document.getElementById("cancel_job").addEventListener("click", cancelCurrentJob);
  document.getElementById("purge_output_btn").addEventListener("click", purgeOutputTree);
  document.getElementById("reload_config_btn").addEventListener("click", reloadConfigFromDisk);
  CONFIG_STATUS_TIMER = window.setInterval(refreshConfigStatus, 3000);
  await refreshConfigStatus();
}
init();
</script>
</body>
</html>
"""


def build_html(*, title: str, default_tab: str) -> str:
    return (
        HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__DEFAULT_TAB__", default_tab)
    )


def make_handler(
    layout: PipelineLayout,
    default_experiment_name: str,
    runner_python: str,
    lama_python_bin: str,
    *,
    html: str,
):
    setup_path = layout.setup_path.resolve()
    loaded_setup_mtime_ns = setup_path.stat().st_mtime_ns if setup_path.exists() else None
    log_dir = layout.root_dir / "pipeline" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()
    next_job_id = 1
    root_cmd_prefix = f"cd {shlex.quote(str(layout.root_dir))} && "

    def iso_for_mtime_ns(mtime_ns: int | None) -> str:
        if mtime_ns is None:
            return ""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime_ns / 1_000_000_000))

    def current_config_status() -> dict[str, object]:
        disk_mtime_ns = setup_path.stat().st_mtime_ns if setup_path.exists() else None
        changed = disk_mtime_ns != loaded_setup_mtime_ns
        return {
            "setup_path": str(setup_path),
            "loaded_mtime": iso_for_mtime_ns(loaded_setup_mtime_ns),
            "disk_mtime": iso_for_mtime_ns(disk_mtime_ns),
            "changed": changed,
        }

    def reload_layout_from_disk() -> dict[str, object]:
        nonlocal layout, loaded_setup_mtime_ns
        layout = PipelineLayout(layout.root_dir)
        loaded_setup_mtime_ns = setup_path.stat().st_mtime_ns if setup_path.exists() else None
        return current_config_status()

    def now_iso() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def now_compact() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def parse_iso(ts: str | None) -> float | None:
        if not ts:
            return None
        try:
            return time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        except Exception:  # noqa: BLE001
            return None

    def elapsed_label(started_at: str | None) -> str:
        t0 = parse_iso(started_at)
        if t0 is None:
            return ""
        sec = max(0, int(time.time() - t0))
        mm, ss = divmod(sec, 60)
        hh, mm = divmod(mm, 60)
        if hh > 0:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        return f"{mm:02d}:{ss:02d}"

    def trim_text(text: str, max_chars: int = 50000) -> str:
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def parse_payload(handler: BaseHTTPRequestHandler) -> dict:
        length = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def create_job(kind: str, step: str | None = None, pipeline_id: str | None = None) -> str:
        nonlocal next_job_id
        with jobs_lock:
            job_id = str(next_job_id)
            next_job_id += 1
            jobs[job_id] = {
                "job_id": job_id,
                "kind": kind,
                "step": step,
                "pipeline_id": pipeline_id,
                "status": "queued",
                "cancel_requested": False,
                "live_log": "",
                "started_at": now_iso(),
                "ended_at": None,
                "result": None,
                "error": None,
                "_proc": None,
                "_active_ctx": {},
            }
        return job_id

    def ctx_to_dict(ctx: PipelineContext) -> dict:
        return {
            "dataset_name": ctx.dataset_name,
            "dataset_variant": ctx.dataset_variant,
            "detection_algorithm": ctx.detection_algorithm,
            "tiling": ctx.tiling,
            "experiment_name": ctx.experiment_name,
            "repair_method": ctx.repair_method,
            "mask_type": ctx.mask_type,
            "extra_vars": dict(ctx.extra_vars or {}),
        }

    def context_fingerprint(ctx: PipelineContext) -> str:
        payload = json.dumps(ctx_to_dict(ctx), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def generate_pipeline_id(ctx: PipelineContext, ts: str, index: int) -> str:
        # Tie ID to both selection and run time while keeping it compact and sortable.
        material = f"{context_fingerprint(ctx)}|{ts}|{index}"
        h = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
        return f"p_{ts}_{h}"

    def read_latest_run_for_fingerprint(fingerprint: str) -> dict | None:
        candidates = []
        for p in log_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if data.get("context_fingerprint") != fingerprint:
                continue
            candidates.append(data)
        if not candidates:
            return None
        candidates.sort(key=lambda d: d.get("ended_at", d.get("started_at", "")), reverse=True)
        return candidates[0]

    def context_label(ctx: PipelineContext) -> str:
        extras = ",".join(f"{k}={v}" for k, v in sorted((ctx.extra_vars or {}).items()))
        return (
            f"{ctx.detection_algorithm} | {ctx.dataset_name}/{ctx.dataset_variant} | "
            f"{ctx.repair_method}/{ctx.mask_type} | {ctx.tiling}" + (f" | {extras}" if extras else "")
        )

    def _expand_flat(selected: str, values: set[str], key: str) -> list[str]:
        allowed = sorted(values)
        if selected == "ALL":
            return allowed
        if selected not in values:
            raise ValueError(f"{key} must be one of {allowed} or ALL")
        return [selected]

    def expand_contexts(payload: dict) -> list[PipelineContext]:
        dataset_sel = str(payload.get("dataset_name", "")).strip()
        dataset_variant_sel = str(payload.get("dataset_variant", "")).strip()
        algo_sel = str(payload.get("detection_algorithm", "")).strip()
        tiling_sel = str(payload.get("tiling", "")).strip()
        repair_sel = str(payload.get("repair_method", "")).strip()
        mask_sel = str(payload.get("mask_type", "")).strip()
        experiment_name = str(payload.get("experiment_name", default_experiment_name)).strip() or default_experiment_name

        datasets = layout.get_valid_datasets()
        if dataset_sel == "ALL":
            dataset_names = sorted(datasets.keys())
        elif dataset_sel in datasets:
            dataset_names = [dataset_sel]
        else:
            raise ValueError(f"dataset_name must be one of {sorted(datasets)} or ALL")

        dataset_pairs: list[tuple[str, str]] = []
        for ds in dataset_names:
            variants = sorted(datasets[ds])
            if dataset_variant_sel == "ALL":
                chosen = variants
            elif dataset_variant_sel in datasets[ds]:
                chosen = [dataset_variant_sel]
            else:
                if dataset_sel == "ALL":
                    continue
                raise ValueError(f"dataset_variant must be one of {variants} or ALL for dataset={ds}")
            for dv in chosen:
                dataset_pairs.append((ds, dv))
        if not dataset_pairs:
            raise ValueError("No dataset/dataset_variant combinations matched selection")

        algos = _expand_flat(algo_sel, layout.get_valid_algos(), "detection_algorithm")
        tilings = _expand_flat(tiling_sel, layout.get_valid_tilings(), "tiling")
        repairs = _expand_flat(repair_sel, layout.get_valid_repair_methods(), "repair_method")
        masks = _expand_flat(mask_sel, layout.get_valid_mask_types(), "mask_type")

        extra_flat_values = layout.get_extra_variation_points()
        extra_flat_keys = sorted(extra_flat_values.keys())
        extra_flat_expanded: dict[str, list[str]] = {}
        for key in extra_flat_keys:
            sel = str(payload.get(key, "")).strip()
            extra_flat_expanded[key] = _expand_flat(sel, extra_flat_values[key], key)

        extra_hier_values = layout.get_extra_hier_variation_points()
        extra_hier_keys = sorted(extra_hier_values.keys())
        extra_hier_expanded: dict[str, list[tuple[str, str]]] = {}
        for key in extra_hier_keys:
            parent_sel = str(payload.get(key, "")).strip()
            child_key = f"{key}_variant"
            child_sel = str(payload.get(child_key, "")).strip()
            parent_map = extra_hier_values[key]
            if parent_sel == "ALL":
                parents = sorted(parent_map.keys())
            elif parent_sel in parent_map:
                parents = [parent_sel]
            else:
                raise ValueError(f"{key} must be one of {sorted(parent_map)} or ALL")

            pairs: list[tuple[str, str]] = []
            for parent in parents:
                variants = sorted(parent_map[parent])
                if child_sel == "ALL":
                    chosen = variants
                elif child_sel in parent_map[parent]:
                    chosen = [child_sel]
                else:
                    if parent_sel == "ALL":
                        continue
                    raise ValueError(f"{child_key} must be one of {variants} or ALL for {key}={parent}")
                for variant in chosen:
                    pairs.append((parent, variant))
            if not pairs:
                raise ValueError(f"No combinations matched for {key}/{child_key}")
            extra_hier_expanded[key] = pairs

        dims: list[tuple[str, list]] = [
            ("dataset_pair", dataset_pairs),
            ("detection_algorithm", algos),
            ("tiling", tilings),
            ("repair_method", repairs),
            ("mask_type", masks),
        ]
        for key in extra_flat_keys:
            dims.append((f"flat:{key}", extra_flat_expanded[key]))
        for key in extra_hier_keys:
            dims.append((f"hier:{key}", extra_hier_expanded[key]))

        contexts: list[PipelineContext] = []
        seen = set()
        for combo in itertools.product(*[values for _, values in dims]):
            picked = {name: value for (name, _), value in zip(dims, combo)}
            dataset_name, dataset_variant = picked["dataset_pair"]
            extra_vars: dict[str, str] = {}
            for key in extra_flat_keys:
                extra_vars[key] = str(picked[f"flat:{key}"])
            for key in extra_hier_keys:
                parent, variant = picked[f"hier:{key}"]
                extra_vars[key] = str(parent)
                extra_vars[f"{key}_variant"] = str(variant)

            ctx = PipelineContext(
                dataset_name=dataset_name,
                dataset_variant=dataset_variant,
                detection_algorithm=str(picked["detection_algorithm"]),
                tiling=str(picked["tiling"]),
                experiment_name=experiment_name,
                repair_method=str(picked["repair_method"]),
                mask_type=str(picked["mask_type"]),
                extra_vars=extra_vars,
            )
            layout.validate_ctx(ctx)
            fp = context_fingerprint(ctx)
            if fp in seen:
                continue
            seen.add(fp)
            contexts.append(ctx)
        return contexts

    def update_job(job_id: str, **fields) -> None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            job.update(fields)

    def append_job_log(job_id: str, text: str) -> None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            job["live_log"] = trim_text(job.get("live_log", "") + text)

    def get_job(job_id: str) -> dict | None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return None
            payload = {k: v for k, v in job.items() if not k.startswith("_")}
            if str(payload.get("status")) in {"queued", "running"}:
                payload["live_elapsed"] = elapsed_label(str(payload.get("started_at") or ""))
            return payload

    def set_active_ctx(job_id: str, fp: str, data: dict | None) -> None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            active = job.setdefault("_active_ctx", {})
            if not isinstance(active, dict):
                active = {}
                job["_active_ctx"] = active
            if data is None:
                active.pop(fp, None)
            else:
                active[fp] = data

    def collect_live_ctx_map() -> dict[str, dict]:
        out: dict[str, dict] = {}
        with jobs_lock:
            for job in jobs.values():
                if job.get("kind") != "run-pipeline":
                    continue
                if job.get("status") not in {"queued", "running"}:
                    continue
                active = job.get("_active_ctx", {})
                if not isinstance(active, dict):
                    continue
                for fp, data in active.items():
                    if not isinstance(data, dict):
                        continue
                    merged = dict(data)
                    merged["live_elapsed"] = elapsed_label(str(data.get("started_at", "")))
                    out[str(fp)] = merged
        return out

    def get_running_pipeline_summary() -> dict | None:
        with jobs_lock:
            running = [
                j for j in jobs.values()
                if j.get("kind") == "run-pipeline" and j.get("status") in {"queued", "running"}
            ]
            if len(running) != 1:
                return None
            job = running[0]
            live_log = str(job.get("live_log", "") or "")
            current_step = ""
            for m in re.finditer(r"running\\s+([A-Za-z0-9_]+)", live_log):
                current_step = m.group(1)
            return {
                "pipeline_id": str(job.get("pipeline_id") or ""),
                "started_at": str(job.get("started_at") or ""),
                "live_elapsed": elapsed_label(str(job.get("started_at") or "")),
                "current_step": current_step,
            }

    def set_job_proc(job_id: str, proc: subprocess.Popen | None) -> None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            job["_proc"] = proc

    def is_cancel_requested(job_id: str) -> bool:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return False
            return bool(job.get("cancel_requested"))

    def request_cancel(job_id: str) -> dict | None:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return None
            job["cancel_requested"] = True
            proc = job.get("_proc")
            status = job.get("status")
        if status == "queued":
            update_job(job_id, status="cancelled", ended_at=now_iso())
            return get_job(job_id)
        if status == "running" and proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception as exc:  # noqa: BLE001
                append_job_log(job_id, f"[cancel] group terminate failed: {exc}\n")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception as kill_exc:  # noqa: BLE001
                    append_job_log(job_id, f"[cancel] group kill failed: {kill_exc}\n")
        return get_job(job_id)

    def run_step_cmd(
        ctx: PipelineContext,
        step_name: str,
        payload: dict,
        *,
        pipeline_id: str | None = None,
        out_log_path: Path | None = None,
        err_log_path: Path | None = None,
    ) -> tuple[list[str], dict[str, str], Path, Path, Path]:
        script = layout.get_script_by_name(ctx=ctx, step_name=step_name)
        if not script.exists():
            raise FileNotFoundError(f"Script not found: {script}")
        input_dir = layout.get_input_dir_by_name(step_name, ctx)
        output_dir = layout.get_output_dir_by_name(step_name, ctx)
        preview_out_log = output_dir / "log.stdout"
        preview_err_log = output_dir / "log.stderr"
        eff_out_log = out_log_path if out_log_path is not None else preview_out_log
        eff_err_log = err_log_path if err_log_path is not None else preview_err_log
        step_cfg = layout.get_expanded_process_step(step_name, ctx)
        step_runtime = step_cfg.get("runtime", {}) if isinstance(step_cfg, dict) else {}
        conda_env = ""
        if isinstance(step_runtime, dict):
            conda_env = str(step_runtime.get("conda_env", "")).strip()
        contract = {
            "variation_points": layout.get_selected_variation_points(ctx),
            "process_step": {
                "name": step_name,
                **step_cfg,
            },
            "resolved": {
                "input": str(input_dir),
                "output": str(output_dir),
            },
            "runtime": {
                "pipeline_id": pipeline_id or "",
                "log_out": str(eff_out_log),
                "log_err": str(eff_err_log),
                "conda_env": conda_env,
                "runner_python": runner_python,
                "lama_python_bin": lama_python_bin,
            },
        }
        contract_json = json.dumps(contract, separators=(",", ":"))
        if conda_env:
            cmd = ["conda", "run", "-n", conda_env, "python", str(script), "--contract-json", contract_json]
        else:
            cmd = [runner_python, str(script), "--contract-json", contract_json]
        cmd.extend(build_extra_args(step_cfg if isinstance(step_cfg, dict) else {}))
        return cmd, {}, script, preview_out_log, preview_err_log

    def execute_step_live(
        ctx: PipelineContext,
        step_name: str,
        payload: dict,
        job_id: str,
        pipeline_id: str | None = None,
        out_log_path: Path | None = None,
        err_log_path: Path | None = None,
        out_writer=None,
        err_writer=None,
    ) -> dict:
        cmd, env_overrides, script, step_out_log, step_err_log = run_step_cmd(
            ctx,
            step_name,
            payload,
            pipeline_id=pipeline_id,
            out_log_path=out_log_path,
            err_log_path=err_log_path,
        )
        step_out_log.parent.mkdir(parents=True, exist_ok=True)
        step_err_log.parent.mkdir(parents=True, exist_ok=True)
        with step_out_log.open("a", encoding="utf-8"):
            pass
        with step_err_log.open("a", encoding="utf-8"):
            pass
        merged_env = None if not env_overrides else env_overrides
        proc = subprocess.Popen(
            cmd,
            cwd=str(layout.root_dir),
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            start_new_session=True,
        )
        set_job_proc(job_id, proc)
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def read_stream(stream, sink: list[str], tag: str) -> None:
            if stream is None:
                return
            for line in iter(stream.readline, ""):
                sink.append(line)
                append_job_log(job_id, f"[{step_name} {tag}] {line}")
                if tag == "stdout":
                    with step_out_log.open("a", encoding="utf-8") as f:
                        f.write(line)
                else:
                    with step_err_log.open("a", encoding="utf-8") as f:
                        f.write(line)
                if tag == "stdout" and out_writer is not None:
                    out_writer(step_name, line)
                if tag == "stderr" and err_writer is not None:
                    err_writer(step_name, line)
            stream.close()

        t_out = threading.Thread(target=read_stream, args=(proc.stdout, stdout_chunks, "stdout"), daemon=True)
        t_err = threading.Thread(target=read_stream, args=(proc.stderr, stderr_chunks, "stderr"), daemon=True)
        t_out.start()
        t_err.start()
        returncode = proc.wait()
        t_out.join()
        t_err.join()
        set_job_proc(job_id, None)

        return {
            "step": step_name,
            "script": str(script),
            "cmd": cmd,
            "returncode": returncode,
            "pipeline_id": pipeline_id or "",
            "out_log": str(out_log_path) if out_log_path is not None else "",
            "err_log": str(err_log_path) if err_log_path is not None else "",
            "step_out_log": str(step_out_log),
            "step_err_log": str(step_err_log),
            "stdout": trim_text("".join(stdout_chunks), max_chars=20000),
            "stderr": trim_text("".join(stderr_chunks), max_chars=20000),
        }

    def run_step_worker(job_id: str, ctx: PipelineContext, step_name: str, payload: dict) -> None:
        if is_cancel_requested(job_id):
            update_job(job_id, status="cancelled", ended_at=now_iso())
            return
        update_job(job_id, status="running")
        try:
            run_stamp = now_compact()
            pipeline_id = generate_pipeline_id(ctx, run_stamp, 1)
            out_log = log_dir / f"{pipeline_id}.out"
            err_log = log_dir / f"{pipeline_id}.err"
            out_log.touch(exist_ok=True)
            err_log.touch(exist_ok=True)
            io_lock = threading.Lock()

            def write_out(step: str, line: str) -> None:
                with io_lock:
                    with out_log.open("a", encoding="utf-8") as f:
                        f.write(f"[{step}] {line}")

            def write_err(step: str, line: str) -> None:
                with io_lock:
                    with err_log.open("a", encoding="utf-8") as f:
                        f.write(f"[{step}] {line}")

            result = execute_step_live(
                ctx,
                step_name,
                payload,
                job_id=job_id,
                pipeline_id=pipeline_id,
                out_log_path=out_log,
                err_log_path=err_log,
                out_writer=write_out,
                err_writer=write_err,
            )
            if is_cancel_requested(job_id):
                status = "cancelled"
            else:
                status = "completed" if result["returncode"] == 0 else "failed"
            update_job(job_id, status=status, result=result, ended_at=now_iso())
        except Exception as exc:  # noqa: BLE001
            append_job_log(job_id, f"[error] {exc}\n")
            status = "cancelled" if is_cancel_requested(job_id) else "failed"
            update_job(job_id, status=status, error=str(exc), ended_at=now_iso())

    def run_pipeline_worker(job_id: str, contexts: list[PipelineContext], payload: dict) -> None:
        if is_cancel_requested(job_id):
            update_job(job_id, status="cancelled", ended_at=now_iso())
            return
        update_job(job_id, status="running")
        runs: list[dict] = []
        overall_ok = True
        try:
            for idx, ctx in enumerate(contexts, start=1):
                if is_cancel_requested(job_id):
                    update_job(job_id, status="cancelled", result={"ok": False, "runs": runs}, ended_at=now_iso())
                    return
                context_fp = context_fingerprint(ctx)
                run_stamp = now_compact()
                pipeline_id = generate_pipeline_id(ctx, run_stamp, idx)
                out_log = log_dir / f"{pipeline_id}.out"
                err_log = log_dir / f"{pipeline_id}.err"
                out_log.touch(exist_ok=True)
                err_log.touch(exist_ok=True)
                io_lock = threading.Lock()
                started_at = now_iso()
                results = []
                step_statuses: list[dict] = []
                append_job_log(job_id, f"[pipeline {idx}/{len(contexts)}] {context_label(ctx)}\n")

                def write_out(step: str, line: str) -> None:
                    with io_lock:
                        with out_log.open("a", encoding="utf-8") as f:
                            f.write(f"[{step}] {line}")

                def write_err(step: str, line: str) -> None:
                    with io_lock:
                        with err_log.open("a", encoding="utf-8") as f:
                            f.write(f"[{step}] {line}")

                run_status = "completed"
                completed_steps: list[str] = []
                set_active_ctx(
                    job_id,
                    context_fp,
                    {
                        "status": "running",
                        "pipeline_id": pipeline_id,
                        "started_at": started_at,
                        "current_step": "",
                        "completed_steps": completed_steps,
                    },
                )
                for step_name in layout.get_step_names():
                    if is_cancel_requested(job_id):
                        run_status = "cancelled"
                        break
                    append_job_log(job_id, f"[pipeline {pipeline_id}] running {step_name}\n")
                    step_started = now_iso()
                    set_active_ctx(
                        job_id,
                        context_fp,
                        {
                            "status": "running",
                            "pipeline_id": pipeline_id,
                            "started_at": started_at,
                            "current_step": step_name,
                            "step_started_at": step_started,
                            "completed_steps": list(completed_steps),
                        },
                    )
                    result = execute_step_live(
                        ctx,
                        step_name,
                        payload,
                        job_id=job_id,
                        pipeline_id=pipeline_id,
                        out_log_path=out_log,
                        err_log_path=err_log,
                        out_writer=write_out,
                        err_writer=write_err,
                    )
                    results.append(result)
                    step_statuses.append(
                        {
                            "step": step_name,
                            "status": "success" if result["returncode"] == 0 else "failed",
                            "started_at": step_started,
                            "ended_at": now_iso(),
                            "returncode": result["returncode"],
                        }
                    )
                    if result["returncode"] != 0:
                        run_status = "failed"
                        overall_ok = False
                        break
                    completed_steps.append(step_name)

                metadata = {
                    "pipeline_id": pipeline_id,
                    "started_at": started_at,
                    "ended_at": now_iso(),
                    "status": run_status,
                    "ok": run_status == "completed",
                    "context": ctx_to_dict(ctx),
                    "context_fingerprint": context_fp,
                    "steps": step_statuses,
                    "out_log": str(out_log),
                    "err_log": str(err_log),
                }
                (log_dir / f"{pipeline_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                runs.append(
                    {
                        "pipeline_id": pipeline_id,
                        "status": run_status,
                        "context": ctx_to_dict(ctx),
                        "results": results,
                        "out_log": str(out_log),
                        "err_log": str(err_log),
                    }
                )
                if run_status == "cancelled":
                    set_active_ctx(job_id, context_fp, None)
                    update_job(job_id, status="cancelled", result={"ok": False, "runs": runs}, ended_at=now_iso())
                    return
                set_active_ctx(job_id, context_fp, None)

            status = "completed" if overall_ok else "failed"
            update_job(
                job_id,
                status=status,
                result={"ok": overall_ok, "runs": runs, "combination_count": len(contexts)},
                ended_at=now_iso(),
            )
        except Exception as exc:  # noqa: BLE001
            append_job_log(job_id, f"[error] {exc}\n")
            update_job(job_id, status="failed", error=str(exc), result={"ok": False, "runs": runs}, ended_at=now_iso())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/options":
                valid_datasets = layout.get_valid_datasets()
                extra_hier = layout.get_extra_hier_variation_points()
                variation_points_hier = {
                    "dataset_name": {
                        "child_key": "dataset_variant",
                        "values": {k: sorted(v) for k, v in valid_datasets.items()},
                    },
                }
                for key, parent_map in extra_hier.items():
                    variation_points_hier[key] = {
                        "child_key": f"{key}_variant",
                        "values": {pk: sorted(pv) for pk, pv in parent_map.items()},
                    }
                variation_points_flat = {
                    "algo_name": sorted(layout.get_valid_algos()),
                    "tiling": sorted(layout.get_valid_tilings()),
                    "repair_method": sorted(layout.get_valid_repair_methods()),
                    "mask_type": sorted(layout.get_valid_mask_types()),
                }
                for key, values in layout.get_extra_variation_points().items():
                    variation_points_flat[key] = sorted(values)
                payload = {
                    "steps": layout.get_step_names(),
                    "step_display_names": {
                        k: v.display_name for k, v in layout.get_step_gui_map().items()
                    },
                    "step_enumerate": {
                        k: v.enumerate for k, v in layout.get_step_gui_map().items()
                    },
                    "variation_order": layout.get_variation_point_order(),
                    "variation_display_names": {
                        k: v.display_name for k, v in layout.get_variation_gui_map().items()
                    },
                    "variation_enumerate": {
                        k: v.enumerate for k, v in layout.get_variation_gui_map().items()
                    },
                    "variation_points_flat": variation_points_flat,
                    "variation_points_hier": variation_points_hier,
                    "root_dir": str(layout.root_dir),
                }
                as_json(self, payload)
                return

            if parsed.path == "/api/config-status":
                as_json(self, current_config_status())
                return

            if parsed.path == "/api/resolve":
                try:
                    query = parse_qs(parsed.query)
                    ctx = make_ctx(query, default_experiment_name, layout)
                    layout.validate_ctx(ctx)
                    step_name = query.get("step", [""])[0]
                    payload = {k: (v[0] if v else "") for k, v in query.items()}
                    cmd, _, _, _, _ = run_step_cmd(ctx=ctx, step_name=step_name, payload=payload)
                    cmd_paste = ""
                    if "--contract-json" in cmd:
                        idx = cmd.index("--contract-json")
                        if idx + 1 < len(cmd):
                            contract_json = cmd[idx + 1]
                            cmd_no_contract = cmd[:idx] + cmd[idx + 2 :]
                            contract_file = "/tmp/pipeline_contract.json"
                            cmd_base = " ".join(shlex.quote(c) for c in cmd_no_contract)
                            cmd_paste = (
                                f"cd {shlex.quote(str(layout.root_dir))}\n"
                                f"cat > {contract_file} <<'JSON'\n"
                                f"{contract_json}\n"
                                f"JSON\n"
                                f"{cmd_base} --contract-json \"$(cat {contract_file})\""
                            )
                    payload = {
                        "step": step_name,
                        "script": str(layout.get_script_by_name(ctx=ctx, step_name=step_name)),
                        "input": str(layout.get_input_dir_by_name(step_name, ctx)),
                        "output": str(layout.get_output_dir_by_name(step_name, ctx)),
                        "cmd": cmd,
                        "cmd_str": root_cmd_prefix + " ".join(shlex.quote(c) for c in cmd),
                        "cmd_paste": cmd_paste,
                    }
                    as_json(self, payload)
                except Exception as exc:  # noqa: BLE001
                    as_json(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/pipeline":
                try:
                    query = parse_qs(parsed.query)
                    ctx = make_ctx(query, default_experiment_name, layout)
                    layout.validate_ctx(ctx)
                    payload = {k: (v[0] if v else "") for k, v in query.items()}
                    steps = []
                    for step_name in layout.get_step_names():
                        cmd, _, _, _, _ = run_step_cmd(ctx=ctx, step_name=step_name, payload=payload)
                        steps.append(
                            {
                                "step": step_name,
                                "script": str(layout.get_script_by_name(ctx=ctx, step_name=step_name)),
                                "input": str(layout.get_input_dir_by_name(step_name, ctx)),
                                "output": str(layout.get_output_dir_by_name(step_name, ctx)),
                                "cmd_str": root_cmd_prefix + " ".join(shlex.quote(c) for c in cmd),
                            }
                        )
                    as_json(self, {"steps": steps})
                except Exception as exc:  # noqa: BLE001
                    as_json(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/runner-plan":
                try:
                    query = parse_qs(parsed.query)
                    q = {k: [str(vv[0])] for k, vv in query.items() if vv}
                    payload = {k: v[0] for k, v in q.items()}
                    contexts = expand_contexts(payload)
                    live_map = collect_live_ctx_map()
                    running_summary = get_running_pipeline_summary()
                    pipelines = []
                    for ctx in contexts:
                        ctx_fp = context_fingerprint(ctx)
                        live = live_map.get(ctx_fp)
                        if not live and running_summary and len(contexts) == 1:
                            live = {
                                "status": "running",
                                "pipeline_id": running_summary.get("pipeline_id", ""),
                                "started_at": running_summary.get("started_at", ""),
                                "live_elapsed": running_summary.get("live_elapsed", ""),
                                "current_step": running_summary.get("current_step", ""),
                                "completed_steps": [],
                            }
                        latest = read_latest_run_for_fingerprint(ctx_fp)
                        step_map = {
                            s.get("step", ""): s for s in (latest.get("steps", []) if latest else [])
                        }
                        steps = []
                        for step_name in layout.get_step_names():
                            cmd, _, _, _, _ = run_step_cmd(ctx=ctx, step_name=step_name, payload=payload)
                            step_cmd_str = root_cmd_prefix + " ".join(shlex.quote(c) for c in cmd)
                            if live:
                                completed = set(live.get("completed_steps", []) or [])
                                current = str(live.get("current_step", "") or "")
                                if step_name in completed:
                                    steps.append(
                                        {
                                            "step": step_name,
                                            "status": "success",
                                            "last_run_at": "",
                                            "live_elapsed": "",
                                            "cmd_str": step_cmd_str,
                                        }
                                    )
                                    continue
                                if step_name == current:
                                    steps.append(
                                        {
                                            "step": step_name,
                                            "status": "running",
                                            "last_run_at": "",
                                            "live_elapsed": str(live.get("live_elapsed", "")),
                                            "cmd_str": step_cmd_str,
                                        }
                                    )
                                    continue
                                steps.append(
                                    {
                                        "step": step_name,
                                        "status": "never",
                                        "last_run_at": "",
                                        "live_elapsed": "",
                                        "cmd_str": step_cmd_str,
                                    }
                                )
                                continue
                            step_info = step_map.get(step_name)
                            if not step_info:
                                steps.append(
                                    {
                                        "step": step_name,
                                        "status": "never",
                                        "last_run_at": "",
                                        "live_elapsed": "",
                                        "cmd_str": step_cmd_str,
                                    }
                                )
                            else:
                                steps.append(
                                    {
                                        "step": step_name,
                                        "status": step_info.get("status", "never"),
                                        "last_run_at": step_info.get("ended_at", ""),
                                        "live_elapsed": "",
                                        "cmd_str": step_cmd_str,
                                    }
                                )
                        group_status = "never"
                        live_elapsed = ""
                        pipeline_id = latest.get("pipeline_id", "") if latest else ""
                        last_run_at = latest.get("ended_at", "") if latest else ""
                        if live:
                            group_status = "running"
                            live_elapsed = str(live.get("live_elapsed", ""))
                            pipeline_id = str(live.get("pipeline_id", "") or pipeline_id)
                            last_run_at = ""
                        if latest:
                            if not live:
                                group_status = "success" if latest.get("status") == "completed" else "failed"
                        pipelines.append(
                            {
                                "context_label": context_label(ctx),
                                "status": group_status,
                                "last_run_at": last_run_at,
                                "live_elapsed": live_elapsed,
                                "pipeline_id": pipeline_id,
                                "out_log": latest.get("out_log", "") if latest else "",
                                "err_log": latest.get("err_log", "") if latest else "",
                                "steps": steps,
                            }
                        )
                    as_json(self, {"pipelines": pipelines})
                except Exception as exc:  # noqa: BLE001
                    as_json(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/tree":
                try:
                    query = parse_qs(parsed.query)
                    ctx = make_ctx(query, default_experiment_name, layout)
                    layout.validate_ctx(ctx)
                    step_name = query.get("step", [""])[0]
                    in_dir = layout.get_input_dir_by_name(step_name, ctx)
                    out_dir = layout.get_output_dir_by_name(step_name, ctx)
                    as_json(
                        self,
                        {
                            "step": step_name,
                            "input": str(in_dir),
                            "output": str(out_dir),
                            "input_tree": build_tree_text(in_dir),
                            "output_tree": build_tree_text(out_dir),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    as_json(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            if parsed.path == "/api/run-status":
                query = parse_qs(parsed.query)
                job_id = query.get("job_id", [""])[0]
                if not job_id:
                    as_json(self, {"error": "missing job_id"}, status=HTTPStatus.BAD_REQUEST)
                    return
                job = get_job(job_id)
                if job is None:
                    as_json(self, {"error": f"job not found: {job_id}"}, status=HTTPStatus.NOT_FOUND)
                    return
                as_json(self, job)
                return

            if parsed.path == "/api/delete-output":
                as_json(self, {"error": "method not allowed"}, status=HTTPStatus.METHOD_NOT_ALLOWED)
                return

            as_json(self, {"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/run-step", "/api/run-pipeline", "/api/run-cancel", "/api/delete-output", "/api/reload-config"}:
                as_json(self, {"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                payload = parse_payload(self)
                if parsed.path == "/api/reload-config":
                    status = reload_layout_from_disk()
                    as_json(self, {"status": "reloaded", "config_status": status})
                    return
                if parsed.path == "/api/delete-output":
                    q = {k: [str(v)] for k, v in payload.items()}
                    ctx = make_ctx(q, default_experiment_name, layout)
                    layout.validate_ctx(ctx)
                    step_name = str(payload.get("step", "")).strip()
                    if step_name not in set(layout.get_step_names()):
                        raise ValueError(f"Unknown step: {step_name}")
                    out_dir = layout.get_output_dir_by_name(step_name, ctx)
                    out_dir = out_dir.resolve()
                    allowed_root = (layout.root_dir / "pipeline_data").resolve()
                    if allowed_root not in out_dir.parents and out_dir != allowed_root:
                        raise ValueError(f"Refusing to delete outside pipeline_data: {out_dir}")
                    if out_dir.exists():
                        shutil.rmtree(out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    as_json(self, {"step": step_name, "output": str(out_dir), "status": "purged"})
                    return
                if parsed.path == "/api/run-cancel":
                    job_id = str(payload.get("job_id", "")).strip()
                    if not job_id:
                        as_json(self, {"error": "missing job_id"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    job = request_cancel(job_id)
                    if job is None:
                        as_json(self, {"error": f"job not found: {job_id}"}, status=HTTPStatus.NOT_FOUND)
                        return
                    as_json(self, job)
                    return
                if parsed.path == "/api/run-step":
                    q = {k: [str(v)] for k, v in payload.items()}
                    ctx = make_ctx(q, default_experiment_name, layout)
                    layout.validate_ctx(ctx)
                    step_name = str(payload.get("step", "")).strip()
                    if step_name not in set(layout.get_step_names()):
                        raise ValueError(f"Unknown step: {step_name}")
                    job_id = create_job(kind="run-step", step=step_name)
                    t = threading.Thread(target=run_step_worker, args=(job_id, ctx, step_name, payload), daemon=True)
                    t.start()
                    as_json(self, {"job_id": job_id, "status": "queued"}, status=HTTPStatus.ACCEPTED)
                    return

                # /api/run-pipeline
                contexts = expand_contexts(payload)
                job_id = create_job(kind="run-pipeline")
                t = threading.Thread(target=run_pipeline_worker, args=(job_id, contexts, payload), daemon=True)
                t.start()
                as_json(
                    self,
                    {
                        "job_id": job_id,
                        "status": "queued",
                        "combination_count": len(contexts),
                    },
                    status=HTTPStatus.ACCEPTED,
                )
            except Exception as exc:  # noqa: BLE001
                as_json(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return Handler


def main() -> None:
    args = parse_args()
    if args.root is None:
        if args.setup:
            args.root = _infer_project_root(Path(args.setup))
        else:
            args.root = Path.cwd()
    else:
        args.root = args.root.resolve()

    if args.setup:
        os.environ["PIPELINE_SETUP_PATH"] = str(Path(args.setup).resolve())

    global PipelineContext, PipelineLayout
    PipelineContext, PipelineLayout = _load_layout_classes(args.root)

    default_tab = args.mode if args.mode is not None else args.default_tab
    title = "Pipeline Config"
    layout = PipelineLayout(args.root)
    html = build_html(title=title, default_tab=default_tab)
    handler = make_handler(
        layout,
        args.experiment_name,
        args.runner_python,
        args.lama_python_bin,
        html=html,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"{title}: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
