from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .planner import build_step_run
from .setup_loader import flatten_choices, load_setup


HTML = """<!doctype html>
<html><head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Pipeliner</title>
<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:16px;background:#f6f8fb;color:#1e2430}
.card{background:#fff;border:1px solid #d6dce8;border-radius:10px;padding:12px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
label{display:block;font-size:12px;color:#5a6578;margin-bottom:6px}
select,input,button{width:100%;box-sizing:border-box;padding:8px;border-radius:8px;border:1px solid #d6dce8}
button{background:#1d6fd6;color:#fff;border-color:#1d6fd6;cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word;margin:0}
.tabs{display:flex;gap:8px;margin-bottom:12px}
.tab{width:auto;padding:8px 12px;background:#fff;color:#1e2430}.tab.active{background:#1d6fd6;color:#fff}
</style></head><body>
<h1 style='margin:0 0 12px'>Pipeliner</h1>
<div class='tabs'>
  <button class='tab active' id='tab_viewer' onclick='setTab("viewer")'>Viewer</button>
  <button class='tab' id='tab_runner' onclick='setTab("runner")'>Runner</button>
  <button class='tab' id='tab_analysis' onclick='setTab("analysis")'>Analysis</button>
</div>
<div class='card'>
  <div class='grid' id='vars'></div>
  <div class='grid' style='margin-top:10px'>
    <div><label>Step</label><select id='step'></select></div>
    <div><label>Python</label><input id='python' value='python3'></div>
  </div>
  <div class='grid' style='margin-top:10px'>
    <button onclick='refreshAll()'>Refresh</button>
    <button onclick='runStep()'>Run Selected Step</button>
  </div>
</div>
<div class='card' id='analysis' style='display:none'><pre>Analysis tab scaffold.</pre></div>
<div class='card' id='selected_card'><h3 style='margin:0 0 8px'>Selected Step</h3><pre id='selected'></pre></div>
<div class='card'><h3 style='margin:0 0 8px'>Run Log</h3><pre id='runlog'></pre></div>
<script>
let OPTS = null;
let TAB = 'viewer';

function setTab(t){
  TAB=t;
  for(const x of ['viewer','runner','analysis']) document.getElementById('tab_'+x).classList.toggle('active', x===t);
  document.getElementById('analysis').style.display = t==='analysis' ? '' : 'none';
  document.getElementById('selected_card').style.display = t==='analysis' ? 'none' : '';
}

function q(){
  const p = new URLSearchParams();
  for(const k of Object.keys(OPTS.variation_points)) p.set(k, document.getElementById('v_'+k).value);
  p.set('step', document.getElementById('step').value);
  p.set('python', document.getElementById('python').value || 'python3');
  return p;
}

async function loadOptions(){
  OPTS = await (await fetch('/api/options')).json();
  const box = document.getElementById('vars');
  box.innerHTML='';
  for(const [k,vals] of Object.entries(OPTS.variation_points)){
    const w=document.createElement('div');
    const label=k.replaceAll('_',' ');
    w.innerHTML = `<label>${label}</label><select id='v_${k}'>${vals.map(v=>`<option>${v}</option>`).join('')}</select>`;
    box.appendChild(w);
  }
  const step = document.getElementById('step');
  step.innerHTML = OPTS.process_steps.map(s=>`<option>${s}</option>`).join('');
  for(const k of Object.keys(OPTS.variation_points)) document.getElementById('v_'+k).onchange = refreshSelected;
  document.getElementById('step').onchange = refreshSelected;
}

async function refreshSelected(){
  const data = await (await fetch('/api/resolve?'+q().toString())).json();
  document.getElementById('selected').textContent = JSON.stringify(data, null, 2);
}

async function runStep(){
  const body = {};
  for(const k of Object.keys(OPTS.variation_points)) body[k] = document.getElementById('v_'+k).value;
  body.step = document.getElementById('step').value;
  body.python = document.getElementById('python').value || 'python3';
  document.getElementById('runlog').textContent = 'Running...';
  const res = await fetch('/api/run-step', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const out = await res.json();
  document.getElementById('runlog').textContent = JSON.stringify(out, null, 2);
  await refreshSelected();
}

async function refreshAll(){ await loadOptions(); await refreshSelected(); document.getElementById('runlog').textContent=''; }
refreshAll();
</script>
</body></html>"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeliner web UI")
    p.add_argument("--setup", "--config", "-config", dest="setup", required=True, help="Path to experiment setup yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--project-root",
        default="",
        help="Project root for resolving relative paths. Default is inferred from setup path.",
    )
    return p.parse_args()


def _json(handler: BaseHTTPRequestHandler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    raw = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _selection(query: dict[str, list[str]], keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        v = query.get(k, [""])[0]
        if v != "":
            out[k] = v
    return out


def _infer_project_root(setup_path: Path) -> Path:
    setup_path = setup_path.resolve()
    if setup_path.parent.name == "pipeline":
        return setup_path.parent.parent
    return setup_path.parent


def _abs_path(project_root: Path, value: str) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((project_root / p).resolve())


def _abs_pathlike(project_root: Path, value):
    if isinstance(value, str):
        return _abs_path(project_root, value)
    if isinstance(value, dict):
        return {k: _abs_pathlike(project_root, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_abs_pathlike(project_root, v) for v in value]
    return value


def make_handler(setup_path: Path, project_root: Path):
    setup = load_setup(setup_path)
    vp_flat = flatten_choices(setup.variation_points)

    def _resolve_step_payload(query: dict[str, list[str]]) -> dict:
        step = query.get("step", [""])[0]
        if not step:
            raise ValueError("missing step")
        selection = _selection(query, list(vp_flat.keys()))
        python_bin = query.get("python", ["python3"])[0]
        sr = build_step_run(setup, step, selection)

        sr.script = _abs_path(project_root, sr.script)
        sr.contract.resolved = _abs_pathlike(project_root, sr.contract.resolved)
        process_step = dict(sr.contract.process_step)
        if "script" in process_step and isinstance(process_step["script"], str):
            process_step["script"] = _abs_path(project_root, process_step["script"])
        if "input" in process_step:
            process_step["input"] = _abs_pathlike(project_root, process_step["input"])
        if "output" in process_step:
            process_step["output"] = _abs_pathlike(project_root, process_step["output"])
        sr.contract.process_step = process_step

        cmd = sr.command(python_bin=python_bin)
        return {
            "step": sr.step_name,
            "script": sr.script,
            "contract": sr.contract.to_dict(),
            "cmd": cmd,
            "cmd_str": " ".join(shlex.quote(x) for x in cmd),
            "cwd": str(project_root),
        }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/options":
                return _json(
                    self,
                    {
                        "variation_points": vp_flat,
                        "process_steps": list(setup.process_steps.keys()),
                    },
                )
            if parsed.path == "/api/resolve":
                query = parse_qs(parsed.query)
                try:
                    payload = _resolve_step_payload(query)
                except Exception as exc:  # noqa: BLE001
                    return _json(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return _json(self, payload)
            return _json(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/run-step":
                return _json(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return _json(self, {"error": "invalid json"}, HTTPStatus.BAD_REQUEST)

            step = str(body.get("step", ""))
            python_bin = str(body.get("python", "python3"))
            selection = {k: str(v) for k, v in body.items() if k in vp_flat.keys()}

            try:
                query = {"step": [step], "python": [python_bin], **{k: [v] for k, v in selection.items()}}
                payload = _resolve_step_payload(query)
                cmd = payload["cmd"]
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
            except Exception as exc:  # noqa: BLE001
                return _json(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

            return _json(
                self,
                {
                    "step": step,
                    "cmd": cmd,
                    "cwd": str(project_root),
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )

    return Handler


def main() -> int:
    args = parse_args()
    setup_path = Path(args.setup).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else _infer_project_root(setup_path)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(setup_path, project_root))
    print(f"Pipeliner web UI on http://{args.host}:{args.port}")
    print(f"setup: {setup_path}")
    print(f"project_root: {project_root}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
