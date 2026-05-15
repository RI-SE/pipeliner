from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Inference Result Viewer")

# Global state to store session info
session_data = {
    "csv_path": Path(os.environ.get("RESULT_VIEWER_CSV", "")) if os.environ.get("RESULT_VIEWER_CSV") else None,
    "image_root": None,
    "overlay_dir": Path(os.environ.get("RESULT_VIEWER_OVERLAY_DIR", "")) if os.environ.get("RESULT_VIEWER_OVERLAY_DIR") else None,
    "run_name": os.environ.get("RESULT_VIEWER_RUN_NAME", "Inference Run"),
}

def init_session(csv_path: Path, image_root: Path, overlay_dir: Path, run_name: str):
    session_data["csv_path"] = csv_path
    session_data["image_root"] = image_root
    session_data["overlay_dir"] = overlay_dir
    session_data["run_name"] = run_name

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Inference Result Viewer</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background: #f0f2f5; }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .controls { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
            .card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .card-images { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 10px; }
            .image-container { flex: 0 0 250px; text-align: center; }
            .image-container img, .image-container canvas { width: 100%; border-radius: 4px; cursor: pointer; border: 1px solid #eee; }
            .image-label { font-size: 11px; color: #666; margin-bottom: 5px; font-weight: bold; }
            .card .info { margin-top: 10px; font-size: 14px; display: flex; justify-content: space-between; align-items: center; }
            .card .status { font-weight: bold; padding: 2px 6px; border-radius: 4px; }
            .status-defect { background: #ffebee; color: #c62828; }
            .status-good { background: #e8f5e9; color: #2e7d32; }
            .slider-container { display: flex; align-items: center; gap: 10px; }
            .slider { flex-grow: 1; }
            .stats { display: flex; gap: 20px; margin-top: 10px; font-size: 14px; color: #666; }
            .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); }
            .modal-content { margin: auto; display: block; max-width: 90%; max-height: 90%; }
            .modal-close { position: absolute; top: 15px; right: 35px; color: white; font-size: 40px; font-weight: bold; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Inference Result Viewer: <span id="run-name"></span></h1>
        </div>
        
        <div class="controls">
            <div class="slider-container">
                <label>Anomaly Threshold:</label>
                <input type="range" id="threshold-slider" class="slider" min="0" max="1" step="0.001" value="0.5">
                <span id="threshold-value">0.500</span>
            </div>
            <div class="stats">
                <span>Total: <span id="stat-total">0</span></span>
                <span>Defects: <span id="stat-defect">0</span></span>
                <span>Good: <span id="stat-good">0</span></span>
            </div>
        </div>

        <div class="grid" id="results-grid"></div>

        <div id="image-modal" class="modal">
            <span class="modal-close">&times;</span>
            <img class="modal-content" id="modal-img">
        </div>

        <script>
            let results = [];
            const grid = document.getElementById('results-grid');
            const slider = document.getElementById('threshold-slider');
            const thresholdValue = document.getElementById('threshold-value');
            const statTotal = document.getElementById('stat-total');
            const statDefect = document.getElementById('stat-defect');
            const statGood = document.getElementById('stat-good');
            const modal = document.getElementById('image-modal');
            const modalImg = document.getElementById('modal-img');
            const modalClose = document.querySelector('.modal-close');

            async function loadResults() {
                const response = await fetch('/api/results');
                const data = await response.json();
                results = data.results;
                document.getElementById('run-name').textContent = data.run_name;
                
                // Find min/max scores to set slider range
                const scores = results.map(r => r.pred_score).filter(Number.isFinite);
                if (scores.length === 0) {
                    slider.min = 0;
                    slider.max = 1;
                    slider.value = 0.5;
                } else {
                    const minScore = Math.min(...scores);
                    const maxScore = Math.max(...scores);
                    slider.min = minScore;
                    slider.max = maxScore;
                    slider.value = (minScore + maxScore) / 2;
                }
                thresholdValue.textContent = parseFloat(slider.value).toFixed(3);
                
                render();
            }

            function render() {
                const threshold = parseFloat(slider.value);
                const isInitial = grid.innerHTML === '';
                let defectCount = 0;
                
                results.forEach(r => {
                    const isDefect = Number.isFinite(r.pred_score) && r.pred_score >= threshold;
                    if (isDefect) defectCount++;
                    
                    if (isInitial) {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.id = `card-${r.stem}`;
                        card.innerHTML = `
                            <div class="card-images">
                                <div class="image-container">
                                    <div class="image-label">Original</div>
                                    <img src="/api/image?path=${encodeURIComponent(r.image_path)}" onclick="openModal(this.src)">
                                </div>
                                <div class="image-container">
                                    <div class="image-label">Heatmap Overlay</div>
                                    <img src="/api/overlay?stem=${encodeURIComponent(r.stem)}" onclick="openModal(this.src)" onerror="this.parentElement.style.display='none'">
                                </div>
                                <div class="image-container">
                                    <div class="image-label">Ground Truth Mask</div>
                                    ${r.mask_path ? `<img src="/api/image?path=${encodeURIComponent(r.mask_path)}" onclick="openModal(this.src)">` : '<div style="height: 100px; display: flex; align-items: center; justify-content: center; background: #eee; color: #999; font-size: 11px;">No GT Mask</div>'}
                                </div>
                                <div class="image-container">
                                    <div class="image-label">Predicted Mask</div>
                                    <canvas id="canvas-${r.stem}" onclick="openModal(this.toDataURL())"></canvas>
                                </div>
                            </div>
                            <div class="info">
                                <div>
                                    <strong>${r.stem}</strong>
                                    <span style="margin-left: 15px; color: #666;">Score: ${Number.isFinite(r.pred_score) ? r.pred_score.toFixed(4) : 'n/a'}</span>
                                </div>
                                <div id="status-${r.stem}">
                                    <span class="status ${isDefect ? 'status-defect' : 'status-good'}">
                                        ${isDefect ? 'DEFECT' : 'GOOD'}
                                    </span>
                                </div>
                            </div>
                        `;
                        grid.appendChild(card);
                    } else {
                        // Update status only
                        const statusDiv = document.getElementById(`status-${r.stem}`);
                        if (statusDiv) {
                            statusDiv.innerHTML = `
                                <span class="status ${isDefect ? 'status-defect' : 'status-good'}">
                                    ${isDefect ? 'DEFECT' : 'GOOD'}
                                </span>
                            `;
                        }
                    }
                    
                    // Update predicted mask on canvas
                    const canvas = document.getElementById(`canvas-${r.stem}`);
                    if (canvas) {
                        const ctx = canvas.getContext('2d');
                        
                        // Store the original raw image data on the canvas element for reuse
                        if (!canvas._rawImageData) {
                            const img = new Image();
                            img.onload = function() {
                                canvas.width = img.width;
                                canvas.height = img.height;
                                ctx.drawImage(img, 0, 0);
                                canvas._rawImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                                updateCanvas(canvas, threshold);
                            };
                            img.src = `/api/raw_map?stem=${encodeURIComponent(r.stem)}`;
                            img.onerror = function() {
                                img.src = `/api/overlay?stem=${encodeURIComponent(r.stem)}`;
                            };
                        } else {
                            updateCanvas(canvas, threshold);
                        }
                    }
                });
                
                statTotal.textContent = results.length;
                statDefect.textContent = defectCount;
                statGood.textContent = results.length - defectCount;
            }

            function updateCanvas(canvas, threshold) {
                const ctx = canvas.getContext('2d');
                const rawData = canvas._rawImageData;
                if (!rawData) return;
                
                const imageData = new ImageData(new Uint8ClampedArray(rawData.data), rawData.width, rawData.height);
                const data = imageData.data;
                const threshold8bit = threshold * 255;
                
                for (let i = 0; i < data.length; i += 4) {
                    const avg = (data[i] + data[i+1] + data[i+2]) / 3;
                    const val = avg >= threshold8bit ? 255 : 0;
                    data[i] = data[i+1] = data[i+2] = val;
                }
                ctx.putImageData(imageData, 0, 0);
            }

            slider.oninput = function() {
                thresholdValue.textContent = parseFloat(this.value).toFixed(3);
                render();
            };

            function openModal(src) {
                modal.style.display = "block";
                modalImg.src = src;
            }

            modalClose.onclick = function() {
                modal.style.display = "none";
            }

            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }

            loadResults();
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/results")
async def get_results():
    csv_path = session_data["csv_path"]
    if not csv_path or not csv_path.exists():
        raise HTTPException(status_code=404, detail="Results CSV not found")
    
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = Path(row["image_path"])
            score = float(row["pred_score"])
            if not math.isfinite(score):
                score = None
            results.append({
                "image_path": row["image_path"],
                "mask_path": row.get("mask_path", ""),
                "stem": img_path.stem,
                "pred_score": score,
                "pred_label": row["pred_label"] == "True"
            })
    
    return {
        "run_name": session_data["run_name"],
        "results": results
    }

@app.get("/api/raw_map")
async def get_raw_map(stem: str):
    overlay_dir = session_data["overlay_dir"]
    if not overlay_dir:
        raise HTTPException(status_code=404, detail="Overlay directory not set")
    
    # Raw maps are saved in a 'raw_maps' sibling directory to 'overlays'
    # or inside 'latest/images/raw_maps'
    raw_map_dir = overlay_dir.parent / "raw_maps"
    raw_map_path = raw_map_dir / f"{stem}_raw.png"
    
    if not raw_map_path.exists():
        raise HTTPException(status_code=404, detail="Raw map not found")
    return FileResponse(raw_map_path)

@app.get("/api/image")
async def get_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/api/overlay")
async def get_overlay(stem: str):
    overlay_dir = session_data["overlay_dir"]
    if not overlay_dir:
        raise HTTPException(status_code=404, detail="Overlay directory not set")
    
    # Try direct match with _overlay.jpg
    overlay_path = overlay_dir / f"{stem}_overlay.jpg"
    
    if not overlay_path.exists():
        # Try in the parent 'images' directory, looking for the original filename
        images_dir = overlay_dir.parent
        matches = list(images_dir.glob(f"**/{stem}.*"))
        if matches:
            overlay_path = matches[0]
            
    if not overlay_path.exists():
        # Fallback: look for a file that matches the beginning of the stem
        # e.g. A0005-20260422_093029_rivet_001 -> A0005-20260422_093029_overlay.jpg
        parts = stem.split('_')
        while len(parts) > 1:
            parts = parts[:-1]
            short_stem = "_".join(parts)
            
            # Try with _overlay.jpg suffix
            alt_path = overlay_dir / f"{short_stem}_overlay.jpg"
            if alt_path.exists():
                overlay_path = alt_path
                break
                
            # Try in the whole images tree with the short stem
            alt_matches = list(overlay_dir.parent.glob(f"**/{short_stem}.*"))
            if alt_matches:
                overlay_path = alt_matches[0]
                break
    
    # FINAL ATTEMPT: Check if the file is in the variant subfolder under images
    if not overlay_path.exists():
        images_dir = overlay_dir.parent
        variant_matches = list(images_dir.glob(f"**/{stem}.*"))
        if variant_matches:
            overlay_path = variant_matches[0]

    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(overlay_path)

def create_app():
    return app
