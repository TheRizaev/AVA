"""Live-demo server: upload a clip, get events, a risk curve and an annotated video back.

Serves the static website (../website) and a small job API:

    POST /api/jobs                 multipart 'file' (.mp4, <= MAX_SECONDS)  -> {"id": ...}
    GET  /api/jobs/{id}            status, stage, progress, result when done
    GET  /api/jobs/{id}/video      annotated H.264 mp4

Runs the same pipeline as the submission in a CPU-friendly profile (small
detector at 960 px, the same 10 fps, no hazard pass); jobs are processed one at a time.
If only the annotated video fails, the job still returns its events and risk curve.

    uvicorn demo.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import av
import numpy as np
import torch

from src import config
from src.analysis import analyze
from src.detection import detect_batch, load_model
from src.events import build_context, detect
from src.render import Renderer
from src.risk import risk_curve_from_analysis
from src.risk_model import ALARM_CAP
from src.signals import SignalTimeline
from src.video import probe

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(tempfile.gettempdir()) / "wiut_demo_jobs"
MAX_SECONDS = 150
SLACK_SECONDS = 3.0            # a "150 s" cut is often longer: 150.02 s re-encoded, 151.02 s stream-copied from a keyframe
MAX_BYTES = 400 * 1024 * 1024
DEMO_WEIGHTS = config.WEIGHTS_DIR / "yolo26s.pt"
DEMO_FPS = config.SAMPLE_FPS   # the rules are tuned for this sampling (track splitting, speeds)
DEMO_IMGSZ = 960
RISK_WARMUP_SEC = 2.0          # no history yet: the demo raises no alarm in the first 2 s (submission unaffected)
KEEP_JOBS = 20

app = FastAPI(title="WIUT traffic events demo")
# the website may be served elsewhere (a static Space whose <meta name="demo-api"> points here)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
jobs: dict[str, dict] = {}
work_q: queue.Queue = queue.Queue()
_model = None


def _cpu_quota() -> int | None:
    """CPUs granted by the container's cgroup limit (Linux); None when there is no limit.

    A Hugging Face CPU Space sees every core of the host but may use only its quota
    (2 vCPUs on cpu-basic): torch's default thread count would oversubscribe it.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]       # cgroup v2
    except (OSError, ValueError):
        try:
            quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip()   # cgroup v1
            period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip()
        except OSError:
            return None
    if quota in ("max", "-1"):
        return None
    return max(1, round(int(quota) / int(period)))


if "OMP_NUM_THREADS" not in os.environ and (_n := _cpu_quota()):
    torch.set_num_threads(_n)


def _get_model():
    global _model
    if _model is None:
        _model = load_model(DEMO_WEIGHTS)
    return _model


class DemoRenderer(Renderer):
    """Reports rendering progress; on footage from another camera, skips the layout overlay and the
    signal lamp (both would be misplaced)."""

    def __init__(self, analysis, events, risk=None, progress=None):
        super().__init__(analysis, events, risk)
        self.progress = progress
        if not analysis.scene_recognised:
            self.signal = SignalTimeline(np.zeros(0), np.zeros(0, int), np.zeros((0, 5)))

    def _draw_layout(self, img, t, sx, sy):
        if self.va.scene_recognised:
            super()._draw_layout(img, t, sx, sy)

    def _draw_hud(self, img, t):   # called once per output frame
        super()._draw_hud(img, t)
        if self.progress is not None and self.va.meta.duration > 0:
            self.progress(min(1.0, t / self.va.meta.duration))


def _run(job_id: str) -> None:
    job = jobs[job_id]
    src = job["dir"] / "input.mp4"
    try:
        job.update(status="running", stage="detecting and tracking", progress=0.0)
        va = analyze(str(src), _get_model(), None, sample_fps=DEMO_FPS, det_imgsz=DEMO_IMGSZ,
                     progress=lambda f: job.__setitem__("progress", round(0.8 * f, 3)))
        job.update(stage="detecting events", progress=0.8)
        ctx = build_context(va)
        events = detect(ctx, config.ENABLED_CLASSES)   # the same classes the submission reports
        job.update(stage="computing accident risk", progress=0.82)
        t, score = risk_curve_from_analysis(va)
        score = np.where(t < RISK_WARMUP_SEC, np.minimum(score, ALARM_CAP), score)
        risk = [[round(float(a), 2), round(float(b), 3)] for a, b in zip(t, score)]
        job.update(stage="rendering annotated video", progress=0.84)
        out = job["dir"] / "annotated.mp4"
        try:   # on 2 vCPUs: ~10 % of the job for 1080p input, ~30 % for 4K (every frame is decoded again)
            DemoRenderer(va, events, risk, progress=lambda f: job.__setitem__("progress", round(0.84 + 0.16 * f, 3))
                         ).render(src, out, fps_out=10)
            video = f"/api/jobs/{job_id}/video"
        except Exception:   # the events and the risk curve are still worth returning
            traceback.print_exc()
            video = None
        job.update(status="done", stage="done", progress=1.0, result={
            "duration": round(va.meta.duration, 2),
            "scene_recognised": va.scene_recognised,
            "events": [e.as_list() for e in events],
            "risk": risk,
            "video": video,
        })
    except Exception as exc:  # reported to the page, never crashes the server
        msg = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, av.error.FFmpegError):
            msg = f"the video could not be decoded to the end (damaged or incomplete file?) - {msg}"
        job.update(status="error", stage="error", error=msg)
        traceback.print_exc()


def _worker() -> None:
    try:   # load the detector and run it once now, so the first upload does not pay for the start-up
        detect_batch(_get_model(), [np.zeros((*config.ANALYSIS_SIZE[::-1], 3), np.uint8)], imgsz=DEMO_IMGSZ)
    except Exception:
        traceback.print_exc()
    while True:
        job_id = work_q.get()
        _run(job_id)
        _prune()


def _prune() -> None:
    done = sorted((j for j in jobs.values() if j["status"] in ("done", "error")), key=lambda j: j["created"])
    for j in done[:-KEEP_JOBS]:
        shutil.rmtree(j["dir"], ignore_errors=True)
        jobs.pop(j["id"], None)


threading.Thread(target=_worker, daemon=True).start()


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".m4v")):
        raise HTTPException(400, "please upload an .mp4 video")
    job_id = uuid.uuid4().hex[:12]
    d = WORK / job_id
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "input.mp4"
    size = 0
    with dst.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"file larger than {MAX_BYTES // 2**20} MB")
            fh.write(chunk)
    try:
        meta = probe(str(dst))
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "could not read this video")
    if meta.duration > MAX_SECONDS + SLACK_SECONDS:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"video is {meta.duration:.1f} s long; the demo accepts up to {MAX_SECONDS} s")
    jobs[job_id] = {"id": job_id, "dir": d, "status": "queued", "stage": "queued", "progress": 0.0,
                    "created": time.time(), "position": work_q.qsize() + 1}
    work_q.put(job_id)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return JSONResponse({k: v for k, v in job.items() if k not in ("dir",)})


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    job = jobs.get(job_id)
    if job is None or job["status"] != "done" or not job["result"]["video"]:
        raise HTTPException(404, "video not ready")
    return FileResponse(job["dir"] / "annotated.mp4", media_type="video/mp4")


app.mount("/", StaticFiles(directory=ROOT / "website", html=True), name="site")
