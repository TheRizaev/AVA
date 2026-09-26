"""Build the website's data files (EDA + results) from the cached analyses.

    python scripts/build_site_data.py --cache work/cache --out website/data --media website/media

Writes eda.json, results.json and copies the EDA images the site shows.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.analysis import VideoAnalysis  # noqa: E402
from src.events import build_context, detect  # noqa: E402
from src.risk import risk_curve_from_analysis  # noqa: E402
from src.scene import lane_angle, polygon, side_of_line, load_layout  # noqa: E402
from src.signals import STATE_NAMES, _runs  # noqa: E402

LIGHTING = {"C3896": "noon", "C3897": "noon", "C3902": "evening sun", "C3905": "dusk"}
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames:format=duration,bit_rate",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    d = json.loads(out)
    s, f = d["streams"][0], d["format"]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"codec": f"{s['codec_name']} {s.get('profile', '')}".strip(), "pix_fmt": s["pix_fmt"],
            "width": s["width"], "height": s["height"], "fps": round(num / den, 3),
            "frames": int(s.get("nb_frames", 0)), "duration": round(float(f["duration"]), 2),
            "bitrate_mbps": round(int(f["bit_rate"]) / 1e6, 1)}


def counts_over_time(va: VideoAnalysis, bin_sec: float = 5.0) -> dict:
    tab = va.tracks
    bins = np.arange(0, va.meta.duration + bin_sec, bin_sec)
    out = {"t": bins[:-1].round(1).tolist()}
    frames_per_bin = np.histogram(np.unique(tab[:, 1]), bins)[0].clip(1)
    for cls, name in COCO.items():
        m = tab[:, 8] == cls
        out[name] = (np.histogram(tab[m, 1], bins)[0] / frames_per_bin).round(2).tolist()  # mean objects per frame
    return out


def throughput(va: VideoAnalysis, tracks) -> dict:
    """Approach vehicles crossing the stop line, per minute."""
    a, b = (np.array(p, float) for p in load_layout()["lines"]["stop_line"])
    sign = np.sign(side_of_line(np.array([[500.0, 380.0]]), a, b)[0])
    times = []
    for tr in tracks:
        if tr.kind != "vehicle":
            continue
        d = -sign * side_of_line(tr.xy, a, b)
        idx = np.flatnonzero((d[:-1] <= 0) & (d[1:] > 0))
        if len(idx):
            times.append(float(tr.t[idx[0] + 1]))
    minutes = np.arange(0, va.meta.duration + 60, 60)
    return {"minute": list(range(len(minutes) - 1)), "vehicles": np.histogram(times, minutes)[0].tolist(),
            "total": len(times)}


def signal_runs(va: VideoAnalysis) -> list:
    sig = va.signal_timeline()
    return [[round(float(sig.t[s]), 1), round(float(sig.t[e - 1]), 1), STATE_NAMES[v]] for s, e, v in _runs(sig.state)]


def speeds(tracks) -> dict:
    sp = []
    for tr in tracks:
        if tr.kind == "vehicle" and len(tr) > 10:
            v = tr.speed(0.5)
            sp.append(v[v > 20])
    sp = np.concatenate(sp) if sp else np.zeros(0)
    h, e = np.histogram(sp, bins=np.arange(0, 700, 25))
    return {"bin_px_s": e[:-1].tolist(), "count": h.tolist(), "median": round(float(np.median(sp)), 1) if len(sp) else 0}


def lane_hist(all_tracks) -> dict:
    appr = polygon("approach")
    a = []
    for tr in all_tracks:
        if tr.kind != "vehicle":
            continue
        m = appr.contains(tr.xy) & (tr.xy[:, 1] > 330) & (tr.xy[:, 1] < 500)
        a.append(lane_angle(tr.xy[m]))
    a = np.concatenate(a)
    h, e = np.histogram(a, bins=np.arange(20, 36, 0.25))
    return {"angle": (e[:-1] + 0.125).round(3).tolist(), "count": h.tolist(),
            "lines": [ln["angle"] for ln in load_layout()["solid_lines"]["lines"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="work/cache")
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--out", default="website/data")
    ap.add_argument("--media", default="website/media")
    ap.add_argument("--eda-images", default="work/eda")
    ap.add_argument("--pred", default="predictions_samples.json",
                    help="official harness output: its events and Part B risk curves are what the site shows")
    args = ap.parse_args()
    out, media = Path(args.out), Path(args.media)
    out.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)

    pred = json.loads(Path(args.pred).read_text())["videos"] if Path(args.pred).exists() else {}
    eda = {"videos": [], "lane_hist": None}
    results = {"videos": []}
    all_tracks = []
    for f in sorted(Path(args.cache).glob("*.npz")):
        name = f.stem
        va = VideoAnalysis.load(f)
        ctx = build_context(va)
        tracks = ctx.tracks
        all_tracks += tracks
        info = ffprobe(Path(args.samples) / f"{name}.MP4")
        thumbs = va.thumbs
        brightness = [round(float(x), 1) for x in thumbs.reshape(len(thumbs), -1).mean(1)] if len(thumbs) else []
        shift = va.reg_H[:, :2, 2]
        kinds = {}
        for tr in tracks:
            kinds[tr.kind] = kinds.get(tr.kind, 0) + 1
        eda["videos"].append({
            "name": name, **info, "lighting": LIGHTING.get(name, ""),
            "brightness": {"t": va.thumb_t.round(1).tolist(), "mean": brightness},
            "registration": {"t": va.reg_t.round(1).tolist(), "dx": shift[:, 0].round(2).tolist(),
                             "dy": shift[:, 1].round(2).tolist()},
            "tracks_by_kind": kinds,
            "counts": counts_over_time(va),
            "throughput": throughput(va, tracks),
            "signal": signal_runs(va),
            "speeds": speeds(tracks),
        })
        official = pred.get(f"{name}.MP4")
        if official:   # the submission's own output (RiskEstimator curve from the harness)
            events = official["events"]
            curve = np.array(official["risk"]) if official["risk"] else np.zeros((0, 2))
        else:          # no harness run yet: rules on the cache + causal replay of the risk
            events = [e.as_list() for e in detect(ctx, config.ENABLED_CLASSES)]
            t, score = risk_curve_from_analysis(va)
            curve = np.stack([t, score], 1) if len(t) else np.zeros((0, 2))
        results["videos"].append({
            "name": name, "duration": round(va.meta.duration, 2),
            "events": events,
            "risk": curve[::3].round(3).tolist(),
            "video": f"media/{name}_annotated.mp4",
        })
        cv2.imwrite(str(media / f"{name}_thumb.jpg"), thumbs[len(thumbs) // 3] if len(thumbs) else np.zeros((360, 640, 3)))
        print("processed", name)
    eda["lane_hist"] = lane_hist(all_tracks)
    (out / "eda.json").write_text(json.dumps(eda))
    (out / "results.json").write_text(json.dumps(results))
    for img in ("occupancy_vehicle.jpg", "occupancy_person.jpg", "flow_field.jpg", "layout_C3896.jpg",
                "lane_groups.jpg", "C3905_vehicle_tracks.jpg", "C3905_person_tracks.jpg"):
        src = Path(args.eda_images) / img
        if src.exists():
            shutil.copy(src, media / img)
    print("wrote", out / "eda.json", out / "results.json")


if __name__ == "__main__":
    main()
