"""Benchmark the ways we can decode the 4K 10-bit 4:2:2 camera files.

The harness (run_submission.py) streams every frame through cv2.VideoCapture for
Part B, so its decode speed is a floor we cannot change. Part A is free to use
PyAV with `skip_frame`, which lets us decode only reference frames.

    python scripts/bench_decode.py samples/C3905.MP4 --seconds 20
"""
from __future__ import annotations

import argparse
import time

import av
import cv2


def bench_cv2(path: str, seconds: float) -> None:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(seconds * fps)
    t0 = time.perf_counter()
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
    dt = time.perf_counter() - t0
    print(f"cv2.VideoCapture full decode: {n} frames in {dt:.1f}s -> {n / dt:.1f} fps "
          f"({dt / seconds:.2f}x realtime), frame {frame.shape} {frame.dtype}")


def bench_pyav(path: str, seconds: float, skip: str, size: tuple[int, int] | None) -> None:
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.skip_frame = skip
    tb = float(stream.time_base)
    n, t0 = 0, time.perf_counter()
    for frame in container.decode(stream):
        t = frame.pts * tb
        if t > seconds:
            break
        if size:
            frame.to_ndarray(width=size[0], height=size[1], format="bgr24")
        else:
            frame.to_ndarray(format="bgr24")
        n += 1
    dt = time.perf_counter() - t0
    container.close()
    print(f"PyAV skip={skip:<7} size={size}: {n} frames ({n / seconds:.1f}/s of video) in {dt:.1f}s "
          f"({dt / seconds:.2f}x realtime)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    bench_cv2(args.video, args.seconds)
    for skip in ("DEFAULT", "NONREF", "NONKEY"):
        bench_pyav(args.video, args.seconds, skip, (1280, 720))
    bench_pyav(args.video, args.seconds, "NONREF", (1920, 1080))


if __name__ == "__main__":
    main()
