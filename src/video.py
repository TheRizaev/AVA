"""Video decoding.

The camera writes 4K H.264 High 4:2:2 10-bit at ~147 Mb/s. NVDEC cannot decode
that profile, so everything runs on the CPU, and decoding is the single most
expensive step of the pipeline. Two tricks keep it cheap:

* ``skip_frame="NONREF"`` makes the decoder drop non-reference (B) frames
  entirely. The camera uses an IBBP GOP, so we get every 3rd frame (~10 fps)
  for a fraction of the cost of a full decode.
* The frame is converted to 8-bit BGR at the analysis resolution inside
  swscale, never materialised at 4K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import av
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float


def probe(path: str) -> VideoMeta:
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 25.0)
        n_frames = stream.frames
        if container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = float(stream.duration * stream.time_base) if stream.duration else n_frames / fps
        if not n_frames:
            n_frames = int(round(duration * fps))
        return VideoMeta(path, stream.codec_context.width, stream.codec_context.height, fps, n_frames,
                         n_frames / fps if n_frames else duration)


def iter_frames(path: str, size: tuple[int, int], skip: str = "NONREF",
                min_step: float = 0.0) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, t_sec, bgr)`` resized to ``size`` = (width, height).

    ``skip`` is passed to the decoder: "DEFAULT" decodes everything, "NONREF"
    skips B-frames, "NONKEY" keeps only I-frames. ``min_step`` additionally
    drops decoded frames closer than this many seconds to the previous one.
    Timestamps are relative to the first frame, so they match the harness
    (frame_index / fps).
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.skip_frame = skip
        fps = float(stream.average_rate or 25.0)
        tb = float(stream.time_base)
        start_pts = stream.start_time or 0
        last_t = -1e9
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = (frame.pts - start_pts) * tb
            idx = int(round(t * fps))
            t = idx / fps
            if t - last_t < min_step - 1e-6:
                continue
            last_t = t
            yield idx, t, frame.to_ndarray(width=size[0], height=size[1], format="bgr24")
