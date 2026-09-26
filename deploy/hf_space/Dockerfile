# Website + live demo as a Hugging Face Space (Docker SDK, cpu-basic hardware; needs a PRO account).
# Built from the folder assembled by scripts/build_space.py.
# Debian 12 pinned: the plain python:3.10-slim tag moved to Debian 13, where package names differ.
FROM python:3.10-slim-bookworm

# ffmpeg (libx264) writes the annotated demo video. ultralytics also pulls in the non-headless
# opencv-python, whose cv2 may overwrite the headless one: it needs libGL and libglib at import.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
# YOLO_CONFIG_DIR is the parent of Ultralytics' settings folder (-> /tmp/Ultralytics)
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user -r requirements-space.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "demo.app:app", "--host", "0.0.0.0", "--port", "7860"]
