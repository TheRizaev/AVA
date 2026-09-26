---
title: AVA - live demo
emoji: 🚦
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Traffic events and accident risk from a CCTV camera
---

# AVA — WIUT Hackathon 2026, computer-vision track

This Space serves the project website and its live demo: upload an `.mp4` (up to 150 s,
400 MB) and get the detected traffic events, an annotated playback and the accident-risk curve.
The demo runs on CPU with the small detector (YOLO26-S, 960 px, ≈10 fps); the submitted pipeline uses
YOLO26-L on a GPU.

Source code, weights and the full pipeline: see the repository linked on the website.
