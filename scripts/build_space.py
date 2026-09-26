"""Assemble (and optionally upload) the Hugging Face Space that serves the website + live demo.

    python scripts/build_space.py                                  # -> work/hf_space
    python scripts/build_space.py --upload <user>/<space-name>     # create the Space and push it
    python scripts/build_space.py --static --upload <user>/<space>  # the website alone (no demo)
    python scripts/build_space.py --static --demo-api https://<user>-<space>.hf.space/api --upload ...
                                                    # the website, with its demo served by a Docker Space

Uploading needs `pip install huggingface_hub` and a write token (`hf auth login`, or the
HF_TOKEN environment variable). The full Space uses the Docker SDK on CPU hardware (Hugging Face
asks for a PRO account for that): the demo runs YOLO26-S at 960 px; the submission itself
(run_submission.py) is not part of the Space. The static build is free: the website only, whose
demo section calls the demo server named by --demo-api (a Docker Space built from this script),
or reports the demo as offline without one.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("src", "demo", "website", "assets")
WEIGHTS = ("yolo26s.pt",)          # the demo detector only
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.stackdump")


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():        # empty it (the folder itself may be open in a shell)
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    for d in DIRS:
        shutil.copytree(ROOT / d, out / d, ignore=SKIP)
    (out / "weights").mkdir()
    for w in WEIGHTS:
        shutil.copy2(ROOT / "weights" / w, out / "weights" / w)
    for f in (ROOT / "deploy" / "hf_space").iterdir():
        shutil.copy2(f, out / f.name)
    return report(out)


def build_static(out: Path, demo_api: str | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    shutil.copytree(ROOT / "website", out, ignore=SKIP, dirs_exist_ok=True)
    shutil.copy2(ROOT / "deploy" / "hf_space_static" / "README.md", out / "README.md")
    if demo_api:                     # website/assets/js/demo.js reads the API base from this tag
        page = out / "index.html"
        html = page.read_text(encoding="utf-8")
        tag = f'<meta name="demo-api" content="{demo_api.rstrip("/")}">'
        anchor = '<meta charset="utf-8">'
        page.write_text(html.replace(anchor, f"{anchor}\n  {tag}", 1), encoding="utf-8")
        assert tag in page.read_text(encoding="utf-8"), "index.html has no <meta charset> to anchor the demo-api tag"
    return report(out)


def report(out: Path) -> Path:
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"built {out} ({size:.0f} MB)")
    return out


def upload(folder: Path, space: str, sdk: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk=sdk, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=space, repo_type="space",
                      commit_message="Website" if sdk == "static" else "Website and live demo",
                      delete_patterns="*")     # the Space mirrors the folder exactly
    print(f"uploaded: https://huggingface.co/spaces/{space}  (the build takes a few minutes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="build folder (default work/hf_space, or work/hf_space_static)")
    ap.add_argument("--static", action="store_true", help="the website alone, as a free static Space")
    ap.add_argument("--demo-api", metavar="URL", help="with --static: the demo server's API base, e.g. https://user-space.hf.space/api")
    ap.add_argument("--upload", metavar="USER/SPACE", help="create the Space if needed and push the folder")
    args = ap.parse_args()
    out = Path(args.out or ROOT / "work" / ("hf_space_static" if args.static else "hf_space"))
    folder = build_static(out, args.demo_api) if args.static else build(out)
    if args.upload:
        upload(folder, args.upload, "static" if args.static else "docker")


if __name__ == "__main__":
    main()
