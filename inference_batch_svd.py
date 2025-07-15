#!/usr/bin/env python
# coding=utf-8
"""
Batch-Wrapper: ruft inference_single.py parallel auf allen GPUs auf.
– nimmt alle PNG/JPG im TEST_DIR
– sucht automatisch den neuesten LoRA-Checkpoint
– verteilt Bilder round-robin auf die vorhandenen GPUs
"""

import os, subprocess, sys, math, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch


# ---------- Pfade anpassen ---------- #
BASE_CKPT   = "stabilityai/stable-video-diffusion-img2vid-xt"
TRAIN_OUT   = Path(os.getenv("OUTPUT_DIR"))           # ← dein Trainings-Output-Ordner
DATA_ROOT_TEST = Path(os.environ["DATA_ROOT_TEST"])
TEST_DIR       = DATA_ROOT_TEST / "test"
SINGLE_RUN  = Path(__file__).parent / "inference_single.py"

VALIDATION_DIR = TRAIN_OUT / "ValidationsLastCheckpoint"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Standard-Parameter für jeden Aufruf
COMMON_ARGS = [
    "--base_ckpt", BASE_CKPT,
    "--output_dir", str(TRAIN_OUT),
    "--num_frames", "17",
    "--width",  "768",
    "--height", "768",
    "--motion_bucket_id", "127",
    "--fps", "16",
    "--num_inference_steps", "25",
    "--decode_chunk_size", "8",
]

# ---------- Hilfen ---------- #
def all_images(folder: Path):
    return sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")])


def run_on_gpu(gpu_id: int, img: Path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out_file = VALIDATION_DIR / img.with_suffix(".mp4").name          # gleicher Name, mp4
    cmd = [
        sys.executable, str(SINGLE_RUN),
        *COMMON_ARGS,
        "--image", str(img),
        "--output", str(out_file),
        "--seed", str(42 + gpu_id),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    print(proc.stdout.strip())


# ---------- Haupt ---------- #
def main():
    imgs = all_images(TEST_DIR)
    if not imgs:
        print("⚠️  Keine Test-Bilder in", TEST_DIR)
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("⚠️  Keine GPU gefunden – benutze CPU single-thread")
        num_gpus = 1

    # ThreadPool zum einfachen Verteilen
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futs = []
        for idx, img in enumerate(imgs):
            gpu = idx % num_gpus
            futs.append(pool.submit(run_on_gpu, gpu, img))

        # Exceptions bubb­len lassen
        for f in as_completed(futs):
            f.result()

    print("🎉  Batch-Inference fertig.")


if __name__ == "__main__":
    main()
