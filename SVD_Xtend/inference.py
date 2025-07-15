#!/usr/bin/env python
# coding=utf-8
import argparse
import os
from pathlib import Path

import torch
import numpy as np
import imageio
from safetensors.torch import load_file
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference für StableVideoDiffusion + LoRA (aus Checkpoints)")
    p.add_argument(
        "--base_ckpt", required=True,
        help="Pfad zur Basispipeline, z.B. /mnt/models/stable-video-diffusion-img2vid-xt")
    p.add_argument(
        "--output_dir", required=True,
        help="Dein OutputTrainedModel-Ordner, z.B. /…/OutputTrainedModel")
    p.add_argument(
        "--image", required=True,
        help="Pfad zum Eingangsbild (PNG/JPG)")
    p.add_argument(
        "--output", default="out.mp4",
        help="Zieldatei (*.mp4 oder *.gif)")
    p.add_argument("--num_frames",           type=int, default=17)
    p.add_argument("--width",                type=int, default=768)
    p.add_argument("--height",               type=int, default=768)
    p.add_argument("--motion_bucket_id",     type=int, default=127)
    p.add_argument("--fps",                  type=int, default=16)
    p.add_argument("--num_inference_steps",  type=int, default=25)
    p.add_argument("--decode_chunk_size",    type=int, default=8)
    p.add_argument("--seed",                 type=int, default=42)
    return p.parse_args()

def find_unet_ckpt(output_dir: Path) -> Path:
    # Suche neueste checkpoint-*/unet/diffusion_pytorch_model.safetensors
    ckpts = sorted(output_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    for ck in reversed(ckpts):
        candidate = ck / "unet" / "diffusion_pytorch_model.safetensors"
        if candidate.exists():
            return candidate
    # Fallback auf top-level (falls train final dort gespeichert wurde)
    top = output_dir / "diffusion_pytorch_model.safetensors"
    if top.exists():
        return top
    raise FileNotFoundError(f"Kein UNet-Checkpoint in {output_dir} gefunden")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Wähle dtype
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 1) Basispipeline laden
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.base_ckpt,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    # 2) UNet-Checkpoint finden und laden
    output_dir = Path(args.output_dir)
    unet_ckpt = find_unet_ckpt(output_dir)
    print("Lade UNet-Weights aus:", unet_ckpt)
    state_dict = load_file(str(unet_ckpt), device="cpu")
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=False)
    print(f"→ geladen; fehlend: {len(missing)}, unerwartet: {len(unexpected)}")

    # 3) Modell auf Device + contiguous setzen
    pipe.unet  = pipe.unet.to(device, dtype=dtype, memory_format=torch.contiguous_format)
    pipe.vae   = pipe.vae.to(device, dtype=dtype, memory_format=torch.contiguous_format)
    pipe.image_encoder = pipe.image_encoder.to(device, dtype=dtype)

    # 4) Bild laden und skalieren
    img = load_image(args.image).resize((args.width, args.height))

    # 5) Inferenz
    generator = torch.manual_seed(args.seed)
    pipe.set_progress_bar_config(disable=False)
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype):
        result = pipe(
            img,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            motion_bucket_id=args.motion_bucket_id,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            generator=generator,
        )
    frames = result.frames[0]

    # 6) Speichern
    out_path = args.output
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        with imageio.get_writer(out_path, fps=args.fps) as vid:
            for frame in frames:
                vid.append_data(np.array(frame))
    elif ext == ".gif":
        frames[0].save(
            out_path,
            save_all=True,
            append_images=[np.array(f) for f in frames[1:]],
            duration=int(1000/args.fps),
            loop=0,
        )
    else:
        raise ValueError("Unbekannte Extension: " + ext)

    print(f"Inferenz fertig – Ergebnis in {out_path}")

if __name__ == "__main__":
    main()
