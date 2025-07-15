#!/usr/bin/env python
# coding=utf-8
"""
Einzel-Inference für Stable Video Diffusion XT + LoRA-Checkpoint.
Aufrufbeispiel:
    python inference_single.py \
        --base_ckpt stabilityai/stable-video-diffusion-img2vid-xt \
        --output_dir ./outputs \
        --image  ./test/foo.png \
        --output ./foo.mp4
"""
import argparse, os, numpy as np, torch, imageio
from pathlib import Path
from safetensors.torch import load_file
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image


# ---------- CLI ---------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="out.mp4")
    p.add_argument("--num_frames", type=int, default=17)
    p.add_argument("--width",      type=int, default=768)
    p.add_argument("--height",     type=int, default=768)
    p.add_argument("--motion_bucket_id",    type=int, default=127)
    p.add_argument("--fps",                 type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--decode_chunk_size",   type=int, default=8)
    p.add_argument("--seed",                type=int, default=42)
    return p.parse_args()


# ---------- Hilfen ---------- #
def find_latest_unet_ckpt(out_dir: Path) -> Path:
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for c in reversed(ckpts):
        f = c / "unet" / "diffusion_pytorch_model.safetensors"
        if f.is_file():
            return f
    top = out_dir / "diffusion_pytorch_model.safetensors"
    if top.is_file():
        return top
    raise FileNotFoundError(f"Kein UNet-Checkpoint in {out_dir}")


# ---------- Haupt ---------- #
def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # passendes dtype wählen
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32)

    # 1) Basispipeline *ohne* Accelerate-Sharding laden
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.base_ckpt,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        local_files_only=True,
        device_map=None,           # ← verhindert Meta-Tensor
        low_cpu_mem_usage=False,   # ← alles sofort echt laden
    ).to(device)

    # 2) letzten LoRA-Checkpoint laden
    ckpt = find_latest_unet_ckpt(outdir)
    print("Lade LoRA-UNet:", ckpt)
    state = load_file(str(ckpt), device="cpu")
    pipe.unet.load_state_dict(state, strict=False)

    # 3) Bild
    img = load_image(args.image).resize((args.width, args.height))

    # 4) Inferenz
    gen = torch.manual_seed(args.seed)
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype):
        res = pipe(
            img,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            motion_bucket_id=args.motion_bucket_id,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            generator=gen,
        )
    frames = res.frames[0]

    # 5) Speichern
    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".mp4":
        with imageio.get_writer(args.output, fps=args.fps) as vid:
            for f in frames:
                vid.append_data(np.array(f))
    elif ext == ".gif":
        frames[0].save(
            args.output,
            save_all=True,
            append_images=[np.array(f) for f in frames[1:]],
            duration=int(1000 / args.fps),
            loop=0,
        )
    else:
        raise ValueError("Extension muss .mp4 oder .gif sein")

    print("✅  Ergebnis gespeichert:", args.output)


if __name__ == "__main__":
    main()
