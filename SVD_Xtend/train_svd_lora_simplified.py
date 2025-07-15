#!/usr/bin/env python
# coding=utf-8
import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, RandomSampler, DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from src.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from peft import LoraConfig
from diffusers.training_utils import cast_training_params
from tqdm.auto import tqdm

# ── Monkey-Patch group_norm für contiguous inputs ──
_orig_gn = F.group_norm
def _patched_group_norm(x, g, w, b, e):
    return _orig_gn(x.contiguous(), g, w, b, e)
F.group_norm = _patched_group_norm

# ── Dummy Dataset (liefert nur Null-Tensoren) ──
class DummyDataset(Dataset):
    def __init__(self, base_folder, width, height, sample_frames):
        self.width, self.height, self.sample_frames = width, height, sample_frames
        self._dummy = torch.zeros((sample_frames, 3, height, width))
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        return {"pixel_values": self._dummy.clone()}

# ── Argument Parser (ignoriert unbekannte args) ──
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_folder",                   required=True)
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--num_frames",      type=int,     default=10)
    p.add_argument("--width",           type=int,     default=768)
    p.add_argument("--height",          type=int,     default=768)
    p.add_argument("--output_dir",                 default="./outputs")
    p.add_argument("--per_gpu_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps",type=int,default=1)
    p.add_argument("--max_train_steps",   type=int,    default=20)
    p.add_argument("--learning_rate",     type=float,  default=1e-4)
    p.add_argument("--scale_lr",          action="store_true")
    p.add_argument(
        "--mixed_precision", choices=["no","fp16","bf16"], default=None
    )
    p.add_argument("--seed",              type=int,    default=42)
    args, _ = p.parse_known_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # ── Accelerator Setup ──
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs"),
        ),
    )

    # ── dtype wählen ──
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # ── Modelle laden ──
    _ = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor"
    )
    _ = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder"
    )
    vae  = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True
    )

    # ── Freeze + LoRA ──
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    lora_config = LoraConfig(
        r=4, lora_alpha=4,
        target_modules=["to_k","to_q","to_v","to_out.0"]
    )
    unet.add_adapter(lora_config)
    if args.mixed_precision in ("fp16","bf16"):
        cast_training_params(unet, dtype=torch.float32)

    # ── DataLoader ──
    ds = DummyDataset(
        args.base_folder, args.width, args.height, args.num_frames
    )
    dl = DataLoader(
        ds,
        sampler=RandomSampler(ds),
        batch_size=args.per_gpu_batch_size,
        num_workers=2,
    )

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate
    )
    total_steps = args.max_train_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # ── Checkpoint-Hook für LoRA speichern ──
    def save_lora_hook(models, weights, output_dir):
        from peft.utils import get_peft_model_state_dict
        from diffusers.utils import convert_state_dict_to_diffusers
        unet_model = models[0]
        lora_state = get_peft_model_state_dict(unet_model)
        lora_state = convert_state_dict_to_diffusers(lora_state)
        # speichere unter output_dir/lora/
        save_path = os.path.join(output_dir, "lora")
        unet_model.save_attn_procs(save_path, safe_serialization=True)
        weights.clear()
    accelerator.register_save_state_pre_hook(save_lora_hook)

    # ── Mit Accelerator vorbereiten ──
    unet, optimizer, dl, scheduler = accelerator.prepare(
        unet, optimizer, dl, scheduler
    )

    # ── Trainings-Loop mit TQDM & Dummy-Loss ──
    progress = tqdm(total=total_steps, desc="Training")
    global_step = 0
    for batch in dl:
        # — Platzhalter — ersetze durch echten UNet-Forward + Loss
        # z.B.:
        #   latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        #   noisy = ...
        #   pred = unet(...)
        #   loss = F.mse_loss(pred, latents)
        loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        progress.update(1)
        progress.set_postfix(loss=loss.item())
        if global_step >= total_steps:
            break

    progress.close()
    accelerator.end_training()

if __name__ == "__main__":
    main()
