#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure
import os
# Head‑Tracking für Intrusion
from head_tracking import BlobHeadTracker, rightmost_x  # type: ignore

# -------------------------------------------------------------
# Konfiguration
# -------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DATA_ROOT_TESTVIDS = os.environ["DATA_ROOT_TESTVIDS"]
# Torch‑Metriken einmalig anlegen (State wird pro Paar zurückgesetzt)
_MSE_METRIC = MeanSquaredError()
_SSIM_METRIC = StructuralSimilarityIndexMeasure(data_range=1.0)


# -------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------

def _str2bool(v: str | bool) -> bool:
    """Argparse‑kompatible Umwandlung von String → Bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _strip_suffix(name: str) -> str:
    """Entfernt optionale Endungen wie `_XX00` aus Dateinamen."""
    return re.sub(r"_[A-Za-z]{2}\d{2}$", "", name)


def _list_basenames(folder: Path) -> set[str]:
    """Menge der Basenames ohne Endung, die ein Video im Ordner haben."""
    basenames: set[str] = set()
    for p in folder.iterdir():
        if p.suffix.lower() in VIDEO_EXTS:
            basenames.add(_strip_suffix(p.stem))
    return basenames


def _find_video(folder: Path, basename: str) -> Path | None:
    """Gibt den Pfad zum Video mit diesem Basename zurück (Ersttreffer)."""
    for p in folder.iterdir():
        if p.suffix.lower() in VIDEO_EXTS and _strip_suffix(p.stem) == basename:
            return p
    return None


def _read_video_frames(path: Path, device: torch.device) -> torch.Tensor:
    """Liest Video → Tensor [T,3,H,W] in float32, Werte∈[0,1]."""
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    if not frames:
        return torch.empty((0, 3, 0, 0), dtype=torch.float32, device=device)

    arr = torch.from_numpy(np.stack(frames, axis=0))  # [T,H,W,3] uint8
    return (arr.permute(0, 3, 1, 2).float() / 255.0).to(device)


def _align(gt: torch.Tensor, gen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Räumliches Resize + temporales Padding (wie im Streamlit‑Code)."""
    if gt.numel() == 0 or gen.numel() == 0:
        return gt, gen

    # --- Resize auf kleinste gemeinsame Größe ---
    _, _, H1, W1 = gt.shape
    _, _, H2, W2 = gen.shape
    H, W = min(H1, H2), min(W1, W2)
    if (H1, W1) != (H, W):
        gt = F.interpolate(gt, size=(H, W), mode="bilinear", align_corners=False)
    if (H2, W2) != (H, W):
        gen = F.interpolate(gen, size=(H, W), mode="bilinear", align_corners=False)

    # --- Padding auf gleiche Länge ---
    T = max(gt.shape[0], gen.shape[0])
    if gt.shape[0] < T:
        gt = torch.cat([gt, gt[-1:].repeat(T - gt.shape[0], 1, 1, 1)], 0)
    if gen.shape[0] < T:
        gen = torch.cat([gen, gen[-1:].repeat(T - gen.shape[0], 1, 1, 1)], 0)

    return gt.contiguous(), gen.contiguous()


# ------------------- Intrusion‑Metrik ------------------------------------

def _intrusion_depth(frames: torch.Tensor) -> float:
    """Maximale normierte Rechts‑Intrusion des lila Blobs (0–1)."""
    tracker = BlobHeadTracker(dE_thr=18, area_min=150, max_shift=120)
    masks, _ = tracker.track(frames)
    W = frames.shape[-1]
    xs = [rightmost_x(m) for m in masks if m is not None and m.any()]
    return max(xs) / (W - 1) if xs else 0.0


def _compute_metrics(
    gt_path: Path,
    gen_path: Path,
    want_mse: bool,
    want_ssim: bool,
    want_intrusion: bool,
    device: torch.device,
) -> Dict[str, float | None]:
    """Berechnet die gewünschten Metriken für ein Videopaar."""
    metrics: Dict[str, float | None] = {"mse": None, "ssim": None, "intrusion": None}

    gt = _read_video_frames(gt_path, device)
    gen = _read_video_frames(gen_path, device)
    if gt.numel() == 0 or gen.numel() == 0:
        raise RuntimeError("Leeres oder nicht lesbares Video")

    gt, gen = _align(gt, gen)

    if want_mse:
        _MSE_METRIC.reset()
        metrics["mse"] = _MSE_METRIC(gt, gen).item()

    if want_ssim:
        _SSIM_METRIC.reset()
        metrics["ssim"] = _SSIM_METRIC(gt, gen).item()

    if want_intrusion:
        depth_gt = _intrusion_depth(gt)
        depth_gen = _intrusion_depth(gen)
        metrics["intrusion"] = abs(depth_gen - depth_gt)

    return metrics


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch‑Berechnung von MSE, SSIM und Intrusion für Videopaare.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt_dir", required=True, type=Path, help="Ordner mit Ground‑Truth‑Videos")
    parser.add_argument("--gen_dir", required=True, type=Path, help="Ordner mit generierten Videos")
    parser.add_argument("--out_file", type=Path, default=Path("metrics_results.txt"), help="Ausgabedatei (.txt)")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch‑Device (cpu / cuda)")
    parser.add_argument("--mse", type=_str2bool, default=True, help="MSE berechnen?")
    parser.add_argument("--ssim", type=_str2bool, default=True, help="SSIM berechnen?")
    parser.add_argument("--intrusion", type=_str2bool, default=True, help="Intrusion berechnen?")

    args = parser.parse_args()

    device = torch.device(args.device)
    missing = []
    if not args.gt_dir.exists():
        missing.append(f"GT-Ordner '{args.gt_dir}'")
    if not args.gen_dir.exists():
        missing.append(f"GEN-Ordner '{args.gen_dir}'")
    if missing:
        sys.exit(f"❌ {' und '.join(missing)} nicht gefunden.")


    common = _list_basenames(args.gt_dir) & _list_basenames(args.gen_dir)
    if not common:
        sys.exit("❌ Keine gemeinsamen Videodateien gefunden.")

    header_cols = ["file"]
    if args.mse:
        header_cols.append("mse")
    if args.ssim:
        header_cols.append("ssim")
    if args.intrusion:
        header_cols.append("intrusion")

    lines: List[str] = ["\t".join(header_cols)]
    sums: Dict[str, float] = {k: 0.0 for k in header_cols if k != "file"}
    counts = 0

    for base in sorted(common):
        p_gt = _find_video(args.gt_dir, base)
        p_gen = _find_video(args.gen_dir, base)
        if p_gt is None or p_gen is None:
            print(f"⚠️  Überspringe {base}: Datei fehlt")
            continue

        try:
            res = _compute_metrics(
                p_gt,
                p_gen,
                want_mse=args.mse,
                want_ssim=args.ssim,
                want_intrusion=args.intrusion,
                device=device,
            )
        except Exception as e:
            print(f"⚠️  Fehler bei {base}: {e}")
            continue

        row = [base]
        for key in ("mse", "ssim", "intrusion"):
            if getattr(args, key):
                val = res[key]
                row.append(f"{val:.6f}" if val is not None else "nan")
                if val is not None:
                    sums[key] += val
        lines.append("\t".join(row))
        counts += 1

    # --- Durchschnitt ---
    if counts:
        avg_row = ["AVERAGE"]
        for key in ("mse", "ssim", "intrusion"):
            if getattr(args, key):
                avg = sums[key] / counts
                avg_row.append(f"{avg:.6f}")
        lines.append("\t".join(avg_row))

    # --- Schreiben ---
    args.out_file.write_text("\n".join(lines))
    print(f"✅ Fertig. Ergebnisse → {args.out_file}")


if __name__ == "__main__":
    main()
