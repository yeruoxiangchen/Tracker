#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

TRACKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACKER_ROOT))

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPCONV_ALGO", "native")

from ar_pose_trellis.condition import ARDinoRayCond
from ar_pose_trellis.objaverse_pose_dataset import ObjaversePoseDataset, custom_collate


class TinyPoseConditionTrainer(nn.Module):
    """CPU-friendly smoke target for ARDinoRayCond.

    This does not replace full TRELLIS SS-flow training. It verifies that the
    generated data, intrinsics/extrinsics, mask-conditioned ray features, and
    ARDinoRayCond are trainable end-to-end before launching a GPU job.
    """

    def __init__(self, patch_side: int = 16, channels: int = 64, tokens: int = 256):
        super().__init__()
        self.patch_side = patch_side
        self.cond = ARDinoRayCond(
            channels=channels,
            dino_channels=3,
            pose_channels=16,
            num_heads=4,
            mlp_ratio=2.0,
            num_init_tokens=tokens,
            use_image_features=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 4),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch["ref_image"].float()
        alpha = batch["alpha"].float()
        b, v, _, h, w = images.shape
        img_flat = (images * alpha).reshape(b * v, 3, h, w)
        patches = F.interpolate(
            img_flat,
            size=(self.patch_side, self.patch_side),
            mode="bilinear",
            align_corners=False,
        )
        patches = patches.reshape(b, v, 3, -1).permute(0, 1, 3, 2).contiguous()
        cond = self.cond(
            patches,
            batch["batch_intrinsics"].float(),
            batch["batch_extrinsics"].float(),
            masks=alpha,
            image_size=h,
            extrinsics_are_c2w=True,
            camera_forward_sign=1.0,
        )
        return self.head(cond.mean(dim=1))


def target_summary(batch: dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    coords = batch["target_coords"].long()
    out = []
    for b in range(batch_size):
        xyz = coords[coords[:, 0] == b][:, 1:].float()
        if xyz.numel() == 0:
            out.append(torch.zeros(4))
            continue
        mean_xyz = xyz.mean(dim=0) / 63.0
        fill = torch.tensor([min(float(xyz.shape[0]) / 8192.0, 1.0)])
        out.append(torch.cat([mean_xyz, fill], dim=0))
    return torch.stack(out, dim=0)


def save_batch_vis(batch: dict[str, torch.Tensor], pred: torch.Tensor, target: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = batch["ref_image"][0].detach().cpu().clamp(0, 1)
    alpha = batch["alpha"][0].detach().cpu().clamp(0, 1)
    tiles = []
    for i in range(images.shape[0]):
        rgb = (images[i].permute(1, 2, 0).numpy() * 255).astype("uint8")
        mask = (alpha[i, 0].numpy() * 255).astype("uint8")
        rgba = Image.fromarray(rgb).convert("RGBA")
        rgba.putalpha(Image.fromarray(mask))
        tiles.append(rgba.convert("RGB"))
    w, h = tiles[0].size
    label_h = 48
    sheet = Image.new("RGB", (w * len(tiles), h + label_h), (20, 20, 20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * w, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, h + 6), f"pred={pred[0].detach().cpu().numpy().round(3).tolist()}", fill=(230, 230, 230))
    draw.text((8, h + 24), f"target={target[0].detach().cpu().numpy().round(3).tolist()}", fill=(160, 220, 160))
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data/ar_pose_trellis/objaverse_pose_smoke")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch_side", type=int, default=16)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "outputs" / "smoke_runs" / f"ss_pose_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vis").mkdir(exist_ok=True)

    dataset = ObjaversePoseDataset(args.data_root, split=args.split, num_views=args.num_views)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        drop_last=True,
    )
    model = TinyPoseConditionTrainer(
        patch_side=args.patch_side,
        channels=args.channels,
        tokens=args.tokens,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log_path = output_dir / "train_log.csv"
    args_path = output_dir / "args.json"
    args_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    rows = []
    step = 0
    while step < args.steps:
        for batch in loader:
            pred = model(batch)
            target = target_summary(batch, pred.shape[0]).to(pred)
            loss = F.mse_loss(torch.sigmoid(pred), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows.append({"step": step, "loss": float(loss.detach().cpu())})
            if step == 0 or (step + 1) == args.steps:
                save_batch_vis(batch, torch.sigmoid(pred), target, output_dir / "vis" / f"step_{step:04d}.jpg")
            print(f"[smoke] step={step} loss={rows[-1]['loss']:.6f}", flush=True)
            step += 1
            if step >= args.steps:
                break

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(rows)

    ckpt_path = output_dir / "ckpt_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "steps": step,
            "last_loss": rows[-1]["loss"] if rows else None,
        },
        ckpt_path,
    )
    print(f"[smoke] wrote {ckpt_path}")
    print(f"[smoke] wrote {log_path}")
    print(f"[smoke] wrote visualizations under {output_dir / 'vis'}")


if __name__ == "__main__":
    main()
