from __future__ import annotations

import torch


def fuse_global_tokens(
    z_clstoken: torch.Tensor,
    z_regtokens: torch.Tensor,
    *,
    mode: str = "concat",
) -> tuple[torch.Tensor, dict]:
    """Fuse per-view DINO global tokens into the Pixal3D conditioning format."""
    tokens = torch.cat([z_clstoken, z_regtokens], dim=1)
    mode = mode.lower()
    if mode == "concat":
        fused = tokens.reshape(1, -1, tokens.shape[-1])
    elif mode == "mean":
        fused = tokens.mean(dim=0, keepdim=True)
    elif mode == "first":
        fused = tokens[:1]
    else:
        raise ValueError(f"Unknown global_fusion: {mode}")
    stats = {
        "global_fusion": mode,
        "num_views": int(tokens.shape[0]),
        "tokens_per_view": int(tokens.shape[1]),
        "global_token_count": int(fused.shape[1]),
        "feature_dim": int(fused.shape[2]),
    }
    return fused, stats
