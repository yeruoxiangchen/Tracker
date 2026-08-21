from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ARPoseRecord:
    image_name: str
    position: tuple[float, float, float]
    euler_deg: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    intrinsics: tuple[float, float, float, float]
    image_size: tuple[int, int]


def euler_xyz_deg_to_quaternion_xyzw(euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    rx, ry, rz = (math.radians(float(v)) for v in euler_deg)
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    # XYZ intrinsic order, sufficient as a deterministic pose descriptor for
    # feature plumbing. Camera convention is still an ablation variable.
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def parse_ar_pose_file(
    path: str | Path,
    *,
    default_intrinsics: tuple[float, float, float, float] = (485.845947, 485.744232, 322.973236, 237.599487),
    default_image_size: tuple[int, int] = (640, 480),
) -> dict[str, ARPoseRecord]:
    """Parse ReconViaGen AR tracker poses.txt rows.

    Full AR rows are comma-separated and start with:

        image_name, tx,ty,tz, euler_x,euler_y,euler_z,
        qx,qy,qz,qw, fx,fy,cx,cy, width,height, ...

    Some prepared AR-session rows only contain:

        image_name, tx,ty,tz, euler_x,euler_y,euler_z

    For those rows, quaternion is derived from Euler angles and intrinsics use
    explicit defaults. This keeps B-stage feature plumbing usable across both
    raw AR captures and prepared point-prior datasets.
    """

    records: dict[str, ARPoseRecord] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 7:
                continue
            image_name = Path(row[0]).name
            try:
                position = tuple(float(v) for v in row[1:4])
                euler_deg = tuple(float(v) for v in row[4:7])
                if len(row) >= 17:
                    quat = tuple(float(v) for v in row[7:11])
                    intrinsics = tuple(float(v) for v in row[11:15])
                    image_size = (int(float(row[15])), int(float(row[16])))
                else:
                    quat = euler_xyz_deg_to_quaternion_xyzw(euler_deg)  # type: ignore[arg-type]
                    intrinsics = tuple(float(v) for v in default_intrinsics)
                    image_size = tuple(int(v) for v in default_image_size)
            except ValueError as exc:
                raise ValueError(f"Failed to parse pose row for {image_name}: {row}") from exc
            records[image_name] = ARPoseRecord(
                image_name=image_name,
                position=position,  # type: ignore[arg-type]
                euler_deg=euler_deg,  # type: ignore[arg-type]
                quaternion_xyzw=quat,  # type: ignore[arg-type]
                intrinsics=intrinsics,  # type: ignore[arg-type]
                image_size=image_size,
            )
    return records


def select_pose_records(image_names: Iterable[str], records: dict[str, ARPoseRecord]) -> list[ARPoseRecord]:
    selected: list[ARPoseRecord] = []
    missing: list[str] = []
    for name in image_names:
        key = Path(name).name
        record = records.get(key)
        if record is None:
            missing.append(key)
        else:
            selected.append(record)
    if missing:
        raise KeyError(f"Missing pose rows for selected images: {missing}")
    return selected


def _normalize_quaternion_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def quaternion_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternions to rotation matrices.

    The AR pose convention may still need task-level calibration, but this
    representation is sufficient for B-stage zero-init feature plumbing and
    later ablations because it preserves a consistent orientation descriptor.
    """

    q = _normalize_quaternion_xyzw(q.float())
    x, y, z, w = q.unbind(dim=-1)
    two = torch.tensor(2.0, dtype=q.dtype, device=q.device)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack((1 - two * (yy + zz), two * (xy - wz), two * (xz + wy)), dim=-1)
    row1 = torch.stack((two * (xy + wz), 1 - two * (xx + zz), two * (yz - wx)), dim=-1)
    row2 = torch.stack((two * (xz - wy), two * (yz + wx), 1 - two * (xx + yy)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def build_pose_token_features(
    pose_records: list[ARPoseRecord],
    *,
    token_grid_side: int,
    image_resolution: int = 518,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build per-view per-token pose/ray features.

    Returns [1, V, token_grid_side^2, F].  The feature vector is deliberately
    compact and deterministic:

    - normalized token center x/y and radius
    - camera-space ray and quaternion-rotated ray
    - normalized camera position
    - quaternion xyzw
    - resized intrinsics
    - normalized view index and valid flag
    """

    if token_grid_side <= 0:
        raise ValueError(f"token_grid_side must be positive, got {token_grid_side}")
    if not pose_records:
        raise ValueError("pose_records is empty")

    dev = torch.device(device)
    positions = torch.tensor([r.position for r in pose_records], dtype=torch.float32, device=dev)
    pos_mean = positions.mean(dim=0, keepdim=True)
    pos_scale = (positions - pos_mean).norm(dim=-1).amax().clamp_min(1.0e-6)
    positions_norm = (positions - pos_mean) / pos_scale

    ys, xs = torch.meshgrid(
        torch.arange(token_grid_side, dtype=torch.float32, device=dev),
        torch.arange(token_grid_side, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    centers_x = (xs + 0.5) * (float(image_resolution) / float(token_grid_side))
    centers_y = (ys + 0.5) * (float(image_resolution) / float(token_grid_side))
    x_norm = centers_x / float(image_resolution) * 2.0 - 1.0
    y_norm = centers_y / float(image_resolution) * 2.0 - 1.0
    r_norm = torch.sqrt((x_norm * x_norm + y_norm * y_norm).clamp_min(0.0))
    base_grid = torch.stack((x_norm, y_norm, r_norm), dim=-1).reshape(-1, 3)

    view_features: list[torch.Tensor] = []
    denom = max(1, len(pose_records) - 1)
    for view_idx, record in enumerate(pose_records):
        fx, fy, cx, cy = record.intrinsics
        orig_w, orig_h = record.image_size
        sx = float(image_resolution) / float(max(1, orig_w))
        sy = float(image_resolution) / float(max(1, orig_h))
        fx_r = fx * sx
        fy_r = fy * sy
        cx_r = cx * sx
        cy_r = cy * sy

        ray = torch.stack(
            (
                (centers_x - cx_r) / max(fx_r, 1.0e-6),
                (centers_y - cy_r) / max(fy_r, 1.0e-6),
                torch.ones_like(centers_x),
            ),
            dim=-1,
        ).reshape(-1, 3)
        ray = ray / ray.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

        quat = torch.tensor(record.quaternion_xyzw, dtype=torch.float32, device=dev)
        rot = quaternion_xyzw_to_matrix(quat.unsqueeze(0))[0]
        ray_world = ray @ rot.transpose(0, 1)
        ray_world = ray_world / ray_world.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

        n_tokens = token_grid_side * token_grid_side
        pos_feat = positions_norm[view_idx].expand(n_tokens, 3)
        quat_feat = _normalize_quaternion_xyzw(quat).expand(n_tokens, 4)
        intr_feat = torch.tensor(
            [fx_r / image_resolution, fy_r / image_resolution, cx_r / image_resolution, cy_r / image_resolution],
            dtype=torch.float32,
            device=dev,
        ).expand(n_tokens, 4)
        view_feat = torch.tensor([float(view_idx) / float(denom), 1.0], dtype=torch.float32, device=dev).expand(n_tokens, 2)
        view_features.append(torch.cat((base_grid, ray, ray_world, pos_feat, quat_feat, intr_feat, view_feat), dim=-1))

    return torch.stack(view_features, dim=0).unsqueeze(0).to(dtype=dtype)


def summarize_pose_features(features: torch.Tensor) -> dict:
    y = features.detach().float()
    return {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "device": str(features.device),
        "feature_dim": int(features.shape[-1]) if features.ndim else None,
        "mean": float(y.mean().item()) if y.numel() else 0.0,
        "std": float(y.std().item()) if y.numel() > 1 else 0.0,
        "abs_max": float(y.abs().max().item()) if y.numel() else 0.0,
    }


def _mask_path_for_image(mask_dir: str | Path, image_name: str | Path) -> Path:
    root = Path(mask_dir)
    image_path = Path(image_name)
    candidates = [
        root / f"{image_path.stem}.png",
        root / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing mask for image={image_path.name} in {root}")


def load_resized_mask_tensor(
    mask_dir: str | Path,
    image_name: str | Path,
    *,
    image_resolution: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    mask_path = _mask_path_for_image(mask_dir, image_name)
    mask = Image.open(mask_path).convert("L").resize(
        (int(image_resolution), int(image_resolution)),
        Image.Resampling.BILINEAR,
    )
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=torch.device(device), dtype=torch.float32)


def build_token_mask_features(
    image_names: list[str],
    *,
    mask_dir: str | Path,
    token_grid_side: int,
    image_resolution: int = 518,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    mask_threshold: int = 127,
    token_min_ratio: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Build per-token foreground mask features.

    Returns [1,V,S,2]:

    - foreground ratio in the token cell
    - foreground hit flag after token_min_ratio threshold

    The function intentionally keeps full-frame geometry.  It does not crop or
    remap intrinsics; crop-aware projection should be a separate ablation.
    """

    if int(image_resolution) % int(token_grid_side) != 0:
        raise ValueError(
            f"image_resolution={image_resolution} must be divisible by token_grid_side={token_grid_side}"
        )
    dev = torch.device(device)
    cell = int(image_resolution) // int(token_grid_side)
    rows: list[torch.Tensor] = []
    summaries: list[dict] = []
    for image_name in image_names:
        mask = load_resized_mask_tensor(mask_dir, image_name, image_resolution=image_resolution, device=dev)
        binary = (mask > (float(mask_threshold) / 255.0)).float()
        cells = binary.reshape(token_grid_side, cell, token_grid_side, cell).permute(0, 2, 1, 3).reshape(
            token_grid_side * token_grid_side,
            cell * cell,
        )
        ratio = cells.mean(dim=-1)
        hit = (ratio >= float(token_min_ratio)).float()
        rows.append(torch.stack((ratio, hit), dim=-1))
        summaries.append(
            {
                "image": str(image_name),
                "foreground_ratio": float(binary.mean().detach().cpu().item()),
                "token_hit_count": int(hit.sum().detach().cpu().item()),
                "token_hit_ratio": float(hit.mean().detach().cpu().item()),
                "token_min_ratio": float(token_min_ratio),
            }
        )
    out = torch.stack(rows, dim=0).unsqueeze(0).to(dtype=dtype)
    return out, {
        "mask_dir": str(mask_dir),
        "feature_dim": 2,
        "mask_threshold": int(mask_threshold),
        "token_min_ratio": float(token_min_ratio),
        "per_view": summaries,
        "foreground_ratio_mean": float(sum(x["foreground_ratio"] for x in summaries) / max(1, len(summaries))),
        "token_hit_count_total": int(sum(x["token_hit_count"] for x in summaries)),
        "token_hit_ratio_mean": float(sum(x["token_hit_ratio"] for x in summaries) / max(1, len(summaries))),
    }


def load_points3d_txt(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    points: list[list[float]] = []
    conf: list[float] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            try:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                err = max(0.0, float(parts[7]))
                conf.append(1.0 / (1.0 + err))
            except ValueError as exc:
                raise ValueError(f"Failed to parse points3D row: {line}") from exc
    if not points:
        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
    return torch.tensor(points, dtype=torch.float32), torch.tensor(conf, dtype=torch.float32)


def load_prior_npz_points(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    with np.load(path) as data:
        if "source_points" in data.files:
            points = np.asarray(data["source_points"], dtype=np.float32)
            if "source_conf" in data.files:
                conf = np.asarray(data["source_conf"], dtype=np.float32)
            elif "prior_conf" in data.files and data["prior_conf"].shape[0] == points.shape[0]:
                conf = np.asarray(data["prior_conf"], dtype=np.float32)
            else:
                conf = np.ones((points.shape[0],), dtype=np.float32)
        elif {"prior_coords", "normalization_center", "normalization_scale"}.issubset(set(data.files)):
            coords = np.asarray(data["prior_coords"], dtype=np.float32)
            center = np.asarray(data["normalization_center"], dtype=np.float32).reshape(1, 3)
            scale = float(np.asarray(data["normalization_scale"], dtype=np.float32).reshape(-1)[0])
            points = ((coords + 0.5) / 64.0 - 0.5) * scale + center
            conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data.files else np.ones((points.shape[0],), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported prior npz keys in {path}: {data.files}")
    points_t = torch.from_numpy(points.astype(np.float32, copy=False))
    conf_t = torch.from_numpy(conf.astype(np.float32, copy=False)).reshape(-1)
    if conf_t.numel() != points_t.shape[0]:
        conf_t = torch.ones((points_t.shape[0],), dtype=torch.float32)
    return points_t, conf_t


def _point_support_summary(
    *,
    enabled: bool,
    point_count_before: int,
    point_count_after: int,
    min_support_views: int,
    min_support_ratio: float,
    visible_counts: torch.Tensor | None = None,
    mask_hit_counts: torch.Tensor | None = None,
    keep: torch.Tensor | None = None,
    per_view_inside: list[int] | None = None,
    per_view_mask_hit: list[int] | None = None,
) -> dict:
    summary = {
        "enabled": bool(enabled),
        "mode": "multiview_mask" if enabled else "none",
        "point_count_before": int(point_count_before),
        "point_count_after": int(point_count_after),
        "kept_ratio": float(point_count_after / max(1, point_count_before)),
        "min_support_views": int(min_support_views),
        "min_support_ratio": float(min_support_ratio),
    }
    if visible_counts is None or mask_hit_counts is None:
        return summary

    visible = visible_counts.float()
    hit = mask_hit_counts.float()
    ratio = torch.where(visible > 0, hit / visible.clamp_min(1.0), torch.zeros_like(hit))

    def _mean(x: torch.Tensor) -> float:
        return float(x.mean().detach().cpu().item()) if x.numel() else 0.0

    summary.update(
        {
            "visible_count_mean": _mean(visible),
            "visible_count_max": int(visible_counts.max().detach().cpu().item()) if visible_counts.numel() else 0,
            "mask_hit_count_mean": _mean(hit),
            "mask_hit_count_max": int(mask_hit_counts.max().detach().cpu().item()) if mask_hit_counts.numel() else 0,
            "support_ratio_mean": _mean(ratio),
            "support_ratio_max": float(ratio.max().detach().cpu().item()) if ratio.numel() else 0.0,
            "per_view_inside_count_before_support_filter": per_view_inside or [],
            "per_view_mask_hit_count_before_support_filter": per_view_mask_hit or [],
        }
    )
    if keep is not None and bool(keep.any()):
        summary.update(
            {
                "kept_visible_count_mean": _mean(visible[keep]),
                "kept_mask_hit_count_mean": _mean(hit[keep]),
                "kept_support_ratio_mean": _mean(ratio[keep]),
            }
        )
    else:
        summary.update(
            {
                "kept_visible_count_mean": 0.0,
                "kept_mask_hit_count_mean": 0.0,
                "kept_support_ratio_mean": 0.0,
            }
        )
    return summary


def qvec_wxyz_to_matrix(qvec: torch.Tensor) -> torch.Tensor:
    q = qvec.float()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = q.unbind(dim=-1)
    two = torch.tensor(2.0, dtype=q.dtype, device=q.device)
    row0 = torch.stack((1 - two * y * y - two * z * z, two * x * y - two * z * w, two * x * z + two * y * w), dim=-1)
    row1 = torch.stack((two * x * y + two * z * w, 1 - two * x * x - two * z * z, two * y * z - two * x * w), dim=-1)
    row2 = torch.stack((two * x * z - two * y * w, two * y * z + two * x * w, 1 - two * x * x - two * y * y), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def parse_colmap_cameras(path: str | Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"Unsupported COLMAP camera model {model!r} in {path}")
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
    return cameras


def parse_colmap_images(path: str | Path) -> dict[str, dict]:
    images: dict[str, dict] = {}
    raw = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(raw):
        line = raw[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = Path(" ".join(parts[9:])).name
        images[name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "qvec": qvec,
            "tvec": tvec,
        }
        if i < len(raw):
            i += 1
    return images


def build_point_projection_features(
    points: torch.Tensor,
    confidences: torch.Tensor,
    pose_records: list[ARPoseRecord],
    *,
    token_grid_side: int,
    image_resolution: int = 518,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    rotation_mode: str = "c2w",
    min_depth: float = 1.0e-4,
    mask_dir: str | Path | None = None,
    image_names: list[str] | None = None,
    mask_threshold: int = 127,
    point_mask_support_min_views: int = 0,
    point_mask_support_min_ratio: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Project sparse 3D prior points into each 37x37 VGGT token grid.

    Returns [1,V,S,7] features:

    hit, log_count, mean_conf, mean_inv_depth, min_inv_depth,
    mean_u_norm, mean_v_norm

    The camera convention is intentionally configurable because AR pose export
    conventions differ. B1 first uses this as a plumbing diagnostic; geometric
    calibration can then be ablated with rotation_mode.
    """

    if rotation_mode not in {"c2w", "w2c"}:
        raise ValueError(f"rotation_mode must be c2w or w2c, got {rotation_mode}")
    dev = torch.device(device)
    n_tokens = int(token_grid_side) * int(token_grid_side)
    n_views = len(pose_records)
    out = torch.zeros((n_views, n_tokens, 7), dtype=torch.float32, device=dev)
    points = points.to(device=dev, dtype=torch.float32)
    confidences = confidences.to(device=dev, dtype=torch.float32).reshape(-1)
    if points.numel() == 0:
        return out.unsqueeze(0).to(dtype=dtype), {
            "point_count": 0,
            "point_count_before_support_filter": 0,
            "per_view_inside_count": [0 for _ in pose_records],
            "per_view_mask_hit_count": [0 for _ in pose_records],
            "inside_count_total": 0,
            "mask_hit_count_total": 0,
            "mask_filter_enabled": bool(mask_dir),
            "point_mask_support_filter": _point_support_summary(
                enabled=False,
                point_count_before=0,
                point_count_after=0,
                min_support_views=point_mask_support_min_views,
                min_support_ratio=point_mask_support_min_ratio,
            ),
            "rotation_mode": rotation_mode,
            "feature_dim": 7,
        }
    if confidences.numel() != points.shape[0]:
        confidences = torch.ones((points.shape[0],), dtype=torch.float32, device=dev)

    support_filter_enabled = bool(mask_dir) and (
        int(point_mask_support_min_views) > 0 or float(point_mask_support_min_ratio) > 0.0
    )
    support_summary = _point_support_summary(
        enabled=False,
        point_count_before=int(points.shape[0]),
        point_count_after=int(points.shape[0]),
        min_support_views=point_mask_support_min_views,
        min_support_ratio=point_mask_support_min_ratio,
    )
    if support_filter_enabled:
        if image_names is None:
            raise ValueError("image_names are required when point_mask_support filtering is used")
        visible_counts = torch.zeros((points.shape[0],), dtype=torch.long, device=dev)
        mask_hit_counts = torch.zeros_like(visible_counts)
        support_per_view_inside: list[int] = []
        support_per_view_mask_hit: list[int] = []
        for view_idx, record in enumerate(pose_records):
            fx, fy, cx, cy = record.intrinsics
            orig_w, orig_h = record.image_size
            sx = float(image_resolution) / float(max(1, orig_w))
            sy = float(image_resolution) / float(max(1, orig_h))
            fx_r = fx * sx
            fy_r = fy * sy
            cx_r = cx * sx
            cy_r = cy * sy

            pos = torch.tensor(record.position, dtype=torch.float32, device=dev).reshape(1, 3)
            quat = torch.tensor(record.quaternion_xyzw, dtype=torch.float32, device=dev)
            rot = quaternion_xyzw_to_matrix(quat.unsqueeze(0))[0]
            centered = points - pos
            cam = centered @ rot if rotation_mode == "c2w" else centered @ rot.transpose(0, 1)
            z = cam[:, 2]
            valid_z = z > float(min_depth)
            u = torch.empty_like(z)
            v = torch.empty_like(z)
            u[valid_z] = float(fx_r) * cam[valid_z, 0] / z[valid_z] + float(cx_r)
            v[valid_z] = float(fy_r) * cam[valid_z, 1] / z[valid_z] + float(cy_r)
            inside = valid_z & (u >= 0) & (u < image_resolution) & (v >= 0) & (v < image_resolution)
            support_per_view_inside.append(int(inside.sum().detach().cpu().item()))
            visible_counts += inside.long()
            if bool(inside.any()):
                mask = load_resized_mask_tensor(mask_dir, image_names[view_idx], image_resolution=image_resolution, device=dev)
                ui = torch.clamp(torch.round(u[inside]).long(), 0, int(image_resolution) - 1)
                vi = torch.clamp(torch.round(v[inside]).long(), 0, int(image_resolution) - 1)
                hit_inside = mask[vi, ui] > (float(mask_threshold) / 255.0)
                hit_indices = inside.nonzero(as_tuple=False).reshape(-1)[hit_inside]
                mask_hit_counts[hit_indices] += 1
                support_per_view_mask_hit.append(int(hit_inside.sum().detach().cpu().item()))
            else:
                support_per_view_mask_hit.append(0)
        visible = visible_counts.float()
        ratio = torch.where(visible > 0, mask_hit_counts.float() / visible.clamp_min(1.0), torch.zeros_like(visible))
        keep = visible_counts > 0
        if int(point_mask_support_min_views) > 0:
            keep = keep & (mask_hit_counts >= int(point_mask_support_min_views))
        if float(point_mask_support_min_ratio) > 0.0:
            keep = keep & (ratio >= float(point_mask_support_min_ratio))
        support_summary = _point_support_summary(
            enabled=True,
            point_count_before=int(points.shape[0]),
            point_count_after=int(keep.sum().detach().cpu().item()),
            min_support_views=point_mask_support_min_views,
            min_support_ratio=point_mask_support_min_ratio,
            visible_counts=visible_counts,
            mask_hit_counts=mask_hit_counts,
            keep=keep,
            per_view_inside=support_per_view_inside,
            per_view_mask_hit=support_per_view_mask_hit,
        )
        points = points[keep]
        confidences = confidences[keep]
        if points.numel() == 0:
            return out.unsqueeze(0).to(dtype=dtype), {
                "point_count": 0,
                "point_count_before_support_filter": int(support_summary["point_count_before"]),
                "per_view_inside_count": [0 for _ in pose_records],
                "per_view_mask_hit_count": [0 for _ in pose_records],
                "inside_count_total": 0,
                "mask_hit_count_total": 0,
                "mask_filter_enabled": bool(mask_dir),
                "point_mask_support_filter": support_summary,
                "rotation_mode": rotation_mode,
                "feature_dim": 7,
            }

    per_view_inside: list[int] = []
    per_view_mask_hit: list[int] = []
    for view_idx, record in enumerate(pose_records):
        fx, fy, cx, cy = record.intrinsics
        orig_w, orig_h = record.image_size
        sx = float(image_resolution) / float(max(1, orig_w))
        sy = float(image_resolution) / float(max(1, orig_h))
        fx_r = fx * sx
        fy_r = fy * sy
        cx_r = cx * sx
        cy_r = cy * sy

        pos = torch.tensor(record.position, dtype=torch.float32, device=dev).reshape(1, 3)
        quat = torch.tensor(record.quaternion_xyzw, dtype=torch.float32, device=dev)
        rot = quaternion_xyzw_to_matrix(quat.unsqueeze(0))[0]
        centered = points - pos
        if rotation_mode == "c2w":
            cam = centered @ rot
        else:
            cam = centered @ rot.transpose(0, 1)
        z = cam[:, 2]
        valid_z = z > float(min_depth)
        u = torch.empty_like(z)
        v = torch.empty_like(z)
        u[valid_z] = float(fx_r) * cam[valid_z, 0] / z[valid_z] + float(cx_r)
        v[valid_z] = float(fy_r) * cam[valid_z, 1] / z[valid_z] + float(cy_r)
        inside = valid_z & (u >= 0) & (u < image_resolution) & (v >= 0) & (v < image_resolution)
        per_view_inside.append(int(inside.sum().item()))
        if not bool(inside.any()):
            per_view_mask_hit.append(0)
            continue

        u_i = u[inside]
        v_i = v[inside]
        z_i = z[inside]
        conf_i = confidences[inside].clamp_min(0.0)
        if mask_dir:
            if image_names is None:
                raise ValueError("image_names are required when mask_dir is used")
            mask = load_resized_mask_tensor(mask_dir, image_names[view_idx], image_resolution=image_resolution, device=dev)
            ui = torch.clamp(torch.round(u_i).long(), 0, int(image_resolution) - 1)
            vi = torch.clamp(torch.round(v_i).long(), 0, int(image_resolution) - 1)
            keep = mask[vi, ui] > (float(mask_threshold) / 255.0)
            per_view_mask_hit.append(int(keep.sum().item()))
            if not bool(keep.any()):
                continue
            u_i = u_i[keep]
            v_i = v_i[keep]
            z_i = z_i[keep]
            conf_i = conf_i[keep]
        else:
            per_view_mask_hit.append(int(inside.sum().item()))
        gx = torch.clamp((u_i / float(image_resolution) * token_grid_side).floor().long(), 0, token_grid_side - 1)
        gy = torch.clamp((v_i / float(image_resolution) * token_grid_side).floor().long(), 0, token_grid_side - 1)
        cell = gy * token_grid_side + gx
        ones = torch.ones_like(conf_i)
        inv_depth = 1.0 / z_i.clamp_min(float(min_depth))
        u_norm = u_i / float(image_resolution) * 2.0 - 1.0
        v_norm = v_i / float(image_resolution) * 2.0 - 1.0

        count = torch.zeros((n_tokens,), dtype=torch.float32, device=dev)
        conf_sum = torch.zeros_like(count)
        inv_sum = torch.zeros_like(count)
        u_sum = torch.zeros_like(count)
        v_sum = torch.zeros_like(count)
        min_inv = torch.zeros_like(count)
        count.index_add_(0, cell, ones)
        conf_sum.index_add_(0, cell, conf_i)
        inv_sum.index_add_(0, cell, inv_depth)
        u_sum.index_add_(0, cell, u_norm)
        v_sum.index_add_(0, cell, v_norm)
        min_inv.scatter_reduce_(0, cell, inv_depth, reduce="amax", include_self=True)

        denom = count.clamp_min(1.0)
        hit = (count > 0).float()
        log_count = torch.log1p(count) / math.log1p(max(1.0, float(count.max().item())))
        out[view_idx, :, 0] = hit
        out[view_idx, :, 1] = log_count
        out[view_idx, :, 2] = conf_sum / denom
        out[view_idx, :, 3] = inv_sum / denom
        out[view_idx, :, 4] = min_inv
        out[view_idx, :, 5] = u_sum / denom
        out[view_idx, :, 6] = v_sum / denom

    return out.unsqueeze(0).to(dtype=dtype), {
        "point_count": int(points.shape[0]),
        "point_count_before_support_filter": int(support_summary["point_count_before"]),
        "per_view_inside_count": per_view_inside,
        "per_view_mask_hit_count": per_view_mask_hit,
        "inside_count_total": int(sum(per_view_inside)),
        "mask_hit_count_total": int(sum(per_view_mask_hit)),
        "mask_filter_enabled": bool(mask_dir),
        "point_mask_support_filter": support_summary,
        "rotation_mode": rotation_mode,
        "feature_dim": 7,
    }


def build_colmap_point_projection_features(
    points: torch.Tensor,
    confidences: torch.Tensor,
    image_names: list[str],
    *,
    colmap_cameras: dict[int, dict],
    colmap_images: dict[str, dict],
    token_grid_side: int,
    image_resolution: int = 518,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    min_depth: float = 1.0e-4,
    mask_dir: str | Path | None = None,
    mask_threshold: int = 127,
    point_mask_support_min_views: int = 0,
    point_mask_support_min_ratio: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Project point prior using COLMAP world-to-camera qvec/tvec."""

    dev = torch.device(device)
    n_tokens = int(token_grid_side) * int(token_grid_side)
    n_views = len(image_names)
    out = torch.zeros((n_views, n_tokens, 7), dtype=torch.float32, device=dev)
    points = points.to(device=dev, dtype=torch.float32)
    confidences = confidences.to(device=dev, dtype=torch.float32).reshape(-1)
    if confidences.numel() != points.shape[0]:
        confidences = torch.ones((points.shape[0],), dtype=torch.float32, device=dev)
    per_view_inside: list[int] = []
    per_view_mask_hit: list[int] = []
    matched_images: list[str] = []
    if points.numel() == 0:
        return out.unsqueeze(0).to(dtype=dtype), {
            "point_count": 0,
            "point_count_before_support_filter": 0,
            "per_view_inside_count": [0 for _ in image_names],
            "per_view_mask_hit_count": [0 for _ in image_names],
            "inside_count_total": 0,
            "mask_hit_count_total": 0,
            "mask_filter_enabled": bool(mask_dir),
            "point_mask_support_filter": _point_support_summary(
                enabled=False,
                point_count_before=0,
                point_count_after=0,
                min_support_views=point_mask_support_min_views,
                min_support_ratio=point_mask_support_min_ratio,
            ),
            "projection_source": "colmap",
            "matched_images": [],
            "feature_dim": 7,
        }

    support_filter_enabled = bool(mask_dir) and (
        int(point_mask_support_min_views) > 0 or float(point_mask_support_min_ratio) > 0.0
    )
    support_summary = _point_support_summary(
        enabled=False,
        point_count_before=int(points.shape[0]),
        point_count_after=int(points.shape[0]),
        min_support_views=point_mask_support_min_views,
        min_support_ratio=point_mask_support_min_ratio,
    )
    if support_filter_enabled:
        visible_counts = torch.zeros((points.shape[0],), dtype=torch.long, device=dev)
        mask_hit_counts = torch.zeros_like(visible_counts)
        support_per_view_inside: list[int] = []
        support_per_view_mask_hit: list[int] = []
        for image_name in image_names:
            key = Path(image_name).name
            meta = colmap_images.get(key)
            if meta is None:
                raise KeyError(f"Missing COLMAP image row for {key}")
            camera = colmap_cameras.get(int(meta["camera_id"]))
            if camera is None:
                raise KeyError(f"Missing COLMAP camera row for camera_id={meta['camera_id']} image={key}")
            sx = float(image_resolution) / float(max(1, int(camera["width"])))
            sy = float(image_resolution) / float(max(1, int(camera["height"])))
            fx = float(camera["fx"]) * sx
            fy = float(camera["fy"]) * sy
            cx = float(camera["cx"]) * sx
            cy = float(camera["cy"]) * sy
            qvec = torch.tensor(meta["qvec"], dtype=torch.float32, device=dev).reshape(1, 4)
            tvec = torch.tensor(meta["tvec"], dtype=torch.float32, device=dev).reshape(1, 3)
            rot = qvec_wxyz_to_matrix(qvec)[0]
            cam = points @ rot.transpose(0, 1) + tvec
            z = cam[:, 2]
            valid_z = z > float(min_depth)
            u = torch.empty_like(z)
            v = torch.empty_like(z)
            u[valid_z] = fx * cam[valid_z, 0] / z[valid_z] + cx
            v[valid_z] = fy * cam[valid_z, 1] / z[valid_z] + cy
            inside = valid_z & (u >= 0) & (u < image_resolution) & (v >= 0) & (v < image_resolution)
            support_per_view_inside.append(int(inside.sum().detach().cpu().item()))
            visible_counts += inside.long()
            if bool(inside.any()):
                mask = load_resized_mask_tensor(mask_dir, key, image_resolution=image_resolution, device=dev)
                ui = torch.clamp(torch.round(u[inside]).long(), 0, int(image_resolution) - 1)
                vi = torch.clamp(torch.round(v[inside]).long(), 0, int(image_resolution) - 1)
                hit_inside = mask[vi, ui] > (float(mask_threshold) / 255.0)
                hit_indices = inside.nonzero(as_tuple=False).reshape(-1)[hit_inside]
                mask_hit_counts[hit_indices] += 1
                support_per_view_mask_hit.append(int(hit_inside.sum().detach().cpu().item()))
            else:
                support_per_view_mask_hit.append(0)
        visible = visible_counts.float()
        ratio = torch.where(visible > 0, mask_hit_counts.float() / visible.clamp_min(1.0), torch.zeros_like(visible))
        keep = visible_counts > 0
        if int(point_mask_support_min_views) > 0:
            keep = keep & (mask_hit_counts >= int(point_mask_support_min_views))
        if float(point_mask_support_min_ratio) > 0.0:
            keep = keep & (ratio >= float(point_mask_support_min_ratio))
        support_summary = _point_support_summary(
            enabled=True,
            point_count_before=int(points.shape[0]),
            point_count_after=int(keep.sum().detach().cpu().item()),
            min_support_views=point_mask_support_min_views,
            min_support_ratio=point_mask_support_min_ratio,
            visible_counts=visible_counts,
            mask_hit_counts=mask_hit_counts,
            keep=keep,
            per_view_inside=support_per_view_inside,
            per_view_mask_hit=support_per_view_mask_hit,
        )
        points = points[keep]
        confidences = confidences[keep]
        if points.numel() == 0:
            return out.unsqueeze(0).to(dtype=dtype), {
                "point_count": 0,
                "point_count_before_support_filter": int(support_summary["point_count_before"]),
                "per_view_inside_count": [0 for _ in image_names],
                "per_view_mask_hit_count": [0 for _ in image_names],
                "inside_count_total": 0,
                "mask_hit_count_total": 0,
                "mask_filter_enabled": bool(mask_dir),
                "point_mask_support_filter": support_summary,
                "projection_source": "colmap",
                "matched_images": [],
                "feature_dim": 7,
            }

    for view_idx, image_name in enumerate(image_names):
        key = Path(image_name).name
        meta = colmap_images.get(key)
        if meta is None:
            raise KeyError(f"Missing COLMAP image row for {key}")
        camera = colmap_cameras.get(int(meta["camera_id"]))
        if camera is None:
            raise KeyError(f"Missing COLMAP camera row for camera_id={meta['camera_id']} image={key}")
        matched_images.append(key)
        sx = float(image_resolution) / float(max(1, int(camera["width"])))
        sy = float(image_resolution) / float(max(1, int(camera["height"])))
        fx = float(camera["fx"]) * sx
        fy = float(camera["fy"]) * sy
        cx = float(camera["cx"]) * sx
        cy = float(camera["cy"]) * sy
        qvec = torch.tensor(meta["qvec"], dtype=torch.float32, device=dev).reshape(1, 4)
        tvec = torch.tensor(meta["tvec"], dtype=torch.float32, device=dev).reshape(1, 3)
        rot = qvec_wxyz_to_matrix(qvec)[0]
        cam = points @ rot.transpose(0, 1) + tvec
        z = cam[:, 2]
        valid_z = z > float(min_depth)
        u = torch.empty_like(z)
        v = torch.empty_like(z)
        u[valid_z] = fx * cam[valid_z, 0] / z[valid_z] + cx
        v[valid_z] = fy * cam[valid_z, 1] / z[valid_z] + cy
        inside = valid_z & (u >= 0) & (u < image_resolution) & (v >= 0) & (v < image_resolution)
        per_view_inside.append(int(inside.sum().item()))
        if not bool(inside.any()):
            per_view_mask_hit.append(0)
            continue

        u_i = u[inside]
        v_i = v[inside]
        z_i = z[inside]
        conf_i = confidences[inside].clamp_min(0.0)
        if mask_dir:
            mask = load_resized_mask_tensor(mask_dir, key, image_resolution=image_resolution, device=dev)
            ui = torch.clamp(torch.round(u_i).long(), 0, int(image_resolution) - 1)
            vi = torch.clamp(torch.round(v_i).long(), 0, int(image_resolution) - 1)
            keep = mask[vi, ui] > (float(mask_threshold) / 255.0)
            per_view_mask_hit.append(int(keep.sum().item()))
            if not bool(keep.any()):
                continue
            u_i = u_i[keep]
            v_i = v_i[keep]
            z_i = z_i[keep]
            conf_i = conf_i[keep]
        else:
            per_view_mask_hit.append(int(inside.sum().item()))
        gx = torch.clamp((u_i / float(image_resolution) * token_grid_side).floor().long(), 0, token_grid_side - 1)
        gy = torch.clamp((v_i / float(image_resolution) * token_grid_side).floor().long(), 0, token_grid_side - 1)
        cell = gy * token_grid_side + gx
        ones = torch.ones_like(conf_i)
        inv_depth = 1.0 / z_i.clamp_min(float(min_depth))
        u_norm = u_i / float(image_resolution) * 2.0 - 1.0
        v_norm = v_i / float(image_resolution) * 2.0 - 1.0

        count = torch.zeros((n_tokens,), dtype=torch.float32, device=dev)
        conf_sum = torch.zeros_like(count)
        inv_sum = torch.zeros_like(count)
        u_sum = torch.zeros_like(count)
        v_sum = torch.zeros_like(count)
        max_inv = torch.zeros_like(count)
        count.index_add_(0, cell, ones)
        conf_sum.index_add_(0, cell, conf_i)
        inv_sum.index_add_(0, cell, inv_depth)
        u_sum.index_add_(0, cell, u_norm)
        v_sum.index_add_(0, cell, v_norm)
        max_inv.scatter_reduce_(0, cell, inv_depth, reduce="amax", include_self=True)

        denom = count.clamp_min(1.0)
        hit = (count > 0).float()
        log_count = torch.log1p(count) / math.log1p(max(1.0, float(count.max().item())))
        out[view_idx, :, 0] = hit
        out[view_idx, :, 1] = log_count
        out[view_idx, :, 2] = conf_sum / denom
        out[view_idx, :, 3] = inv_sum / denom
        out[view_idx, :, 4] = max_inv
        out[view_idx, :, 5] = u_sum / denom
        out[view_idx, :, 6] = v_sum / denom

    return out.unsqueeze(0).to(dtype=dtype), {
        "point_count": int(points.shape[0]),
        "point_count_before_support_filter": int(support_summary["point_count_before"]),
        "per_view_inside_count": per_view_inside,
        "per_view_mask_hit_count": per_view_mask_hit,
        "inside_count_total": int(sum(per_view_inside)),
        "mask_hit_count_total": int(sum(per_view_mask_hit)),
        "mask_filter_enabled": bool(mask_dir),
        "point_mask_support_filter": support_summary,
        "projection_source": "colmap",
        "matched_images": matched_images,
        "feature_dim": 7,
    }
