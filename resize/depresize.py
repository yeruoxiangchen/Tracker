from PIL import Image
import numpy as np
import math
import os

# --- 修改为你的路径与点 ---
rgb_path = "/home/zjr/Tracker/resize/RealSenseRecorder/3/color/1761723173108.png"
depth_path = "/home/zjr/Tracker/resize/RealSenseRecorder/3/depth/1761723173108.png"

A = (487, 110)
B = (426, 50)
C = (313, 30)
D = (258, 445)

fx = 386.689
fy = 386.689
cx = 321.525
cy = 244.724

A4_short_m = 0.210  # meters

def load_depth(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth file not found: {path}")
    d = Image.open(path)
    arr = np.array(d)
    return arr

def get_depth_at(arr, pt, window=5):
    h, w = arr.shape[:2]
    u, v = int(pt[0]), int(pt[1])
    half = window // 2
    x0, x1 = max(u-half, 0), min(u+half+1, w)
    y0, y1 = max(v-half, 0), min(v+half+1, h)
    patch = arr[y0:y1, x0:x1]
    if patch.ndim == 3:
        patch = patch[:, :, 0]
    patch_valid = patch[patch > 0]
    if patch_valid.size == 0:
        return 0
    return float(np.median(patch_valid))

def backproject(u, v, z, fx, fy, cx, cy):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

# Try depth-based method
try:
    depth = load_depth(depth_path)
    print("Depth image loaded. dtype =", depth.dtype, "shape =", depth.shape)
    sample_nonzero = depth[depth>0]
    median_val = float(np.median(sample_nonzero)) if sample_nonzero.size>0 else 0.0
    print("Median non-zero depth value (raw) =", median_val)
    if np.issubdtype(depth.dtype, np.integer) and median_val > 1000:
        depth_scale = 1000.0
    elif np.issubdtype(depth.dtype, np.floating) and median_val < 100:
        depth_scale = 1.0
    else:
        depth_scale = 1000.0

    zC_raw = get_depth_at(depth, C, window=5)
    zD_raw = get_depth_at(depth, D, window=5)
    zC = zC_raw / depth_scale if zC_raw>0 else 0.0
    zD = zD_raw / depth_scale if zD_raw>0 else 0.0

    print("Depths (m): C =", zC, " D =", zD)

    if zC > 0 and zD > 0:
        P_C = backproject(C[0], C[1], zC, fx, fy, cx, cy)
        P_D = backproject(D[0], D[1], zD, fx, fy, cx, cy)
        dist = np.linalg.norm(P_C - P_D)
        print(f"3D C = {P_C}, 3D D = {P_D}, distance CD = {dist:.6f} m")
    else:
        raise ValueError("Missing depth at C or D; fallback to A-B method.")

except Exception as e:
    print("Depth-based failed or missing:", e)
    # fallback using A-B known length
    uA, vA = A
    uB, vB = B
    d_pix = math.hypot(uA - uB, vA - vB)
    f_avg = (fx + fy) / 2.0
    Z_paper = f_avg * A4_short_m / d_pix
    P_C = backproject(C[0], C[1], Z_paper, fx, fy, cx, cy)
    P_D = backproject(D[0], D[1], Z_paper, fx, fy, cx, cy)
    dist = np.linalg.norm(P_C - P_D)
    print(f"(Fallback) Estimated paper Z = {Z_paper:.6f} m")
    print(f"(Fallback) 3D C = {P_C}, 3D D = {P_D}, approx distance CD = {dist:.6f} m")
    print("NOTE: fallback assumes C/D lie on same plane as A-B and is only approximate.")
