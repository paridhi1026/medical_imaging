import os
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from joblib import Parallel, delayed

from .config import Config


def _ensure_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    # if image is 0..255, scale to 0..1
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _resize_gray(path: str, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img)
    return _ensure_float01(arr)


def normalize_global(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Min-max normalization strictly over non-zero brain tissue pixels (img > 1e-4).
    Background pixels (<= 1e-4) remain exactly 0.0.
    """
    mask = img > 1e-4
    if not np.any(mask):
        return np.zeros_like(img, dtype=np.float32)
    tissue_vals = img[mask]
    mn = float(tissue_vals.min())
    mx = float(tissue_vals.max())
    if mx - mn < eps:
        out = np.zeros_like(img, dtype=np.float32)
        out[mask] = 0.5
        return out
    out = np.zeros_like(img, dtype=np.float32)
    out[mask] = ((img[mask] - mn) / (mx - mn + eps)).astype(np.float32)
    return np.clip(out, 0.0, 1.0)



def normalize_local_block(img: np.ndarray, block: int = 8, eps: float = 1e-8) -> np.ndarray:
    """
    Vectorized local min-max normalization within each block×block region.
    Output is still in [0,1].
    """
    H, W = img.shape
    if H % block != 0 or W % block != 0:
        raise ValueError(f"Image {H}x{W} not divisible by local_block={block}")

    bh, bw = H // block, W // block
    x = img.reshape(bh, block, bw, block)

    # per-block min/max
    mn = x.min(axis=(1, 3), keepdims=True)
    mx = x.max(axis=(1, 3), keepdims=True)

    denom = (mx - mn)
    out = (x - mn) / (denom + eps)

    # if denom~0 → make block 0
    out = np.where(denom < eps, 0.0, out)

    return out.reshape(H, W).astype(np.float32)


def apply_norm(img: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.norm_mode == "global":
        return normalize_global(img)
    if cfg.norm_mode == "local":
        return normalize_local_block(img, block=cfg.local_block)
    raise ValueError(f"Unknown cfg.norm_mode={cfg.norm_mode}")


def _load_one(path: str, cfg: Config) -> Optional[np.ndarray]:
    try:
        img = _resize_gray(path, cfg.img_size)
        img = apply_norm(img, cfg)
        return img.reshape(-1).astype(np.float32)
    except Exception:
        return None


def load_images_matrix(paths: List[str], cfg: Config, ncore: int = 1) -> Tuple[np.ndarray, List[str]]:
    """
    Load images -> normalize -> flatten into X (N, img_size^2).
    Returns X and kept_paths aligned to rows of X.
    """
    ncore = max(1, int(ncore))
    if ncore == 1:
        rows = []
        kept = []
        for p in paths:
            r = _load_one(p, cfg)
            if r is not None and np.isfinite(r).all():
                rows.append(r)
                kept.append(p)
        if not rows:
            return np.zeros((0, cfg.img_size * cfg.img_size), dtype=np.float32), []
        return np.stack(rows, axis=0), kept

    # parallel
    loaded = Parallel(n_jobs=ncore, backend="loky")(
        delayed(_load_one)(p, cfg) for p in paths
    )
    rows = []
    kept = []
    for p, r in zip(paths, loaded):
        if r is not None and np.isfinite(r).all():
            rows.append(r)
            kept.append(p)
    if not rows:
        return np.zeros((0, cfg.img_size * cfg.img_size), dtype=np.float32), []
    return np.stack(rows, axis=0), kept
