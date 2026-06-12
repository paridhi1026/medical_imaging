#!/usr/bin/env python3
"""
Hybrid PatchNMF anomaly evaluation (structure-aware residual + optional latent Mahalanobis + optional rarity score).

Stability tweak (conference-friendly):
- Drop border blocks (suppresses skull/outer-edge dominance)
- Use stable "top-p mean" of block map instead of max / pure quantile spikes

Dataset structure supported:
  DATASET_ROOT/
    Trainig/notumor
    Testing/notumor
    Testing/Ischemoa

Outputs:
- roc_raw.png (raw direction)
- roc.png (best direction, possibly inverted)
- metrics.json
- calib.json (if --save-calib)
- debug_*.png (if --debug)
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from nmfcore.config import Config
from nmfcore.preprocess import load_images_matrix

try:
    from scipy.ndimage import gaussian_filter
except Exception as e:
    raise SystemExit(
        "Missing dependency: scipy (needed for gaussian_filter). Install with: pip install scipy\n"
        f"Original error: {e}"
    )

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_images_flat(folder: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(folder):
        return out
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def infer_k_from_model_path(model_path: str) -> Optional[int]:
    m = re.search(r"k(\d+)", os.path.basename(model_path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def block_mean_map(x_img: np.ndarray, block: int) -> np.ndarray:
    """Downsample (H,W) to (H/block, W/block) by block mean."""
    H, W = x_img.shape
    bh, bw = H // block, W // block
    if bh <= 0 or bw <= 0:
        raise ValueError(f"block={block} too large for image {H}x{W}")
    x = x_img[: bh * block, : bw * block].reshape(bh, block, bw, block)
    return x.mean(axis=(1, 3))


def structure_residual_map(
    img: np.ndarray,
    rec: np.ndarray,
    *,
    sigma_small: float = 2.0,
    sigma_large: float = 6.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Structure-aware residual on |img - rec| using DoG on the residual magnitude:
      R = |img-rec|
      R1 = G(R, sigma_small)
      R2 = G(R1, sigma_large)
      S = max(0, R1 - R2)
    """
    R = np.abs(img - rec).astype(np.float32)
    R1 = gaussian_filter(R, sigma=float(sigma_small))
    R2 = gaussian_filter(R1, sigma=float(sigma_large))
    S = np.maximum(0.0, R1 - R2).astype(np.float32)
    denom = float(np.mean(S) + eps)
    return (S / denom).astype(np.float32)


def intensity_mask(img: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo = np.percentile(img, float(lo_q))
    hi = np.percentile(img, float(hi_q))
    return ((img > lo) & (img < hi)).astype(np.float32)


def scalar_from_blockmap(m: np.ndarray, score_mode: str, score_quantile: float) -> float:
    if score_mode == "mean":
        return float(np.mean(m))
    if score_mode == "quantile":
        q = float(np.clip(score_quantile, 0.0, 1.0))
        return float(np.quantile(m.reshape(-1), q))
    raise ValueError(f"Unknown --score-mode: {score_mode}")


def build_valid_block_mask(bh: int, bw: int, drop_border: int) -> np.ndarray:
    """
    Returns boolean mask (bh,bw) selecting valid blocks.
    drop_border=1 drops a 1-block ring around the border.
    """
    m = np.ones((bh, bw), dtype=bool)
    db = int(max(0, drop_border))
    if db == 0:
        return m
    if bh <= 2 * db or bw <= 2 * db:
        return np.ones((bh, bw), dtype=bool)
    m[:db, :] = False
    m[-db:, :] = False
    m[:, :db] = False
    m[:, -db:] = False
    return m


def stable_topmean_score(block_map: np.ndarray, *, top_p: float = 0.95, valid_mask: np.ndarray | None = None) -> float:
    """
    Stable score: mean of the top (1-top_p) fraction of blocks.
    Equivalent to: mean( m[m >= quantile(m, top_p)] ) on valid blocks.
    """
    m = np.asarray(block_map, dtype=np.float32)
    v = m[valid_mask] if valid_mask is not None else m.reshape(-1)
    if v.size == 0:
        return 0.0
    p = float(np.clip(top_p, 0.0, 1.0))
    cut = float(np.quantile(v, p))
    sel = v[v >= cut]
    return float(sel.mean()) if sel.size else float(cut)


def safe_inv_cov(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    cov = np.array(cov, dtype=np.float64)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=np.float64)
    return np.linalg.inv(cov)


def try_get_latents(bundle, X: np.ndarray, batch: int = 64) -> Optional[np.ndarray]:
    """
    Best-effort latent extraction. Returns (N,K) or None.

    Tries these APIs if present:
      - bundle.transform_images(X, batch=?)
      - bundle.transform(X, batch=?)
      - bundle.encode_images(X, batch=?)
      - bundle.nmf.transform(X)
      - bundle.model.transform(X)
    """
    cand = [
        getattr(bundle, "transform_images", None),
        getattr(bundle, "transform", None),
        getattr(bundle, "encode_images", None),
    ]
    for fn in cand:
        if callable(fn):
            try:
                Z = fn(X, batch=batch)
            except TypeError:
                try:
                    Z = fn(X)
                except Exception:
                    continue
            except Exception:
                continue
            Z = np.asarray(Z)
            if Z.ndim == 2 and Z.shape[0] == X.shape[0]:
                return Z.astype(np.float32)

    for obj_name in ["nmf", "model"]:
        obj = getattr(bundle, obj_name, None)
        if obj is None:
            continue
        fn = getattr(obj, "transform", None)
        if callable(fn):
            try:
                Z = np.asarray(fn(X))
                if Z.ndim == 2 and Z.shape[0] == X.shape[0]:
                    return Z.astype(np.float32)
            except Exception:
                pass
    return None


@dataclass
class Calib:
    """Calibration stats derived from training-normal images."""
    res_mu: float
    res_sig: float
    rar_mu: float
    rar_sig: float
    lat_mu: Optional[List[float]] = None
    lat_invcov: Optional[List[List[float]]] = None
    block_mu: Optional[List[float]] = None
    block_sig: Optional[List[float]] = None
    img_size: int = 128
    block: int = 8
    lo_q: float = 20.0
    hi_q: float = 99.5
    sigma_small: float = 2.0
    sigma_large: float = 6.0


def compute_train_stats(
    bundle,
    Xtrain: np.ndarray,
    *,
    img_size: int,
    block: int,
    lo_q: float,
    hi_q: float,
    sigma_small: float,
    sigma_large: float,
    recon_batch: int,
    score_mode: str,
    score_quantile: float,
    use_stable_score: bool,
    drop_border: int,
    top_p: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    s = int(img_size)
    Xrec = bundle.reconstruct_images(Xtrain, batch=int(recon_batch))
    bh = s // block
    bw = s // block
    B = bh * bw

    valid_mask = build_valid_block_mask(bh, bw, drop_border=drop_border)

    block_maps = np.zeros((Xtrain.shape[0], B), dtype=np.float32)
    res_scores = np.zeros((Xtrain.shape[0],), dtype=np.float32)

    for i in range(Xtrain.shape[0]):
        img = Xtrain[i].reshape(s, s)
        rec = Xrec[i].reshape(s, s)

        msk = intensity_mask(img, lo_q, hi_q)
        S = structure_residual_map(img, rec, sigma_small=sigma_small, sigma_large=sigma_large) * msk

        bm = block_mean_map(S, block).astype(np.float32)
        block_maps[i] = bm.reshape(-1)

        if use_stable_score:
            res_scores[i] = float(stable_topmean_score(bm, top_p=top_p, valid_mask=valid_mask))
        else:
            res_scores[i] = float(scalar_from_blockmap(bm, score_mode, score_quantile))

    block_mu = block_maps.mean(axis=0)
    block_sig = block_maps.std(axis=0) + 1e-8

    Z = (block_maps - block_mu[None, :]) / block_sig[None, :]

    # rarity: mean z^2 over VALID blocks only (drop border rings)
    v = valid_mask.reshape(-1)
    rar_scores = (Z[:, v] * Z[:, v]).mean(axis=1).astype(np.float32)

    latents = try_get_latents(bundle, Xtrain, batch=max(16, int(recon_batch)))
    return res_scores, rar_scores, block_mu, block_sig, latents


def build_calib(
    bundle,
    Xtrain: np.ndarray,
    *,
    img_size: int,
    block: int,
    lo_q: float,
    hi_q: float,
    sigma_small: float,
    sigma_large: float,
    recon_batch: int,
    score_mode: str,
    score_quantile: float,
    use_stable_score: bool,
    drop_border: int,
    top_p: float,
) -> Calib:
    res_scores, rar_scores, block_mu, block_sig, latents = compute_train_stats(
        bundle, Xtrain,
        img_size=img_size, block=block,
        lo_q=lo_q, hi_q=hi_q,
        sigma_small=sigma_small, sigma_large=sigma_large,
        recon_batch=recon_batch,
        score_mode=score_mode, score_quantile=score_quantile,
        use_stable_score=use_stable_score,
        drop_border=drop_border,
        top_p=top_p,
    )

    res_mu = float(np.mean(res_scores))
    res_sig = float(np.std(res_scores) + 1e-8)
    rar_mu = float(np.mean(rar_scores))
    rar_sig = float(np.std(rar_scores) + 1e-8)

    lat_mu_list = None
    invcov_list = None
    if latents is not None and latents.shape[0] >= 10:
        mu = latents.mean(axis=0).astype(np.float64)
        cov = np.cov(latents.astype(np.float64).T)
        invcov = safe_inv_cov(cov, ridge=1e-6)
        lat_mu_list = mu.tolist()
        invcov_list = invcov.tolist()

    return Calib(
        res_mu=res_mu,
        res_sig=res_sig,
        rar_mu=rar_mu,
        rar_sig=rar_sig,
        lat_mu=lat_mu_list,
        lat_invcov=invcov_list,
        block_mu=block_mu.astype(np.float32).tolist(),
        block_sig=block_sig.astype(np.float32).tolist(),
        img_size=int(img_size),
        block=int(block),
        lo_q=float(lo_q),
        hi_q=float(hi_q),
        sigma_small=float(sigma_small),
        sigma_large=float(sigma_large),
    )


def load_calib(path: str) -> Calib:
    with open(path, "r") as f:
        d = json.load(f)
    return Calib(**d)


def save_calib(cal: Calib, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(cal.__dict__, f, indent=2)



def score_dataset(
    bundle,
    X: np.ndarray,
    *,
    cal,
    recon_batch: int,
    score_mode: str,
    score_quantile: float,
    w_res: float,
    w_lat: float,
    w_rar: float,
):

    N = X.shape[0]
    s = int(cal.img_size)
    block = int(cal.block)

    # reconstruction
    Xrec = bundle.reconstruct_images(X, batch=int(recon_batch))

    imgs = X.reshape(N, s, s)
    recs = Xrec.reshape(N, s, s)

    # structure-aware residual (vectorized)
    R = np.abs(imgs - recs).astype(np.float32)

    R1 = gaussian_filter(R, sigma=(0, cal.sigma_small, cal.sigma_small))
    R2 = gaussian_filter(R1, sigma=(0, cal.sigma_large, cal.sigma_large))
    S = np.maximum(0.0, R1 - R2)

    # intensity mask (vectorized)
    lo = np.percentile(imgs, cal.lo_q, axis=(1,2), keepdims=True)
    hi = np.percentile(imgs, cal.hi_q, axis=(1,2), keepdims=True)
    mask = ((imgs > lo) & (imgs < hi)).astype(np.float32)

    S *= mask

    # block pooling (vectorized)
    bh = s // block
    bw = s // block

    S = S[:, :bh*block, :bw*block]
    S = S.reshape(N, bh, block, bw, block)
    block_maps_arr = S.mean(axis=(2,4)).astype(np.float32)  # (N,bh,bw)

    flat = block_maps_arr.reshape(N, -1)

    # residual score
    if score_mode == "quantile":
        res_scores = np.quantile(flat, score_quantile, axis=1)
    else:
        res_scores = flat.mean(axis=1)

    # rarity score
    if cal.block_mu is not None and cal.block_sig is not None:
        mu = np.array(cal.block_mu, dtype=np.float32)[None, :]
        sig = np.array(cal.block_sig, dtype=np.float32)[None, :]
        z = (flat - mu) / (sig + 1e-8)
        rar_scores = (z * z).mean(axis=1)
    else:
        rar_scores = np.zeros_like(res_scores)

    # latent score (unchanged logic)
    lat_scores = np.zeros_like(res_scores)
    latents = try_get_latents(bundle, X, batch=max(16, int(recon_batch)))
    if latents is not None and cal.lat_mu is not None and cal.lat_invcov is not None:
        mu = np.array(cal.lat_mu, dtype=np.float64)
        invcov = np.array(cal.lat_invcov, dtype=np.float64)
        D = latents.astype(np.float64) - mu[None, :]
        md2 = np.einsum("ni,ij,nj->n", D, invcov, D)
        lat_scores = np.sqrt(np.maximum(0.0, md2)).astype(np.float32)

    # z-normalization
    res_z = (res_scores - float(cal.res_mu)) / float(cal.res_sig + 1e-8)
    rar_z = (rar_scores - float(cal.rar_mu)) / float(cal.rar_sig + 1e-8)

    if cal.lat_mu is not None and cal.lat_invcov is not None and latents is not None:
        lat_mu_s = float(np.mean(lat_scores))
        lat_sig_s = float(np.std(lat_scores) + 1e-8)
        lat_z = (lat_scores - lat_mu_s) / lat_sig_s
    else:
        lat_z = np.zeros_like(res_z)

    hybrid = (float(w_res) * res_z) + (float(w_rar) * rar_z) + (float(w_lat) * lat_z)

    # convert block maps to list for compatibility
    block_maps = [block_maps_arr[i] for i in range(N)]
    struct_maps = [None] * N  # unchanged structure map handling

    return hybrid.astype(np.float32), block_maps, struct_maps, latents



def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, out_png: str) -> float:
    if len(np.unique(y_true)) < 2:
        plt.figure(figsize=(6, 5))
        plt.title(f"{title} (AUC=nan)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{title} (AUC={roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return float(roc_auc)



def roc_arrays(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return ROC arrays as python lists (fpr, tpr, thr)."""
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thr": []}
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return {"fpr": fpr.astype(float).tolist(), "tpr": tpr.astype(float).tolist(), "thr": thr.astype(float).tolist()}


def best_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, int]]:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    k = int(np.argmax(j))
    best_thr = float(thr[k])

    y_pred = (y_score >= best_thr).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return best_thr, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset-root", required=True, help="Root that contains Trainig/ and Testing/")
    ap.add_argument("--train-normal", default="Trainig/notumor", help="Relative to --dataset-root")
    ap.add_argument("--test-normal", default="Testing/notumor", help="Relative to --dataset-root")
    ap.add_argument("--test-anom", default="Testing/Ischemoa", help="Relative to --dataset-root")

    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--norm-mode", choices=["global", "local"], default="local")
    ap.add_argument("--local-block", type=int, default=8)

    ap.add_argument("--model", required=True, help=".joblib PatchNMFBundle")

    ap.add_argument("--mask-lo-q", type=float, default=20.0)
    ap.add_argument("--mask-hi-q", type=float, default=99.5)
    ap.add_argument("--sigma-small", type=float, default=2.0)
    ap.add_argument("--sigma-large", type=float, default=6.0)
    ap.add_argument("--block", type=int, default=16)

    # scoring
    ap.add_argument("--score-mode", choices=["mean", "quantile"], default="quantile")
    ap.add_argument("--score-quantile", type=float, default=0.995)
    ap.add_argument("--patch-batch", type=int, default=16)

    # stability tweak
    ap.add_argument("--use-stable-score", action="store_true",
                    help="Use stable score: mean of top-p blocks after dropping border ring.")
    ap.add_argument("--drop-border", type=int, default=1,
                    help="Number of block rings to drop at border (default 1). Used when --use-stable-score.")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="Top-p quantile for stable score (default 0.95 => top 5%% mean). Used when --use-stable-score.")

    # hybrid weights
    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--w-lat", type=float, default=0.5)
    ap.add_argument("--w-rar", type=float, default=0.5)

    # calibration freeze
    ap.add_argument("--calib", default=None, help="Load calibration JSON (for frozen results)")
    ap.add_argument("--save-calib", default=None, help="Save calibration JSON (to freeze results)")

    ap.add_argument("--out", required=True)

    args = ap.parse_args()
    ensure_dir(args.out)

    cfg = Config(
        base_path=args.dataset_root,
        img_size=int(args.img_size),
        normal_class="",
        norm_mode=args.norm_mode,
        local_block=int(args.local_block),
        random_state=0,
    )

    bundle = joblib.load(args.model)

    train_dir = os.path.join(args.dataset_root, args.train_normal)
    test_n_dir = os.path.join(args.dataset_root, args.test_normal)
    test_a_dir = os.path.join(args.dataset_root, args.test_anom)

    if args.calib is not None:
        cal = load_calib(args.calib)
        calib_path_used = os.path.abspath(args.calib)
    else:
        train_paths = list_images_flat(train_dir)
        if len(train_paths) == 0:
            raise SystemExit(f"No training-normal images found: {train_dir}")
        Xtr, _ = load_images_matrix(train_paths, cfg)
        cal = build_calib(
            bundle, Xtr,
            img_size=int(args.img_size),
            block=int(args.block),
            lo_q=float(args.mask_lo_q),
            hi_q=float(args.mask_hi_q),
            sigma_small=float(args.sigma_small),
            sigma_large=float(args.sigma_large),
            recon_batch=int(args.patch_batch),
            score_mode=args.score_mode,
            score_quantile=float(args.score_quantile),
            use_stable_score=bool(args.use_stable_score),
            drop_border=int(args.drop_border),
            top_p=float(args.top_p),
        )
        calib_path_used = None
        if args.save_calib is not None:
            save_calib(cal, args.save_calib)
            calib_path_used = os.path.abspath(args.save_calib)

    n_paths = list_images_flat(test_n_dir)
    a_paths = list_images_flat(test_a_dir)
    if len(n_paths) == 0:
        raise SystemExit(f"No test-normal images found: {test_n_dir}")
    if len(a_paths) == 0:
        raise SystemExit(f"No test-anom images found: {test_a_dir}")

    Xn, _ = load_images_matrix(n_paths, cfg)
    Xa, _ = load_images_matrix(a_paths, cfg)

    scores_n, _, _, _ = score_dataset(
        bundle, Xn, cal=cal,
        recon_batch=int(args.patch_batch),
        score_mode=args.score_mode,
        score_quantile=float(args.score_quantile),
        use_stable_score=bool(args.use_stable_score),
        drop_border=int(args.drop_border),
        top_p=float(args.top_p),
        w_res=float(args.w_res), w_lat=float(args.w_lat), w_rar=float(args.w_rar),
    )
    scores_a, _, _, _ = score_dataset(
        bundle, Xa, cal=cal,
        recon_batch=int(args.patch_batch),
        score_mode=args.score_mode,
        score_quantile=float(args.score_quantile),
        use_stable_score=bool(args.use_stable_score),
        drop_border=int(args.drop_border),
        top_p=float(args.top_p),
        w_res=float(args.w_res), w_lat=float(args.w_lat), w_rar=float(args.w_rar),
    )

    y_true = np.concatenate([np.zeros_like(scores_n, dtype=int), np.ones_like(scores_a, dtype=int)])
    y_score_raw = np.concatenate([scores_n, scores_a]).astype(np.float32)

    roc_raw_path = os.path.join(args.out, "roc_raw.png")
    raw_auc = plot_roc(y_true, y_score_raw, f"ROC raw ({os.path.basename(args.model)})", roc_raw_path)

    roc_raw = roc_arrays(y_true, y_score_raw)

    # auto-invert if it improves AUC
    y_score_inv = (-y_score_raw).astype(np.float32)
    if len(np.unique(y_true)) >= 2:
        inv_auc = auc(*roc_curve(y_true, y_score_inv)[:2])
    else:
        inv_auc = float("nan")

    use_invert = False
    y_score = y_score_raw
    best_auc = raw_auc
    if np.isfinite(inv_auc) and (not np.isfinite(raw_auc) or inv_auc > raw_auc):
        use_invert = True
        y_score = y_score_inv
        best_auc = inv_auc

    roc_path = os.path.join(args.out, "roc.png")
    title = f"ROC ({'auto-inverted ' if use_invert else ''}{os.path.basename(args.model)})"
    _ = plot_roc(y_true, y_score, title, roc_path)

    roc_best = roc_arrays(y_true, y_score)

    thr, cm = best_threshold_youden(y_true, y_score)

    metrics = {
        "model": os.path.abspath(args.model),
        "dataset_root": os.path.abspath(args.dataset_root),
        "train_normal": train_dir,
        "test_normal": test_n_dir,
        "test_anom": test_a_dir,
        "img_size": int(args.img_size),
        "norm_mode": args.norm_mode,
        "local_block": int(args.local_block),
        "block": int(args.block),
        "mask_lo_q": float(args.mask_lo_q),
        "mask_hi_q": float(args.mask_hi_q),
        "sigma_small": float(args.sigma_small),
        "sigma_large": float(args.sigma_large),
        "score_mode": args.score_mode,
        "score_quantile": float(args.score_quantile),
        "patch_batch": int(args.patch_batch),
        "use_stable_score": bool(args.use_stable_score),
        "drop_border": int(args.drop_border),
        "top_p": float(args.top_p),
        "weights": {"res": float(args.w_res), "lat": float(args.w_lat), "rar": float(args.w_rar)},
        "calib_used": calib_path_used,
        "n_normal": int(len(scores_n)),
        "n_anom": int(len(scores_a)),
        "auc_raw": float(raw_auc) if np.isfinite(raw_auc) else None,
        "auc_inverted": float(inv_auc) if np.isfinite(inv_auc) else None,
        "invert_used": bool(use_invert),
        "auc": float(best_auc) if np.isfinite(best_auc) else None,
        "roc_curve_raw": roc_raw,
        "roc_curve_best": roc_best,
        "thr": float(thr),
        "confusion": cm,
        "scores_normal": {"mean": float(np.mean(scores_n)), "std": float(np.std(scores_n)), "min": float(np.min(scores_n)), "max": float(np.max(scores_n))},
        "scores_anom": {"mean": float(np.mean(scores_a)), "std": float(np.std(scores_a)), "min": float(np.min(scores_a)), "max": float(np.max(scores_a))},
    }

    
    # Additional compact score summaries for analysis
    def _q(x, qs=(0.5, 0.9, 0.95, 0.99)):
        x = np.asarray(x, dtype=float)
        return {str(q): float(np.quantile(x, q)) for q in qs}

    metrics["score_summary"] = {
        "raw": {"quantiles": _q(y_score_raw), "mean": float(np.mean(y_score_raw)), "std": float(np.std(y_score_raw))},
        "best": {"quantiles": _q(y_score), "mean": float(np.mean(y_score)), "std": float(np.std(y_score))},
    }
    
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(" ", roc_raw_path)
    print(" ", roc_path)
    print(" ", os.path.join(args.out, "metrics.json"))
    if calib_path_used is not None:
        print(" ", calib_path_used)
    print(f"AUC(raw)={raw_auc}  AUC(best)={best_auc}  invert={use_invert}  thr={thr}")


if __name__ == "__main__":
    main()
