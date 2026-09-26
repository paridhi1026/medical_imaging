#!/usr/bin/env python3
"""
Diagnostic PatchNMF Anomaly Evaluator.

Key fixes implemented per diagnostic protocol:
1. NO per-image normalization in structure_residual_map (preserves absolute reconstruction error magnitude).
2. Diagnostic intensity_mask (does NOT discard top/bottom intensity percentiles).
3. Rarity floor protection (uses 10th percentile floor on block_sig to prevent division-by-near-zero explosion).
4. Component-wise AUC breakdown: computes independent AUCs for:
   - Raw MAE (mean absolute reconstruction error)
   - Raw MSE (mean squared reconstruction error)
   - Structure Mean (mean spatial residual)
   - Structure Q99.5 (99.5th percentile localized residual)
   - Rarity (calibrated spatial block z^2)
   - Latent (Mahalanobis distance in latent space)
   - Hybrid Combined score
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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


def intensity_mask(img: np.ndarray, lo_q: float = 0.0, hi_q: float = 100.0) -> np.ndarray:
    """
    Diagnostic mode: retain all pixel intensities so stroke hypodensity
    or hyperdensity is NOT clipped out.
    """
    return np.ones_like(img, dtype=np.float32)


def structure_residual_map(
    img: np.ndarray,
    rec: np.ndarray,
    *,
    use_dog: bool = False,
    sigma_small: float = 2.0,
    sigma_large: float = 6.0,
) -> np.ndarray:
    """
    Structure residual map.
    IMPORTANT: Retains absolute magnitude. Does NOT normalize by per-image mean!
    """
    R = np.abs(img - rec).astype(np.float32)

    if use_dog:
        R1 = gaussian_filter(R, sigma=float(sigma_small))
        R2 = gaussian_filter(R, sigma=float(sigma_large))
        S = np.maximum(0.0, R1 - R2).astype(np.float32)
        return S

    return R


def scalar_from_blockmap(m: np.ndarray, score_mode: str, score_quantile: float) -> float:
    if score_mode == "mean":
        return float(np.mean(m))
    if score_mode == "quantile":
        q = float(np.clip(score_quantile, 0.0, 1.0))
        return float(np.quantile(m.reshape(-1), q))
    raise ValueError(f"Unknown --score-mode: {score_mode}")


def safe_inv_cov(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    cov = np.array(cov, dtype=np.float64)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=np.float64)
    return np.linalg.inv(cov)


def try_get_latents(bundle, X: np.ndarray, batch: int = 64) -> Optional[np.ndarray]:
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
    use_dog: bool = False
    sigma_small: float = 2.0
    sigma_large: float = 6.0


def compute_train_stats(
    bundle,
    Xtrain: np.ndarray,
    *,
    img_size: int,
    block: int,
    use_dog: bool,
    sigma_small: float,
    sigma_large: float,
    recon_batch: int,
    score_mode: str,
    score_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    s = int(img_size)
    Xrec = bundle.reconstruct_images(Xtrain, batch=int(recon_batch))
    bh = s // block
    bw = s // block
    B = bh * bw

    block_maps = np.zeros((Xtrain.shape[0], B), dtype=np.float32)
    res_scores = np.zeros((Xtrain.shape[0],), dtype=np.float32)

    for i in range(Xtrain.shape[0]):
        img = Xtrain[i].reshape(s, s)
        rec = Xrec[i].reshape(s, s)

        S = structure_residual_map(img, rec, use_dog=use_dog, sigma_small=sigma_small, sigma_large=sigma_large)
        bm = block_mean_map(S, block).astype(np.float32)
        block_maps[i] = bm.reshape(-1)
        res_scores[i] = float(scalar_from_blockmap(bm, score_mode, score_quantile))

    block_mu = block_maps.mean(axis=0)
    block_sig = block_maps.std(axis=0)

    # Use 10th percentile floor on std to prevent zero/near-zero division explosion
    sig_floor = float(np.percentile(block_sig, 10))
    block_sig = np.maximum(block_sig, sig_floor)

    Z = (block_maps - block_mu[None, :]) / block_sig[None, :]
    rar_scores = (Z * Z).mean(axis=1).astype(np.float32)

    latents = try_get_latents(bundle, Xtrain, batch=max(16, int(recon_batch)))
    return res_scores, rar_scores, block_mu, block_sig, latents


def build_calib(
    bundle,
    Xtrain: np.ndarray,
    *,
    img_size: int,
    block: int,
    use_dog: bool,
    sigma_small: float,
    sigma_large: float,
    recon_batch: int,
    score_mode: str,
    score_quantile: float,
) -> Calib:
    res_scores, rar_scores, block_mu, block_sig, latents = compute_train_stats(
        bundle, Xtrain,
        img_size=img_size, block=block,
        use_dog=use_dog, sigma_small=sigma_small, sigma_large=sigma_large,
        recon_batch=recon_batch,
        score_mode=score_mode, score_quantile=score_quantile,
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
        use_dog=bool(use_dog),
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


def evaluate_component_scores(
    bundle,
    X: np.ndarray,
    *,
    cal: Calib,
    recon_batch: int,
    score_mode: str,
    score_quantile: float,
    w_res: float,
    w_lat: float,
    w_rar: float,
) -> Dict[str, np.ndarray]:
    """
    Computes all component anomaly scores independently:
      - raw_mae
      - raw_mse
      - struct_mean
      - struct_q995
      - rarity
      - latent
      - hybrid
    """
    s = int(cal.img_size)
    block = int(cal.block)
    bh = s // block
    bw = s // block
    B = bh * bw

    block_mu = np.array(cal.block_mu, dtype=np.float32).reshape(1, B) if cal.block_mu is not None else None
    block_sig = np.array(cal.block_sig, dtype=np.float32).reshape(1, B) if cal.block_sig is not None else None

    Xrec = bundle.reconstruct_images(X, batch=int(recon_batch))

    N = X.shape[0]
    raw_mae = np.zeros(N, dtype=np.float32)
    raw_mse = np.zeros(N, dtype=np.float32)
    struct_mean = np.zeros(N, dtype=np.float32)
    struct_q995 = np.zeros(N, dtype=np.float32)
    rar_scores = np.zeros(N, dtype=np.float32)
    res_scores = np.zeros(N, dtype=np.float32)

    block_maps: List[np.ndarray] = []
    struct_maps: List[np.ndarray] = []

    for i in range(N):
        img = X[i].reshape(s, s)
        rec = Xrec[i].reshape(s, s)

        diff_abs = np.abs(img - rec)
        raw_mae[i] = float(np.mean(diff_abs))
        raw_mse[i] = float(np.mean((img - rec) ** 2))

        S = structure_residual_map(img, rec, use_dog=cal.use_dog, sigma_small=cal.sigma_small, sigma_large=cal.sigma_large)
        struct_maps.append(S)
        struct_mean[i] = float(np.mean(S))
        struct_q995[i] = float(np.quantile(S, 0.995))

        bm = block_mean_map(S, block).astype(np.float32)
        block_maps.append(bm)

        res_scores[i] = float(scalar_from_blockmap(bm, score_mode, score_quantile))

        if block_mu is not None and block_sig is not None:
            v = bm.reshape(1, -1)
            z = (v - block_mu) / block_sig
            rar_scores[i] = float(np.mean(z * z))
        else:
            rar_scores[i] = 0.0

    lat_scores = np.zeros(N, dtype=np.float32)
    latents = try_get_latents(bundle, X, batch=max(16, int(recon_batch)))
    if latents is not None and cal.lat_mu is not None and cal.lat_invcov is not None:
        mu = np.array(cal.lat_mu, dtype=np.float64)
        invcov = np.array(cal.lat_invcov, dtype=np.float64)
        D = latents.astype(np.float64) - mu[None, :]
        md2 = np.einsum("ni,ij,nj->n", D, invcov, D)
        lat_scores = np.sqrt(np.maximum(0.0, md2)).astype(np.float32)

    res_z = (res_scores - float(cal.res_mu)) / float(cal.res_sig + 1e-8)
    rar_z = (rar_scores - float(cal.rar_mu)) / float(cal.rar_sig + 1e-8)

    if cal.lat_mu is not None and cal.lat_invcov is not None and latents is not None:
        lat_mu_s = float(np.mean(lat_scores))
        lat_sig_s = float(np.std(lat_scores) + 1e-8)
        lat_z = (lat_scores - lat_mu_s) / lat_sig_s
    else:
        lat_z = np.zeros_like(res_z)

    hybrid = (float(w_res) * res_z) + (float(w_rar) * rar_z) + (float(w_lat) * lat_z)

    return {
        "raw_mae": raw_mae,
        "raw_mse": raw_mse,
        "struct_mean": struct_mean,
        "struct_q995": struct_q995,
        "rarity": rar_scores,
        "latent": lat_scores,
        "hybrid": hybrid.astype(np.float32),
        "_block_maps": block_maps,
        "_struct_maps": struct_maps,
    }


def compute_auc_pair(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float, bool]:
    """Returns (raw_auc, inv_auc, best_auc, is_inverted)"""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan"), False

    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_score)
    raw_auc = float(auc(fpr_raw, tpr_raw))

    fpr_inv, tpr_inv, _ = roc_curve(y_true, -y_score)
    inv_auc = float(auc(fpr_inv, tpr_inv))

    if inv_auc > raw_auc:
        return raw_auc, inv_auc, inv_auc, True
    return raw_auc, inv_auc, raw_auc, False


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
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return float(roc_auc)


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


def save_debug(
    out_dir: str,
    prefix: str,
    X: np.ndarray,
    cfg: Config,
    struct_maps: List[np.ndarray],
    scores: np.ndarray,
    y_true: np.ndarray,
    thr: float,
    *,
    n: int = 12,
) -> None:
    ensure_dir(out_dir)
    s = cfg.img_size
    idx = np.linspace(0, len(scores) - 1, num=min(n, len(scores)), dtype=int)

    for j, i in enumerate(idx):
        img = X[i].reshape(s, s)
        sm = struct_maps[i]
        y = int(y_true[i])
        pred = int(scores[i] >= thr)
        tag = "TP" if (y == 1 and pred == 1) else "TN" if (y == 0 and pred == 0) else "FP" if (y == 0 and pred == 1) else "FN"

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"Input ({'Anom' if y==1 else 'Normal'})")
        axes[0].axis("off")

        axes[1].imshow(sm, cmap="hot")
        axes[1].set_title(f"Residual Map ({tag} score={scores[i]:.3f})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"debug_{prefix}_{j:02d}_{tag}.png"), dpi=120)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--train-normal", default="Training/after", help="Relative to --dataset-root")
    ap.add_argument("--test-normal", default="Testing/after", help="Relative to --dataset-root")
    ap.add_argument("--test-anom", default="Testing/Ischemia", help="Relative to --dataset-root")

    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--norm-mode", choices=["global", "local"], default="global")
    ap.add_argument("--local-block", type=int, default=8)

    ap.add_argument("--model", required=True, help=".joblib PatchNMFBundle")

    ap.add_argument("--use-dog", action="store_true", help="Enable DoG bandpass filter on residual")
    ap.add_argument("--sigma-small", type=float, default=2.0)
    ap.add_argument("--sigma-large", type=float, default=6.0)
    ap.add_argument("--block", type=int, default=16)

    ap.add_argument("--score-mode", choices=["mean", "quantile"], default="mean")
    ap.add_argument("--score-quantile", type=float, default=0.995)
    ap.add_argument("--patch-batch", type=int, default=16)

    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--w-lat", type=float, default=0.0)
    ap.add_argument("--w-rar", type=float, default=0.0)

    ap.add_argument("--mask-lo-q", type=float, default=0.0)
    ap.add_argument("--mask-hi-q", type=float, default=100.0)

    ap.add_argument("--calib", default=None, help="Load calibration JSON")
    ap.add_argument("--save-calib", default=None, help="Save calibration JSON")

    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-n", type=int, default=12)

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
            use_dog=args.use_dog,
            sigma_small=float(args.sigma_small),
            sigma_large=float(args.sigma_large),
            recon_batch=int(args.patch_batch),
            score_mode=args.score_mode,
            score_quantile=float(args.score_quantile),
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

    dict_n = evaluate_component_scores(
        bundle, Xn, cal=cal,
        recon_batch=int(args.patch_batch),
        score_mode=args.score_mode,
        score_quantile=float(args.score_quantile),
        w_res=float(args.w_res), w_lat=float(args.w_lat), w_rar=float(args.w_rar),
    )
    dict_a = evaluate_component_scores(
        bundle, Xa, cal=cal,
        recon_batch=int(args.patch_batch),
        score_mode=args.score_mode,
        score_quantile=float(args.score_quantile),
        w_res=float(args.w_res), w_lat=float(args.w_lat), w_rar=float(args.w_rar),
    )

    y_true = np.concatenate([np.zeros(Xn.shape[0], dtype=int), np.ones(Xa.shape[0], dtype=int)])

    component_keys = ["raw_mae", "raw_mse", "struct_mean", "struct_q995", "rarity", "latent", "hybrid"]
    component_results = {}

    print("\n" + "=" * 75)
    print(f"DIAGNOSTIC COMPONENT-WISE AUC RESULTS (Model: {os.path.basename(args.model)})")
    print("=" * 75)
    print(f"{'Component':<22} | {'Raw AUC':<10} | {'Inv AUC':<10} | {'Best AUC':<10} | {'Inverted?'}")
    print("-" * 75)

    for key in component_keys:
        sn = dict_n[key]
        sa = dict_a[key]
        scores_all = np.concatenate([sn, sa]).astype(np.float32)

        raw_auc, inv_auc, best_auc, is_inv = compute_auc_pair(y_true, scores_all)
        component_results[key] = {
            "raw_auc": raw_auc,
            "inv_auc": inv_auc,
            "best_auc": best_auc,
            "is_inverted": is_inv,
            "normal_mean": float(np.mean(sn)),
            "normal_std": float(np.std(sn)),
            "anom_mean": float(np.mean(sa)),
            "anom_std": float(np.std(sa)),
        }

        inv_str = "YES" if is_inv else "No"
        print(f"{key:<22} | {raw_auc:.4f}     | {inv_auc:.4f}     | {best_auc:.4f}     | {inv_str}")

    print("=" * 75 + "\n")

    # Save ROC plot for raw_mae, raw_mse, struct_mean, and hybrid
    plot_roc(y_true, np.concatenate([dict_n["raw_mae"], dict_a["raw_mae"]]), "ROC Raw MAE", os.path.join(args.out, "roc_raw_mae.png"))
    plot_roc(y_true, np.concatenate([dict_n["raw_mse"], dict_a["raw_mse"]]), "ROC Raw MSE", os.path.join(args.out, "roc_raw_mse.png"))
    plot_roc(y_true, np.concatenate([dict_n["struct_mean"], dict_a["struct_mean"]]), "ROC Structure Mean", os.path.join(args.out, "roc_struct_mean.png"))

    hybrid_scores = np.concatenate([dict_n["hybrid"], dict_a["hybrid"]]).astype(np.float32)
    roc_path = os.path.join(args.out, "roc.png")
    best_hybrid_auc = plot_roc(y_true, hybrid_scores, f"ROC Hybrid ({os.path.basename(args.model)})", roc_path)

    thr, cm = best_threshold_youden(y_true, hybrid_scores)

    metrics = {
        "model": os.path.abspath(args.model),
        "dataset_root": os.path.abspath(args.dataset_root),
        "train_normal": train_dir,
        "test_normal": test_n_dir,
        "test_anom": test_a_dir,
        "img_size": int(args.img_size),
        "norm_mode": args.norm_mode,
        "block": int(args.block),
        "use_dog": bool(args.use_dog),
        "calib_used": calib_path_used,
        "n_normal": int(Xn.shape[0]),
        "n_anom": int(Xa.shape[0]),
        "component_results": component_results,
        "hybrid_auc": float(best_hybrid_auc) if np.isfinite(best_hybrid_auc) else None,
        "thr": float(thr),
        "confusion": cm,
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.debug:
        k = infer_k_from_model_path(args.model) or "k"
        prefix = f"diag_k{k}"
        Xall = np.concatenate([Xn, Xa], axis=0)
        structs_all = dict_n["_struct_maps"] + dict_a["_struct_maps"]
        save_debug(args.out, prefix, Xall, cfg, structs_all, hybrid_scores, y_true, thr, n=int(args.debug_n))

    print(f"Metrics saved to: {os.path.join(args.out, 'metrics.json')}")


if __name__ == "__main__":
    main()