#!/usr/bin/env python3
"""
Structure-aware residual evaluation for PatchNMF (CT anomaly detection).

Key upgrades vs plain MSE block residual:
1) Brain-focused mask (simple intensity-based foreground mask).
2) Edge suppression (downweight strong gradients so skull edges don't dominate).
3) Structure-normalized residual: |x-xrec| / (local_std(x)+eps).

AUC sign fix:
- By default, if raw AUC < 0.5, we flip score sign (y_score := -y_score)
  and recompute ROC/AUC.

Dataset structure (your current layout):
  data_root/Training/notumor
  data_root/Testing/notumor
  data_root/Testing/Ischemia

Outputs:
  out_dir/roc.png
  out_dir/roc_raw.png
  out_dir/metrics.json
  out_dir/debug_*.png  (if --debug)

Compatibility:
- Accepts old flags: --mask-mode, --mask-lo-q, --mask-hi-q, --patch-batch, --io-ncore
  (some are aliases).
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from nmfcore.config import Config
from nmfcore.preprocess import load_images_matrix

try:
    from scipy.ndimage import gaussian_filter, sobel, binary_closing, binary_fill_holes
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_images_flat(folder: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def infer_k_from_model_path(model_path: str) -> Optional[int]:
    m = re.search(r"k(\d+)", os.path.basename(model_path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def infer_blockstats_path(model_path: str) -> Optional[str]:
    k = infer_k_from_model_path(model_path)
    if k is None:
        return None
    d = os.path.dirname(model_path)
    cand = os.path.join(d, f"blockstats_k{k}.json")
    return cand if os.path.exists(cand) else None


def load_blockstats(path: str) -> Dict:
    with open(path, "r") as f:
        b = json.load(f)
    if "block" not in b:
        raise ValueError(f"blockstats missing key 'block': {path}")
    if "eps" not in b:
        b["eps"] = 1e-8
    return b


def block_mean_map(img: np.ndarray, block: int) -> np.ndarray:
    H, W = img.shape
    bh, bw = H // block, W // block
    if bh <= 0 or bw <= 0:
        raise ValueError(f"block={block} too large for image {H}x{W}")
    x = img[: bh * block, : bw * block].reshape(bh, block, bw, block)
    return x.mean(axis=(1, 3))


def local_std_map(img: np.ndarray, sigma: float, eps: float) -> np.ndarray:
    if _HAVE_SCIPY:
        mu = gaussian_filter(img, sigma=sigma)
        mu2 = gaussian_filter(img * img, sigma=sigma)
        var = np.maximum(mu2 - mu * mu, 0.0)
        return np.sqrt(var + eps)
    return np.full_like(img, float(np.std(img) + eps), dtype=np.float32)


def edge_weight(img: np.ndarray, alpha: float) -> np.ndarray:
    if _HAVE_SCIPY:
        gx = sobel(img, axis=1, mode="nearest")
        gy = sobel(img, axis=0, mode="nearest")
        g = np.sqrt(gx * gx + gy * gy)
        denom = np.percentile(g, 95) + 1e-8
        gn = g / denom
        return 1.0 / (1.0 + alpha * gn)
    return np.ones_like(img, dtype=np.float32)


def brain_mask(img: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo = np.percentile(img, lo_q)
    hi = np.percentile(img, hi_q)
    m = (img > lo) & (img < hi)
    if _HAVE_SCIPY:
        m = binary_closing(m, iterations=2)
        m = binary_fill_holes(m)
    return m.astype(np.float32)


def compute_structure_residual(
    img: np.ndarray,
    rec: np.ndarray,
    *,
    lo_q: float,
    hi_q: float,
    local_sigma: float,
    edge_alpha: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = brain_mask(img, lo_q=lo_q, hi_q=hi_q)
    w_edge = edge_weight(img, alpha=edge_alpha)

    res = np.abs(img - rec).astype(np.float32)
    lstd = local_std_map(img.astype(np.float32), sigma=local_sigma, eps=eps)
    res_norm = (res / (lstd + eps)) * mask * w_edge
    return res_norm.astype(np.float32), mask, w_edge


def score_from_blocks(block_map: np.ndarray, mode: str, q: float, topk: int) -> float:
    v = block_map.reshape(-1)
    if mode == "mean":
        return float(np.mean(v))
    if mode == "quantile":
        qq = float(np.clip(q, 0.0, 1.0))
        return float(np.quantile(v, qq))
    if mode == "topk":
        k = int(max(1, min(topk, v.size)))
        part = np.partition(v, -k)[-k:]
        return float(np.mean(part))
    raise ValueError(f"Unknown score mode: {mode}")


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


def best_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, int]]:
    if len(np.unique(y_true)) < 2:
        thr = float(np.quantile(y_score, 0.99))
    else:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        thr = float(thr[int(np.argmax(j))])

    y_pred = (y_score >= thr).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return thr, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def save_debug_panels(
    out_dir: str,
    tag: str,
    X: np.ndarray,
    Xrec: np.ndarray,
    res_norm_list: List[np.ndarray],
    block_list: List[np.ndarray],
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thr: float,
    cfg: Config,
    n: int = 10,
) -> None:
    ensure_dir(out_dir)
    s = cfg.img_size

    # Prefer TP/FP/FN/TN examples, then fill.
    chosen: List[Tuple[str, int]] = []
    for want in ["TP", "FP", "FN", "TN"]:
        for i in range(len(scores)):
            if want == "TP" and y_true[i] == 1 and y_pred[i] == 1:
                chosen.append((want, i)); break
            if want == "FP" and y_true[i] == 0 and y_pred[i] == 1:
                chosen.append((want, i)); break
            if want == "FN" and y_true[i] == 1 and y_pred[i] == 0:
                chosen.append((want, i)); break
            if want == "TN" and y_true[i] == 0 and y_pred[i] == 0:
                chosen.append((want, i)); break

    # Fill remaining with evenly spaced indices.
    need = min(n, len(scores)) - len(chosen)
    if need > 0:
        idx = np.linspace(0, len(scores) - 1, num=need, dtype=int)
        for i in idx:
            if all(i != j for _, j in chosen):
                chosen.append(("S", int(i)))

    for j, (lab, i) in enumerate(chosen[: min(n, len(scores))]):
        img = X[i].reshape(s, s)
        rec = Xrec[i].reshape(s, s)
        rn = res_norm_list[i]
        bm = block_list[i]
        sc = float(scores[i])

        plt.figure(figsize=(12, 7))
        plt.suptitle(f"{tag}_{lab} | y={int(y_true[i])} pred={int(y_pred[i])} | score={sc:.4g} | thr={thr:.4g}")

        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap="gray"); plt.title("Original (scaled)"); plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(rec, cmap="gray"); plt.title("Reconstruction"); plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(rn, cmap="hot"); plt.title("Structure residual"); plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, 3, 4)
        plt.imshow(bm, cmap="hot"); plt.title("Residual (blocks)"); plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, 3, 5)
        up = np.kron(bm, np.ones((cfg.local_block, cfg.local_block), dtype=np.float32))[:s, :s]
        t = np.percentile(up, 95)
        overlay = (up >= t).astype(np.float32)
        plt.imshow(img, cmap="gray")
        plt.imshow(overlay, alpha=0.35)
        plt.title("Overlay: top 5% blocks")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.hist(bm.reshape(-1), bins=40)
        plt.title("Block residual histogram")
        plt.grid(True)

        out = os.path.join(out_dir, f"debug_{tag}_{lab}_i{j:05d}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Evaluate PatchNMF with structure-aware residual scoring.")

    ap.add_argument("--data-root", default="./dataset2/Brain_Stroke_CT_Dataset/", help="Dataset root")
    ap.add_argument("--test-normal", default="Testing/notumor", help="Relative to data-root")
    ap.add_argument("--test-anom", default="Testing/Ischemia", help="Relative to data-root")

    ap.add_argument("--model", required=True, help=".joblib PatchNMFBundle")
    ap.add_argument("--blockstats", default=None, help="blockstats_k?.json (optional; inferred if omitted)")

    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--norm-mode", choices=["global", "local"], default="local")
    ap.add_argument("--local-block", type=int, default=8)

    ap.add_argument("--mask-lo-q", type=float, default=20.0)
    ap.add_argument("--mask-hi-q", type=float, default=99.5)
    ap.add_argument("--local-sigma", type=float, default=2.0)
    ap.add_argument("--edge-alpha", type=float, default=4.0)

    ap.add_argument("--score-mode", choices=["topk", "quantile", "mean"], default="topk")
    ap.add_argument("--score-quantile", type=float, default=0.99)
    ap.add_argument("--topk", type=int, default=8)

    ap.add_argument("--patch-batch", type=int, default=16)
    ap.add_argument("--io-ncore", type=int, default=1)

    ap.add_argument("--auto-invert", action="store_true", default=True)
    ap.add_argument("--no-auto-invert", dest="auto_invert", action="store_false")

    ap.add_argument("--out", default="./dataset2/eval_structure_residual")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-n", type=int, default=10)

    ap.add_argument("--mask-mode", choices=["zscore", "quantile"], default=None,
                    help="Compatibility alias for score-mode (zscore->topk, quantile->quantile)")
    args = ap.parse_args()

    if args.mask_mode is not None:
        args.score_mode = "quantile" if args.mask_mode == "quantile" else "topk"

    ensure_dir(args.out)

    cfg = Config(
        base_path=args.data_root,
        img_size=args.img_size,
        normal_class="",
        norm_mode=args.norm_mode,
        local_block=args.local_block,
        random_state=0,
    )

    bundle = joblib.load(args.model)

    bpath = args.blockstats or infer_blockstats_path(args.model)
    if bpath and os.path.exists(bpath):
        bstats = load_blockstats(bpath)
        block = int(bstats.get("block", args.local_block))
        eps = float(bstats.get("eps", 1e-8))
    else:
        bstats = {"block": int(args.local_block), "eps": 1e-8}
        block = int(args.local_block)
        eps = 1e-8
        bpath = None

    normal_dir = os.path.join(args.data_root, args.test_normal)
    anom_dir = os.path.join(args.data_root, args.test_anom)

    normal_paths = list_images_flat(normal_dir)
    anom_paths = list_images_flat(anom_dir)
    if len(normal_paths) == 0:
        raise SystemExit(f"No images found in normal test folder: {normal_dir}")
    if len(anom_paths) == 0:
        raise SystemExit(f"No images found in anomalous test folder: {anom_dir}")

    Xn, _ = load_images_matrix(normal_paths, cfg)
    Xa, _ = load_images_matrix(anom_paths, cfg)

    Xn_rec = bundle.reconstruct_images(Xn, batch=int(args.patch_batch))
    Xa_rec = bundle.reconstruct_images(Xa, batch=int(args.patch_batch))

    def process(X, Xrec):
        scores = np.zeros((X.shape[0],), dtype=np.float32)
        res_list: List[np.ndarray] = []
        blk_list: List[np.ndarray] = []
        s = cfg.img_size
        for i in range(X.shape[0]):
            img = X[i].reshape(s, s).astype(np.float32)
            rec = Xrec[i].reshape(s, s).astype(np.float32)
            res_norm, _, _ = compute_structure_residual(
                img, rec,
                lo_q=args.mask_lo_q,
                hi_q=args.mask_hi_q,
                local_sigma=args.local_sigma,
                edge_alpha=args.edge_alpha,
                eps=eps,
            )
            bm = block_mean_map(res_norm, block).astype(np.float32)
            scores[i] = score_from_blocks(bm, mode=args.score_mode, q=args.score_quantile, topk=args.topk)
            res_list.append(res_norm)
            blk_list.append(bm)
        return scores, res_list, blk_list

    scores_n, res_n, blk_n = process(Xn, Xn_rec)
    scores_a, res_a, blk_a = process(Xa, Xa_rec)

    y_true = np.concatenate([np.zeros_like(scores_n, dtype=int), np.ones_like(scores_a, dtype=int)])
    y_score_raw = np.concatenate([scores_n, scores_a]).astype(np.float32)

    roc_raw_path = os.path.join(args.out, "roc_raw.png")
    auc_raw = plot_roc(y_true, y_score_raw, f"ROC raw ({os.path.basename(args.model)})", roc_raw_path)

    y_score = y_score_raw.copy()
    invert_applied = False
    if args.auto_invert and np.isfinite(auc_raw) and auc_raw < 0.5:
        y_score = -y_score_raw
        invert_applied = True

    roc_path = os.path.join(args.out, "roc.png")
    auc_used = plot_roc(
        y_true, y_score,
        f"ROC ({'auto-inverted ' if invert_applied else ''}{os.path.basename(args.model)})",
        roc_path,
    )

    thr, cm = best_threshold_youden(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)

    metrics = {
        "model": os.path.abspath(args.model),
        "blockstats": os.path.abspath(bpath) if bpath else None,
        "img_size": int(args.img_size),
        "norm_mode": args.norm_mode,
        "local_block": int(args.local_block),
        "structure_residual": {
            "mask_lo_q": float(args.mask_lo_q),
            "mask_hi_q": float(args.mask_hi_q),
            "local_sigma": float(args.local_sigma),
            "edge_alpha": float(args.edge_alpha),
            "have_scipy": bool(_HAVE_SCIPY),
        },
        "scoring": {
            "score_mode": args.score_mode,
            "score_quantile": float(args.score_quantile),
            "topk": int(args.topk),
        },
        "n_normal": int(len(scores_n)),
        "n_anom": int(len(scores_a)),
        "auc_raw": float(auc_raw),
        "auc_used": float(auc_used),
        "invert_applied": bool(invert_applied),
        "thr": float(thr),
        "confusion": cm,
        "scores_normal": {"mean": float(np.mean(scores_n)), "std": float(np.std(scores_n)),
                          "min": float(np.min(scores_n)), "max": float(np.max(scores_n))},
        "scores_anom": {"mean": float(np.mean(scores_a)), "std": float(np.std(scores_a)),
                        "min": float(np.min(scores_a)), "max": float(np.max(scores_a))},
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.debug:
        X_all = np.concatenate([Xn, Xa], axis=0)
        Xrec_all = np.concatenate([Xn_rec, Xa_rec], axis=0)
        res_all = res_n + res_a
        blk_all = blk_n + blk_a
        tag = f"resStruct_k{infer_k_from_model_path(args.model) or 'k'}_{args.score_mode}_b{args.patch_batch}"
        save_debug_panels(args.out, tag, X_all, Xrec_all, res_all, blk_all, y_score, y_true, y_pred, thr, cfg, n=args.debug_n)

    print("Saved:")
    print(" ", roc_path)
    print(" ", roc_raw_path)
    print(" ", os.path.join(args.out, "metrics.json"))
    print(f"AUC_raw={auc_raw}  AUC_used={auc_used}  invert_applied={invert_applied}  thr={thr}")


if __name__ == "__main__":
    main()

