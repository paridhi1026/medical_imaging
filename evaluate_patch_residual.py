#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import joblib

from nmfcore.data import train_test_paths, list_class_images, make_binary_labels
from nmfcore.preprocess import load_images_matrix
from nmfcore.metrics import roc_stats, confusion_at_threshold, threshold_by_quantile, best_f1_threshold
from nmfcore.patches import patchify


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_roc(out_png: str, fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str) -> None:
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.4f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _threshold_youden(y_true: np.ndarray, scores: np.ndarray) -> float:
    roc = roc_stats(scores, y_true)
    fpr = roc["fpr"]
    tpr = roc["tpr"]
    thr = roc["thr"]
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _extract_k_from_model(model_path: str) -> str:
    base = os.path.basename(model_path)
    base = base.replace(".joblib", "")
    # accept patchnmf_k100 or nmf_k50
    for tok in base.split("_"):
        if tok.startswith("k") and tok[1:].isdigit():
            return tok[1:]
    digits = "".join([c for c in base if c.isdigit()])
    return digits if digits else "?"


def _topk_mean(x: np.ndarray, k: int) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    k = max(1, min(int(k), x.size))
    part = np.partition(x, -k)[-k:]
    return float(part.mean())


def _patch_residual_scores(
    bundle,
    X: np.ndarray,
    agg: str,
    q: float,
    topk: int,
    patch_batch: int,
    mask_mode: str,
    mask_lo_q: float,
    mask_hi_q: float,
    debug_dir: str,
    debug_n: int,
    debug_tag: str,
) -> Tuple[np.ndarray, Dict]:
    """
    Patch-residual scoring:
      For each image:
        - patchify(img)
        - reconstruct patches via W @ H (no unpatchify needed)
        - residual per patch = mean((P - Prec)^2)
        - aggregate residuals to scalar score using agg

    mask_mode:
      - "none": no mask; include all pixels in patch residual
      - "quantile": compute per-image mask using percentiles; patches outside mask are down-weighted by zeroing masked pixels
    """
    s = bundle.cfg.img_size
    patch = int(bundle.patch)
    stride = int(bundle.stride)

    scores = np.zeros((X.shape[0],), dtype=np.float32)

    # debug summary
    dbg = {
        "agg": agg,
        "q": float(q),
        "topk": int(topk),
        "patch": patch,
        "stride": stride,
        "mask_mode": mask_mode,
        "mask_lo_q": float(mask_lo_q),
        "mask_hi_q": float(mask_hi_q),
        "patch_batch": int(patch_batch),
    }

    if debug_dir:
        ensure_dir(debug_dir)

    for i in range(X.shape[0]):
        img = X[i].reshape(s, s).astype(np.float32, copy=False)

        if mask_mode == "quantile":
            lo = np.percentile(img, mask_lo_q)
            hi = np.percentile(img, mask_hi_q)
            mask = ((img > lo) & (img < hi)).astype(np.float32)
        else:
            mask = None

        # patches from image
        P = patchify(img, patch=patch, stride=stride)  # (npatch, p*p)

        if mask is not None:
            # apply pixel-level mask to patches (mask out background/saturated)
            M = patchify(mask, patch=patch, stride=stride)
            P_eff = P * M
        else:
            P_eff = P

        # reconstruct patches in batches
        n_p = P_eff.shape[0]
        if n_p == 0:
            scores[i] = 0.0
            continue

        # scale then transform
        # (MinMaxScaler may produce tiny negatives from float errors; clip)
        residuals = np.zeros((n_p,), dtype=np.float32)

        for j0 in range(0, n_p, patch_batch):
            j1 = min(n_p, j0 + patch_batch)
            Ps = bundle.scaler.transform(P_eff[j0:j1])
            Ps = np.clip(Ps, 0.0, None)

            W = bundle.nmf.transform(Ps)           # (b, k)
            Prec_s = W @ bundle.nmf.components_    # (b, p*p)

            Prec = bundle.scaler.inverse_transform(Prec_s)
            Prec = np.clip(Prec, 0.0, 1.0)

            # residual computed against the same effective patch representation
            d = (P_eff[j0:j1] - Prec).astype(np.float32, copy=False)
            residuals[j0:j1] = (d * d).mean(axis=1)

        # aggregate residuals
        if agg == "max":
            score = float(residuals.max())
        elif agg == "topk":
            score = _topk_mean(residuals, topk)
        elif agg == "quantile":
            score = float(np.quantile(residuals, q))
        else:
            raise ValueError(f"Unknown agg='{agg}'. Choose from: max, topk, quantile.")

        scores[i] = score

        # compact debug: save 1 panel showing image + heatmap of patch residual grid
        if debug_dir and i < debug_n:
            # reshape residuals to patch grid
            nH = (s - patch) // stride + 1
            nW = (s - patch) // stride + 1
            if residuals.size == nH * nW:
                R = residuals.reshape(nH, nW)
            else:
                R = None

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 2, 1)
            ax.set_title("Input (scaled)")
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            ax = fig.add_subplot(1, 2, 2)
            ax.set_title(f"Patch residual map ({agg})\nscore={score:.3g}")
            if R is not None:
                im = ax.imshow(R, cmap="hot")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.1, 0.5, "Residual grid reshape failed", fontsize=10)
            ax.axis("off")

            fig.tight_layout()
            outp = os.path.join(debug_dir, f"debug_patchres_{debug_tag}_i{i:05d}.png")
            plt.savefig(outp, dpi=180)
            plt.close(fig)

    return scores, dbg


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True, help="Path to saved patchnmf_k*.joblib bundle")
    ap.add_argument("--base", required=True, help="Dataset base (expects Training/ and Testing/)")
    ap.add_argument("--use", choices=["testing", "training"], default="testing")
    ap.add_argument("--out", required=True, help="Output directory for metrics/plots")

    # scoring
    ap.add_argument("--agg", choices=["max", "topk", "quantile"], default="quantile",
                    help="Aggregate patch residuals into a scalar anomaly score.")
    ap.add_argument("--q", type=float, default=0.995, help="Quantile for agg=quantile.")
    ap.add_argument("--topk", type=int, default=4, help="Top-k mean for agg=topk.")
    ap.add_argument("--patch-batch", type=int, default=4096,
                    help="Batch size for patch transform/reconstruction (bigger = faster, more RAM).")

    # patch residual masking
    ap.add_argument("--mask-mode", choices=["none", "quantile"], default="quantile",
                    help="Optional pixel mask to suppress background/saturated regions.")
    ap.add_argument("--mask-lo-q", type=float, default=20.0)
    ap.add_argument("--mask-hi-q", type=float, default=99.5)

    # thresholding
    ap.add_argument("--threshold", type=float, default=None, help="Explicit threshold (overrides all).")
    ap.add_argument("--threshold-mode", choices=["quantile", "best_f1", "youden"], default="quantile",
                    help="quantile uses normals (val-mse file if provided); best_f1/youden use labeled eval set.")
    ap.add_argument("--thr-quantile", type=float, default=0.99)
    ap.add_argument("--val-mse", default=None,
                    help="Optional: val_mse_k*.npy from train_patch.py for clean quantile thresholding.")

    # I/O parallelism for loading images
    ap.add_argument("--io-ncore", type=int, default=1)

    # debug
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--debug-n", type=int, default=0)

    args = ap.parse_args()
    ensure_dir(args.out)

    bundle = joblib.load(args.model)
    cfg = bundle.cfg

    # dataset root
    train_root, test_root = train_test_paths(args.base)
    root = test_root if args.use == "testing" else train_root

    classes = list_class_images(root)
    if not classes:
        raise RuntimeError(f"No class folders found under: {root}")

    # build all paths + labels
    all_paths, _ = make_binary_labels(classes, cfg.normal_class)
    X, kept = load_images_matrix(all_paths, cfg, ncore=args.io_ncore)

    if X.shape[0] == 0:
        print("DEBUG: base=", args.base)
        print("DEBUG: root=", root)
        for cls, files in classes.items():
            print(f"DEBUG: class={cls} n_files={len(files)} example={files[0] if files else None}")
        raise RuntimeError("No images loaded after preprocessing.")

    # align labels with kept
    kept_set = set(kept)
    y_kept = []
    kept_classes = []
    for cls_name, paths in classes.items():
        for p in paths:
            if p in kept_set:
                kept_classes.append(cls_name)
                y_kept.append(0 if cls_name == cfg.normal_class else 1)
    y_kept = np.asarray(y_kept, dtype=np.int32)

    debug_tag = f"k{_extract_k_from_model(args.model)}_{args.agg}"
    scores, dbg = _patch_residual_scores(
        bundle=bundle,
        X=X,
        agg=args.agg,
        q=args.q,
        topk=args.topk,
        patch_batch=args.patch_batch,
        mask_mode=args.mask_mode,
        mask_lo_q=args.mask_lo_q,
        mask_hi_q=args.mask_hi_q,
        debug_dir=args.debug_dir,
        debug_n=args.debug_n,
        debug_tag=debug_tag,
    )

    # threshold selection
    thr_source = ""
    if args.threshold is not None:
        thr = float(args.threshold)
        thr_source = "explicit"
    else:
        if args.threshold_mode == "quantile":
            if args.val_mse is not None and os.path.isfile(args.val_mse):
                val_mse = np.load(args.val_mse)
                thr = threshold_by_quantile(val_mse, args.thr_quantile)
                thr_source = f"val_mse_quantile(q={args.thr_quantile})"
            else:
                # fallback threshold on normals in eval root
                normal_scores = scores[y_kept == 0]
                thr = threshold_by_quantile(normal_scores, args.thr_quantile)
                thr_source = f"fallback_normals_in_{args.use}_quantile(q={args.thr_quantile})"
        elif args.threshold_mode == "best_f1":
            thr, best_f1 = best_f1_threshold(scores, y_kept)
            thr = float(thr)
            thr_source = "best_f1_on_eval_set"
        else:
            thr = _threshold_youden(y_kept, scores)
            thr_source = "youden_on_eval_set"

    roc = roc_stats(scores, y_kept)
    conf = confusion_at_threshold(scores, y_kept, thr)

    # per-class
    per_class = {}
    for cls_name in sorted(set(kept_classes)):
        idx = [i for i, c in enumerate(kept_classes) if c == cls_name]
        if not idx:
            continue
        cls_s = scores[idx]
        cls_pred = (cls_s > thr).astype(int)
        per_class[cls_name] = {
            "n": int(len(idx)),
            "mean_score": float(cls_s.mean()),
            "flagged_fraction": float(cls_pred.mean()),
        }

    out = {
        "model": args.model,
        "root": root,
        "score_mode": "patch_residual",
        "agg": args.agg,
        "q": float(args.q),
        "topk": int(args.topk),
        "patch": int(bundle.patch),
        "stride": int(bundle.stride),
        "patch_batch": int(args.patch_batch),
        "mask_mode": args.mask_mode,
        "mask_lo_q": float(args.mask_lo_q),
        "mask_hi_q": float(args.mask_hi_q),
        "threshold": float(thr),
        "threshold_source": thr_source,
        "roc_auc": float(roc["auc"]),
        "roc_curve": {
            "fpr": roc["fpr"].tolist(),
            "tpr": roc["tpr"].tolist(),
            "thr": roc["thr"].tolist(),
        },
        "confusion_matrix_TN_FP_FN_TP": conf["cm"].tolist(),
        "per_class": per_class,
        "debug": dbg,
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    plot_roc(
        os.path.join(args.out, "roc.png"),
        roc["fpr"], roc["tpr"], roc["auc"],
        title=f"ROC ({os.path.basename(args.model)})"
    )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
