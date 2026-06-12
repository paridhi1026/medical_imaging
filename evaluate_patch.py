#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib

from nmfcore.data import train_test_paths, list_class_images, make_binary_labels
from nmfcore.preprocess import load_images_matrix
from nmfcore.metrics import roc_stats, confusion_at_threshold, threshold_by_quantile, best_f1_threshold


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def block_mse_map(diff2_img: np.ndarray, block: int) -> np.ndarray:
    H, W = diff2_img.shape
    if H % block != 0 or W % block != 0:
        raise ValueError(f"Image {H}x{W} not divisible by block {block}")
    bh, bw = H // block, W // block
    x = diff2_img.reshape(bh, block, bw, block)
    return x.mean(axis=(1, 3))


def topk_mean(x: np.ndarray, k: int) -> float:
    flat = x.reshape(-1)
    k = min(k, flat.size)
    return float(np.sort(flat)[-k:].mean())


def quantile_score(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x.reshape(-1), q))


def load_blockstats(path: str):
    with open(path, "r") as f:
        d = json.load(f)
    mu = np.array(d["mu"], dtype=np.float32)
    sigma = np.array(d["sigma"], dtype=np.float32)
    eps = float(d.get("eps", 1e-8))
    block = int(d["block"])
    mask_lo_q = float(d.get("mask_lo_q", 20.0))
    mask_hi_q = float(d.get("mask_hi_q", 99.5))
    return block, eps, mu, sigma, mask_lo_q, mask_hi_q


def save_debug_panel(out_dir, idx, orig, rec, Z, block, zthr=3.0, tag=""):
    os.makedirs(out_dir, exist_ok=True)
    resid = np.abs(orig - rec)
    Z_up = np.kron(Z, np.ones((block, block), dtype=np.float32))
    Z_up = Z_up[:orig.shape[0], :orig.shape[1]]
    hot = (Z_up > zthr).astype(np.float32)

    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(2,3,1); ax.set_title("Original"); ax.imshow(orig, cmap="gray"); ax.axis("off")
    ax = fig.add_subplot(2,3,2); ax.set_title("Recon"); ax.imshow(rec, cmap="gray"); ax.axis("off")
    ax = fig.add_subplot(2,3,3); ax.set_title("|Residual|"); ax.imshow(resid, cmap="hot"); ax.axis("off")
    ax = fig.add_subplot(2,3,4); ax.set_title("Z-map"); im=ax.imshow(Z, cmap="hot"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.axis("off")
    ax = fig.add_subplot(2,3,5); ax.set_title(f"Overlay Z>{zthr}"); ax.imshow(orig, cmap="gray"); ax.imshow(hot, cmap="spring", alpha=0.35); ax.axis("off")
    ax = fig.add_subplot(2,3,6); ax.set_title("Z hist"); ax.hist(Z.reshape(-1), bins=60); ax.grid(True)

    fig.tight_layout(rect=[0,0.02,1,0.95])
    fig.suptitle(tag, fontsize=11)
    out = os.path.join(out_dir, f"debug_{tag}_i{idx:05d}.png")
    plt.savefig(out, dpi=180)
    plt.close(fig)
    return out


def threshold_youden(y_true, scores):
    roc = roc_stats(scores, y_true)
    j = roc["tpr"] - roc["fpr"]
    return float(roc["thr"][int(np.argmax(j))])


def _extract_k_from_model(model_path: str) -> str:
    base = os.path.basename(model_path)
    # accepts patchnmf_k100.joblib or nmf_k50.joblib
    for token in base.replace(".joblib", "").split("_"):
        if token.startswith("k") and token[1:].isdigit():
            return token[1:]
    # fallback
    digits = "".join([c for c in base if c.isdigit()])
    return digits if digits else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--blockstats", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--use", choices=["testing","training"], default="testing")
    ap.add_argument("--out", required=True)

    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--agg", choices=["topk","quantile","max"], default="quantile")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--q", type=float, default=0.995)

    ap.add_argument("--threshold-mode", choices=["quantile","best_f1","youden"], default="quantile")
    ap.add_argument("--thr-quantile", type=float, default=0.99)
    ap.add_argument("--val-mse", default=None)

    ap.add_argument("--io-ncore", type=int, default=1)

    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--debug-n", type=int, default=0)
    ap.add_argument("--debug-zthr", type=float, default=3.0)
    args = ap.parse_args()

    ensure_dir(args.out)
    bundle = joblib.load(args.model)
    cfg = bundle.cfg

    b_block, b_eps, mu, sigma, mask_lo_q, mask_hi_q = load_blockstats(args.blockstats)
    if b_block != args.block:
        raise ValueError(f"Block mismatch: blockstats={b_block} vs --block={args.block}")

    train_root, test_root = train_test_paths(args.base)
    root = test_root if args.use == "testing" else train_root
    if not os.path.isdir(root):
        raise RuntimeError(f"Root not found: {root} (base={args.base})")

    classes = list_class_images(root)
    if not classes:
        raise RuntimeError(f"No class folders found under: {root}")

    all_paths, _ = make_binary_labels(classes, cfg.normal_class)
    X, kept = load_images_matrix(all_paths, cfg, ncore=args.io_ncore)

    if X.shape[0] == 0:
        # better diagnostics
        print("DEBUG: base=", args.base)
        print("DEBUG: root=", root)
        for cls, files in classes.items():
            print(f"DEBUG: class={cls} n_files={len(files)} example={files[0] if files else None}")
        raise RuntimeError("No images loaded after preprocessing. (Check extensions / file permissions / corrupt files)")

    # labels aligned to kept
    kept_set = set(kept)
    y, kept_classes = [], []
    for cls, files in classes.items():
        for f in files:
            if f in kept_set:
                kept_classes.append(cls)
                y.append(0 if cls == cfg.normal_class else 1)
    y = np.asarray(y, dtype=np.int32)

    # reconstruction
    Xrec = bundle.reconstruct_images(X, batch=16)

    # compute per-image scores using masked residual -> block Z -> aggregate
    s = cfg.img_size
    scores = np.zeros((X.shape[0],), dtype=np.float32)

    ktag = _extract_k_from_model(args.model)

    for i in range(X.shape[0]):
        img = X[i].reshape(s, s)
        rec = Xrec[i].reshape(s, s)

        lo = np.percentile(img, mask_lo_q)
        hi = np.percentile(img, mask_hi_q)
        mask = ((img > lo) & (img < hi)).astype(np.float32)

        diff2 = ((img - rec) ** 2) * mask
        E = block_mse_map(diff2, args.block)
        Z = (E - mu) / (sigma + b_eps)
        Zpos = np.maximum(Z, 0.0)  # positive-Z only

        if args.agg == "max":
            scores[i] = float(Zpos.max())
        elif args.agg == "topk":
            scores[i] = float(topk_mean(Zpos, args.topk))
        else:
            scores[i] = float(quantile_score(Zpos, args.q))

        if args.debug_dir and i < args.debug_n:
            save_debug_panel(
                args.debug_dir, i, img, rec, Zpos, args.block,
                zthr=args.debug_zthr,
                tag=f"{args.agg}_k{ktag}"
            )

    # threshold
    thr_source = ""
    if args.threshold_mode == "quantile":
        if args.val_mse and os.path.isfile(args.val_mse):
            val_mse = np.load(args.val_mse)
            thr = threshold_by_quantile(val_mse, args.thr_quantile)
            thr_source = f"val_mse_quantile(q={args.thr_quantile})"
        else:
            thr = threshold_by_quantile(scores[y == 0], args.thr_quantile)
            thr_source = f"fallback_normals_quantile(q={args.thr_quantile})"
    elif args.threshold_mode == "best_f1":
        thr, bestf1 = best_f1_threshold(scores, y)
        thr_source = "best_f1_on_eval_set"
    else:
        thr = threshold_youden(y, scores)
        thr_source = "youden_on_eval_set"

    roc = roc_stats(scores, y)
    conf = confusion_at_threshold(scores, y, thr)

    # per-class summary
    per_class = {}
    for cls in sorted(set(kept_classes)):
        idx = [i for i,c in enumerate(kept_classes) if c == cls]
        scls = scores[idx]
        per_class[cls] = {
            "n": int(len(idx)),
            "mean_score": float(scls.mean()),
            "flagged_fraction": float((scls > thr).mean()),
        }

    out = {
        "model": args.model,
        "blockstats": args.blockstats,
        "root": root,
        "agg": args.agg,
        "topk": int(args.topk),
        "q": float(args.q),
        "block": int(args.block),
        "threshold": float(thr),
        "threshold_source": thr_source,
        "roc_auc": float(roc["auc"]),
        "roc_curve": {"fpr": roc["fpr"].tolist(), "tpr": roc["tpr"].tolist(), "thr": roc["thr"].tolist()},
        "confusion_matrix_TN_FP_FN_TP": conf["cm"].tolist(),
        "per_class": per_class,
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    plt.figure()
    plt.plot(roc["fpr"], roc["tpr"])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={roc['auc']:.4f})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "roc.png"), dpi=160)
    plt.close()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
