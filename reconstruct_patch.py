#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt

from nmfcore.data import train_test_paths, list_class_images
from nmfcore.preprocess import load_images_matrix


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def save_triplet(out_png, orig, rec):
    resid = np.abs(orig - rec)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("Original"); plt.imshow(orig, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Reconstruction"); plt.imshow(rec, cmap="gray"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("|Residual|"); plt.imshow(resid, cmap="hot"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--use", choices=["testing","training"], default="testing")
    ap.add_argument("--out", required=True)

    ap.add_argument("--io-ncore", type=int, default=1)
    ap.add_argument("--save-n", type=int, default=0, help="Save N example recon panels")
    args = ap.parse_args()

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "examples"))

    bundle = joblib.load(args.model)
    cfg = bundle.cfg

    train_root, test_root = train_test_paths(args.base)
    root = test_root if args.use == "testing" else train_root
    if not os.path.isdir(root):
        raise RuntimeError(f"root not found: {root}")

    classes = list_class_images(root)

    # load all images (all classes)
    all_paths = []
    all_cls = []
    for cls, files in classes.items():
        for f in files:
            all_paths.append(f)
            all_cls.append(cls)

    X, kept = load_images_matrix(all_paths, cfg, ncore=args.io_ncore)
    if X.shape[0] == 0:
        raise RuntimeError("No images loaded after preprocessing.")

    # align classes to kept
    kept_set = set(kept)
    kept_cls = []
    for f, c in zip(all_paths, all_cls):
        if f in kept_set:
            kept_cls.append(c)

    Xrec = bundle.reconstruct_images(X, batch=16)
    mse = ((X - Xrec) ** 2).mean(axis=1).astype(np.float32)

    out = {
        "model": args.model,
        "root": root,
        "n": int(X.shape[0]),
        "mean_mse": float(mse.mean()),
        "std_mse": float(mse.std()),
        "per_class": {}
    }

    for cls in sorted(set(kept_cls)):
        idx = [i for i,c in enumerate(kept_cls) if c == cls]
        out["per_class"][cls] = {
            "n": int(len(idx)),
            "mean_mse": float(mse[idx].mean()),
            "std_mse": float(mse[idx].std()),
        }

    np.save(os.path.join(args.out, "recon_mse.npy"), mse)
    with open(os.path.join(args.out, "recon_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # optional example panels
    s = cfg.img_size
    nsave = min(int(args.save_n), X.shape[0])
    for i in range(nsave):
        orig = X[i].reshape(s, s)
        rec = Xrec[i].reshape(s, s)
        save_triplet(os.path.join(args.out, "examples", f"recon_{i:05d}.png"), orig, rec)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
