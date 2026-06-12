#!/usr/bin/env python3
import os, json, argparse, traceback
import numpy as np
import joblib

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

from nmfcore.config import Config
from nmfcore.data import train_test_paths, list_class_images
from nmfcore.preprocess import load_images_matrix
from nmfcore.model_patch import PatchNMFBundle


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def block_mse_map(diff2_img: np.ndarray, block: int) -> np.ndarray:
    H, W = diff2_img.shape
    if H % block != 0 or W % block != 0:
        raise ValueError(f"Image {H}x{W} not divisible by block {block}")
    bh, bw = H // block, W // block
    x = diff2_img.reshape(bh, block, bw, block)
    return x.mean(axis=(1, 3))


def fit_blockstats(bundle: PatchNMFBundle, X_val: np.ndarray, block: int,
                   mask_lo_q=20.0, mask_hi_q=99.5, batch_recon: int = 16):
    """
    Compute per-block residual energy mean/std on val normals.
    Uses a mask to ignore background and saturated pixels.
    """
    s = bundle.cfg.img_size
    Xrec = bundle.reconstruct_images(X_val, batch=batch_recon)

    maps = []
    for i in range(X_val.shape[0]):
        img = X_val[i].reshape(s, s)
        lo = np.percentile(img, mask_lo_q)
        hi = np.percentile(img, mask_hi_q)
        mask = ((img > lo) & (img < hi)).astype(np.float32)

        diff2 = ((X_val[i] - Xrec[i]) ** 2).reshape(s, s) * mask
        maps.append(block_mse_map(diff2, block))
    maps = np.stack(maps, axis=0)

    return {
        "block": int(block),
        "eps": 1e-8,
        "mu": maps.mean(axis=0).astype(np.float32).tolist(),
        "sigma": maps.std(axis=0).astype(np.float32).tolist(),
        "mask_lo_q": float(mask_lo_q),
        "mask_hi_q": float(mask_hi_q),
        "patch": int(bundle.patch),
        "stride": int(bundle.stride),
    }


def _train_one_k(k: int, args, cfg: Config, X_train: np.ndarray, X_val: np.ndarray) -> dict:
    try:
        nmf = NMF(
            n_components=k,
            init="nndsvda",
            random_state=args.seed,
            max_iter=args.max_iter,
        )
        scaler = MinMaxScaler()
        bundle = PatchNMFBundle(cfg=cfg, nmf=nmf, scaler=scaler, patch=args.patch, stride=args.stride)

        bundle.fit_on_patches(X_train, max_patches=args.max_patches, seed=args.seed)

        # val recon and val mse distribution
        Xrec_val = bundle.reconstruct_images(X_val, batch=args.recon_batch)
        val_mse = ((X_val - Xrec_val) ** 2).mean(axis=1).astype(np.float32)
        thr = float(np.quantile(val_mse, args.thr_quantile))

        # blockstats
        bstats = fit_blockstats(
            bundle, X_val,
            block=args.block,
            mask_lo_q=args.mask_lo_q,
            mask_hi_q=args.mask_hi_q,
            batch_recon=args.recon_batch
        )

        model_path = os.path.join(args.out, "models", f"patchnmf_k{k}.joblib")
        val_mse_path = os.path.join(args.out, "models", f"val_mse_k{k}.npy")
        bstats_path = os.path.join(args.out, "models", f"blockstats_k{k}.json")

        joblib.dump(bundle, model_path)
        np.save(val_mse_path, val_mse)
        with open(bstats_path, "w") as f:
            json.dump(bstats, f, indent=2)

        return {
            "k": int(k),
            "ok": True,
            "model_path": model_path,
            "val_mse_mean": float(val_mse.mean()),
            "val_mse_std": float(val_mse.std()),
            "thr_quantile": float(args.thr_quantile),
            "thr_value": thr,
            "val_mse_path": val_mse_path,
            "blockstats_path": bstats_path,
            "patch": int(args.patch),
            "stride": int(args.stride),
            "block": int(args.block),
            "norm_mode": cfg.norm_mode,
            "local_block": int(cfg.local_block),
        }

    except Exception as e:
        return {
            "k": int(k),
            "ok": False,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--normal-class", default="notumor")

    ap.add_argument("--norm-mode", choices=["global", "local"], default="global")
    ap.add_argument("--local-block", type=int, default=8)

    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)

    ap.add_argument("--k-list", default="10,20,30,50,75,100")
    ap.add_argument("--max-iter", type=int, default=800)
    ap.add_argument("--max-patches", type=int, default=0, help="Subsample patches for NMF fit (0=no subsample)")

    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--mask-lo-q", type=float, default=20.0)
    ap.add_argument("--mask-hi-q", type=float, default=99.5)

    ap.add_argument("--thr-quantile", type=float, default=0.99)

    ap.add_argument("--ncore", type=int, default=1,
                    help="Parallel workers across k values.")
    ap.add_argument("--io-ncore", type=int, default=1,
                    help="Parallel workers for image loading/preprocessing.")
    ap.add_argument("--recon-batch", type=int, default=16)

    args = ap.parse_args()

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "models"))

    cfg = Config(
        base_path=args.base,
        img_size=args.img_size,
        normal_class=args.normal_class,
        norm_mode=args.norm_mode,
        local_block=args.local_block,
        random_state=args.seed,
    )

    # dataset paths
    train_root, _ = train_test_paths(cfg.base_path)
    if not os.path.isdir(train_root):
        raise RuntimeError(f"Training folder not found: {train_root}")

    classes = list_class_images(train_root)
    good = classes.get(cfg.normal_class, [])
    if not good:
        raise RuntimeError(f"No normal images found for class='{cfg.normal_class}' in {train_root}")

    # split normals only
    good_train, good_tmp = train_test_split(good, train_size=args.train_frac, random_state=args.seed, shuffle=True)
    val_size = args.val_frac / (1.0 - args.train_frac)
    good_val, good_hold = train_test_split(good_tmp, train_size=val_size, random_state=args.seed, shuffle=True)

    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump({"good_train": good_train, "good_val": good_val, "good_hold": good_hold}, f, indent=2)

    # load images (parallel if io-ncore>1)
    X_train, kept_train = load_images_matrix(good_train, cfg, ncore=args.io_ncore)
    X_val, kept_val = load_images_matrix(good_val, cfg, ncore=args.io_ncore)

    print("X_train:", X_train.shape, "kept:", len(kept_train))
    print("X_val  :", X_val.shape, "kept:", len(kept_val))
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise RuntimeError("No images loaded after preprocessing (check paths / file formats).")

    ks = sorted({int(x) for x in args.k_list.split(",") if x.strip()})
    print(f"Running PatchNMF k-scan: ks={ks} | ncore={args.ncore} | io_ncore={args.io_ncore} | norm={cfg.norm_mode}")

    results = Parallel(n_jobs=max(1, int(args.ncore)), backend="loky", verbose=10)(
        delayed(_train_one_k)(k, args, cfg, X_train, X_val) for k in ks
    )

    results.sort(key=lambda r: r["k"])
    with open(os.path.join(args.out, "k_scan_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print(f"Finished: {len(ok)} ok, {len(bad)} failed")
    for r in ok:
        print(f"[k={r['k']}] val_mse_mean={r['val_mse_mean']:.6g} thr(q={r['thr_quantile']})={r['thr_value']:.6g}")
    if bad:
        print("Failures:")
        for r in bad:
            print(f"[k={r['k']}] {r.get('error')}")


if __name__ == "__main__":
    main()
