#!/usr/bin/env python3

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_curve

from nmfcore.config import Config
from nmfcore.preprocess import load_images_matrix


# ----------------------------------------------------------
# utilities
# ----------------------------------------------------------

def find_eval_dirs(root):

    evals = []

    for name in os.listdir(root):

        full = os.path.join(root, name)

        if not os.path.isdir(full):
            continue

        m = re.search(r"k(\d+)", name)

        if m:
            evals.append((int(m.group(1)), full))

    evals.sort(key=lambda x: x[0])
    return evals


def compute_metrics(cm):

    tn = cm["tn"]
    fp = cm["fp"]
    fn = cm["fn"]
    tp = cm["tp"]

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision, recall, acc, f1


# ----------------------------------------------------------
# metric plots
# ----------------------------------------------------------

def plot_metrics(evals, outdir):

    Ks = []
    aucs = []
    prec = []
    rec = []
    acc = []
    f1 = []

    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []

    for k, folder in evals:

        mfile = os.path.join(folder, "metrics.json")
        if not os.path.exists(mfile):
            continue

        data = json.load(open(mfile))

        Ks.append(k)

        aucs.append(data.get("auc", np.nan))

        cm = data["confusion"]

        p, r, a, f = compute_metrics(cm)

        prec.append(p)
        rec.append(r)
        acc.append(a)
        f1.append(f)

        tn_list.append(cm["tn"])
        fp_list.append(cm["fp"])
        fn_list.append(cm["fn"])
        tp_list.append(cm["tp"])

    plt.figure(figsize=(10,6))
    plt.plot(Ks, aucs, label="AUC")
    plt.plot(Ks, f1, label="F1")
    plt.plot(Ks, acc, label="Accuracy")
    plt.plot(Ks, prec, label="Precision")
    plt.plot(Ks, rec, label="Recall")
    plt.legend()
    plt.xlabel("K")
    plt.grid(True)
    plt.savefig(os.path.join(outdir,"metrics_vs_k.png"))
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(Ks, tp_list, label="TP")
    plt.plot(Ks, fp_list, label="FP")
    plt.plot(Ks, fn_list, label="FN")
    plt.plot(Ks, tn_list, label="TN")
    plt.legend()
    plt.xlabel("K")
    plt.grid(True)
    plt.savefig(os.path.join(outdir,"confusion_vs_k.png"))
    plt.close()


# ----------------------------------------------------------
# HEP-style efficiency plot
# ----------------------------------------------------------

def plot_hep_curve(evals, outdir):

    plt.figure(figsize=(8,6))

    for k, folder in evals:

        mfile = os.path.join(folder, "metrics.json")

        if not os.path.exists(mfile):
            continue

        data = json.load(open(mfile))

        roc = data.get("roc_curve_raw")

        if roc is None:
            continue

        fpr = np.array(roc["fpr"])
        tpr = np.array(roc["tpr"])

        background_rejection = 1 - fpr
        signal_eff = tpr

        plt.plot(signal_eff, background_rejection, label=f"K={k}")

        # mark key points
        for target in [0.5, 0.8, 0.9]:
            idx = np.argmin(np.abs(signal_eff - target))
            plt.scatter(signal_eff[idx], background_rejection[idx])

    plt.xlabel("Signal efficiency (TPR)")
    plt.ylabel("Background rejection (1 - FPR)")
    plt.title("HEP-style anomaly performance")
    plt.legend(fontsize=7)
    plt.grid(True)

    plt.savefig(os.path.join(outdir,"hep_efficiency_curve.png"))
    plt.close()


# ----------------------------------------------------------
# example reconstruction
# ----------------------------------------------------------

def plot_examples(model_path, normal_img, anom_img, cfg, outdir):

    bundle = joblib.load(model_path)

    X,_ = load_images_matrix([normal_img, anom_img], cfg)
    Xrec = bundle.reconstruct_images(X, batch=16)

    labels = ["normal","ischemia"]

    for i in range(2):

        img = X[i].reshape(cfg.img_size,cfg.img_size)
        rec = Xrec[i].reshape(cfg.img_size,cfg.img_size)

        plt.figure(figsize=(8,4))

        plt.subplot(1,2,1)
        plt.imshow(img,cmap='gray')
        plt.title("Original")

        plt.subplot(1,2,2)
        plt.imshow(rec,cmap='gray')
        plt.title("Reconstructed")

        plt.savefig(os.path.join(outdir,f"example_{labels[i]}.png"))
        plt.close()


# ----------------------------------------------------------
# main
# ----------------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--model-example", required=True)
    ap.add_argument("--normal-example", required=True)
    ap.add_argument("--anom-example", required=True)

    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--norm-mode", default="local")
    ap.add_argument("--local-block", type=int, default=8)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    evals = find_eval_dirs(args.eval_root)

    plot_metrics(evals, args.out)
    plot_hep_curve(evals, args.out)

    cfg = Config(
        base_path="",
        img_size=args.img_size,
        normal_class="",
        norm_mode=args.norm_mode,
        local_block=args.local_block,
        random_state=0
    )

    plot_examples(
        args.model_example,
        args.normal_example,
        args.anom_example,
        cfg,
        args.out
    )


if __name__ == "__main__":
    main()
