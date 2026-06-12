#!/usr/bin/env python3
"""
Plot confusion matrices (per k) and performance curves vs k from metrics.json files
stored inside a zip (or a directory).

Usage examples
--------------
# From a zip that contains eval folders:
python plot_k_scan.py --input /path/to/eval_hybrid_k100.zip --out plots_eval

# From a directory that contains many k-run subfolders:
python plot_k_scan.py --input /path/to/eval_runs_dir --out plots_eval

# If you have multiple zips:
python plot_k_scan.py --input /path/to/eval_hybrid_k100.zip /path/to/eval_hybrid_k100_repeat.zip --out plots_all

Notes
-----
- Expects each run folder to contain a metrics.json (spelling as "metrics.json").
- Extracts k primarily from the folder name (e.g., "...k100..." -> k=100).
  If not found, tries reading from metrics.json keys (k, rank, n_components).
- Plots:
  1) Confusion matrix heatmap for each k (saved as separate PNGs)
  2) Curves vs k: AUC, F1, Efficiency (TPR/Recall), FPR, Precision, Accuracy
  3) Any other numeric scalar parameters in metrics.json (top N by variance), vs k
"""

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

K_PATTERNS = [
    re.compile(r"(?:^|[^0-9])k\s*[_-]?\s*(\d+)(?:[^0-9]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[^0-9])rank\s*[_-]?\s*(\d+)(?:[^0-9]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[^0-9])components\s*[_-]?\s*(\d+)(?:[^0-9]|$)", re.IGNORECASE),
]

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(x)
        if isinstance(x, (int, float)):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs == "":
                return None
            v = float(xs)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
    except Exception:
        return None
    return None

def find_k_from_name(name: str) -> Optional[int]:
    for pat in K_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def find_k_from_metrics(m: Dict[str, Any]) -> Optional[int]:
    for key in ["k", "rank", "n_components", "components", "nmf_rank", "K"]:
        if key in m:
            v = m.get(key)
            if isinstance(v, (int, float)) and int(v) == v:
                return int(v)
            if isinstance(v, str) and v.strip().isdigit():
                return int(v.strip())
    # Sometimes nested:
    for key in ["config", "params", "model", "nmf"]:
        if isinstance(m.get(key), dict):
            kk = find_k_from_metrics(m[key])
            if kk is not None:
                return kk
    return None

def find_metrics_json_files(root: Path) -> List[Path]:
    return list(root.rglob("metrics.json"))

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_cm(m: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (tn, fp, fn, tp) if found.
    Common keys: tn, fp, fn, tp or confusion_matrix dict/list.
    """
    # Flat keys
    keys_lower = {k.lower(): k for k in m.keys()}
    if all(k in keys_lower for k in ["tn", "fp", "fn", "tp"]):
        tn = m[keys_lower["tn"]]
        fp = m[keys_lower["fp"]]
        fn = m[keys_lower["fn"]]
        tp = m[keys_lower["tp"]]
        try:
            return int(tn), int(fp), int(fn), int(tp)
        except Exception:
            pass

    # confusion_matrix dict
    for cm_key in ["confusion_matrix", "cm", "conf_mat"]:
        if cm_key in m and isinstance(m[cm_key], dict):
            cm = m[cm_key]
            cm_l = {k.lower(): k for k in cm.keys()}
            if all(k in cm_l for k in ["tn", "fp", "fn", "tp"]):
                try:
                    return int(cm[cm_l["tn"]]), int(cm[cm_l["fp"]]), int(cm[cm_l["fn"]]), int(cm[cm_l["tp"]])
                except Exception:
                    pass

    # confusion_matrix as 2x2 list: [[tn, fp],[fn,tp]]
    if "confusion_matrix" in m and isinstance(m["confusion_matrix"], list):
        cm = m["confusion_matrix"]
        try:
            if len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
                tn, fp = cm[0]
                fn, tp = cm[1]
                return int(tn), int(fp), int(fn), int(tp)
        except Exception:
            pass

    return None

def compute_derived_metrics(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)              # efficiency/TPR
    fpr = fp / (fp + tn + eps)
    tnr = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "precision": precision,
        "recall": recall,
        "efficiency": recall,
        "fpr": fpr,
        "tnr": tnr,
        "accuracy": accuracy,
        "f1": f1,
    }

def extract_zip_to_temp(zip_path: Path, tempdir: Path) -> Path:
    out = tempdir / zip_path.stem
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out)
    return out

def flatten_numeric_scalars(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """
    Collect numeric scalar values from a nested dict, excluding obvious per-sample arrays.
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_numeric_scalars(v, key))
        elif isinstance(v, (list, tuple)):
            # skip arrays/lists (often per-sample scores)
            continue
        else:
            fv = safe_float(v)
            if fv is not None:
                out[key] = fv
    return out


# -------------------------
# Main processing
# -------------------------

def collect_runs(inputs: List[Path]) -> List[Dict[str, Any]]:
    tempdir = Path(tempfile.mkdtemp(prefix="k_scan_"))
    roots: List[Path] = []
    try:
        for inp in inputs:
            if inp.is_dir():
                roots.append(inp)
            elif inp.is_file() and inp.suffix.lower() == ".zip":
                roots.append(extract_zip_to_temp(inp, tempdir))
            else:
                raise ValueError(f"Unsupported input: {inp}")

        runs: List[Dict[str, Any]] = []
        for root in roots:
            for mj in find_metrics_json_files(root):
                m = load_json(mj)

                # Guess k from folder name or metrics content
                rel = str(mj.parent)
                k = find_k_from_name(rel)
                if k is None:
                    k = find_k_from_metrics(m)

                # pull primary metrics
                auc = None
                for key in ["auc", "AUC", "roc_auc", "rocAuc", "auroc", "AUROC"]:
                    if key in m:
                        auc = safe_float(m[key])
                        if auc is not None:
                            break
                # sometimes nested:
                if auc is None:
                    for nest in ["metrics", "eval", "results"]:
                        if isinstance(m.get(nest), dict):
                            for key in ["auc", "roc_auc", "auroc"]:
                                if key in m[nest]:
                                    auc = safe_float(m[nest][key])
                                    if auc is not None:
                                        break

                cm = get_cm(m)
                derived = {}
                if cm is not None:
                    tn, fp, fn, tp = cm
                    derived = compute_derived_metrics(tn, fp, fn, tp)

                # other scalar params (useful for plotting vs k)
                scalars = flatten_numeric_scalars(m)
                # avoid duplicating derived metrics keys if present in file
                for dk, dv in derived.items():
                    scalars[f"derived.{dk}"] = dv

                runs.append({
                    "k": k,
                    "metrics_path": str(mj),
                    "root": str(root),
                    "folder": str(mj.parent),
                    "auc": auc,
                    "cm": cm,
                    "derived": derived,
                    "scalars": scalars,
                })

        # Keep tempdir for the duration of script; we delete in caller with shutil.rmtree.
        return runs, tempdir
    except Exception:
        shutil.rmtree(tempdir, ignore_errors=True)
        raise

def plot_confusion_matrix(cm: Tuple[int, int, int, int], title: str, out_path: Path) -> None:
    tn, fp, fn, tp = cm
    mat = [[tn, fp],
           [fn, tp]]

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(mat)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    # Annotate
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_curve(xs: List[int], ys: List[float], ylabel: str, title: str, out_path: Path) -> None:
    fig = plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="One or more .zip files or directories containing runs")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    ap.add_argument("--top_params", type=int, default=12, help="Number of additional scalar params (by variance) to plot vs k")
    args = ap.parse_args()

    inputs = [Path(p).expanduser().resolve() for p in args.input]
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    runs, tempdir = collect_runs(inputs)
    try:
        # Filter out runs without k
        runs = [r for r in runs if r["k"] is not None]
        if len(runs) == 0:
            raise RuntimeError("No runs found with identifiable k and metrics.json")

        # If multiple runs share same k (repeats), we keep them all and also create averaged curves later
        runs_sorted = sorted(runs, key=lambda r: (r["k"], r["folder"]))

        # --- Confusion matrices per run (and per k) ---
        cm_dir = outdir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        for r in runs_sorted:
            if r["cm"] is None:
                continue
            k = r["k"]
            folder_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(r["folder"]).name)
            out_path = cm_dir / f"cm_k{k}_{folder_tag}.png"
            title = f"Confusion Matrix (k={k})\n{Path(r['folder']).name}"
            plot_confusion_matrix(r["cm"], title, out_path)

        # --- Aggregate by k (mean over repeats) for curves ---
        # Collect per-k lists
        by_k: Dict[int, Dict[str, List[float]]] = {}
        for r in runs_sorted:
            k = r["k"]
            by_k.setdefault(k, {})
            if r["auc"] is not None:
                by_k[k].setdefault("auc", []).append(r["auc"])
            for key in ["f1", "efficiency", "recall", "precision", "fpr", "accuracy", "tnr"]:
                if key in r["derived"]:
                    by_k[k].setdefault(key, []).append(r["derived"][key])

        ks = sorted(by_k.keys())

        def mean(lst: List[float]) -> Optional[float]:
            if not lst:
                return None
            return sum(lst) / len(lst)

        # Core performance curves
        core_keys = [
            ("auc", "AUC", "AUC vs k"),
            ("f1", "F1", "F1 vs k"),
            ("efficiency", "Efficiency (TPR/Recall)", "Efficiency vs k"),
            ("precision", "Precision", "Precision vs k"),
            ("fpr", "False Positive Rate", "FPR vs k"),
            ("accuracy", "Accuracy", "Accuracy vs k"),
        ]

        perf_dir = outdir / "performance_curves"
        perf_dir.mkdir(exist_ok=True)

        for key, ylabel, title in core_keys:
            ys = []
            xs = []
            for k in ks:
                v = mean(by_k[k].get(key, []))
                if v is None:
                    continue
                xs.append(k)
                ys.append(v)
            if len(xs) >= 2:
                plot_curve(xs, ys, ylabel, title, perf_dir / f"{key}_vs_k.png")

        # --- Additional scalar params vs k (top variance across ks) ---
        # Build per-k scalar means
        scalar_by_k: Dict[int, Dict[str, float]] = {}
        for k in ks:
            # average all scalars across runs for this k
            accum: Dict[str, List[float]] = {}
            for r in runs_sorted:
                if r["k"] != k:
                    continue
                for sk, sv in r["scalars"].items():
                    # skip huge per-sample arrays already excluded; also skip derived.* duplicates maybe ok
                    accum.setdefault(sk, []).append(sv)
            scalar_by_k[k] = {sk: mean(vals) for sk, vals in accum.items() if mean(vals) is not None}

        # Compute variance across k for each scalar
        all_scalar_keys = set()
        for k in ks:
            all_scalar_keys.update(scalar_by_k[k].keys())

        variances: List[Tuple[str, float]] = []
        for sk in all_scalar_keys:
            vals = [scalar_by_k[k].get(sk) for k in ks]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
            variances.append((sk, var))

        # Pick top params by variance, excluding ones already plotted
        already = set(["auc"] + [f"derived.{k}" for k in ["f1","efficiency","recall","precision","fpr","accuracy","tnr"]])
        variances = [(sk, var) for sk, var in sorted(variances, key=lambda x: x[1], reverse=True)
                     if sk not in already]
        top = variances[: max(0, args.top_params)]

        params_dir = outdir / "parameter_curves"
        params_dir.mkdir(exist_ok=True)

        for sk, _ in top:
            xs, ys = [], []
            for k in ks:
                v = scalar_by_k[k].get(sk)
                if v is None:
                    continue
                xs.append(k)
                ys.append(v)
            if len(xs) >= 2:
                safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sk)
                plot_curve(xs, ys, sk, f"{sk} vs k", params_dir / f"{safe_name}_vs_k.png")

        # --- Write a small summary table as CSV ---
        csv_path = outdir / "summary_by_k.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            cols = ["k", "auc", "f1", "efficiency", "precision", "fpr", "accuracy"]
            f.write(",".join(cols) + "\n")
            for k in ks:
                row = [
                    str(k),
                    str(mean(by_k[k].get("auc", [])) or ""),
                    str(mean(by_k[k].get("f1", [])) or ""),
                    str(mean(by_k[k].get("efficiency", [])) or ""),
                    str(mean(by_k[k].get("precision", [])) or ""),
                    str(mean(by_k[k].get("fpr", [])) or ""),
                    str(mean(by_k[k].get("accuracy", [])) or ""),
                ]
                f.write(",".join(row) + "\n")

        print(f"Saved plots to: {outdir}")
        print(f"Saved summary CSV to: {csv_path}")

    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    main()
