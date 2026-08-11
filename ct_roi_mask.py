#!/usr/bin/env python3
"""
ct_roi_mask.py  –  Brain-tissue ROI masking for JPEG brain CT scans.


Typical brain CT windowing:
    WL = 40 HU,  WW = 80 HU   → visible range ~ -1000 … 3071 HU
    pixel ≈ (HU + 1024) / (3071 + 1024) * 255

Useful approximate pixel thresholds for 8-bit JPEG brain CTs:
    Background (air)     :  HU < -900   →  pixel  <  10
    Skull / dense bone   :  HU > +400   →  pixel  > 200  (adjustable)
    Brain tissue (keep)  :  10 ≤ pixel ≤ 200

We expose a single function  `build_brain_mask(img_gray, cfg)`  that returns
a float32 binary mask (1 = keep, 0 = ignore).

The mask is used in two places in your pipeline:
  1. During patch extraction  (ignore patches whose mean mask value < min_mask_frac)
  2. During block-MSE computation  (zero-out masked regions so skull/air don't
     inflate reconstruction error)
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path


# ---------------------------------------------------------------------------
# Default HU-equivalent thresholds for 8-bit JPEG brain CT
# ---------------------------------------------------------------------------
DEFAULT_BG_HI     = 10    # pixels ≤ this  → background / air  (HU ≈ -900)
DEFAULT_SKULL_LO  = 200   # pixels ≥ this  → skull / bone      (HU ≈ +400)
DEFAULT_OPEN_K    = 3     # morphological opening kernel size  (noise removal)
DEFAULT_CLOSE_K   = 9     # morphological closing kernel size  (fill small holes)


class BrainROIConfig:
    """
    Holds all thresholds used to build the brain-tissue mask.

    Parameters
    ----------
    bg_hi     : upper pixel threshold for background (black region).
                Pixels at or below this are treated as air/background.
    skull_lo  : lower pixel threshold for skull.
                Pixels at or above this are treated as bone.
    open_k    : morphological opening kernel size (removes noise at mask edge).
    close_k   : morphological closing kernel size (fills small brain-tissue gaps).
    min_mask_frac : for patch filtering – minimum fraction of 'keep' pixels in a
                    patch for it to be included in NMF training.
    """
    def __init__(
        self,
        bg_hi:         int   = DEFAULT_BG_HI,
        skull_lo:      int   = DEFAULT_SKULL_LO,
        open_k:        int   = DEFAULT_OPEN_K,
        close_k:       int   = DEFAULT_CLOSE_K,
        min_mask_frac: float = 0.5,
    ):
        self.bg_hi         = int(bg_hi)
        self.skull_lo      = int(skull_lo)
        self.open_k        = int(open_k)
        self.close_k       = int(close_k)
        self.min_mask_frac = float(min_mask_frac)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "BrainROIConfig":
        return cls(**d)


# ---------------------------------------------------------------------------
# Core masking logic
# ---------------------------------------------------------------------------

def build_brain_mask(img_gray: np.ndarray, cfg: BrainROIConfig) -> np.ndarray:
    """
    Build a binary brain-tissue mask for a single grayscale CT slice.

    Steps
    -----
    1. Threshold out background (dark pixels ≤ bg_hi).
    2. Threshold out skull   (bright pixels ≥ skull_lo).
    3. Keep middle-intensity pixels (brain tissue).
    4. Morphological opening  → remove speckling at air/tissue boundary.
    5. Find the largest connected component (the brain itself).
    6. Morphological closing  → fill small holes inside the brain region.

    Parameters
    ----------
    img_gray : H×W uint8 numpy array (0-255, single channel).
    cfg      : BrainROIConfig instance.

    Returns
    -------
    mask : H×W float32 array.  1.0 = brain tissue,  0.0 = background/skull.
    """
    assert img_gray.ndim == 2, "Expected a grayscale 2-D image"
    assert img_gray.dtype == np.uint8, "Expected uint8 (0-255) input"

    # Step 1+2+3 – intensity-based tissue band
    tissue = ((img_gray > cfg.bg_hi) & (img_gray < cfg.skull_lo)).astype(np.uint8)

    # Step 4 – opening removes thin noise / partial-volume voxels at borders
    if cfg.open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_k, cfg.open_k))
        tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, k)

    # Step 5 – largest connected component = brain parenchyma
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    if n_labels <= 1:
        # no tissue found – return empty mask
        return np.zeros(img_gray.shape, dtype=np.float32)

    # label 0 is background; find largest non-background component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    brain = (labels == largest_label).astype(np.uint8)

    # Step 6 – closing fills small holes (ventricles, CSF spaces)
    if cfg.close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k, cfg.close_k))
        brain = cv2.morphologyEx(brain, cv2.MORPH_CLOSE, k)

    return brain.astype(np.float32)


# ---------------------------------------------------------------------------
# Patch-level utility – used during NMF patch extraction
# ---------------------------------------------------------------------------

def patch_is_brain(mask_patch: np.ndarray, min_frac: float = 0.5) -> bool:
    """
    Return True if at least `min_frac` of the patch pixels are brain tissue.
    Use this to filter patches before feeding them to NMF.
    """
    return float(mask_patch.mean()) >= min_frac


# ---------------------------------------------------------------------------
# Diagnostic / visualisation helpers
# ---------------------------------------------------------------------------

def overlay_mask(img_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Return an RGB image with the brain mask overlaid in green.

    Parameters
    ----------
    img_gray : H×W uint8 grayscale image.
    mask     : H×W float32 mask (0 or 1).
    alpha    : blend strength for overlay.

    Returns
    -------
    rgb : H×W×3 uint8.
    """
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    green = np.zeros_like(rgb)
    green[:, :, 1] = 255  # pure green channel
    blend = cv2.addWeighted(rgb, 1.0, green, alpha, 0)
    # apply overlay only where mask == 1
    out = rgb.copy()
    out[mask == 1] = blend[mask == 1]
    return out


def compute_mask_stats(img_gray: np.ndarray, mask: np.ndarray) -> dict:
    """
    Return a dict of diagnostic statistics for a (image, mask) pair.
    Useful for checking that the HU-equivalent thresholds are sensible.
    """
    total   = img_gray.size
    n_brain = int(mask.sum())
    n_bg    = int((img_gray <= DEFAULT_BG_HI).sum())
    n_skull = int((img_gray >= DEFAULT_SKULL_LO).sum())
    brain_px = img_gray[mask == 1]
    return {
        "total_pixels"      : total,
        "brain_pixels"      : n_brain,
        "brain_fraction"    : round(n_brain / total, 4),
        "background_pixels" : n_bg,
        "skull_pixels"      : n_skull,
        "brain_mean_px"     : round(float(brain_px.mean()), 2) if len(brain_px) else None,
        "brain_std_px"      : round(float(brain_px.std()),  2) if len(brain_px) else None,
        "brain_min_px"      : int(brain_px.min())  if len(brain_px) else None,
        "brain_max_px"      : int(brain_px.max())  if len(brain_px) else None,
        # approximate HU equivalents (linear mapping 0-255 → -1024 … 3071 HU)
        "brain_mean_hu_approx"  : round(float(brain_px.mean())  / 255 * (3071 + 1024) - 1024, 1) if len(brain_px) else None,
    }


def pixel_to_hu_approx(pixel: float,
                        hu_min: float = -1024.0,
                        hu_max: float =  3071.0) -> float:
    """
    Approximate mapping from 8-bit JPEG pixel value to HU.
    Only valid if the JPEG was generated with a full-range window.
    Real clinical DICOM exports often use a narrower window (e.g. WW=80),
    so treat this as an estimate only.
    """
    return pixel / 255.0 * (hu_max - hu_min) + hu_min


def hu_to_pixel_approx(hu: float,
                        hu_min: float = -1024.0,
                        hu_max: float =  3071.0) -> float:
    return (hu - hu_min) / (hu_max - hu_min) * 255.0


# ---------------------------------------------------------------------------
# Stand-alone diagnostic runner
# ---------------------------------------------------------------------------

def _run_diagnostics(args):
    """
    Process every JPEG in `--img-dir`, save diagnostic overlays and a JSON
    report so you can visually verify the HU-equivalent thresholding logic.
    """
    import glob, random

    roi_cfg = BrainROIConfig(
        bg_hi         = args.bg_hi,
        skull_lo      = args.skull_lo,
        open_k        = args.open_k,
        close_k       = args.close_k,
        min_mask_frac = args.min_mask_frac,
    )
    print("\n=== BrainROIConfig ===")
    print(json.dumps(roi_cfg.to_dict(), indent=2))

    # Approximate HU equivalents for the chosen thresholds
    bg_hu    = pixel_to_hu_approx(args.bg_hi)
    skull_hu = pixel_to_hu_approx(args.skull_lo)
    print(f"\nHU-equivalent thresholds (approximate):")
    print(f"  bg_hi   = {args.bg_hi:3d} px  →  ~{bg_hu:+.0f} HU  (background / air below this)")
    print(f"  skull_lo= {args.skull_lo:3d} px  →  ~{skull_hu:+.0f} HU  (skull / bone above this)")
    print(f"  Brain tissue band: {args.bg_hi+1} – {args.skull_lo-1} px  (~{bg_hu+1:.0f} to {skull_hu-1:.0f} HU)\n")

    # collect image paths
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for pat in patterns:
        paths += glob.glob(os.path.join(args.img_dir, "**", pat), recursive=True)
    paths = sorted(set(paths))
    print(f"Found {len(paths)} images in '{args.img_dir}'")
    if not paths:
        print("No images found – check --img-dir path.")
        return

    if args.max_samples and len(paths) > args.max_samples:
        random.seed(42)
        paths = random.sample(paths, args.max_samples)
        print(f"Sampled {len(paths)} images for diagnostics.")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    all_stats = []
    for i, p in enumerate(paths):
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            print(f"  [WARN] Could not read: {p}")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # resize to --img-size for consistency
        if args.img_size:
            img_gray = cv2.resize(img_gray, (args.img_size, args.img_size),
                                  interpolation=cv2.INTER_AREA)

        mask = build_brain_mask(img_gray, roi_cfg)
        stats = compute_mask_stats(img_gray, mask)
        stats["path"] = p
        all_stats.append(stats)

        # Save diagnostic strip: original | threshold map | brain overlay
        strip = _make_diagnostic_strip(img_gray, mask, roi_cfg)
        fname = Path(p).stem + f"_diag_{i:04d}.jpg"
        cv2.imwrite(os.path.join(overlay_dir, fname), strip)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(paths)}] brain_frac={stats['brain_fraction']:.3f}  "
                  f"mean_px={stats['brain_mean_px']}  ~{stats['brain_mean_hu_approx']} HU")

    # summary
    fracs  = [s["brain_fraction"] for s in all_stats if s["brain_fraction"] is not None]
    means  = [s["brain_mean_px"]  for s in all_stats if s["brain_mean_px"]  is not None]
    summary = {
        "n_images"            : len(all_stats),
        "brain_fraction_mean" : round(float(np.mean(fracs)),  4) if fracs else None,
        "brain_fraction_std"  : round(float(np.std(fracs)),   4) if fracs else None,
        "brain_fraction_min"  : round(float(np.min(fracs)),   4) if fracs else None,
        "brain_fraction_max"  : round(float(np.max(fracs)),   4) if fracs else None,
        "mean_pixel_mean"     : round(float(np.mean(means)),  2) if means else None,
        "roi_config"          : roi_cfg.to_dict(),
        "hu_equiv_bg_hi"      : round(bg_hu,    1),
        "hu_equiv_skull_lo"   : round(skull_hu, 1),
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(out_dir, "mask_diagnostics.json"), "w") as f:
        json.dump({"summary": summary, "per_image": all_stats}, f, indent=2)
    with open(os.path.join(out_dir, "roi_config.json"), "w") as f:
        json.dump(roi_cfg.to_dict(), f, indent=2)

    print(f"\nDiagnostic overlays saved to  : {overlay_dir}")
    print(f"Full stats saved to           : {os.path.join(out_dir, 'mask_diagnostics.json')}")
    print(f"ROI config saved to           : {os.path.join(out_dir, 'roi_config.json')}")

    # Sanity-check warnings
    print("\n=== Sanity checks ===")
    low_frac  = [s for s in all_stats if (s["brain_fraction"] or 0) < 0.05]
    high_frac = [s for s in all_stats if (s["brain_fraction"] or 0) > 0.60]
    if low_frac:
        print(f"  [WARN] {len(low_frac)} images have brain_fraction < 5% – "
              f"thresholds may be too aggressive or images are non-standard.")
    if high_frac:
        print(f"  [WARN] {len(high_frac)} images have brain_fraction > 60% – "
              f"skull may not be fully excluded. Try lowering --skull-lo.")
    if not low_frac and not high_frac:
        print("  [OK]  Brain fraction looks healthy across all images.")

    mean_hu = summary.get("hu_equiv_bg_hi")
    if mean_hu is not None:
        print(f"  [INFO] bg_hi threshold ≈ {mean_hu:.0f} HU  "
              f"(expected: ~ -900 to -800 HU for air/background)")
    mean_hu2 = summary.get("hu_equiv_skull_lo")
    if mean_hu2 is not None:
        print(f"  [INFO] skull_lo threshold ≈ {mean_hu2:.0f} HU  "
              f"(expected: +300 to +700 HU for cortical bone)")


def _make_diagnostic_strip(img_gray: np.ndarray, mask: np.ndarray,
                            cfg: BrainROIConfig) -> np.ndarray:
    """
    Build a side-by-side diagnostic image:
      [original] | [intensity-band raw] | [final brain mask] | [overlay]
    """
    H, W = img_gray.shape

    # Panel 1 – original
    p1 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Panel 2 – raw intensity band (background=blue, tissue=white, skull=red)
    p2 = np.zeros((H, W, 3), dtype=np.uint8)
    bg_mask    = img_gray <= cfg.bg_hi
    skull_mask = img_gray >= cfg.skull_lo
    tissue_mask = (~bg_mask) & (~skull_mask)
    p2[bg_mask]    = (200, 50,  50)   # blue  – background
    p2[skull_mask] = (50,  50, 200)   # red   – skull
    p2[tissue_mask]= (220, 220, 220)  # white – tissue

    # Panel 3 – final binary mask
    p3 = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Panel 4 – overlay
    p4 = overlay_mask(img_gray, mask)

    # add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, label in [(p1, "Original"), (p2, "HU-equiv bands"), (p3, "Brain mask"), (p4, "Overlay")]:
        cv2.putText(panel, label, (4, 16), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return np.concatenate([p1, p2, p3, p4], axis=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    ap = argparse.ArgumentParser(
        description="Diagnose and validate HU-equivalent brain ROI masking on JPEG CT images."
    )
    ap.add_argument("--img-dir",   required=True,
                    help="Directory containing JPEG CT slices (scanned recursively).")
    ap.add_argument("--out-dir",   default="roi_diag",
                    help="Output directory for overlays and stats JSON.")
    ap.add_argument("--img-size",  type=int, default=128,
                    help="Resize images to this square size before masking (0=no resize).")
    ap.add_argument("--max-samples", type=int, default=50,
                    help="Maximum number of images to process (0=all).")

    # threshold controls
    ap.add_argument("--bg-hi",         type=int,   default=DEFAULT_BG_HI,
                    help=f"Pixel threshold for background/air (default={DEFAULT_BG_HI}). "
                         f"Pixels ≤ this are excluded.")
    ap.add_argument("--skull-lo",      type=int,   default=DEFAULT_SKULL_LO,
                    help=f"Pixel threshold for skull (default={DEFAULT_SKULL_LO}). "
                         f"Pixels ≥ this are excluded.")
    ap.add_argument("--open-k",        type=int,   default=DEFAULT_OPEN_K,
                    help="Morphological opening kernel size.")
    ap.add_argument("--close-k",       type=int,   default=DEFAULT_CLOSE_K,
                    help="Morphological closing kernel size.")
    ap.add_argument("--min-mask-frac", type=float, default=0.5,
                    help="Min brain-tissue fraction for a patch to be kept.")
    return ap


if __name__ == "__main__":
    _run_diagnostics(_build_parser().parse_args())