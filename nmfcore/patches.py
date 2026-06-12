import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def patchify(img2d: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """
    img2d: (H,W)
    returns patches: (n_patches, patch*patch)
    Uses sliding_window_view for speed.
    """
    H, W = img2d.shape
    if patch > H or patch > W:
        raise ValueError("patch larger than image")
    if stride <= 0:
        raise ValueError("stride must be >=1")

    windows = sliding_window_view(img2d, (patch, patch))  # (H-p+1, W-p+1, p, p)
    windows = windows[::stride, ::stride, :, :]          # stride sampling
    patches = windows.reshape(-1, patch * patch).astype(np.float32, copy=False)
    return patches


def unpatchify(patches: np.ndarray, out_hw: tuple, patch: int, stride: int) -> np.ndarray:
    """
    Reconstruct image via overlap-add with averaging.
    patches: (n_patches, patch*patch)
    """
    H, W = out_hw
    out = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)

    # compute grid
    nH = (H - patch) // stride + 1
    nW = (W - patch) // stride + 1
    if patches.shape[0] != nH * nW:
        raise ValueError(f"patch count mismatch: got {patches.shape[0]} expected {nH*nW}")

    idx = 0
    for i in range(nH):
        y0 = i * stride
        for j in range(nW):
            x0 = j * stride
            p = patches[idx].reshape(patch, patch)
            out[y0:y0+patch, x0:x0+patch] += p
            wgt[y0:y0+patch, x0:x0+patch] += 1.0
            idx += 1

    out = out / np.maximum(wgt, 1e-8)
    return out
