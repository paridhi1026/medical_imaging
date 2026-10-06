import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

from .config import Config
from .patches import patchify, unpatchify


class PatchNMFBundle:
    """
    Patch-based NMF:
      - Train NMF on patches extracted from normal images.
      - Reconstruct full images by patch reconstruction + overlap-add.
    """

    def __init__(self, cfg: Config, nmf: NMF, scaler=None, patch: int = 16, stride: int = 8):
        self.cfg = cfg
        self.nmf = nmf
        self.scaler = scaler  # kept for backward compatibility if unpickling
        self.patch = int(patch)
        self.stride = int(stride)

    def _all_patches(self, X: np.ndarray, max_patches: int = 0, seed: int = 42) -> np.ndarray:
        """
        X: (N, S*S) flattened images
        returns P: (M, patch*patch) patches across all images
        """
        s = self.cfg.img_size
        patches_list = []
        for i in range(X.shape[0]):
            img = X[i].reshape(s, s)
            patches_list.append(patchify(img, self.patch, self.stride))
        P = np.concatenate(patches_list, axis=0) if patches_list else np.zeros((0, self.patch*self.patch), np.float32)

        if max_patches and P.shape[0] > max_patches:
            rng = np.random.default_rng(seed)
            idx = rng.choice(P.shape[0], size=int(max_patches), replace=False)
            P = P[idx]
        return P

    def fit_on_patches(self, X_train: np.ndarray, max_patches: int = 0, seed: int = 42) -> None:
        """
        Fit NMF directly on non-negative patches without feature-wise MinMax scaling distortion.
        """
        P = self._all_patches(X_train, max_patches=max_patches, seed=seed)
        if P.shape[0] == 0:
            raise RuntimeError("No patches to train on.")

        Ps = np.clip(P, 0.0, 1.0)
        self.nmf.fit(Ps)

    def reconstruct_images(self, X: np.ndarray, batch: int = 16) -> np.ndarray:
        """
        Reconstruct images by patch decomposition + overlap-add.
        Returns Xrec flattened: (N, S*S)
        """
        s = self.cfg.img_size
        out = np.zeros_like(X, dtype=np.float32)

        for i in range(X.shape[0]):
            img = X[i].reshape(s, s)
            P = patchify(img, self.patch, self.stride)               # (m, p*p)
            Ps = np.clip(P, 0.0, 1.0)

            W = self.nmf.transform(Ps)
            Prec = W @ self.nmf.components_
            Prec = np.clip(Prec, 0.0, 1.0)

            rec2d = unpatchify(Prec, (s, s), self.patch, self.stride)
            out[i] = rec2d.reshape(-1)

        return out

