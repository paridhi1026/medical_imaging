import os
from typing import Dict, List, Tuple


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def train_test_paths(base: str) -> Tuple[str, str]:
    train_root = os.path.join(base, "Training")
    test_root = os.path.join(base, "Testing")
    return train_root, test_root


def list_class_images(root: str) -> Dict[str, List[str]]:
    """
    Returns {class_name: [filepaths...]} for all subfolders in root.
    """
    out: Dict[str, List[str]] = {}
    if not os.path.isdir(root):
        return out

    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = []
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith(IMG_EXTS):
                files.append(os.path.join(cls_dir, fn))
        out[cls] = files
    return out


def make_binary_labels(classes: Dict[str, List[str]], normal_class: str):
    """
    Returns (all_paths, y_binary) where y=0 for normal_class else 1.
    Note: build/load may skip unreadable files; caller must realign with kept.
    """
    all_paths = []
    y = []
    for cls, paths in classes.items():
        for p in paths:
            all_paths.append(p)
            y.append(0 if cls == normal_class else 1)
    return all_paths, y
