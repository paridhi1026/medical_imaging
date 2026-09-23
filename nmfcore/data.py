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


def extract_patient_id(path: str) -> str:
    """
    Extract Subject/Patient ID from filename.
    Matches patterns like '00000_10000' in '00000_10000.png_slice-0000.png'.
    """
    import re
    fn = os.path.basename(path)
    m = re.search(r"(\d+_\d+)", fn)
    if m:
        return m.group(1)
    return fn.split(".")[0].split("_")[0]


def patient_group_split(paths: List[str], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Group filepaths by Patient ID and split Patient IDs into Train, Val, and Holdout sets.
    Guarantees zero data leakage across splits.
    """
    import random
    from collections import defaultdict

    patient_to_paths = defaultdict(list)
    for p in paths:
        pid = extract_patient_id(p)
        patient_to_paths[pid].append(p)

    unique_patients = sorted(list(patient_to_paths.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))

    train_pids = set(unique_patients[:n_train])
    val_pids = set(unique_patients[n_train:n_train + n_val])
    hold_pids = set(unique_patients[n_train + n_val:])

    train_paths = [p for pid in train_pids for p in patient_to_paths[pid]]
    val_paths = [p for pid in val_pids for p in patient_to_paths[pid]]
    hold_paths = [p for pid in hold_pids for p in patient_to_paths[pid]]

    return train_paths, val_paths, hold_paths


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

