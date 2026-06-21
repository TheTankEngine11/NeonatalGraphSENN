"""Small data helpers shared by the thesis scripts."""

import os
import random
from typing import Optional

import numpy as np
import scipy.sparse
import torch


def set_seed(seed: int = 42) -> torch.Generator:
    """Set the common random seeds and return a torch Generator."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return generator


def set_random_seed(seed_or_config=42) -> torch.Generator:
    """Compatibility helper for the old training script seed dictionary."""
    if isinstance(seed_or_config, dict):
        seed_data = int(seed_or_config.get("randseeddata", 42))
        seed_other = int(seed_or_config.get("randseedother", seed_data))
        torch.manual_seed(seed_other)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_other)
            torch.cuda.manual_seed_all(seed_other)
        np.random.seed(seed_data)
        random.seed(seed_data)
        generator = torch.Generator()
        generator.manual_seed(seed_other)
        torch.backends.cudnn.deterministic = True
        return generator
    return set_seed(int(seed_or_config))


def prepare_graphs_labels(features, labels, adj_matrix, masks=None):
    """Prepare a list of PyG Data objects with x, edge_index, y, and optional y_mask."""
    from torch_geometric.data import Data

    pyg_data_list = []
    adj_sparse = scipy.sparse.coo_matrix(adj_matrix)
    edge_index = torch.tensor(np.array([adj_sparse.row, adj_sparse.col]), dtype=torch.long)

    for i in range(features.shape[0]):
        data = Data(
            x=torch.tensor(features[i], dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(labels[i], dtype=torch.float32),
        )
        if masks is not None:
            y_mask = np.asarray(masks[i])
            if y_mask.ndim > 1:
                y_mask = np.squeeze(y_mask)
            data.y_mask = torch.tensor(y_mask, dtype=torch.float32)
        pyg_data_list.append(data)
    return pyg_data_list


def load_fold_arrays(data_folder, fold, split: str = "test", mmap_mode: Optional[str] = "r"):
    """Load `{split}data.npy` and `{split}labels.npy` from a CV fold folder."""
    fold_dir = os.path.join(str(data_folder), f"fold_{fold}")
    x_path = os.path.join(fold_dir, f"{split}data.npy")
    y_path = os.path.join(fold_dir, f"{split}labels.npy")
    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"Missing data file: {x_path}")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"Missing label file: {y_path}")
    return np.load(x_path, mmap_mode=mmap_mode), np.load(y_path, mmap_mode=mmap_mode)


def make_loader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False):
    """Create the PyG DataLoader pattern used throughout the scripts."""
    from torch_geometric.loader import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )


def thin_overlapping_windows(
    x,
    y,
    fs: int = 32,
    t_overlap_non: int = 10,
    t_overlap_seiz: int = 11,
    return_indices: bool = False,
):
    """Thin overlapping EEG windows using the thesis validation convention."""
    idx_seiz = np.where(y == 1)[0]
    idx_non = np.where(y == 0)[0]

    t_window = x.shape[-1] / fs
    skip_non = int(t_window / (t_window - t_overlap_non))
    skip_seiz = int(t_window / (t_window - t_overlap_seiz))
    skip_non = max(skip_non, 1)
    skip_seiz = max(skip_seiz, 1)

    keep_idx = np.sort(np.concatenate([idx_non[0::skip_non], idx_seiz[0::skip_seiz]]))
    if return_indices:
        return x[keep_idx], y[keep_idx], keep_idx
    return x[keep_idx], y[keep_idx]
