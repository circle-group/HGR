

from .evaluator import get_all_metrics
from .dataio.download import ensure_dataset, read_test_idx
from .dataio.data_splits import load_train_test_smiles, load_full_smiles


__all__ = [
    "get_all_metrics",
    "ensure_dataset",
    "read_test_idx",
    "load_train_test_smiles",
    "load_full_smiles",
]

# Public API (stable-ish)
#
# - get_all_metrics(gen, test_smiles=None, train_smiles=None, metrics=None, ...):
#     Main evaluation entrypoint. Computes validity/uniqueness/novelty/FCD/NSPDK/curvature/etc.
# - ensure_dataset(dataset_name, ...):
#     Ensure RingDiv/RingDiv300k raw files exist (download if enabled) and return paths.
# - read_test_idx(test_idx_npz_path) -> (test_idx, meta):
#     Read `*_test_idx.npz` and return indices plus metadata (n_total/seed/test_ratio/...).
# - load_train_test_smiles(dataset_name, ...) -> (train_smiles, test_smiles):
#     Load SMILES from `*_property.csv` and split by `*_test_idx.npz`.
# - load_full_smiles(dataset_name, ...) -> full_smiles:
#     Load all SMILES (in the csv row order).