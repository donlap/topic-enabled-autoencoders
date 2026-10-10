"""Build the split of a dataset from its raw file, and store the split."""

import os

import numpy as np

from tea.data import loaders, prepare, sources, split


def interactions_path(data_dir, dataset):
    """Return the path of the interactions CSV file of a dataset."""
    return os.path.join(data_dir, "interactions", f"{dataset}.csv.gz")


def split_path(data_dir, dataset):
    """Return the path of the stored split of a dataset."""
    return os.path.join(data_dir, "splits", f"{dataset}.npz")


def build(dataset, data_dir):
    """Download the raw file if needed, build the split, and store the split.

    The interactions are always written to a CSV file and read back before filtering,
    because the parsed types of the item ids determine the order of the items.
    """
    raw = sources.fetch(sources.INTERACTIONS[dataset], os.path.join(data_dir, "raw", dataset))
    csv = interactions_path(data_dir, dataset)
    if not os.path.exists(csv):
        os.makedirs(os.path.dirname(csv), exist_ok=True)
        prepare.write_interactions(loaders.LOADERS[dataset](raw), csv)
    prepared = prepare.prepare(prepare.read_interactions(csv))
    parts = split.temporal_split(prepared)
    out = split_path(data_dir, dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, vocab=np.array([str(v) for v in prepared["vocab"]]), **parts)
    return load(dataset, data_dir)


def load(dataset, data_dir):
    """Load the stored split of a dataset as a dictionary of arrays."""
    with np.load(split_path(data_dir, dataset)) as z:
        return {k: z[k] for k in z.files}
