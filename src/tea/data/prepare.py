"""Filter the interactions of a dataset and index its users and items.

The filters are applied in a fixed order, because the order changes the result.
First, the items are restricted to the 5000 items with the most interactions.
Next, the users and items are restricted to a 5-core.
If more than 5000 users remain, a fixed random sample of 5000 users is kept, followed by another 5-core.
Finally, only the users with at least three distinct timestamps are kept.
"""

import os

import numpy as np
import pandas as pd

KCORE = 5
MAX_USERS = 5000
MAX_ITEMS = 5000
MIN_TIMESTAMPS = 3
SAMPLE_SEED = 12345


def write_interactions(df, path):
    """Drop missing and non-positive rows, and write the interactions to a CSV file.

    The file is written under a temporary name in the same folder and then renamed, so a stopped run leaves no partial file.
    """
    df = df.dropna()
    df = df[df["count"] > 0]
    folder, name = os.path.split(path)
    temporary = os.path.join(folder, "partial-" + name)
    df.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_interactions(path):
    """Read the interactions from the CSV file that write_interactions created.

    The values "null", "nan", and "none" are kept as strings, because they can be item ids.
    """
    return pd.read_csv(path, keep_default_na=False)


def kcore(df, k):
    """Keep the users and items that have at least k distinct interactions, until no user or item is removed."""
    if k <= 1:
        return df
    while True:
        n0 = len(df)
        uc = df.groupby("user")["item"].transform("nunique")
        df = df[uc >= k]
        ic = df.groupby("item")["user"].transform("nunique")
        df = df[ic >= k]
        if len(df) == n0 or df.empty:
            return df


def prepare(df, k=KCORE, max_users=MAX_USERS, max_items=MAX_ITEMS,
            min_timestamps=MIN_TIMESTAMPS, sample_seed=SAMPLE_SEED):
    """Filter the interactions and index the users and items in sorted order.

    Returns the interactions as an array with the columns user index, item index, ts, and count,
    sorted by user and then by time, together with the numbers of users and items and the item ids.
    """
    if max_items and df["item"].nunique() > max_items:
        top = df["item"].value_counts().nlargest(max_items).index
        df = df[df["item"].isin(top)]
    df = kcore(df, k)
    users = df["user"].unique()
    if max_users and len(users) > max_users:
        rng = np.random.RandomState(sample_seed)
        keep = rng.choice(users, size=max_users, replace=False)
        df = df[df["user"].isin(keep)]
        df = kcore(df, k)
    seq = df.groupby("user")["ts"].transform("nunique")
    df = df[seq >= min_timestamps]
    if df.empty:
        raise RuntimeError("no interactions are left after filtering")
    uids = {u: i for i, u in enumerate(sorted(df["user"].unique()))}
    iids = {it: j for j, it in enumerate(sorted(df["item"].unique()))}
    df = df.assign(u=df["user"].map(uids), i=df["item"].map(iids))
    df = df.sort_values(["u", "ts"]).reset_index(drop=True)
    vocab = [None] * len(iids)
    for it, j in iids.items():
        vocab[j] = it
    return dict(interactions=df[["u", "i", "ts", "count"]].to_numpy(),
                n_users=len(uids), n_items=len(iids), vocab=np.array(vocab, dtype=object))
