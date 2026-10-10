"""Parsers that convert each raw file into a table of interactions.

Each parser returns a data frame with the columns user, item, ts, and count.
The column ts is a Unix time in seconds. The column count is the weight of the interaction.
"""

import ast
import gzip
import io
import json
import zipfile

import numpy as np
import pandas as pd


def amazon(path):
    """Read an Amazon 2023 rating file, keeping every rating as the count."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f)
    return pd.DataFrame({"user": df["user_id"].astype(str),
                         "item": df["parent_asin"].astype(str),
                         "ts": (df["timestamp"].astype(np.int64) // 1000),
                         "count": df["rating"].astype(float).to_numpy()})


def movielens(path):
    """Read the ratings of MovieLens 1M, keeping every rating as the count."""
    with zipfile.ZipFile(path) as z:
        with z.open("ml-1m/ratings.dat") as f:
            data = f.read()
    df = pd.read_csv(io.BytesIO(data), sep="::", engine="python", header=None,
                     names=["user", "item", "rating", "ts"])
    return pd.DataFrame({"user": df["user"].astype(np.int64),
                         "item": df["item"].astype(np.int64),
                         "ts": df["ts"].astype(np.int64),
                         "count": df["rating"].astype(float).to_numpy()})


def retail_sheets(path):
    """Read both sheets of the Online Retail II workbook into one data frame."""
    with zipfile.ZipFile(path) as z:
        xlsx = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
        with z.open(xlsx) as f:
            buf = io.BytesIO(f.read())
    df = pd.concat(pd.read_excel(buf, sheet_name=None, engine="openpyxl").values(), ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def retail(path):
    """Read the invoices of Online Retail II, without cancellations and returns."""
    df = retail_sheets(path)
    cust = "Customer ID" if "Customer ID" in df.columns else "CustomerID"
    df = df.dropna(subset=[cust])
    df = df[df["Quantity"] > 0]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    ts = pd.to_datetime(df["InvoiceDate"]).astype("int64") // 10**9
    return pd.DataFrame({"user": df[cust].astype(np.int64),
                         "item": df["StockCode"].astype(str),
                         "ts": ts.astype(np.int64),
                         "count": df["Quantity"].astype(float).clip(lower=1.0)})


def gowalla(path):
    """Read the Gowalla check-ins, with a location as the item."""
    df = pd.read_csv(path, sep="\t", header=None, compression="gzip",
                     names=["user", "ts_iso", "lat", "lon", "item"],
                     usecols=["user", "ts_iso", "item"])
    dt = pd.to_datetime(df["ts_iso"], errors="coerce", utc=True, format="ISO8601")
    ts = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    out = pd.DataFrame({"user": df["user"].astype(str), "item": df["item"].astype(str),
                        "ts": ts, "count": 1.0}).dropna(subset=["ts"])
    out["ts"] = out["ts"].astype(np.int64)
    return out[out["ts"] > 0]


def mcauley_records(path):
    """Yield the records of a gzipped file of JSON or Python dictionaries, one per line."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                try:
                    d = ast.literal_eval(line)
                except Exception:
                    continue
            if isinstance(d, dict):
                yield d


def steam(path):
    """Read the Steam reviews, with a game as the item and the review date as the time."""
    rows = []
    for d in mcauley_records(path):
        u = d.get("username") or d.get("user_id")
        it = d.get("product_id") or d.get("item_id") or d.get("id")
        t = d.get("date") or d.get("timestamp") or d.get("time")
        if u is not None and it is not None and t is not None:
            rows.append((str(u), str(it), t))
    df = pd.DataFrame(rows, columns=["user", "item", "ts"])
    t = df["ts"]
    if pd.api.types.is_numeric_dtype(t):
        t = t.astype(np.int64)
        if int(t.max()) > 1_000_000_000_000:
            t = t // 1000
    else:
        dt = pd.to_datetime(t, errors="coerce", utc=True)
        t = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    out = pd.DataFrame({"user": df["user"].astype(str), "item": df["item"].astype(str),
                        "ts": t, "count": 1.0}).dropna(subset=["ts"])
    out["ts"] = out["ts"].astype(np.int64)
    return out[out["ts"] > 0]


LOADERS = {"books": amazon, "cds": amazon, "gowalla": gowalla, "ml-1m": movielens,
           "retail": retail, "steam": steam}
