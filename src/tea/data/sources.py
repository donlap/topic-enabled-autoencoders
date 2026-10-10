"""Raw files of the six datasets, the URLs of these files, and their SHA-256 digests."""

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass

DATASETS = ("books", "cds", "gowalla", "ml-1m", "retail", "steam")

_AMAZON = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass(frozen=True)
class Source:
    """A raw file, the URL from which the file is downloaded, and the SHA-256 digest of the file."""

    filename: str
    url: str
    sha256: str


INTERACTIONS = {
    "books": Source("Books.csv.gz", _AMAZON + "benchmark/5core/rating_only/Books.csv.gz",
                    "28c5104d0c2f7a7e842da3431606e21a245d8fb317e3434cd20b5c5a7b521615"),
    "cds": Source("CDs_and_Vinyl.csv.gz", _AMAZON + "benchmark/5core/rating_only/CDs_and_Vinyl.csv.gz",
                  "53811902ffb01eb22da31a35e0775f3b8351baebe7e1ab38b85b6c4aee689c20"),
    "gowalla": Source("loc-gowalla_totalCheckins.txt.gz",
                      "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
                      "c1c3e19effba649b6c89aeab3c1f9459fad88cfdc2b460fc70fd54e295d83ea0"),
    "ml-1m": Source("ml-1m.zip", "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                    "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"),
    "retail": Source("online_retail_ii.zip", "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
                     "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"),
    "steam": Source("steam_reviews.json.gz", "https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
                    "40c6343deedf488689987749c5f8435233da48c163fbaa2a7766c215d584bc61"),
}


def sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source, folder, timeout=900):
    """Download a source into a folder unless the file is already there, check its digest, and return its path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, source.filename)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        request = urllib.request.Request(source.url, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response, open(path + ".tmp", "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(path + ".tmp", path)
    if sha256(path) != source.sha256:
        raise ValueError(f"{path} does not have the SHA-256 digest {source.sha256}")
    return path
