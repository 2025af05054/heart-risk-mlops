"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heart.config import CLEAN_PATH, COLUMNS, RAW_PATH, TARGET, VALID_CATEGORIES

log = logging.getLogger(__name__)

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# Fallback: a widely used GitHub copy of the same 303 Cleveland records,
# re-coded (0-based categories, inverted target). We convert it back to the
# original UCI coding so both sources produce an identical schema.
MIRROR_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv"
)

_MIRROR_RECODE = {
    "cp": {0: 4, 1: 2, 2: 3, 3: 1},
    "restecg": {0: 2, 1: 0, 2: 1},
    "slope": {0: 3, 1: 2, 2: 1},
    "thal": {0: np.nan, 1: 6, 2: 3, 3: 7},
}


def _fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def mirror_to_uci(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the GitHub mirror's coding back to the original UCI coding."""
    out = df.copy()
    for col, mapping in _MIRROR_RECODE.items():
        out[col] = out[col].map(mapping)
    out["ca"] = out["ca"].where(out["ca"] <= 3, np.nan)  # 4 = missing in mirror
    out["num"] = 1 - out.pop("target")  # mirror: 1 = healthy
    return out[COLUMNS]


def download(dest: Path = RAW_PATH, force: bool = False) -> Path:
    """Download the raw Cleveland file (UCI first, mirror as fallback)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("Raw data already present at %s", dest)
        return dest
    try:
        dest.write_bytes(_fetch(UCI_URL))
        log.info("Downloaded from UCI repository")
    except Exception as exc:  # network blocked, UCI down, etc.
        log.warning("UCI download failed (%s); using GitHub mirror", exc)
        tmp = dest.with_suffix(".mirror.csv")
        tmp.write_bytes(_fetch(MIRROR_URL))
        df = mirror_to_uci(pd.read_csv(tmp, encoding="utf-8-sig"))
        df.to_csv(dest, header=False, index=False, na_rep="?")
        tmp.unlink()
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the header-less UCI file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame.

    * binarise the 0-4 diagnosis into target (0 = absent, 1 = present)
    * coerce types, drop exact duplicates
    * mark out-of-range categorical codes and physiologically impossible
      values (chol or trestbps == 0) as missing
    Missing values are kept as NaN and imputed inside the model pipeline,
    so the same logic runs at training and serving time.
    """
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df[TARGET] = (df.pop("num") > 0).astype(int)
    for col, valid in VALID_CATEGORIES.items():
        df.loc[~df[col].isin(valid), col] = np.nan
    for col in ("chol", "trestbps"):
        df.loc[df[col] <= 0, col] = np.nan
    df.loc[~df["ca"].isin([0, 1, 2, 3]), "ca"] = np.nan
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = load_raw(download())
    df = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    log.info("Saved %d rows to %s; missing per column: %s",
             len(df), CLEAN_PATH, df.isna().sum()[lambda s: s > 0].to_dict())


if __name__ == "__main__":
    main()
