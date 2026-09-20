"""Single entry point for loading REAL TikTok shortlist data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REAL_DIR = ROOT / "data" / "drive-download-20260920T011427Z-1-001"
REAL_PATH = REAL_DIR / "top_500_audios.parquet"

# Map standard names -> real column names (None = synthesize / missing)
COLUMN_MAP = {
    "video_id": None,          # sequential id assigned on load
    "music_id": "music_id",
    "post_date": "create_time",  # unix seconds
    "caption": None,             # not in dump
    "total_views": None,         # NOT in dump — see warning below
}

# Set manually if you know the collection date. None => max(post_date).
SNAPSHOT_DATE: pd.Timestamp | None = None

# Seeded synthetic totals only used when total_views is missing from the dump.
SYNTHETIC_VIEWS_SEED = 42

_CACHE: dict | None = None


def _music_id_str(series: pd.Series) -> pd.Series:
    return series.round().astype("int64").astype(str)


def inspect_real_table(path: Path = REAL_PATH) -> pd.DataFrame:
    """Print schema summary and return the raw frame."""
    raw = pd.read_parquet(path)
    print("=== REAL DATA INSPECTION ===")
    print(f"path: {path}")
    print(f"rows: {len(raw):,}")
    print(f"columns: {list(raw.columns)}")
    print("dtypes:")
    print(raw.dtypes.to_string())
    print("null counts:")
    print(raw.isna().sum().to_string())
    print("head:")
    print(raw.head(3).to_string())
    if "create_time" in raw.columns:
        ts = pd.to_datetime(raw["create_time"], unit="s")
        print(f"create_time range: {ts.min()} -> {ts.max()}")
    if "music_id" in raw.columns:
        print(
            f"music_ids: {raw['music_id'].nunique():,}  "
            f"videos: {len(raw):,}  "
            f"usage min/median/max: "
            f"{raw['music_id'].value_counts().min()}/"
            f"{int(raw['music_id'].value_counts().median())}/"
            f"{raw['music_id'].value_counts().max()}"
        )
    missing = [k for k, v in COLUMN_MAP.items() if v is None]
    print(f"COLUMN_MAP missing real sources for: {missing}")
    if COLUMN_MAP["total_views"] is None:
        print(
            "WARNING: dump has no plays/views/likes. "
            "Synthesizing total_views so assumed-decay expansion can run. "
            "Replace COLUMN_MAP['total_views'] when real counts are available."
        )
    print("=== END INSPECTION ===")
    return raw


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    global SNAPSHOT_DATE

    post_col = COLUMN_MAP["post_date"]
    music_col = COLUMN_MAP["music_id"]

    out = pd.DataFrame()
    out["music_id"] = _music_id_str(raw[music_col])
    out["post_date"] = (
        pd.to_datetime(raw[post_col], unit="s", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    out["video_id"] = np.arange(1, len(out) + 1, dtype=np.int64)

    if COLUMN_MAP["caption"] and COLUMN_MAP["caption"] in raw.columns:
        out["caption"] = raw[COLUMN_MAP["caption"]].astype(str)
    else:
        out["caption"] = ""

    # Extra engagement columns if present
    for extra in ("play_count", "digg_count", "share_count", "collect_count", "likes", "shares"):
        if extra in raw.columns and extra not in out.columns:
            out[extra] = raw[extra]

    views_col = COLUMN_MAP["total_views"]
    if views_col and views_col in raw.columns:
        out["total_views"] = pd.to_numeric(raw[views_col], errors="coerce").fillna(0).astype(np.int64)
    elif "play_count" in out.columns:
        out["total_views"] = out["play_count"].fillna(0).astype(np.int64)
    else:
        rng = np.random.default_rng(SYNTHETIC_VIEWS_SEED)
        # Age-aware lognormal placeholder (older posts tend to have more views)
        age_days = (out["post_date"].max() - out["post_date"]).dt.days.clip(lower=0).to_numpy()
        log_mean = 7.5 + 0.002 * np.minimum(age_days, 365)
        out["total_views"] = np.maximum(1, rng.lognormal(log_mean, 0.8).astype(np.int64))

    if SNAPSHOT_DATE is None:
        SNAPSHOT_DATE = pd.Timestamp(out["post_date"].max())
        print(
            f"WARNING: SNAPSHOT_DATE unset - defaulting to max(post_date) = {SNAPSHOT_DATE.date()}. "
            "Set loader.SNAPSHOT_DATE manually if the collection date differs."
        )
    else:
        SNAPSHOT_DATE = pd.Timestamp(SNAPSHOT_DATE)

    print(
        f"Loaded {len(out):,} videos across {out['music_id'].nunique():,} music_ids "
        f"(window {out['post_date'].min().date()} -> {SNAPSHOT_DATE.date()})"
    )
    return out


def load_videos(*, inspect: bool = False) -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None and not inspect:
        return _CACHE["videos"].copy()

    raw = inspect_real_table() if inspect else pd.read_parquet(REAL_PATH)
    videos = _normalize(raw)
    _CACHE = {"videos": videos}
    return videos.copy()


def get_snapshot_date() -> pd.Timestamp:
    if SNAPSHOT_DATE is None:
        load_videos()
    assert SNAPSHOT_DATE is not None
    return pd.Timestamp(SNAPSHOT_DATE)


def get_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    videos = load_videos()
    start = pd.Timestamp(videos["post_date"].min())
    end = get_snapshot_date()
    return start, end


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / name)


def processed_path(name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    return PROCESSED / name


if __name__ == "__main__":
    load_videos(inspect=True)
