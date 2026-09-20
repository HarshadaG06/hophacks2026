"""Shared audio lifespan bounds — used by s4_lifespan and s5_burstiness."""
from __future__ import annotations

import pandas as pd

MIN_VIDEOS = 50
MIN_LIFESPAN_DAYS = 14
N_BINS = 20
MIN_VIDEO_AGE_DAYS = 7
LIFESPAN_MODE = "last_post"  # last_post | trimmed | snapshot
TOP_VIDEO_QUANTILE = 0.90
LIFESPAN_MODES = ("last_post", "trimmed", "snapshot")
DAYS_PER_WEEK = 7
BURST_UNIT = "week"  # Kleinberg event offsets and viz time axis


def audio_bounds(
    posts: pd.Series, mode: str, snapshot: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Return (audio_start, audio_end, lifespan_days) for one music_id."""
    if mode == "trimmed":
        start = pd.Timestamp(posts.quantile(0.01))
        end = pd.Timestamp(posts.quantile(0.99))
    elif mode == "snapshot":
        start = pd.Timestamp(posts.min())
        end = pd.Timestamp(snapshot)
    else:
        start = pd.Timestamp(posts.min())
        end = pd.Timestamp(posts.max())
    lifespan = int((end - start).days)
    return start, end, lifespan


def day_to_u(day_index: float, lifespan_days: int) -> float:
    return float(max(0.0, min(1.0, day_index / max(lifespan_days, 1))))


def lifespan_weeks(lifespan_days: int) -> int:
    return max(1, (int(lifespan_days) + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK)


def week_offset_to_u(week_offset: float, lifespan_days: int) -> float:
    """Map a week offset (from audio_start) to normalized lifespan u."""
    return float(
        max(0.0, min(1.0, week_offset * DAYS_PER_WEEK / max(lifespan_days, 1)))
    )
