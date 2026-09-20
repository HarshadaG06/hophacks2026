"""
Estimate daily watch-date views by spreading each video's SNAPSHOT total_views
with an ASSUMED power-law decay kernel — not a fitted model.

Per-video curve fitting is impossible with one data point per video.

TODO: swap in a Hawkes / SEISMIC-style self-exciting point process later.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from loader import get_snapshot_date, load_videos, processed_path

# Tunables
B_VALUES = [0.8, 1.2, 2.0]
DEFAULT_B = 1.2
C = 1.0
MAX_AGE = 90


def _expand_one_b(
    post_dates: np.ndarray,
    totals: np.ndarray,
    music_ids: np.ndarray,
    snapshot: np.datetime64,
    B: float,
) -> pd.DataFrame:
    """Vectorized expansion grouped by lifetime T."""
    T = ((snapshot - post_dates) / np.timedelta64(1, "D")).astype(np.int32)
    ok = (T >= 0) & (totals > 0)
    post_dates = post_dates[ok]
    totals = totals[ok].astype(np.float64)
    music_ids = music_ids[ok]
    T = T[ok]
    horizons = np.minimum(T, MAX_AGE)

    # Accumulate (music_id, day_offset_from_window_start) — use dict of arrays then concat
    # Faster: for each unique horizon, broadcast kernel
    parts = []
    for h in np.unique(horizons):
        idx = np.flatnonzero(horizons == h)
        t = np.arange(int(h) + 1, dtype=np.float64)
        k = np.power(t + C, -B)
        k = k / k.sum()
        # shape (n_videos, h+1)
        daily = totals[idx, None] * k[None, :]
        dates = post_dates[idx, None] + t[None, :].astype("timedelta64[D]")
        mids = np.repeat(music_ids[idx], t.size)
        parts.append(
            pd.DataFrame(
                {
                    "music_id": mids,
                    "date": dates.ravel(),
                    "views": daily.ravel(),
                }
            )
        )

    if not parts:
        return pd.DataFrame(columns=["music_id", "date", "views"])

    long = pd.concat(parts, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    # Clip to snapshot
    long = long[long["date"] <= pd.Timestamp(snapshot)]
    agg = long.groupby(["music_id", "date"], as_index=False)["views"].sum()
    return agg


def main() -> None:
    t0 = time.perf_counter()
    videos = load_videos()
    snapshot = np.datetime64(get_snapshot_date().normalize())

    n = len(videos)
    zero_share = float((videos["total_views"] <= 0).mean())
    age = (
        (pd.Timestamp(snapshot) - pd.to_datetime(videos["post_date"])).dt.days
    )
    young_share = float((age < MAX_AGE).mean())
    print(f"share total_views==0: {zero_share:.2%}")
    print(
        f"share videos younger than MAX_AGE={MAX_AGE}d (right-censored under kernel): "
        f"{young_share:.2%}"
    )

    post_dates = videos["post_date"].to_numpy(dtype="datetime64[ns]")
    totals = videos["total_views"].to_numpy()
    music_ids = videos["music_id"].to_numpy()

    for B in B_VALUES:
        t_b = time.perf_counter()
        # ASSUMED kernel k(t)=(t+C)^(-B), not fitted from data.
        daily = _expand_one_b(post_dates, totals, music_ids, snapshot, B)
        name = f"daily_by_watch_date_b{B}.parquet"
        daily.to_parquet(processed_path(name), index=False)
        print(
            f"B={B}: {len(daily):,} rows -> {name}  ({time.perf_counter() - t_b:.1f}s)"
        )

    print(f"videos: {n:,}  runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
