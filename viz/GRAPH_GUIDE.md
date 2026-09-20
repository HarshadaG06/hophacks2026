# Graph guide — Audio Lifespan Explorer

This page explains every chart in the viz (`viz/index.html`). The main question:

**When within an audio’s life are the most-viewed videos concentrated?**

Each audio gets its own **lifespan** from first video post to last (`audio_start` → `audio_end`). All “normalized” positions use **u** ∈ [0, 1]:

- **u = 0** — first day a video using this audio was posted  
- **u = 1** — last day a video was posted  
- **u = 0.5** — halfway through that audio’s posting history  

Views are **final snapshot counts** at the collection date, not a reconstructed watch timeline.

---

## How to interact

| Action | Effect |
|--------|--------|
| **Click a dot** | Select that audio; updates daily/normalized panels **and** marks the selected audio on both histograms (red dashed line) |
| **Shift + drag** | Brush a rectangle on a scatter plot; dims other audios and **recomputes both histograms** for the brushed subset only |
| **Hover a dot** | Tooltip with `music_id`, metrics, cluster |
| **Tercile toggle** (mean curve) | Switch between all / short / medium / long lifespan audios |

---

## 1. Selected audio — daily posting

Two stacked bar charts for the **currently selected** audio.

### Top chart — total views per day

- **X-axis:** Days since first video (`0` … lifespan)  
- **Y-axis:** Sum of `total_views` for videos **posted that day**  
- **Blue bars:** How much view mass landed on each posting day  
- **Red dashed line:** **Views centroid** — the views-weighted average lifespan position, drawn at `views_centroid_u × max_day`  
- **Shaded bands:** Kleinberg posting bursts, aligned via `start_u`/`end_u` × lifespan (same x-axis as the bars). Legend shows only levels present for this audio (L1, L2, …), each matching band color.

### Bottom chart — videos posted per day

- **X-axis:** Same day index  
- **Y-axis:** Number of videos posted that day  
- **Orange bars:** Posting volume (when creators used this audio)  
- **Same burst bands** as above — bursts are detected from **post times**, not views

**How to read it:** Compare bars (when things were posted) to the views centroid line (where views concentrate). If the centroid sits in a burst band, high-view videos tended to be posted during that burst.

---

## 2. Same audio — normalized lifespan

A **single-audio** view of the full lifespan compressed to u ∈ [0, 1], split into **20 equal bins**.

- **X-axis:** Normalized lifespan position (0% → 100%)  
- **Y-axis:** Share of that audio’s total (each curve sums to 100% across bins)  
- **Solid blue line:** Share of **views** in each bin  
- **Dashed orange line:** Share of **videos posted** in each bin  
- **Red dashed line:** Views centroid (`views_centroid_u`)

**How to read it:**

- Blue **above** orange in a bin → views overweight that part of the lifespan (big videos landed there)  
- Blue **below** orange → lots of posts but relatively fewer views  
- Centroid line shows the single-number summary of where views sit on average  

---

## 3. views_centroid_u distribution

Histogram of `views_centroid_u` across audios. **Updates when you brush** (shows brushed subset only). **Red dashed line** = selected audio.

- **X-axis:** `views_centroid_u` (0 = views concentrated at start, 1 = at end)  
- **Y-axis:** Number of audios in each bin  

**Caption below:** Whether a 1- vs 2-component Gaussian mixture model (BIC) fits the distribution better — a hint at “early vs late concentration” groups.

**Typical pattern in this dataset:** Mass around u ≈ 0.7–0.8, meaning views skew toward the **later** part of each audio’s posting history (partly due to snapshot age bias — older posts had more time to accumulate views).

---

## 4. Views vs count centroid

Scatter plot — one dot per audio, colored by **KMeans cluster**.

- **X-axis:** `views_centroid_u` — views-weighted mean posting position  
- **Y-axis:** `count_centroid_u` — unweighted mean posting position (when videos were posted, ignoring view counts)  
- **Dashed diagonal:** Where views centroid equals count centroid  

| Position | Meaning |
|----------|---------|
| **On diagonal** | Views spread roughly like posting; no strong late/early skew |
| **Above diagonal** (`views_centroid_u > count_centroid_u`) | Views land **later** than typical posts — e.g. early posts didn’t dominate views |
| **Below diagonal** | Views land **earlier** than typical posts — e.g. early viral hits, then many low-view posts |

**Use case:** Quickly spot audios where “when people posted” diverges from “where the views are.”

---

## 5. Mean normalized curve

Average of the 20-bin curves (panel 2) across many audios.

- **X-axis:** Normalized lifespan (u)  
- **Y-axis:** Mean share of views / videos per bin  
- **Toggle:** All audios, or only short / medium / long lifespan (terciles by `lifespan_days`)

**How to read it:** Do short-lived sounds peak early? Do long-lived sounds keep accumulating views late? Compare blue vs orange and switch terciles to see lifespan-length effects.

---

## 6. Cluster summary

Text summary of **KMeans clusters** (from `s6_explore.py`), not a chart.

Each row shows:

- **Cluster id** and **n** (number of audios)  
- **Avg views u** — mean `views_centroid_u`  
- **Avg lift** — mean `views_burst_lift` (see below)  
- **Avg lifespan** — mean days from first to last post  

The selected audio’s cluster is highlighted. If you brushed a scatter plot, stats reflect only brushed audios.

**Cluster features used:** lifespan metrics (`views_centroid_u`, `peak_u_views`, `share_views_first25`, `top_video_centroid_u`, `log_lifespan_days`) plus Kleinberg burst metrics (`kleinberg_max_level`, `top_burst_length_share`, `share_of_views_in_bursts`).

---

## 7. Kleinberg max level (histogram)

Distribution of max burst depth across audios. **Updates when you brush.** Bar colors match burst level colors (L1–L6). **Red dashed line** = selected audio’s level.

- **X-axis:** `kleinberg_max_level` — deepest nested burst detected from **post timestamps**  
- **Y-axis:** Number of audios  

Higher level = more hierarchical posting structure (bursts within bursts).

---

## 8. Burst vs view centroid

Scatter — one dot per audio, colored by cluster. Linked brushing with panel 4.

- **X-axis:** `burst_views_centroid_u` — views-weighted mean lifespan position of videos posted **inside Kleinberg bursts**  
- **Y-axis:** `views_centroid_u` — views-weighted mean position over **all** videos  
- **Dashed diagonal:** Where burst-period views align with overall view concentration  

**How to read it:** Points on the diagonal mean views during bursts sit at the same normalized position as views overall. Above the diagonal → burst-period posts carry views that land **later** than the audio’s average; below → burst views concentrate **earlier**.

---

## Kleinberg burst levels (L0–L3)

On the daily posting charts, translucent bands mark **Kleinberg (2002)** bursts detected from video **post times** ([paper](https://www.cs.cornell.edu/home/kleinber/bhs.pdf)).

| Label | Meaning |
|-------|---------|
| **L1** | First-level burst — a period of elevated posting vs background |
| **L2** | Burst **inside** an L1 window — tighter cluster |
| **L3** | Deeper nesting — even more intense sub-cluster |

- **Lighter band** → lower level (broader)  
- **Darker band** → higher level (narrower, more intense)  

Only intervals with **level ≥ 1** are drawn. Burst **features** (e.g. `share_of_videos_in_bursts`) also use level ≥ 1.

**Important:** Bursts describe **when videos were posted**, not when views were watched. Views are overlaid as bars and the centroid line.

---

## Key metrics (quick reference)

| Metric | Meaning |
|--------|---------|
| `views_centroid_u` | Views-weighted mean lifespan position |
| `count_centroid_u` | Unweighted mean posting position |
| `centroid_gap` | `views_centroid_u − count_centroid_u` |
| `top_video_centroid_u` | Mean u of top-decile videos by views |
| `kleinberg_max_level` | Deepest Kleinberg burst level |
| `share_of_views_in_bursts` | Fraction of total views from videos posted inside bursts |
| `views_burst_lift` | `share_of_views_in_bursts / share_of_videos_in_bursts` — **> 1** means burst-period posts over-index on views |
| `burst_views_centroid_u` | Views-weighted mean u for videos posted inside bursts |

---

## Caveats

1. **Snapshot views:** `total_views` is the count at collection time. Older videos had longer to accumulate views, which pushes `views_centroid_u` toward earlier u unless corrected. Check `age_views_spearman` in the data.  
2. **Synthetic views:** If the raw dump lacks view counts, `loader.py` assigns placeholder totals — metrics are structurally valid but absolute view numbers may not be real.  
3. **Bursts ≠ views:** Kleinberg bands are from post timestamps; a posting burst does not guarantee a view spike on those days.

---

## Footnote on the page

> Bursts are detected from video posting times (Kleinberg 2002). Views are final counts at the snapshot date. Older videos have had longer to accumulate views.
