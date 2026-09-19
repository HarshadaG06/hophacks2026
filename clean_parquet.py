"""Stream-clean a large Parquet file with Polars."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import polars as pl


REQUIRED_COLUMNS = ("music_id", "create_time", "is_ad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows with no music_id, a bad create_time value, "
            "or is_ad = 1 from a Parquet file."
        )
    )
    parser.add_argument("input", type=Path, help="Input Parquet file")
    parser.add_argument("output", type=Path, help="Cleaned output Parquet file")
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path (default: <output>.cleanup_report.json)",
    )
    parser.add_argument(
        "--timestamp-format",
        help=(
            "Optional strptime format for string timestamps, such as "
            "'%%Y-%%m-%%d %%H:%%M:%%S'. Polars infers the format when omitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output and report if they already exist",
    )
    return parser.parse_args()


def music_id_is_valid(dtype: pl.DataType) -> pl.Expr:
    value = pl.col("music_id")
    if dtype == pl.String or dtype == pl.Categorical or dtype == pl.Enum:
        return value.is_not_null() & value.cast(pl.String).str.strip_chars().ne("")
    return value.is_not_null()


def timestamp_is_valid(
    dtype: pl.DataType, timestamp_format: str | None
) -> pl.Expr:
    value = pl.col("create_time")

    if dtype.is_temporal():
        return value.is_not_null()
    if dtype.is_integer():
        # Unix timestamps at or before the epoch are treated as invalid.
        return value.is_not_null() & value.gt(0)
    if dtype.is_float():
        return value.is_not_null() & value.is_finite() & value.gt(0)
    if dtype == pl.String or dtype == pl.Categorical or dtype == pl.Enum:
        text = value.cast(pl.String).str.strip_chars()
        parsed = text.str.to_datetime(
            format=timestamp_format,
            strict=False,
            exact=True,
        )
        return value.is_not_null() & text.ne("") & parsed.is_not_null()

    raise TypeError(
        "Unsupported create_time type "
        f"{dtype!s}; expected a date/datetime, string, integer, or float"
    )


def percentage(count: int, total: int) -> float:
    return round((count / total * 100) if total else 0.0, 6)


def clean_parquet(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    timestamp_format: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")

    existing = [path for path in (output_path, report_path) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite: {paths} (use --overwrite)")

    source = pl.scan_parquet(input_path, low_memory=True)
    schema = source.collect_schema()
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in schema]
    if missing_columns:
        raise ValueError(
            "Input is missing required column(s): " + ", ".join(missing_columns)
        )

    valid_music = music_id_is_valid(schema["music_id"]).fill_null(False)
    valid_timestamp = timestamp_is_valid(
        schema["create_time"], timestamp_format
    ).fill_null(False)
    is_ad = (
        pl.col("is_ad").cast(pl.Int64, strict=False).eq(1).fill_null(False)
    )

    # Attribute each deleted row to the first rule it fails. These categories
    # are mutually exclusive, so their counts sum to the unique deleted total.
    missing_music = ~valid_music
    bad_timestamp = valid_music & ~valid_timestamp
    ad = valid_music & valid_timestamp & is_ad
    keep = valid_music & valid_timestamp & ~is_ad

    stats = (
        source.select(
            pl.len().alias("input_rows"),
            missing_music.sum().alias("missing_music_id"),
            bad_timestamp.sum().alias("bad_create_time"),
            ad.sum().alias("is_ad"),
            keep.sum().alias("output_rows"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        source.filter(keep).sink_parquet(
            temporary_output,
            compression="zstd",
            maintain_order=True,
            mkdir=True,
        )
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    input_rows = int(stats["input_rows"])
    output_rows = int(stats["output_rows"])
    reasons = {
        "missing_music_id": int(stats["missing_music_id"]),
        "bad_create_time": int(stats["bad_create_time"]),
        "is_ad": int(stats["is_ad"]),
    }
    deleted_rows = input_rows - output_rows
    report: dict[str, Any] = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "deleted_rows": deleted_rows,
        "deleted_proportion_percent": percentage(deleted_rows, input_rows),
        "deletions_by_reason": {
            reason: {
                "rows": count,
                "proportion_of_input_percent": percentage(count, input_rows),
                "proportion_of_deleted_percent": percentage(count, deleted_rows),
            }
            for reason, count in reasons.items()
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report_path = args.report or args.output.with_name(
        args.output.name + ".cleanup_report.json"
    )

    try:
        report = clean_parquet(
            input_path=args.input,
            output_path=args.output,
            report_path=report_path,
            timestamp_format=args.timestamp_format,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    print(f"Report written to: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
