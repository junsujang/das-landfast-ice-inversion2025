"""Recorded -> canonical channel numbers across DAS configuration epochs.

The interrogator's channel numbering shifted several times during the
experiment. The cable bend is a fixed physical feature, so the channel it lands
on in each epoch (`mid_idx` in data/bend_change_shortest.csv) reads out that
epoch's shift directly: ``shift = reference_bend_channel - mid_idx``. Picks are
moved onto one canonical numbering, in which the bend is channel 594, before
they are compared with each other.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

Rules = tuple[tuple[str, int], ...]


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_time(value: str) -> datetime:
    """ISO time, 'Z' or space separated, as an aware UTC datetime."""
    text = str(value).strip().replace("Z", "+00:00")
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    return _utc(datetime.fromisoformat(text))


def epoch_shift_rules_from_bend_csv(
    path: str | Path,
    *,
    reference_bend_channel: float = 594.0,
    time_column: str = "change_time",
    bend_channel_column: str = "mid_idx",
) -> Rules:
    """One ``(start time, shift)`` rule per epoch, from the observed-bend table.

    Recordings earlier than the first row get shift 0, i.e. are assumed to be
    in the canonical numbering already. Rows that repeat the previous shift are
    dropped, which does not change the mapping.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = sorted(
        (parse_time(r[time_column]), int(round(reference_bend_channel - float(r[bend_channel_column]))))
        for r in rows
        if str(r.get(time_column, "")).strip() and str(r.get(bend_channel_column, "")).strip()
    )
    if not parsed:
        raise ValueError(f"No usable rows in bend-change table {path}")
    rules, previous = [], None
    for t, shift in parsed:
        if shift != previous:
            rules.append((t.isoformat(), shift))
            previous = shift
    return tuple(rules)


def channel_shift_for_datetime(dt: datetime, rules: Sequence[tuple[str, int]]) -> int:
    """Additive recorded -> canonical channel offset in force at `dt`."""
    t = _utc(dt)
    shift = 0
    for start, offset in sorted((parse_time(t0), int(off)) for t0, off in rules):
        if t < start:
            break
        shift = offset
    return shift


def canonical_channel_count(number_of_loci: int, dt: datetime, rules: Sequence[tuple[str, int]]) -> int:
    """Number of canonical channels 0..N-1 covered by a recording.

    Recorded channels 0..L-1 map to canonical shift..L-1+shift, so the highest
    canonical channel is L-1+shift whichever way the shift goes.
    """
    return max(0, int(number_of_loci) + channel_shift_for_datetime(dt, rules))
