"""Single source of truth for paths and the time split.

Every date in this project derives from here. Nothing downstream may hardcode a date.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SQL = ROOT / "sql"

# --- Time split -------------------------------------------------------------
# The transaction log ends 2020-09-22. Weeks are half-open intervals [start, end).
DATA_END = date(2020, 9, 22)

VAL_START = date(2020, 9, 16)
VAL_END = VAL_START + timedelta(days=7)  # 2020-09-23, exclusive

TRAIN_START = VAL_START - timedelta(days=7)  # 2020-09-09
TRAIN_END = VAL_START  # exclusive

# Feature windows never cross into the target week: features for a week starting at
# `as_of` may only read transactions with t_dat < as_of.
FEATURE_LOOKBACK_DAYS = 90

# --- Model constants --------------------------------------------------------
K = 12  # MAP@12 cutoff — the competition metric

# Retrieval budget per customer. recall@100 is 0.16, @300 is 0.27, @500 is 0.33 — but MAP@12 is
# flat across all three (reports/rolling.md), so this buys headroom for a better ranker, not
# accuracy today.
N_CANDIDATES = 300
NEG_SAMPLE_RATIO = 20  # negatives per positive in TRAINING only, never in validation


def week_window(as_of: date) -> tuple[date, date]:
    """The 7-day target window that starts at `as_of`, as a half-open [start, end) pair."""
    return as_of, as_of + timedelta(days=7)


def feature_window(as_of: date, lookback_days: int = FEATURE_LOOKBACK_DAYS) -> tuple[date, date]:
    """The half-open [start, as_of) window features may read. Excludes `as_of` itself."""
    return as_of - timedelta(days=lookback_days), as_of
