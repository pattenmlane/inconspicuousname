"""Match **merged-timeline** precomputed trigger ints to per-day `TradingState.timestamp`."""
from __future__ import annotations

import bisect
from typing import Any


def window_active_local_legacy(ts_local: int, triggers: list[int], w: int) -> bool:
    """Old (wrong for multi-day merged JSON) bisect on local timestamps only."""
    if not triggers:
        return False
    lo = int(ts_local) - w
    i = bisect.bisect_right(triggers, int(ts_local))
    j = bisect.bisect_left(triggers, lo)
    return j < i


def window_active(state: Any, ts_local: int, triggers: list[int], w: int, day_cum_offset: dict[int, int]) -> bool:
    """
    If `state.day_num` is set (Prosperity4BT `TestRunner` after day_num wire-up),
    map local clock to merged absolute time using `day_cum_offset` from the signal JSON.
    Otherwise fall back to legacy local-only matching (single-day tapes only).
    """
    if not triggers:
        return False
    d = getattr(state, "day_num", None)
    if d is None or int(d) < 0 or not day_cum_offset:
        return window_active_local_legacy(ts_local, triggers, w)
    abs_ts = int(day_cum_offset[int(d)]) + int(ts_local)
    lo = abs_ts - w
    i = bisect.bisect_right(triggers, abs_ts)
    j = bisect.bisect_left(triggers, lo)
    return j < i
