"""Known fair-probe export folders (two sessions) for pooled validation."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def all_session_dirs() -> tuple[Path, ...]:
    """278076 (this package) and 248329 sibling folder when present."""
    dirs: list[Path] = [HERE]
    other = HERE.parent / "r2_osmium_fair_248329"
    if other.is_dir() and other.resolve() != HERE.resolve():
        dirs.append(other)
    return tuple(dirs)
