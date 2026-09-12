"""Resolve HPC nested Amber trajectories (``$TRAJ_DIR/<run>/mdcrd_v``)."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional, Sequence

GENERIC_TRAJ_STEMS = frozenset(
    {
        "mdcrd",
        "mdcrd_v",
        "crd",
        "dcd",
        "xtc",
        "trj",
        "nc",
        "prod",
        "equil",
    }
)
TRAJ_EXTENSIONS = (".trj", ".xtc", ".dcd", ".nc", ".crd")


def trajectory_output_id(traj_path: Path | str) -> str:
    """Id for a trajectory file (``<run>_mdcrd_v`` on the nested HPC layout)."""
    path = Path(traj_path)
    stem = path.stem
    parent = path.parent.name
    if stem.lower() in GENERIC_TRAJ_STEMS and parent not in ("", ".", ".."):
        return f"{parent}_{stem}"
    return stem


def unique_trajectory_ids(paths: Sequence[Path | str]) -> list[str]:
    """``trajectory_output_id`` with ``_1``, ``_2``, … on collisions."""
    counts: dict[str, int] = {}
    unique: list[str] = []
    for path in paths:
        tid = trajectory_output_id(path)
        n = counts.get(tid, 0)
        counts[tid] = n + 1
        unique.append(tid if n == 0 else f"{tid}_{n}")
    return unique


def concrete_traj_files(path: Path | str) -> list[Path]:
    """Resolve a glob hit to one or more readable trajectory files.

    Supports HPC nested layouts such as ``$TRAJ_DIR/<run_id>/mdcrd_v``
    (extensionless Amber mdcrd) as well as ``mdcrd_v.trj`` / directory hits.
    """
    path = Path(path)
    if path.is_file():
        return [path]

    out: list[Path] = []
    if path.is_dir():
        for name in ("mdcrd_v", "mdcrd"):
            base = path / name
            if base.is_file():
                out.append(base)
                continue
            for ext in TRAJ_EXTENSIONS:
                cand = Path(f"{base}{ext}")
                if cand.is_file():
                    out.append(cand)
        if out:
            return out
        for ext in ("*.xtc", "*.trj", "*.dcd", "*.nc", "*.crd"):
            out.extend(sorted(path.glob(ext)))
        return out

    for ext in TRAJ_EXTENSIONS:
        cand = Path(f"{path}{ext}")
        if cand.is_file():
            out.append(cand)
    return out


def expand_trajectories(patterns: Sequence[str]) -> list[Path]:
    """Expand globs / paths into concrete trajectory files.

    Typical HPC pattern::

        --trajectories "$TRAJ_DIR/*/mdcrd_v"
    """
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches and not any(ch in Path(pat).name for ch in "*?[]"):
            for ext in TRAJ_EXTENSIONS:
                matches.extend(sorted(glob.glob(f"{pat}{ext}")))
        elif not matches and Path(pat).name in GENERIC_TRAJ_STEMS:
            matches = sorted(glob.glob(f"{pat}.*"))

        if not matches:
            alt = sorted(glob.glob(pat.rstrip("/")))
            if not alt and pat.endswith("*/mdcrd_v"):
                alt = sorted(glob.glob(pat[: -len("/mdcrd_v")]))
            if alt:
                matches = alt

        if matches:
            for m in matches:
                paths.extend(concrete_traj_files(Path(m)))
        else:
            p = Path(pat)
            resolved = concrete_traj_files(p)
            if resolved:
                paths.extend(resolved)
            elif p.is_dir():
                for ext in ("*.xtc", "*.trj", "*.dcd", "*.nc", "*.crd"):
                    paths.extend(sorted(p.glob(ext)))

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if not p.is_file():
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def format_no_trajectories_error(patterns: Sequence[str]) -> str:
    """Build a diagnostic message when trajectory globs match nothing."""
    lines = [
        "No trajectories found for the given patterns:",
        *[f"  {pat!r}" for pat in patterns],
        "",
        "Hints:",
        "  • Quote the glob so Python expands it: --trajectories \"$TRAJ_DIR/*/mdcrd_v\"",
        "  • Put a space before every line-continuation backslash (and no space after \\).",
        "  • Check the nested layout exists, e.g.:",
    ]
    for pat in patterns:
        norm = pat.replace("\\", "/")
        traj_dir: Optional[Path] = None
        if "/*/mdcrd_v" in norm:
            traj_dir = Path(norm.split("/*/mdcrd_v", 1)[0])
        elif norm.endswith("/*"):
            traj_dir = Path(norm[:-2])
        else:
            parent = Path(pat).parent
            if "*" not in parent.name:
                traj_dir = parent
        if traj_dir is None:
            continue
        lines.append(f"      ls \"{traj_dir}\" | head")
        if traj_dir.is_dir():
            kids = sorted(traj_dir.iterdir())[:8]
            if not kids:
                lines.append(f"    (directory exists but is empty: {traj_dir})")
            else:
                lines.append(f"    Found under {traj_dir}:")
                for kid in kids:
                    marker = ""
                    if kid.is_dir():
                        md = kid / "mdcrd_v"
                        marker = (
                            " [has mdcrd_v]"
                            if md.is_file()
                            else " [NO mdcrd_v]"
                        )
                    lines.append(f"      - {kid.name}{marker}")
                if len(list(traj_dir.iterdir())) > 8:
                    lines.append("      - ...")
        else:
            lines.append(f"    (path does not exist: {traj_dir})")
    return "\n".join(lines)
