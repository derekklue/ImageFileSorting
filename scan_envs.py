#!/usr/bin/env python3
"""
Scan a directory for Python virtual environments and flag ones that look unused.

Usage:
    python scan_envs.py /Users/durrrek
    python scan_envs.py /Users/durrrek --stale-days 90
    python scan_envs.py /Users/durrrek --csv envs.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Virtual env detection ─────────────────────────────────────────────────────

# These markers, when found in a directory, identify it as a venv root
_VENV_MARKERS = [
    'pyvenv.cfg',
    'bin/python',
    'bin/python3',
    'Scripts/python.exe',   # Windows
]

# Common names people give their venv directories
_VENV_DIR_NAMES = {
    '.venv', 'venv', 'env', '.env', 'virtualenv', '.virtualenv',
    'pyenv', '.pyenv', 'venv3', '.venv3',
}


def is_venv(path: Path) -> bool:
    if path.name in _VENV_DIR_NAMES:
        return any((path / m).exists() for m in _VENV_MARKERS)
    return (path / 'pyvenv.cfg').exists()


def find_venvs(root: Path) -> list[Path]:
    """Walk root and return every venv directory found."""
    found: list[Path] = []
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.Trash', 'Library'}

    for dirpath_str, dirnames, _ in os.walk(root, followlinks=False):
        dirpath = Path(dirpath_str)
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith('.')
            or d in _VENV_DIR_NAMES  # still descend into hidden venv dirs
        ]

        for d in list(dirnames):
            candidate = dirpath / d
            if is_venv(candidate):
                found.append(candidate)
                dirnames.remove(d)  # don't descend into the venv itself

    return found


# ── Staleness signals ─────────────────────────────────────────────────────────

def _days_since(timestamp: float) -> float:
    return (time.time() - timestamp) / 86400


def _newest_mtime(path: Path, extensions: set[str], depth: int = 3) -> Optional[float]:
    """Most recent mtime of files matching extensions within depth levels."""
    newest: Optional[float] = None
    for root_str, dirnames, filenames in os.walk(path):
        rel = Path(root_str).relative_to(path)
        if len(rel.parts) >= depth:
            dirnames.clear()
            continue
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                try:
                    t = (Path(root_str) / f).stat().st_mtime
                    if newest is None or t > newest:
                        newest = t
                except OSError:
                    pass
    return newest


def _git_last_commit_days(project_dir: Path) -> Optional[float]:
    """Days since the last git commit in project_dir, or None if not a git repo."""
    try:
        result = subprocess.run(
            ['git', '-C', str(project_dir), 'log', '-1', '--format=%ct'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _days_since(float(result.stdout.strip()))
    except Exception:
        pass
    return None


def _python_interpreter_ok(venv: Path) -> bool:
    """Check whether the venv's Python binary actually executes."""
    candidates = [
        venv / 'bin' / 'python',
        venv / 'bin' / 'python3',
        venv / 'Scripts' / 'python.exe',
    ]
    for py in candidates:
        if py.exists():
            try:
                r = subprocess.run(
                    [str(py), '--version'],
                    capture_output=True, timeout=5,
                )
                return r.returncode == 0
            except Exception:
                return False
    return False


def _installed_packages(venv: Path) -> list[str]:
    """Return list of top-level installed package names."""
    candidates = [
        venv / 'bin' / 'pip',
        venv / 'bin' / 'pip3',
        venv / 'Scripts' / 'pip.exe',
    ]
    for pip in candidates:
        if pip.exists():
            try:
                r = subprocess.run(
                    [str(pip), 'list', '--format=freeze'],
                    capture_output=True, text=True, timeout=15,
                )
                if r.returncode == 0:
                    return [line.split('==')[0] for line in r.stdout.splitlines() if line]
            except Exception:
                pass
    return []


def _venv_disk_size_mb(venv: Path) -> float:
    total = 0
    try:
        for dirpath, _, filenames in os.walk(venv):
            for f in filenames:
                try:
                    total += (Path(dirpath) / f).stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total / (1024 * 1024)


def _newest_access_in_venv(venv: Path) -> Optional[float]:
    """Most recent atime of any file inside the venv."""
    newest: Optional[float] = None
    site_packages = list(venv.glob('lib/*/site-packages'))
    search_root = site_packages[0] if site_packages else venv
    try:
        for dirpath, _, filenames in os.walk(search_root):
            for f in filenames:
                try:
                    t = (Path(dirpath) / f).stat().st_atime
                    if newest is None or t > newest:
                        newest = t
                except OSError:
                    pass
    except Exception:
        pass
    return newest


# ── Analysis result ───────────────────────────────────────────────────────────

@dataclass
class EnvInfo:
    venv_path: Path
    project_dir: Path
    size_mb: float
    python_ok: bool
    packages: list[str] = field(default_factory=list)

    # Staleness signals (days; None = unknown)
    days_since_venv_access: Optional[float] = None
    days_since_project_py_edit: Optional[float] = None
    days_since_git_commit: Optional[float] = None
    days_since_venv_mtime: Optional[float] = None

    @property
    def staleness_score(self) -> int:
        """0 = active, higher = more likely stale. Max ~5."""
        score = 0
        if not self.python_ok:
            score += 2
        thresholds = [
            (self.days_since_venv_access,      90),
            (self.days_since_project_py_edit,  180),
            (self.days_since_git_commit,        180),
            (self.days_since_venv_mtime,       180),
        ]
        for days, limit in thresholds:
            if days is not None and days > limit:
                score += 1
        return score

    @property
    def verdict(self) -> str:
        if not self.python_ok:
            return 'BROKEN'
        s = self.staleness_score
        if s >= 4:
            return 'STALE'
        if s >= 2:
            return 'POSSIBLY STALE'
        return 'ACTIVE'

    def _fmt_days(self, d: Optional[float]) -> str:
        if d is None:
            return 'unknown'
        if d < 1:
            return 'today'
        return f'{int(d)}d ago'

    def summary_lines(self) -> list[str]:
        lines = [
            f"  venv last accessed  : {self._fmt_days(self.days_since_venv_access)}",
            f"  .py files edited    : {self._fmt_days(self.days_since_project_py_edit)}",
            f"  git last commit     : {self._fmt_days(self.days_since_git_commit)}",
            f"  venv last modified  : {self._fmt_days(self.days_since_venv_mtime)}",
            f"  python interpreter  : {'ok' if self.python_ok else 'BROKEN'}",
            f"  disk usage          : {self.size_mb:.0f} MB",
            f"  packages            : {len(self.packages)}"
            + (f" ({', '.join(self.packages[:5])}{'…' if len(self.packages) > 5 else ''})" if self.packages else ""),
        ]
        return lines


def analyse_venv(venv: Path) -> EnvInfo:
    project_dir = venv.parent

    venv_mtime = None
    try:
        venv_mtime = _days_since(venv.stat().st_mtime)
    except OSError:
        pass

    venv_access = _newest_access_in_venv(venv)
    access_days = _days_since(venv_access) if venv_access else None

    py_mtime = _newest_mtime(project_dir, {'.py', '.ipynb'}, depth=4)
    py_days = _days_since(py_mtime) if py_mtime else None

    git_days = _git_last_commit_days(project_dir)

    python_ok = _python_interpreter_ok(venv)
    packages = _installed_packages(venv) if python_ok else []
    size_mb = _venv_disk_size_mb(venv)

    return EnvInfo(
        venv_path=venv,
        project_dir=project_dir,
        size_mb=size_mb,
        python_ok=python_ok,
        packages=packages,
        days_since_venv_access=access_days,
        days_since_project_py_edit=py_days,
        days_since_git_commit=git_days,
        days_since_venv_mtime=venv_mtime,
    )


# ── Output ────────────────────────────────────────────────────────────────────

_VERDICT_COLOUR = {
    'BROKEN':         '\033[91m',  # red
    'STALE':          '\033[93m',  # yellow
    'POSSIBLY STALE': '\033[94m',  # blue
    'ACTIVE':         '\033[92m',  # green
}
_RESET = '\033[0m'


def _colour(text: str, verdict: str) -> str:
    return _VERDICT_COLOUR.get(verdict, '') + text + _RESET


def print_report(envs: list[EnvInfo], stale_only: bool = False) -> None:
    shown = [e for e in envs if not stale_only or e.verdict in ('BROKEN', 'STALE', 'POSSIBLY STALE')]

    # Group by verdict
    groups = {'BROKEN': [], 'STALE': [], 'POSSIBLY STALE': [], 'ACTIVE': []}
    for e in shown:
        groups[e.verdict].append(e)

    print()
    print('=' * 70)
    print(f"  Python Environment Audit  ({len(envs)} found)")
    print('=' * 70)

    total_stale_mb = sum(
        e.size_mb for e in envs
        if e.verdict in ('BROKEN', 'STALE')
    )

    for verdict in ('BROKEN', 'STALE', 'POSSIBLY STALE', 'ACTIVE'):
        bucket = groups[verdict]
        if not bucket:
            continue
        bucket.sort(key=lambda e: e.size_mb, reverse=True)
        print(f"\n{_colour(f'── {verdict} ({len(bucket)})', verdict)}")
        for e in bucket:
            label = _colour(f'[{verdict}]', verdict)
            print(f"\n  {label}  {e.venv_path}")
            for line in e.summary_lines():
                print(line)

    print()
    print(f"  Reclaimable (BROKEN + STALE): {total_stale_mb:.0f} MB")
    print()


def write_csv(envs: list[EnvInfo], output: Path) -> None:
    fields = [
        'verdict', 'venv_path', 'project_dir', 'size_mb',
        'python_ok', 'package_count', 'top_packages',
        'days_since_venv_access', 'days_since_project_py_edit',
        'days_since_git_commit', 'days_since_venv_mtime',
    ]
    with open(output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for e in envs:
            w.writerow({
                'verdict':                    e.verdict,
                'venv_path':                  str(e.venv_path),
                'project_dir':                str(e.project_dir),
                'size_mb':                    round(e.size_mb, 1),
                'python_ok':                  e.python_ok,
                'package_count':              len(e.packages),
                'top_packages':               ', '.join(e.packages[:10]),
                'days_since_venv_access':     round(e.days_since_venv_access, 0) if e.days_since_venv_access else '',
                'days_since_project_py_edit': round(e.days_since_project_py_edit, 0) if e.days_since_project_py_edit else '',
                'days_since_git_commit':      round(e.days_since_git_commit, 0) if e.days_since_git_commit else '',
                'days_since_venv_mtime':      round(e.days_since_venv_mtime, 0) if e.days_since_venv_mtime else '',
            })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='Directory to scan (e.g. /Users/durrrek)')
    parser.add_argument('--stale-days', type=int, default=90,
                        help='Days of inactivity before flagging as possibly stale (default: 90)')
    parser.add_argument('--stale-only', action='store_true',
                        help='Only show non-active environments')
    parser.add_argument('--csv', metavar='FILE', help='Also write results to a CSV file')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: path not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {root} for Python environments…", flush=True)
    venvs = find_venvs(root)
    print(f"Found {len(venvs)} environment(s). Analysing…", flush=True)

    results: list[EnvInfo] = []
    for i, venv in enumerate(venvs, 1):
        print(f"  [{i}/{len(venvs)}] {venv}", flush=True)
        results.append(analyse_venv(venv))

    results.sort(key=lambda e: (
        {'BROKEN': 0, 'STALE': 1, 'POSSIBLY STALE': 2, 'ACTIVE': 3}[e.verdict],
        -e.size_mb,
    ))

    print_report(results, stale_only=args.stale_only)

    if args.csv:
        out = Path(args.csv)
        write_csv(results, out)
        print(f"CSV written to {out.resolve()}")


if __name__ == '__main__':
    main()
