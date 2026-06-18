from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .analyzer import AnalysisResult
from .store import StoredFile

console = Console()


def _fmt_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _short_path(path: str, max_len: int = 55) -> str:
    p = Path(path)
    s = str(p.parent)
    return s if len(s) <= max_len else '…/' + '/'.join(p.parts[-3:-1])


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(result: AnalysisResult) -> None:
    console.print()
    dup_size = sum(
        min(a.size, b.size)
        for a, b, _ in result.near_duplicates
    )
    console.print(Panel.fit(
        f"[bold]Files indexed:[/bold]       {result.total_files:,}\n"
        f"[bold]Total size:[/bold]          {_fmt_size(result.total_size)}\n"
        f"[bold]Clusters found:[/bold]      {len(result.clusters)}\n"
        f"[bold]Isolated files:[/bold]      {len(result.orphans)}\n"
        f"[bold]Near-duplicate pairs:[/bold] {len(result.near_duplicates)}"
        + (f"  [dim](~{_fmt_size(dup_size)} reclaimable)[/dim]" if dup_size else ""),
        title="[bold cyan]Analysis Summary[/bold cyan]",
        border_style="cyan",
    ))


# ── Clusters ──────────────────────────────────────────────────────────────────

def print_clusters(result: AnalysisResult, top_n: int = 20) -> None:
    if not result.clusters:
        console.print("\n[yellow]No clusters found.[/yellow]")
        return

    shown = result.clusters[:top_n]
    console.print(
        f"\n[bold cyan]Clusters[/bold cyan]  "
        f"[dim](showing {len(shown)} of {len(result.clusters)})[/dim]"
    )
    console.print(
        "[dim]Files within a cluster share semantic content — "
        "same subject, same project, same context.[/dim]\n"
    )

    for c in shown:
        size_str = _fmt_size(c.total_size)
        console.print(
            f"  [bold]Cluster {c.id:>3}[/bold]  "
            f"[green]{c.size}[/green] files  [dim]{size_str}[/dim]"
        )
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("name", style="cyan", no_wrap=True, max_width=40)
        t.add_column("type", width=9)
        t.add_column("size", justify="right", width=9)
        t.add_column("folder", style="dim")
        for f in c.files[:8]:
            t.add_row(f.filename, f.file_type, _fmt_size(f.size), _short_path(f.path))
        if c.size > 8:
            t.add_row(f"[dim]… {c.size - 8} more[/dim]", "", "", "")
        console.print(t)


# ── Near-duplicates ───────────────────────────────────────────────────────────

def print_near_duplicates(result: AnalysisResult, top_n: int = 30) -> None:
    if not result.near_duplicates:
        console.print("\n[green]No near-duplicates found.[/green]")
        return

    shown = result.near_duplicates[:top_n]
    console.print(
        f"\n[bold red]Near-Duplicates[/bold red]  "
        f"[dim](showing {len(shown)} of {len(result.near_duplicates)} pairs — "
        "score ≥ 0.97)[/dim]"
    )
    console.print("[dim]These pairs are nearly identical in content. "
                  "The smaller file is the likely candidate to remove.[/dim]\n")

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Score", justify="right", width=7)
    t.add_column("File A", style="cyan", max_width=36)
    t.add_column("File B", style="cyan", max_width=36)
    t.add_column("Remove?", width=36)

    for a, b, score in shown:
        remove = b.filename if a.size >= b.size else a.filename
        t.add_row(f"{score:.4f}", a.filename, b.filename, f"[dim]{remove}[/dim]")

    console.print(t)


# ── Isolated files ────────────────────────────────────────────────────────────

def print_orphans(result: AnalysisResult, top_n: int = 40) -> None:
    if not result.orphans:
        console.print("\n[green]No isolated files.[/green]")
        return

    shown = sorted(result.orphans, key=lambda f: f.size, reverse=True)[:top_n]
    console.print(
        f"\n[bold yellow]Isolated Files[/bold yellow]  "
        f"[dim](showing {len(shown)} of {len(result.orphans)} — sorted by size)[/dim]"
    )
    console.print(
        "[dim]These have no close neighbours — unique, forgotten, or one-offs "
        "that don't fit any theme.[/dim]\n"
    )

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("File", style="cyan", max_width=42)
    t.add_column("Type", width=9)
    t.add_column("Size", justify="right", width=9)
    t.add_column("Folder", style="dim")

    for f in shown:
        t.add_row(f.filename, f.file_type, _fmt_size(f.size), _short_path(f.path))

    if len(result.orphans) > top_n:
        console.print(f"[dim]  … and {len(result.orphans) - top_n} more[/dim]")

    console.print(t)


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(result: AnalysisResult, output_path: Path) -> None:
    rows: list[dict] = []

    for cluster in result.clusters:
        for f in cluster.files:
            rows.append({
                'path':        f.path,
                'filename':    f.filename,
                'file_type':   f.file_type,
                'size_bytes':  f.size,
                'category':    'cluster',
                'cluster_id':  cluster.id,
                'cluster_size': cluster.size,
                'note':        '',
            })

    for f in result.orphans:
        rows.append({
            'path':        f.path,
            'filename':    f.filename,
            'file_type':   f.file_type,
            'size_bytes':  f.size,
            'category':    'isolated',
            'cluster_id':  '',
            'cluster_size': '',
            'note':        'no similar neighbours found',
        })

    dup_seen: set[str] = set()
    for a, b, score in result.near_duplicates:
        for f in (a, b):
            if f.path not in dup_seen:
                dup_seen.add(f.path)
                rows.append({
                    'path':        f.path,
                    'filename':    f.filename,
                    'file_type':   f.file_type,
                    'size_bytes':  f.size,
                    'category':    'near-duplicate',
                    'cluster_id':  '',
                    'cluster_size': '',
                    'note':        f'similarity {score:.4f}',
                })

    fieldnames = ['path', 'filename', 'file_type', 'size_bytes',
                  'category', 'cluster_id', 'cluster_size', 'note']
    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
