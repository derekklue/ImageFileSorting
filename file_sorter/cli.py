from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich import box

from .scanner import scan_directory
from .embedder import Embedder
from .store import VectorStore
from .analyzer import Analyzer
from . import report as rpt

app = typer.Typer(
    name="file-sorter",
    help="Map your files by semantic similarity using vision embeddings.",
    no_args_is_help=True,
)
console = Console()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _store(db: str) -> VectorStore:
    return VectorStore(db_path=db)


# ── scan ──────────────────────────────────────────────────────────────────────

@app.command()
def scan(
    paths: List[Path] = typer.Argument(..., help="Directories to scan"),
    db: str = typer.Option(".file_sorter_db", "--db", help="Vector database path"),
    recursive: bool = typer.Option(True, help="Scan subdirectories"),
    skip_larger_mb: Optional[float] = typer.Option(
        None, "--skip-larger-mb", help="Skip files larger than N MB"
    ),
    device: str = typer.Option("cpu", help="Inference device: cpu / cuda / mps"),
    force: bool = typer.Option(False, "--force", help="Re-embed already-processed files"),
    types: Optional[str] = typer.Option(
        None, "--types",
        help="Comma-separated file types to include: image,video,audio,document,code,text,archive,binary"
    ),
) -> None:
    """Scan directories and embed files into the vector database."""
    for p in paths:
        if not p.exists():
            console.print(f"[red]Path not found: {p}[/red]")
            raise typer.Exit(1)

    store = _store(db)
    embedder = Embedder(device=device)
    max_size = int(skip_larger_mb * 1024 * 1024) if skip_larger_mb else None
    filter_types = set(types.split(',')) if types else None

    all_files = []
    for p in paths:
        all_files.extend(
            scan_directory(p, recursive=recursive, max_size=max_size, file_types=filter_types)
        )

    console.print(
        f"Found [bold]{len(all_files):,}[/bold] files across "
        f"[cyan]{len(paths)}[/cyan] path(s)"
    )
    if not all_files:
        return

    # Warn about model download on first use
    console.print(
        "[dim]Note: first run downloads the CLIP model (~350 MB). "
        "Subsequent runs use the local cache.[/dim]"
    )

    skipped = processed = errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding…", total=len(all_files))

        for info in all_files:
            progress.update(task, description=f"[cyan]{info.name[:45]}[/cyan]")

            if not force and store.is_processed(info.path, info.mtime):
                skipped += 1
                progress.advance(task)
                continue

            embedding = embedder.embed_file(info)
            if embedding is None:
                errors += 1
            else:
                store.add_file(info, embedding)
                processed += 1

            progress.advance(task)

    console.print(
        f"\n[green]{processed:,}[/green] embedded  "
        f"[yellow]{skipped:,}[/yellow] skipped  "
        f"[red]{errors:,}[/red] errors  "
        f"→ [cyan]{store.count():,}[/cyan] total in DB"
    )


# ── analyze ───────────────────────────────────────────────────────────────────

@app.command()
def analyze(
    db: str = typer.Option(".file_sorter_db", "--db"),
    similarity: float = typer.Option(0.82, help="Cluster threshold (cosine similarity 0–1)"),
    near_dup: float = typer.Option(0.97, help="Near-duplicate threshold (0–1)"),
    top_clusters: int = typer.Option(20, help="Clusters to display"),
    top_dups: int = typer.Option(30, help="Duplicate pairs to display"),
    top_orphans: int = typer.Option(40, help="Isolated files to display"),
) -> None:
    """Cluster the database and report duplicates, groups, and isolated files."""
    store = _store(db)
    n = store.count()
    if n == 0:
        console.print("[red]Database is empty. Run 'scan' first.[/red]")
        raise typer.Exit(1)

    console.print(f"Analysing [bold]{n:,}[/bold] files…")
    if n > 10_000:
        console.print(
            f"[yellow]Large collection ({n:,} files). "
            "Building the similarity matrix may take a minute.[/yellow]"
        )

    analyzer = Analyzer(store, similarity_threshold=similarity, near_dup_threshold=near_dup)
    result = analyzer.analyze()

    rpt.print_summary(result)
    rpt.print_near_duplicates(result, top_n=top_dups)
    rpt.print_clusters(result, top_n=top_clusters)
    rpt.print_orphans(result, top_n=top_orphans)


# ── similar ───────────────────────────────────────────────────────────────────

@app.command()
def similar(
    file: Path = typer.Argument(..., help="File to find neighbours for"),
    db: str = typer.Option(".file_sorter_db", "--db"),
    n: int = typer.Option(15, help="Number of results"),
    device: str = typer.Option("cpu", help="Inference device"),
) -> None:
    """Find files most similar to a given file."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    store = _store(db)
    if store.count() == 0:
        console.print("[red]Database is empty. Run 'scan' first.[/red]")
        raise typer.Exit(1)

    from .scanner import FileInfo, get_file_type
    stat = os.stat(file)
    info = FileInfo(
        path=file,
        size=stat.st_size,
        mtime=stat.st_mtime,
        file_type=get_file_type(file),
        extension=file.suffix.lower(),
    )

    embedder = Embedder(device=device)
    embedding = embedder.embed_file(info)
    if embedding is None:
        console.print("[red]Could not embed the file.[/red]")
        raise typer.Exit(1)

    results = store.get_similar(embedding, n=n, exclude_path=file)
    if not results:
        console.print("[yellow]No similar files found in the database.[/yellow]")
        return

    def fmt_size(s: int) -> str:
        for u in ('B', 'KB', 'MB', 'GB'):
            if s < 1024:
                return f"{s:.0f} {u}"
            s //= 1024
        return f"{s} TB"

    t = Table(
        title=f"Similar to [cyan]{file.name}[/cyan]",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
    )
    t.add_column("Score", justify="right", width=7)
    t.add_column("File", style="cyan", max_width=42)
    t.add_column("Type", width=9)
    t.add_column("Size", justify="right", width=9)
    t.add_column("Folder", style="dim")

    for sim in results:
        f = sim.stored
        t.add_row(
            f"{sim.score:.4f}",
            f.filename,
            f.file_type,
            fmt_size(f.size),
            str(Path(f.path).parent),
        )

    console.print(t)


# ── visualize ─────────────────────────────────────────────────────────────────

@app.command()
def visualize(
    db: str = typer.Option(".file_sorter_db", "--db"),
    output: Path = typer.Option(Path("file_map.html"), "--output", "-o"),
    n_neighbors: int = typer.Option(15, help="UMAP n_neighbors parameter"),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist parameter"),
) -> None:
    """Generate an interactive 2-D HTML similarity map (requires umap-learn + plotly)."""
    store = _store(db)
    files = store.get_all(include_embeddings=True)

    if not files:
        console.print("[red]Database is empty. Run 'scan' first.[/red]")
        raise typer.Exit(1)

    console.print(f"Building 2-D map for [bold]{len(files):,}[/bold] files…")

    try:
        from .visualize import build_html
        build_html(files, output=output, n_neighbors=n_neighbors, min_dist=min_dist)
        console.print(f"[green]Map saved:[/green] [cyan]{output.resolve()}[/cyan]")
        console.print("[dim]Open in a browser to explore interactively.[/dim]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


# ── stats ─────────────────────────────────────────────────────────────────────

@app.command()
def stats(
    db: str = typer.Option(".file_sorter_db", "--db"),
) -> None:
    """Show a breakdown of what's in the database."""
    store = _store(db)
    files = store.get_all()

    if not files:
        console.print("[yellow]Database is empty.[/yellow]")
        return

    from collections import Counter

    type_counts = Counter(f.file_type for f in files)
    type_sizes:  dict[str, int] = {}
    for f in files:
        type_sizes[f.file_type] = type_sizes.get(f.file_type, 0) + f.size

    total_size = sum(f.size for f in files)

    def fmt(s: int) -> str:
        for u in ('B', 'KB', 'MB', 'GB', 'TB'):
            if s < 1024:
                return f"{s:.1f} {u}"
            s /= 1024
        return f"{s:.1f} PB"

    t = Table(title="Database Statistics", box=box.SIMPLE, header_style="bold")
    t.add_column("File Type",  style="cyan")
    t.add_column("Count",      justify="right")
    t.add_column("% Count",    justify="right", width=9)
    t.add_column("Size",       justify="right")
    t.add_column("% Size",     justify="right", width=9)

    for ftype, count in type_counts.most_common():
        size = type_sizes[ftype]
        t.add_row(
            ftype,
            f"{count:,}",
            f"{100*count/len(files):.1f}%",
            fmt(size),
            f"{100*size/total_size:.1f}%" if total_size else "—",
        )

    t.add_section()
    t.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(files):,}[/bold]",
        "",
        f"[bold]{fmt(total_size)}[/bold]",
        "",
    )

    console.print(t)


# ── export ────────────────────────────────────────────────────────────────────

@app.command()
def export(
    db: str = typer.Option(".file_sorter_db", "--db"),
    output: Path = typer.Option(Path("file_analysis.csv"), "--output", "-o"),
    similarity: float = typer.Option(0.82, help="Cluster threshold"),
    near_dup: float = typer.Option(0.97, help="Near-duplicate threshold"),
) -> None:
    """Run analysis and export results to a CSV file."""
    store = _store(db)
    if store.count() == 0:
        console.print("[red]Database is empty. Run 'scan' first.[/red]")
        raise typer.Exit(1)

    console.print("Running analysis…")
    analyzer = Analyzer(store, similarity_threshold=similarity, near_dup_threshold=near_dup)
    result = analyzer.analyze()

    rpt.export_csv(result, output)
    console.print(f"[green]Exported:[/green] [cyan]{output.resolve()}[/cyan]")
    rpt.print_summary(result)


# ── cleanup ───────────────────────────────────────────────────────────────────

@app.command()
def cleanup(
    db: str = typer.Option(".file_sorter_db", "--db"),
) -> None:
    """Remove DB entries for files that no longer exist on disk."""
    store = _store(db)
    removed = store.remove_missing()
    console.print(
        f"[green]Removed {removed:,} stale entries.[/green] "
        f"DB now has [cyan]{store.count():,}[/cyan] files."
    )
