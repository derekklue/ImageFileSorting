"""Interactive 2-D similarity map using UMAP + Plotly.

Install optional deps first:
    pip install "file-sorter[visualize]"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .store import StoredFile

logger = logging.getLogger(__name__)

_TYPE_COLORS = {
    'image':    '#4CAF50',
    'video':    '#2196F3',
    'audio':    '#9C27B0',
    'document': '#FF9800',
    'code':     '#00BCD4',
    'text':     '#8BC34A',
    'archive':  '#795548',
    'binary':   '#9E9E9E',
}


def build_html(
    files: list[StoredFile],
    output: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    try:
        import umap
    except ImportError:
        raise RuntimeError(
            "umap-learn is not installed. Run: pip install 'file-sorter[visualize]'"
        )
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise RuntimeError(
            "plotly is not installed. Run: pip install 'file-sorter[visualize]'"
        )

    with_emb = [f for f in files if f.embedding is not None]
    if len(with_emb) < 4:
        raise RuntimeError("Need at least 4 embedded files to generate a map.")

    emb = np.array([f.embedding for f in with_emb], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-8)

    logger.info("Running UMAP on %d files…", len(with_emb))
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(with_emb) - 1),
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
    )
    coords = reducer.fit_transform(emb)

    # Group by file type for separate traces
    type_groups: dict[str, list[int]] = {}
    for i, f in enumerate(with_emb):
        type_groups.setdefault(f.file_type, []).append(i)

    traces = []
    for ftype, indices in sorted(type_groups.items()):
        fs = [with_emb[i] for i in indices]
        xs = coords[indices, 0]
        ys = coords[indices, 1]

        hover = [
            f"<b>{f.filename}</b><br>"
            f"Type: {f.file_type}<br>"
            f"Size: {f.size / 1024:.1f} KB<br>"
            f"Path: {Path(f.path).parent}"
            for f in fs
        ]

        traces.append(go.Scatter(
            x=xs, y=ys,
            mode='markers',
            name=ftype,
            marker=dict(
                size=7,
                color=_TYPE_COLORS.get(ftype, '#607D8B'),
                opacity=0.8,
                line=dict(width=0),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            customdata=[f.path for f in fs],
        ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text="File Similarity Map", font=dict(size=18)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            hovermode='closest',
            legend=dict(title='File type'),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#1a1a2e',
            font=dict(color='#e0e0e0'),
            margin=dict(l=20, r=20, t=50, b=20),
        ),
    )

    fig.write_html(
        str(output),
        include_plotlyjs='cdn',
        full_html=True,
        config={'scrollZoom': True, 'displayModeBar': True},
    )
    logger.info("Map written to %s", output)
