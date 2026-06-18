# File Similarity Scorer

Vision model + vector DB for mapping file content and similarity — helps you figure out what to keep when you're near a storage limit.

Images go through a **CLIP vision encoder**. Text files, PDFs, and code go through the **CLIP text encoder**. Both land in the same 512-dimensional space, so a vacation photo clusters near a text file called "vacation notes", and a folder full of game assets clusters together regardless of whether they're images, docs, or config files.

---

## Install

```bash
pip install -e .

# Optional: video keyframe extraction
pip install -e ".[video]"

# Optional: interactive 2-D HTML map
pip install -e ".[visualize]"
```

> **First run** downloads the CLIP ViT-B/32 model (~350 MB) and caches it locally.

---

## Usage

### 1. Scan your files

```bash
# Scan desktop
python main.py scan ~/Desktop

# Scan multiple paths recursively
python main.py scan ~/Desktop ~/Downloads ~/Documents

# Skip huge files and limit to images + documents
python main.py scan ~/Desktop --skip-larger-mb 500 --types image,document

# Use GPU for faster embedding (if available)
python main.py scan ~/Desktop --device cuda
```

The database is stored in `.file_sorter_db/` by default. Subsequent runs skip already-processed files (checked by mtime).

### 2. Analyse

```bash
python main.py analyze
```

Outputs:

- **Near-duplicates** — pairs scoring ≥ 0.97 cosine similarity, with estimated reclaimable space
- **Clusters** — groups of semantically related files (same subject, project, or context)
- **Isolated files** — files with no close neighbours; likely forgotten one-offs or misplaced items

```bash
# Tune the clustering sensitivity
python main.py analyze --similarity 0.80 --near-dup 0.95
```

### 3. Find what's similar to a specific file

```bash
python main.py similar ~/Desktop/mystery_screenshot.png
```

Useful when you find a file and can't remember why you have it — shows the 15 most similar files in the database.

### 4. Interactive 2-D map

```bash
python main.py visualize --output map.html
```

Opens as a browser page. Each dot is a file, coloured by type. Dots close together share semantic content. Zoom and hover to explore clusters.

### 5. Export to CSV

```bash
python main.py export --output analysis.csv
```

Columns: `path, filename, file_type, size_bytes, category, cluster_id, cluster_size, note`

Open in Excel/Numbers to sort and filter. Add a "keep/delete" column and work through it.

### 6. Other commands

```bash
python main.py stats           # Breakdown by file type and size
python main.py cleanup         # Remove DB entries for deleted files
python main.py --help          # Full command list
```

---

## How it works

| File type | Embedding strategy |
|-----------|-------------------|
| Image | CLIP vision encoder (pixel content) |
| Video | CLIP vision encoder on a keyframe (requires `[video]`) |
| PDF / DOCX | CLIP text encoder on extracted text |
| Code / txt / md | CLIP text encoder on file content |
| Audio / archive / binary | CLIP text encoder on filename + folder context |

All embeddings share a 512-dim space — cross-modal similarity is meaningful.

**Clustering** uses DBSCAN on the cosine distance matrix. Files within `1 - similarity_threshold` cosine distance end up in the same cluster. Files with no neighbours within that radius are labelled "isolated".

**Near-duplicate detection** is a threshold pass over the same matrix (default ≥ 0.97).

**Vector store**: ChromaDB (local, on-disk HNSW index). Incremental — only new or changed files are re-embedded on subsequent scans.

---

## Performance

| Files | Scan time (CPU) | Analysis time |
|-------|----------------|---------------|
| 1 000 | ~3 min | <1 s |
| 5 000 | ~15 min | ~5 s |
| 15 000 | ~45 min | ~60 s |

GPU (`--device cuda` or `--device mps`) speeds up the scan phase 5–10×. Analysis is CPU-bound (numpy matrix multiply).

For collections > 15 000 files the analysis works on the first 15 000 entries; bump the limit in `analyzer.py` (`_EXACT_LIMIT`) if you have more RAM.
