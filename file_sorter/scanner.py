from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

IMAGE_EXTS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
    '.tiff', '.tif', '.ico', '.heic', '.heif', '.avif',
}
VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus', '.aiff'}
DOC_EXTS   = {'.pdf', '.docx', '.doc', '.odt', '.rtf', '.pptx', '.ppt', '.xlsx', '.xls'}
CODE_EXTS  = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.sh', '.bash', '.zsh',
    '.fish', '.lua', '.r', '.m', '.sql', '.html', '.css', '.scss', '.vue',
}
TEXT_EXTS  = {'.txt', '.md', '.rst', '.csv', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.log'}
ARCHIVE_EXTS = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.dmg', '.iso', '.pkg'}

_TYPE_MAP = (
    ('image',   IMAGE_EXTS),
    ('video',   VIDEO_EXTS),
    ('audio',   AUDIO_EXTS),
    ('document', DOC_EXTS),
    ('code',    CODE_EXTS),
    ('text',    TEXT_EXTS),
    ('archive', ARCHIVE_EXTS),
)


def get_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    for type_name, exts in _TYPE_MAP:
        if ext in exts:
            return type_name
    return 'binary'


@dataclass
class FileInfo:
    path: Path
    size: int
    mtime: float
    file_type: str
    extension: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem


def scan_directory(
    root: Path | str,
    recursive: bool = True,
    min_size: int = 0,
    max_size: Optional[int] = None,
    skip_hidden: bool = True,
    file_types: Optional[set[str]] = None,
) -> Iterator[FileInfo]:
    """Yield FileInfo for every accessible file under root."""
    root = Path(root)

    if recursive:
        walker = os.walk(root)
    else:
        try:
            entries = list(os.scandir(root))
        except PermissionError:
            return
        walker = [(str(root), [], [e.name for e in entries if e.is_file()])]

    for dirpath_str, dirnames, filenames in walker:
        dirpath = Path(dirpath_str)

        if skip_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        for filename in filenames:
            if skip_hidden and filename.startswith('.'):
                continue

            filepath = dirpath / filename

            try:
                stat = filepath.stat()
            except (OSError, PermissionError):
                continue

            if not filepath.is_file():
                continue

            size = stat.st_size
            if size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue

            file_type = get_file_type(filepath)
            if file_types and file_type not in file_types:
                continue

            yield FileInfo(
                path=filepath,
                size=size,
                mtime=stat.st_mtime,
                file_type=file_type,
                extension=filepath.suffix.lower(),
            )
