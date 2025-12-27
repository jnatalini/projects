#!/usr/bin/env python3
"""
Always-on Ubuntu "agentic agent" for organizing and indexing media files.
Single-file implementation with optional external libraries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import json
import logging
import logging.handlers
import os
import queue
import re
import signal
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:
    FileSystemEventHandler = None
    Observer = None

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except Exception:
    Image = None
    TAGS = None

try:
    import mutagen
    from mutagen import File as MutagenFile
except Exception:
    mutagen = None
    MutagenFile = None

try:
    from ebooklib import epub
except Exception:
    epub = None

try:
    import pypdf
except Exception:
    pypdf = None

DEFAULT_CONFIG: Dict[str, Any] = {
    "pictures_dirs": [],
    "music_dirs": [],
    "ebooks_dirs": [],
    "other_dirs": [],
    "extensions": {
        "pictures": [".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp", ".tif", ".tiff"],
        "music": [".mp3", ".flac", ".m4a", ".wav", ".ogg"],
        "ebooks": [".epub", ".pdf", ".mobi", ".azw3"],
    },
    "organize": {
        "auto_organize": False,
        "dry_run": True,
        "require_confirm": False,
        "pictures_target": str(Path.home() / "Pictures"),
        "music_target": str(Path.home() / "Music"),
        "ebooks_target": str(Path.home() / "Books"),
        "pictures_template": "{year}/{month}/{day}/{name}",
        "music_template": "{artist}/{album}/{track:02d} - {title}{ext}",
        "ebooks_template": "{author}/{series_or_title}/{title}{ext}",
        "normalize_filenames": True,
    },
    "index": {
        "backend": "sqlite",
        "path": str(Path.home() / ".agent_index" / "agent.db"),
    },
    "reports": {
        "dir": str(Path.home() / "agent_reports"),
        "daily_time": "02:00",
    },
    "scan": {
        "interval_seconds": 300,
        "debounce_seconds": 5,
        "use_watchdog": True,
        "partial_hash_bytes": 0,
    },
    "resources": {
        "max_workers": 4,
        "max_cpu_load": 0.0,
    },
    "logging": {
        "level": "INFO",
        "file": str(Path.home() / ".agent_index" / "agent.log"),
        "audit_file": str(Path.home() / ".agent_index" / "audit.log"),
    },
}

MEDIA_TYPES = ("picture", "music", "ebook", "other")


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if not path:
        return config

    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        if not yaml:
            raise RuntimeError("PyYAML is not installed. Install with: pip install pyyaml")
        user_cfg = yaml.safe_load(text) or {}
    else:
        user_cfg = json.loads(text)
    return deep_merge(config, user_cfg)


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def setup_logging(config: Dict[str, Any], cli_log_file: Optional[str], cli_level: Optional[str]) -> None:
    log_cfg = config.get("logging", {})
    log_path = Path(cli_log_file or log_cfg.get("file", "agent.log")).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = (cli_level or log_cfg.get("level", "INFO")).upper()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    handler.setFormatter(formatter)
    handler.setLevel(getattr(logging, level, logging.INFO))
    logger.addHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(getattr(logging, level, logging.INFO))
    logger.addHandler(console)

    audit_path = Path(log_cfg.get("audit_file", "audit.log")).expanduser()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    audit_handler = logging.handlers.RotatingFileHandler(
        audit_path, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    audit_handler.setFormatter(formatter)
    audit_logger.addHandler(audit_handler)


def now_ts() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds")


def normalize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w.\- ]+", "_", name)
    return name


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_path(path: Path, config: Dict[str, Any]) -> str:
    ext = path.suffix.lower()
    ext_cfg = config.get("extensions", {})
    if ext in ext_cfg.get("pictures", []):
        return "picture"
    if ext in ext_cfg.get("music", []):
        return "music"
    if ext in ext_cfg.get("ebooks", []):
        return "ebook"
    return "other"


def file_hash(path: Path, partial_bytes: int = 0) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        if partial_bytes and path.stat().st_size > partial_bytes * 2:
            hasher.update(f.read(partial_bytes))
            f.seek(-partial_bytes, os.SEEK_END)
            hasher.update(f.read(partial_bytes))
        else:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class FileEntry:
    path: str
    media_type: str
    size: int
    mtime: float
    hash: Optional[str]


class Indexer:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                media_type TEXT,
                size INTEGER,
                mtime REAL,
                hash TEXT,
                added_timestamp TEXT,
                last_seen TEXT,
                removed INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS pictures_metadata (
                file_id INTEGER PRIMARY KEY,
                camera TEXT,
                resolution TEXT,
                date_taken TEXT,
                tags TEXT,
                FOREIGN KEY(file_id) REFERENCES files(id)
            );
            CREATE TABLE IF NOT EXISTS music_metadata (
                file_id INTEGER PRIMARY KEY,
                artist TEXT,
                album TEXT,
                title TEXT,
                track INTEGER,
                duration REAL,
                bitrate INTEGER,
                genre TEXT,
                FOREIGN KEY(file_id) REFERENCES files(id)
            );
            CREATE TABLE IF NOT EXISTS ebooks_metadata (
                file_id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                series TEXT,
                language TEXT,
                pages INTEGER,
                FOREIGN KEY(file_id) REFERENCES files(id)
            );
            CREATE TABLE IF NOT EXISTS duplicates (
                group_id INTEGER,
                file_id INTEGER,
                PRIMARY KEY (group_id, file_id)
            );
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
            CREATE INDEX IF NOT EXISTS idx_files_media ON files(media_type);
            CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
            CREATE INDEX IF NOT EXISTS idx_music_artist ON music_metadata(artist);
            CREATE INDEX IF NOT EXISTS idx_pictures_date ON pictures_metadata(date_taken);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            DELETE FROM duplicates;
            DELETE FROM pictures_metadata;
            DELETE FROM music_metadata;
            DELETE FROM ebooks_metadata;
            DELETE FROM files;
            DELETE FROM state;
            """
        )
        self.conn.commit()

    def get_file(self, path: str) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM files WHERE path = ?", (path,))
        return cur.fetchone()

    def upsert_file(self, entry: FileEntry) -> int:
        cur = self.conn.cursor()
        existing = self.get_file(entry.path)
        if existing:
            cur.execute(
                """
                UPDATE files
                SET media_type=?, size=?, mtime=?, hash=?, last_seen=?, removed=0
                WHERE path=?
                """,
                (entry.media_type, entry.size, entry.mtime, entry.hash, now_ts(), entry.path),
            )
            self.conn.commit()
            return int(existing["id"])
        cur.execute(
            """
            INSERT INTO files (path, media_type, size, mtime, hash, added_timestamp, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.path,
                entry.media_type,
                entry.size,
                entry.mtime,
                entry.hash,
                now_ts(),
                now_ts(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def mark_removed(self, path: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE files SET removed=1, last_seen=? WHERE path=?",
            (now_ts(), path),
        )
        self.conn.commit()

    def update_metadata(self, file_id: int, media_type: str, meta: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        if media_type == "picture":
            cur.execute(
                """
                INSERT OR REPLACE INTO pictures_metadata
                (file_id, camera, resolution, date_taken, tags)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    meta.get("camera"),
                    meta.get("resolution"),
                    meta.get("date_taken"),
                    meta.get("tags"),
                ),
            )
        elif media_type == "music":
            cur.execute(
                """
                INSERT OR REPLACE INTO music_metadata
                (file_id, artist, album, title, track, duration, bitrate, genre)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    meta.get("artist"),
                    meta.get("album"),
                    meta.get("title"),
                    meta.get("track"),
                    meta.get("duration"),
                    meta.get("bitrate"),
                    meta.get("genre"),
                ),
            )
        elif media_type == "ebook":
            cur.execute(
                """
                INSERT OR REPLACE INTO ebooks_metadata
                (file_id, title, author, series, language, pages)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    meta.get("title"),
                    meta.get("author"),
                    meta.get("series"),
                    meta.get("language"),
                    meta.get("pages"),
                ),
            )
        self.conn.commit()

    def update_state(self, key: str, value: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def get_state(self, key: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def query(self, sql: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()


class MetadataExtractor:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger("metadata")

    def extract(self, path: Path, media_type: str) -> Dict[str, Any]:
        try:
            if media_type == "picture":
                return self._extract_picture(path)
            if media_type == "music":
                return self._extract_music(path)
            if media_type == "ebook":
                return self._extract_ebook(path)
        except Exception:
            self.logger.exception("Metadata extraction failed for %s", path)
        return {}

    def _extract_picture(self, path: Path) -> Dict[str, Any]:
        if not Image:
            self.logger.warning("Pillow not installed; skipping image metadata for %s", path)
            return {}
        data: Dict[str, Any] = {}
        with Image.open(path) as img:
            data["resolution"] = f"{img.width}x{img.height}"
            exif = img._getexif() if hasattr(img, "_getexif") else None
            if exif and TAGS:
                exif_data = {TAGS.get(k, str(k)): v for k, v in exif.items()}
                data["camera"] = exif_data.get("Model")
                data["date_taken"] = exif_data.get("DateTimeOriginal")
        return data

    def _extract_music(self, path: Path) -> Dict[str, Any]:
        if not MutagenFile:
            self.logger.warning("mutagen not installed; skipping audio metadata for %s", path)
            return {}
        audio = MutagenFile(path)
        if not audio:
            return {}
        data: Dict[str, Any] = {}
        data["duration"] = getattr(audio.info, "length", None)
        data["bitrate"] = getattr(audio.info, "bitrate", None)
        tags = audio.tags or {}
        def tag_val(key: str) -> Optional[str]:
            val = tags.get(key)
            if isinstance(val, list):
                return str(val[0])
            if val is None:
                return None
            return str(val)
        data["artist"] = tag_val("TPE1") or tag_val("artist")
        data["album"] = tag_val("TALB") or tag_val("album")
        data["title"] = tag_val("TIT2") or tag_val("title")
        data["genre"] = tag_val("TCON") or tag_val("genre")
        track_raw = tag_val("TRCK") or tag_val("tracknumber")
        if track_raw:
            try:
                data["track"] = int(str(track_raw).split("/")[0])
            except Exception:
                data["track"] = None
        return data

    def _extract_ebook(self, path: Path) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if path.suffix.lower() == ".epub" and epub:
            book = epub.read_epub(str(path))
            data["title"] = " ".join(book.get_metadata("DC", "title")[0][0:1] or [])
            data["author"] = " ".join(book.get_metadata("DC", "creator")[0][0:1] or [])
            data["language"] = " ".join(book.get_metadata("DC", "language")[0][0:1] or [])
        elif path.suffix.lower() == ".pdf" and pypdf:
            reader = pypdf.PdfReader(str(path))
            meta = reader.metadata or {}
            data["title"] = getattr(meta, "title", None) or meta.get("/Title")
            data["author"] = getattr(meta, "author", None) or meta.get("/Author")
            data["pages"] = len(reader.pages)
        return data


class Organizer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config["organize"]
        self.logger = logging.getLogger("organize")
        self.audit = logging.getLogger("audit")

    def suggest_path(self, path: Path, media_type: str, meta: Dict[str, Any]) -> Optional[Path]:
        if media_type == "picture":
            base = Path(self.cfg["pictures_target"]).expanduser()
            template = self.cfg["pictures_template"]
            date_taken = meta.get("date_taken")
            dt_obj = None
            if date_taken:
                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    with contextlib.suppress(Exception):
                        dt_obj = dt.datetime.strptime(date_taken, fmt)
            if not dt_obj:
                dt_obj = dt.datetime.fromtimestamp(path.stat().st_mtime)
            mapping = {
                "year": f"{dt_obj.year:04d}",
                "month": f"{dt_obj.month:02d}",
                "day": f"{dt_obj.day:02d}",
                "name": path.stem,
                "ext": path.suffix.lower(),
            }
            return base / template.format(**mapping)
        if media_type == "music":
            base = Path(self.cfg["music_target"]).expanduser()
            template = self.cfg["music_template"]
            mapping = {
                "artist": meta.get("artist") or "Unknown Artist",
                "album": meta.get("album") or "Unknown Album",
                "title": meta.get("title") or path.stem,
                "track": meta.get("track") or 0,
                "ext": path.suffix.lower(),
            }
            return base / template.format(**mapping)
        if media_type == "ebook":
            base = Path(self.cfg["ebooks_target"]).expanduser()
            template = self.cfg["ebooks_template"]
            series_or_title = meta.get("series") or meta.get("title") or path.stem
            mapping = {
                "author": meta.get("author") or "Unknown Author",
                "series_or_title": series_or_title,
                "title": meta.get("title") or path.stem,
                "ext": path.suffix.lower(),
            }
            return base / template.format(**mapping)
        return None

    def maybe_move(self, path: Path, target: Path, dry_run: bool) -> Optional[Path]:
        if path.resolve() == target.resolve():
            return None
        target_parent = target.parent
        safe_mkdir(target_parent)
        final_target = target
        if self.cfg.get("normalize_filenames", True):
            final_target = final_target.with_name(normalize_filename(final_target.name))
        if final_target.exists():
            stem = final_target.stem
            suffix = final_target.suffix
            for i in range(1, 1000):
                candidate = final_target.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    final_target = candidate
                    break
        if dry_run:
            self.logger.info("Dry-run move: %s -> %s", path, final_target)
            return final_target
        try:
            path.rename(final_target)
            self.audit.info("MOVE %s -> %s", path, final_target)
            return final_target
        except Exception:
            self.logger.exception("Failed to move %s -> %s", path, final_target)
            return None


class Reporter:
    def __init__(self, indexer: Indexer, config: Dict[str, Any]) -> None:
        self.indexer = indexer
        self.cfg = config["reports"]
        self.logger = logging.getLogger("report")

    def _report_path(self, name: str) -> Path:
        report_dir = Path(self.cfg["dir"]).expanduser()
        safe_mkdir(report_dir)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return report_dir / f"{name}_{ts}.txt"

    def generate_all(self) -> List[Path]:
        paths = [
            self.generate_new_content_report(),
            self.generate_storage_report(),
            self.generate_duplicates_report(),
            self.generate_missing_metadata_report(),
        ]
        self.logger.info("Reports generated: %s", ", ".join(str(p) for p in paths if p))
        return [p for p in paths if p]

    def generate_new_content_report(self) -> Path:
        report_path = self._report_path("new_content")
        yesterday = (dt.datetime.utcnow() - dt.timedelta(days=1)).isoformat(timespec="seconds")
        rows = self.indexer.query(
            """
            SELECT media_type, COUNT(*) as count
            FROM files
            WHERE added_timestamp >= ? AND removed=0
            GROUP BY media_type
            """,
            (yesterday,),
        )
        with report_path.open("w", encoding="utf-8") as f:
            f.write("New content report (last 24h)\n")
            for row in rows:
                f.write(f"{row['media_type']}: {row['count']}\n")
        return report_path

    def generate_storage_report(self) -> Path:
        report_path = self._report_path("storage_usage")
        rows = self.indexer.query(
            """
            SELECT media_type, SUM(size) as total_size
            FROM files
            WHERE removed=0
            GROUP BY media_type
            """
        )
        with report_path.open("w", encoding="utf-8") as f:
            f.write("Storage usage report\n")
            for row in rows:
                f.write(f"{row['media_type']}: {row['total_size'] or 0} bytes\n")
        return report_path

    def generate_duplicates_report(self) -> Path:
        report_path = self._report_path("duplicates")
        rows = self.indexer.query(
            """
            SELECT hash, COUNT(*) as count
            FROM files
            WHERE hash IS NOT NULL AND removed=0
            GROUP BY hash
            HAVING count > 1
            """
        )
        with report_path.open("w", encoding="utf-8") as f:
            f.write("Duplicates report\n")
            for row in rows:
                f.write(f"Hash {row['hash']} has {row['count']} duplicates\n")
        return report_path

    def generate_missing_metadata_report(self) -> Path:
        report_path = self._report_path("missing_metadata")
        rows = self.indexer.query(
            """
            SELECT f.path, f.media_type
            FROM files f
            LEFT JOIN music_metadata m ON f.id = m.file_id
            LEFT JOIN pictures_metadata p ON f.id = p.file_id
            LEFT JOIN ebooks_metadata e ON f.id = e.file_id
            WHERE f.removed=0 AND (
                (f.media_type='music' AND (m.artist IS NULL OR m.title IS NULL)) OR
                (f.media_type='picture' AND (p.date_taken IS NULL)) OR
                (f.media_type='ebook' AND (e.title IS NULL OR e.author IS NULL))
            )
            """
        )
        with report_path.open("w", encoding="utf-8") as f:
            f.write("Missing metadata report\n")
            for row in rows:
                f.write(f"{row['media_type']}: {row['path']}\n")
        return report_path


class FileWatcher:
    def __init__(self, paths: List[Path], queue: queue.Queue, config: Dict[str, Any]) -> None:
        self.paths = paths
        self.queue = queue
        self.cfg = config["scan"]
        self.logger = logging.getLogger("watcher")
        self.observer: Optional[Any] = None

    def start(self) -> None:
        if not self.cfg.get("use_watchdog", True) or not Observer or not FileSystemEventHandler:
            self.logger.info("Watchdog disabled or unavailable.")
            return
        handler = self._build_handler()
        self.observer = Observer()
        for path in self.paths:
            if path.exists():
                self.observer.schedule(handler, str(path), recursive=True)
        self.observer.start()
        self.logger.info("Watchdog started for %d paths", len(self.paths))

    def stop(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)

    def _build_handler(self) -> FileSystemEventHandler:
        queue_ref = self.queue
        logger = self.logger

        class Handler(FileSystemEventHandler):
            def on_created(self, event) -> None:
                if not event.is_directory:
                    queue_ref.put(("created", event.src_path))

            def on_modified(self, event) -> None:
                if not event.is_directory:
                    queue_ref.put(("modified", event.src_path))

            def on_moved(self, event) -> None:
                if not event.is_directory:
                    queue_ref.put(("moved", event.dest_path))

            def on_deleted(self, event) -> None:
                if not event.is_directory:
                    queue_ref.put(("deleted", event.src_path))
                    logger.info("Deleted: %s", event.src_path)

        return Handler()


class Agent:
    def __init__(self, config: Dict[str, Any], once: bool = False, reindex: bool = False) -> None:
        self.config = config
        self.once = once
        self.reindex = reindex
        self.logger = logging.getLogger("agent")
        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.indexer = Indexer(config["index"]["path"])
        self.extractor = MetadataExtractor(config)
        self.organizer = Organizer(config)
        self.reporter = Reporter(self.indexer, config)
        self.watch_paths = self._gather_watch_paths()
        self.watcher = FileWatcher(self.watch_paths, self.queue, config)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config["resources"].get("max_workers", 4)
        )

    def _gather_watch_paths(self) -> List[Path]:
        paths = []
        for key in ("pictures_dirs", "music_dirs", "ebooks_dirs", "other_dirs"):
            for entry in self.config.get(key, []):
                paths.append(Path(entry).expanduser())
        return paths

    def start(self) -> None:
        self.logger.info("Agent starting.")
        if self.reindex:
            self.logger.info("Reindex requested; resetting index.")
            self.indexer.reset()
        self._install_signals()
        self.watcher.start()
        if self.once:
            self.scan_all()
            self.reporter.generate_all()
            self.shutdown()
            return
        self.run_loop()

    def shutdown(self) -> None:
        self.logger.info("Agent shutting down.")
        self.stop_event.set()
        self.watcher.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.indexer.close()

    def _install_signals(self) -> None:
        def handler(signum, frame) -> None:
            self.logger.info("Received signal %s; shutting down.", signum)
            self.shutdown()
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def run_loop(self) -> None:
        scan_interval = self.config["scan"].get("interval_seconds", 300)
        next_scan = time.time()
        next_report = self._next_daily_report_time()
        while not self.stop_event.is_set():
            self._drain_queue()
            now = time.time()
            if now >= next_scan:
                self.scan_all()
                next_scan = now + scan_interval
            if now >= next_report:
                self.reporter.generate_all()
                next_report = self._next_daily_report_time()
            time.sleep(1)

    def _next_daily_report_time(self) -> float:
        daily_time = self.config["reports"].get("daily_time", "02:00")
        hour, minute = [int(x) for x in daily_time.split(":")]
        now = dt.datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        return target.timestamp()

    def _drain_queue(self) -> None:
        while True:
            try:
                event, path = self.queue.get_nowait()
            except queue.Empty:
                break
            if event == "deleted":
                self.indexer.mark_removed(path)
                continue
            self.executor.submit(self.process_path, Path(path))

    def scan_all(self) -> None:
        self.logger.info("Scanning watched directories.")
        for root in self.watch_paths:
            if not root.exists():
                self.logger.warning("Watched directory missing: %s", root)
                continue
            for path in self._walk(root):
                self.executor.submit(self.process_path, path)
        self.indexer.update_state("last_scan", now_ts())

    def _walk(self, root: Path) -> Iterable[Path]:
        for entry in os.scandir(root):
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                yield from self._walk(path)
            elif entry.is_file(follow_symlinks=False):
                yield path

    def process_path(self, path: Path) -> None:
        debounce = self.config["scan"].get("debounce_seconds", 5)
        if not self._wait_for_stable(path, debounce):
            return
        if not path.exists():
            self.indexer.mark_removed(str(path))
            return
        media_type = classify_path(path, self.config)
        try:
            stat = path.stat()
        except Exception:
            self.logger.exception("Failed to stat %s", path)
            return
        existing = self.indexer.get_file(str(path))
        if existing and existing["size"] == stat.st_size and existing["mtime"] == stat.st_mtime:
            self.indexer.update_state("last_seen", now_ts())
            return
        file_hash_val = None
        try:
            file_hash_val = file_hash(path, self.config["scan"].get("partial_hash_bytes", 0))
        except Exception:
            self.logger.exception("Hashing failed for %s", path)
        entry = FileEntry(
            path=str(path),
            media_type=media_type,
            size=stat.st_size,
            mtime=stat.st_mtime,
            hash=file_hash_val,
        )
        file_id = self.indexer.upsert_file(entry)
        meta = self.extractor.extract(path, media_type)
        if meta:
            self.indexer.update_metadata(file_id, media_type, meta)
        if media_type in ("picture", "music", "ebook"):
            self._maybe_organize(path, media_type, meta)

    def _wait_for_stable(self, path: Path, debounce_seconds: int) -> bool:
        if debounce_seconds <= 0:
            return True
        try:
            prev = path.stat().st_size
        except Exception:
            return False
        time.sleep(debounce_seconds)
        try:
            return path.exists() and path.stat().st_size == prev
        except Exception:
            return False

    def _maybe_organize(self, path: Path, media_type: str, meta: Dict[str, Any]) -> None:
        cfg = self.config["organize"]
        if not cfg.get("auto_organize", False):
            return
        target = self.organizer.suggest_path(path, media_type, meta)
        if not target:
            return
        dry_run = bool(cfg.get("dry_run", True))
        moved = self.organizer.maybe_move(path, target, dry_run=dry_run)
        if moved and not dry_run:
            self.indexer.mark_removed(str(path))
            self.process_path(moved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Always-on media organizer agent.")
    parser.add_argument("--config", help="Path to config YAML/JSON", default=None)
    parser.add_argument("--once", help="Run one scan and exit", action="store_true")
    parser.add_argument("--reindex", help="Force full reindex", action="store_true")
    parser.add_argument("--log-file", help="Override log file path")
    parser.add_argument("--log-level", help="Logging level")
    parser.add_argument("--list", dest="list_media", help="List media type entries")
    parser.add_argument("--since", help="List entries since ISO date (YYYY-MM-DD)")
    parser.add_argument("--search", help="Search query like 'artist=Beatles'")
    parser.add_argument("--report", action="store_true", help="Generate reports and exit")
    return parser.parse_args()


def run_list(indexer: Indexer, media_type: str, since: Optional[str]) -> None:
    params: List[Any] = [media_type]
    sql = "SELECT path, added_timestamp FROM files WHERE media_type=? AND removed=0"
    if since:
        sql += " AND added_timestamp >= ?"
        params.append(since)
    sql += " ORDER BY added_timestamp DESC LIMIT 200"
    rows = indexer.query(sql, tuple(params))
    for row in rows:
        print(f"{row['added_timestamp']} {row['path']}")


def run_search(indexer: Indexer, query: str) -> None:
    key, _, value = query.partition("=")
    key = key.strip().lower()
    value = value.strip()
    if key in {"artist", "album", "title", "genre"}:
        rows = indexer.query(
            """
            SELECT f.path, m.artist, m.album, m.title
            FROM files f
            JOIN music_metadata m ON f.id = m.file_id
            WHERE m.{} LIKE ? AND f.removed=0
            """.format(key),
            (f"%{value}%",),
        )
    elif key in {"author"}:
        rows = indexer.query(
            """
            SELECT f.path, e.author, e.title
            FROM files f
            JOIN ebooks_metadata e ON f.id = e.file_id
            WHERE e.author LIKE ? AND f.removed=0
            """,
            (f"%{value}%",),
        )
    else:
        rows = indexer.query(
            "SELECT path FROM files WHERE path LIKE ? AND removed=0",
            (f"%{value}%",),
        )
    for row in rows:
        print(row["path"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config, args.log_file, args.log_level)
    logger = logging.getLogger("main")
    logger.info("Starting agent with config: %s", args.config or "default")

    agent = Agent(config, once=args.once, reindex=args.reindex)
    if args.list_media:
        run_list(agent.indexer, args.list_media, args.since)
        return
    if args.search:
        run_search(agent.indexer, args.search)
        return
    if args.report:
        agent.reporter.generate_all()
        return
    agent.start()


if __name__ == "__main__":
    main()
