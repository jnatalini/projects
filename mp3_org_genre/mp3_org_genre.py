#!/usr/bin/env python3
"""
MP3 library organizer by genre with local inference.

Dependencies:
  - mutagen

Install:
  pip install mutagen
"""

import argparse
import concurrent.futures as futures
import csv
import json
import logging
import os
import queue
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from mutagen.id3 import ID3, ID3NoHeaderError, TPE1, TIT2, TALB, TCON, TRCK, TDRC
from mutagen.mp3 import MP3


GENRE_BUCKETS = [
    "Pop",
    "Rock",
    "Hip-Hop",
    "Electronic",
    "R&B",
    "Country",
    "Jazz",
    "Classical",
    "Reggae",
    "Latin",
    "Middle Eastern / International",
]

GENRE_NORMALIZATION = {
    "hip hop": "Hip-Hop",
    "hip-hop": "Hip-Hop",
    "rap": "Hip-Hop",
    "electronic": "Electronic",
    "edm": "Electronic",
    "dance": "Electronic",
    "house": "Electronic",
    "techno": "Electronic",
    "trance": "Electronic",
    "r&b": "R&B",
    "rnb": "R&B",
    "soul": "R&B",
    "classical": "Classical",
    "orchestral": "Classical",
    "symphonic": "Classical",
    "latin": "Latin",
    "reggaeton": "Latin",
    "salsa": "Latin",
    "bachata": "Latin",
    "reggae": "Reggae",
    "rock": "Rock",
    "alternative": "Rock",
    "indie": "Rock",
    "hard rock": "Rock",
    "pop": "Pop",
    "country": "Country",
    "jazz": "Jazz",
    "middle eastern": "Middle Eastern / International",
    "arabic": "Middle Eastern / International",
    "persian": "Middle Eastern / International",
    "turkish": "Middle Eastern / International",
    "north african": "Middle Eastern / International",
    "world": "Middle Eastern / International",
    "international": "Middle Eastern / International",
}

GENRE_BLACKLIST = {
    "music",
    "entertainment",
    "unknown",
    "unclassified",
    "other",
    "misc",
    "miscellaneous",
    "na",
    "n/a",
    "none",
}

ILLEGAL_PATH_CHARS = r'\/:*?"<>|'


@dataclass
class TrackInfo:
    path: str
    artist: Optional[str] = None
    title: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    track: Optional[str] = None
    year: Optional[str] = None
    guessed: bool = False


@dataclass
class LookupResult:
    genre: Optional[str]
    provider: Optional[str]
    confidence: float
    source_url: Optional[str]
    timestamp: float


def load_mapping(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        logging.warning("Mapping file not found: %s", path)
        return {}
    _, ext = os.path.splitext(path)
    try:
        if ext.lower() in {".json"}:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            mapping = {}
            for k, v in data.items():
                key = normalize_key(str(k))
                genre = str(v).strip() if v is not None else ""
                if key and genre:
                    mapping[key] = genre
            return mapping
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            mapping = {}
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2:
                    key = normalize_key(row[0])
                    genre = row[1].strip()
                    if key and genre:
                        mapping[key] = genre
            return mapping
    except Exception as exc:
        logging.warning("Failed to load mapping file %s: %s", path, exc)
        return {}


def load_mapping_keys(path: Optional[str]) -> set:
    if not path:
        return set()
    if not os.path.exists(path):
        return set()
    _, ext = os.path.splitext(path)
    keys = set()
    try:
        if ext.lower() in {".json"}:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key in data.keys():
                norm = normalize_key(str(key))
                if norm:
                    keys.add(norm)
            return keys
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                key = normalize_key(row[0]) if row[0] else ""
                if key:
                    keys.add(key)
        return keys
    except Exception as exc:
        logging.warning("Failed to load mapping keys from %s: %s", path, exc)
        return set()


def append_missing_artist(path: str, artist: str):
    if not path or not artist:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([artist, ""])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize MP3s by genre/artist/album.")
    parser.add_argument("inputs", nargs="+", help="Input files or folders")
    parser.add_argument(
        "--target-root",
        default=None,
        help="Target library root (default: use each file's source directory)",
    )
    parser.add_argument(
        "--target-root-mode",
        choices=["source", "fixed"],
        default="source",
        help="Use each file's source directory or a fixed --target-root",
    )
    parser.add_argument("--mode", choices=["move", "copy"], default="move")
    parser.add_argument("--conflict", choices=["skip", "overwrite", "rename"], default="rename")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument(
        "--artist-genre-map",
        default=None,
        help="CSV or JSON map of artist->genre",
    )
    parser.add_argument(
        "--album-genre-map",
        default=None,
        help="CSV or JSON map of album->genre",
    )
    parser.add_argument(
        "--title-genre-map",
        default=None,
        help="CSV or JSON map of title->genre",
    )
    parser.add_argument(
        "--keyword-genre-map",
        default=None,
        help="CSV or JSON map of keyword->genre (matched against filename/path)",
    )
    parser.add_argument(
        "--learn-map",
        default=None,
        help="CSV file to append learned artist->genre mappings (interactive)",
    )
    parser.add_argument(
        "--learn-scope",
        choices=["artist", "album", "title"],
        default="artist",
        help="Which field to learn when prompting (default: artist)",
    )
    parser.add_argument("--id3-version", choices=["3", "4"], default="3")
    parser.add_argument("--rename-pattern", choices=["original", "track-title", "artist-title"], default="original")
    parser.add_argument("--unknown-artist", default="Unknown Artist")
    parser.add_argument("--unknown-album", default="Unknown Album")
    parser.add_argument("--unknown-genre", default="Unknown Genre")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--report-file", default="mp3_org_report.csv")
    parser.add_argument("--review-file", default="mp3_org_review.json")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--mismatch-policy", choices=["keep", "replace", "both", "flag"], default="flag")
    parser.add_argument("--confirm-existing", dest="confirm_existing", action="store_true")
    parser.add_argument("--no-confirm-existing", dest="confirm_existing", action="store_false")
    parser.set_defaults(confirm_existing=True)
    return parser.parse_args()


def configure_logging(log_file: Optional[str]):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def is_hidden(path: str) -> bool:
    return any(part.startswith(".") for part in path.split(os.sep))


def discover_files(inputs: List[str], skip_hidden: bool) -> List[str]:
    files = []
    for inp in inputs:
        if os.path.isfile(inp):
            if inp.lower().endswith(".mp3"):
                if not (skip_hidden and is_hidden(inp)):
                    files.append(os.path.abspath(inp))
            continue
        for root, dirs, filenames in os.walk(inp):
            if skip_hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in filenames:
                if name.lower().endswith(".mp3"):
                    full = os.path.join(root, name)
                    if skip_hidden and is_hidden(full):
                        continue
                    files.append(os.path.abspath(full))
    return files


def normalize_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = unicodedata.normalize("NFKC", value)
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace(" feat. ", " ft. ")
    return value


def normalize_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = normalize_text(value) or ""
    value = re.sub(r"\b(feat|featuring|ft|with)\b.*", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"[^\w\s]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value or None


def sanitize_path_component(value: str) -> str:
    cleaned = "".join("_" if c in ILLEGAL_PATH_CHARS else c for c in value)
    cleaned = cleaned.strip().strip(".")
    return cleaned or "_"


def parse_filename(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"^(?P<artist>.+?)\s*-\s*(?P<title>.+)$", name)
    if match:
        return match.group("artist"), match.group("title"), None
    match = re.match(r"^\d+\s*-\s*(?P<title>.+)$", name)
    if match:
        return None, match.group("title"), None
    return None, name, None


def read_tags(path: str) -> Optional[TrackInfo]:
    try:
        audio = MP3(path)
    except Exception:
        logging.warning("Unreadable MP3: %s", path)
        return None
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    info = TrackInfo(path=path)
    info.artist = tags.get("TPE1").text[0] if tags.get("TPE1") else None
    info.title = tags.get("TIT2").text[0] if tags.get("TIT2") else None
    info.album = tags.get("TALB").text[0] if tags.get("TALB") else None
    info.genre = tags.get("TCON").text[0] if tags.get("TCON") else None
    info.track = tags.get("TRCK").text[0] if tags.get("TRCK") else None
    info.year = tags.get("TDRC").text[0] if tags.get("TDRC") else None
    info.artist = normalize_text(info.artist)
    info.title = normalize_text(info.title)
    info.album = normalize_text(info.album)
    info.genre = normalize_text(info.genre)
    if not (info.artist and info.title and info.album):
        artist, title, album = parse_filename(path)
        if not info.artist and artist:
            info.artist = normalize_text(artist)
            info.guessed = True
        if not info.title and title:
            info.title = normalize_text(title)
            info.guessed = True
        if not info.album and album:
            info.album = normalize_text(album)
            info.guessed = True
    return info


def normalize_genre(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = value.strip().lower()
    if key in GENRE_BLACKLIST:
        return None
    return GENRE_NORMALIZATION.get(key, value)


def map_genre_bucket(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = normalize_genre(value)
    if not normalized:
        return None
    key = normalized.lower()
    return GENRE_NORMALIZATION.get(key, normalized)


def extract_candidates_from_path(path: str) -> List[str]:
    parts = [p for p in path.split(os.sep) if p]
    recent = parts[-4:]
    candidates = []
    for part in recent:
        cleaned = normalize_text(part)
        if cleaned:
            candidates.append(cleaned)
    return candidates


def infer_genre_local(
    info: TrackInfo,
    args: argparse.Namespace,
    artist_map: Dict[str, str],
    album_map: Dict[str, str],
    title_map: Dict[str, str],
    keyword_map: Dict[str, str],
) -> LookupResult:
    candidates: List[Tuple[str, float, str]] = []
    if info.artist:
        mapped = artist_map.get(normalize_key(info.artist) or "")
        if mapped:
            return LookupResult(
                genre=mapped,
                provider="artist-map",
                confidence=0.98,
                source_url=None,
                timestamp=time.time(),
            )
    if info.genre:
        normalized_tag = normalize_genre(info.genre)
        if normalized_tag:
            candidates.append((normalized_tag, 0.9, "tag"))
    if info.album:
        mapped = album_map.get(normalize_key(info.album) or "")
        if mapped:
            candidates.append((mapped, 0.9, "album-map"))
    if info.title:
        mapped = title_map.get(normalize_key(info.title) or "")
        if mapped:
            candidates.append((mapped, 0.85, "title-map"))

    path_candidates = extract_candidates_from_path(info.path)
    for value in path_candidates:
        normalized = normalize_genre(value)
        if normalized in GENRE_BUCKETS:
            candidates.append((normalized, 0.7, "path"))
        key = value.lower()
        if key in keyword_map:
            candidates.append((keyword_map[key], 0.65, "keyword-map"))
        for keyword, genre in keyword_map.items():
            if keyword and keyword in key:
                candidates.append((genre, 0.6, "keyword-match"))

    for key, genre in GENRE_NORMALIZATION.items():
        if key in (info.title or "").lower() or key in (info.artist or "").lower() or key in (info.album or "").lower():
            candidates.append((genre, 0.55, "keyword-builtin"))

    if not candidates:
        return LookupResult(genre=None, provider=None, confidence=0.0, source_url=None, timestamp=time.time())

    candidates = [
        (map_genre_bucket(genre) or genre, confidence, source)
        for genre, confidence, source in candidates
        if genre
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_genre, best_conf, best_source = candidates[0]
    if best_conf < args.min_confidence:
        return LookupResult(genre=None, provider=None, confidence=best_conf, source_url=None, timestamp=time.time())
    return LookupResult(
        genre=best_genre,
        provider=best_source,
        confidence=best_conf,
        source_url=None,
        timestamp=time.time(),
    )


def prompt_learn_genre(
    info: TrackInfo,
    scope: str,
    args: argparse.Namespace,
    artist_map: Dict[str, str],
    album_map: Dict[str, str],
    title_map: Dict[str, str],
) -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    label = None
    key = None
    if scope == "artist" and info.artist:
        label = f"artist '{info.artist}'"
        key = info.artist.lower()
    elif scope == "album" and info.album:
        label = f"album '{info.album}'"
        key = info.album.lower()
    elif scope == "title" and info.title:
        label = f"title '{info.title}'"
        key = info.title.lower()
    if not key or not label:
        return None

    prompt = (
        f"Learn genre for {label}? "
        f"Enter one of {', '.join(GENRE_BUCKETS)} or leave blank to skip: "
    )
    try:
        response = input(prompt).strip()
    except EOFError:
        return None
    if not response:
        return None
    normalized = map_genre_bucket(response)
    if not normalized:
        print("Unrecognized genre, skipping.")
        return None

    if scope == "artist":
        artist_map[key] = normalized
    elif scope == "album":
        album_map[key] = normalized
    else:
        title_map[key] = normalized
    return normalized


def append_mapping(path: str, key: str, genre: str):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([key, genre])


def update_tags(path: str, info: TrackInfo, genre: Optional[str], id3_version: str, dry_run: bool) -> bool:
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    updated = False
    if info.artist and (not tags.get("TPE1") or tags["TPE1"].text[0] != info.artist):
        tags.setall("TPE1", [TPE1(encoding=3, text=info.artist)])
        updated = True
    if info.title and (not tags.get("TIT2") or tags["TIT2"].text[0] != info.title):
        tags.setall("TIT2", [TIT2(encoding=3, text=info.title)])
        updated = True
    if info.album and (not tags.get("TALB") or tags["TALB"].text[0] != info.album):
        tags.setall("TALB", [TALB(encoding=3, text=info.album)])
        updated = True
    if info.track and (not tags.get("TRCK") or tags["TRCK"].text[0] != info.track):
        tags.setall("TRCK", [TRCK(encoding=3, text=info.track)])
        updated = True
    if info.year and (not tags.get("TDRC") or tags["TDRC"].text[0] != info.year):
        tags.setall("TDRC", [TDRC(encoding=3, text=info.year)])
        updated = True
    if genre and (not tags.get("TCON") or tags["TCON"].text[0] != genre):
        tags.setall("TCON", [TCON(encoding=3, text=genre)])
        updated = True
    if updated and not dry_run:
        tags.save(path, v2_version=3 if id3_version == "3" else 4)
    return updated


def build_target_path(
    info: TrackInfo,
    genre: Optional[str],
    args: argparse.Namespace,
    target_root: str,
) -> str:
    artist = sanitize_path_component(info.artist or args.unknown_artist)
    album = sanitize_path_component(info.album or args.unknown_album)
    genre_folder = sanitize_path_component(genre or args.unknown_genre)
    filename = os.path.basename(info.path)
    if args.rename_pattern == "track-title" and info.title:
        track_prefix = f"{info.track} - " if info.track else ""
        filename = f"{track_prefix}{info.title}.mp3"
    elif args.rename_pattern == "artist-title" and info.artist and info.title:
        filename = f"{info.artist} - {info.title}.mp3"
    filename = sanitize_path_component(filename)
    return os.path.join(target_root, genre_folder, artist, album, filename)


def resolve_source_root(path: str, input_roots: List[str]) -> str:
    matches = [root for root in input_roots if path.startswith(root + os.sep) or path == root]
    if not matches:
        return os.path.dirname(path)
    return max(matches, key=len)


def resolve_conflict(path: str, policy: str) -> Optional[str]:
    if not os.path.exists(path):
        return path
    if policy == "skip":
        return None
    if policy == "overwrite":
        return path
    if policy == "rename":
        base, ext = os.path.splitext(path)
        for idx in range(1, 1000):
            candidate = f"{base} ({idx}){ext}"
            if not os.path.exists(candidate):
                return candidate
    return None


def move_or_copy(src: str, dst: str, mode: str, dry_run: bool):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if dry_run:
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def process_file(
    info: TrackInfo,
    args: argparse.Namespace,
    artist_map: Dict[str, str],
    artist_keys: set,
    album_map: Dict[str, str],
    title_map: Dict[str, str],
    keyword_map: Dict[str, str],
    input_roots: List[str],
    report_queue: queue.Queue,
    review_queue: queue.Queue,
):
    if not info:
        return
    lookup = None
    artist_key = normalize_key(info.artist) if info.artist else None
    has_artist_mapping = bool(artist_key and artist_map.get(artist_key))
    needs_lookup = (not info.genre) or args.confirm_existing or info.guessed or has_artist_mapping
    if needs_lookup:
        lookup = infer_genre_local(info, args, artist_map, album_map, title_map, keyword_map)

    final_genre = info.genre
    mismatch_note = None
    if (not lookup or not lookup.genre) and info.artist and args.artist_genre_map:
        artist_key = normalize_key(info.artist) or info.artist.lower()
        if artist_key not in artist_keys:
            append_missing_artist(args.artist_genre_map, info.artist)
            artist_map[artist_key] = ""
            artist_keys.add(artist_key)
    if not lookup or not lookup.genre:
        if args.learn_map:
            learned = prompt_learn_genre(info, args.learn_scope, args, artist_map, album_map, title_map)
            if learned:
                lookup = LookupResult(
                    genre=learned,
                    provider=f"{args.learn_scope}-learned",
                    confidence=0.95,
                    source_url=None,
                    timestamp=time.time(),
                )
                key_value = None
                if args.learn_scope == "artist" and info.artist:
                    key_value = info.artist.lower()
                elif args.learn_scope == "album" and info.album:
                    key_value = info.album.lower()
                elif args.learn_scope == "title" and info.title:
                    key_value = info.title.lower()
                if key_value:
                    append_mapping(args.learn_map, key_value, learned)
    if lookup and lookup.genre:
        inferred_genre = normalize_genre(lookup.genre)
        local_genre = normalize_genre(info.genre)
        if not local_genre:
            final_genre = inferred_genre
        elif local_genre and inferred_genre and local_genre.lower() != inferred_genre.lower():
            if args.mismatch_policy == "replace":
                final_genre = inferred_genre
                mismatch_note = "replaced-local"
            elif args.mismatch_policy == "both":
                final_genre = f"{local_genre}; {inferred_genre}"
                mismatch_note = "kept-both"
            elif args.mismatch_policy == "flag":
                mismatch_note = "mismatch-flag"
            else:
                final_genre = local_genre
        else:
            final_genre = local_genre or inferred_genre

    final_genre = map_genre_bucket(final_genre) or final_genre
    updated = update_tags(info.path, info, final_genre, args.id3_version, args.dry_run)
    if args.target_root_mode == "fixed":
        target_root = os.path.expanduser(args.target_root or "~")
    else:
        target_root = resolve_source_root(info.path, input_roots)
    target = build_target_path(info, final_genre, args, target_root)
    should_move = True
    if os.path.exists(target):
        try:
            if os.path.samefile(info.path, target):
                final_target = target
                should_move = False
            else:
                final_target = resolve_conflict(target, args.conflict)
        except OSError:
            final_target = resolve_conflict(target, args.conflict)
    else:
        final_target = target
    moved = False
    if final_target and should_move:
        move_or_copy(info.path, final_target, args.mode, args.dry_run)
        moved = True
    else:
        mismatch_note = (mismatch_note or "") + " collision-skip"

    report_queue.put(
        {
            "original_path": info.path,
            "new_path": final_target or "",
            "artist": info.artist or "",
            "title": info.title or "",
            "album": info.album or "",
            "old_genre": info.genre or "",
            "new_genre": final_genre or "",
            "source": lookup.provider if lookup else "",
            "confidence": lookup.confidence if lookup else "",
            "notes": mismatch_note or "",
            "updated_tags": updated,
            "moved": moved,
        }
    )

    if mismatch_note:
        review_queue.put(
            {
                "path": info.path,
                "local_genre": info.genre,
                "inferred_genre": lookup.genre if lookup else None,
                "note": mismatch_note,
                "source": lookup.provider if lookup else None,
                "confidence": lookup.confidence if lookup else None,
            }
        )


def write_report(report_path: str, report_rows: List[Dict[str, str]]):
    if not report_rows:
        return
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)


def write_review(review_path: str, items: List[Dict[str, str]]):
    if not items:
        return
    with open(review_path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2)


def main():
    args = parse_args()
    configure_logging(args.log_file)
    artist_map = load_mapping(args.artist_genre_map)
    artist_keys = load_mapping_keys(args.artist_genre_map)
    album_map = load_mapping(args.album_genre_map)
    title_map = load_mapping(args.title_genre_map)
    keyword_map = load_mapping(args.keyword_genre_map)

    files = discover_files(args.inputs, args.skip_hidden)
    input_roots = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            input_roots.append(os.path.abspath(inp))
        elif os.path.isfile(inp):
            input_roots.append(os.path.abspath(os.path.dirname(inp)))
    if not input_roots:
        input_roots = [os.path.expanduser("~")]
    logging.info("Discovered %d mp3 files", len(files))
    report_queue = queue.Queue()
    review_queue = queue.Queue()

    def load_info(path: str) -> Optional[TrackInfo]:
        return read_tags(path)

    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(load_info, p): p for p in files}
        infos = []
        for f in futures.as_completed(future_map):
            info = f.result()
            if info:
                infos.append(info)

    for info in infos:
        process_file(
            info,
            args=args,
            artist_map=artist_map,
            artist_keys=artist_keys,
            album_map=album_map,
            title_map=title_map,
            keyword_map=keyword_map,
            input_roots=input_roots,
            report_queue=report_queue,
            review_queue=review_queue,
        )

    report_rows = []
    while not report_queue.empty():
        report_rows.append(report_queue.get())
    review_rows = []
    while not review_queue.empty():
        review_rows.append(review_queue.get())

    write_report(args.report_file, report_rows)
    write_review(args.review_file, review_rows)
    logging.info("Completed. Report: %s Review: %s", args.report_file, args.review_file)


if __name__ == "__main__":
    main()
