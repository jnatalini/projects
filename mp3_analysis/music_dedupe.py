#!/usr/bin/env python3
import argparse
import csv
import difflib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from mutagen import File as MutagenFile
except Exception:  # pragma: no cover - optional dependency
    MutagenFile = None

STOP_REQUESTED = False
STOP_LOCK = threading.Lock()


@dataclass
class TrackInfo:
    path: str
    title: str
    artist: str
    album: str
    album_artist: str
    track_number: str
    disc_number: str
    genre: str
    year: str
    duration: float
    bitrate: int
    sample_rate: int
    size: int
    vbr: Optional[bool]
    has_art: bool
    art_size: int
    tag_fields: int
    norm_title: str
    norm_artist: str
    norm_album: str
    norm_genre: str
    track_key: str
    audio_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def set_stop_requested(signum, frame):  # pragma: no cover - signal handler
    global STOP_REQUESTED
    with STOP_LOCK:
        STOP_REQUESTED = True


def stop_requested() -> bool:
    with STOP_LOCK:
        return STOP_REQUESTED


def setup_logging(log_file: str, level: str, console: bool) -> logging.Logger:
    logger = logging.getLogger("music_dedupe")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def load_config(config_path: Optional[str]) -> Dict[str, object]:
    config = {
        "quality_weights": {
            "bitrate": 0.5,
            "sample_rate": 0.1,
            "metadata": 0.25,
            "artwork": 0.1,
            "size_efficiency": 0.05,
        },
        "bitrate_thresholds": [128, 192, 256, 320],
        "prefer_vbr": False,
        "duration_tolerance": 2.0,
        "size_tolerance_kb": 5,
        "fuzzy_title_threshold": 0.86,
        "ignore_patterns": [],
        "whitelist_patterns": [],
        "follow_symlinks": False,
        "use_album_art": True,
        "tie_breaker": "shortest_path",
    }
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = None
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyYAML is required for YAML config files. Install with: pip install pyyaml"
            ) from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or {}

    def deep_update(base, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base

    return deep_update(config, data)


def normalize_text(value: str) -> str:
    value = value or ""
    value = value.strip().lower()
    value = value.replace("feat.", "ft.").replace("featuring", "ft.")
    return " ".join(value.split())


def normalize_genre(value: str) -> str:
    value = normalize_text(value)
    replacements = {
        "hip hop": "hip-hop",
        "hiphop": "hip-hop",
        "r&b": "rnb",
    }
    return replacements.get(value, value)


def tag_get(tags, keys: Iterable[str]) -> str:
    if not tags:
        return ""
    for key in keys:
        if key in tags:
            val = tags.get(key)
            if isinstance(val, list):
                return str(val[0])
            return str(val)
    return ""


def safe_int(value: Optional[str]) -> int:
    try:
        return int(str(value).split("/")[0])
    except Exception:
        return 0


def read_mp3_metadata(path: Path, config: Dict[str, object], logger: logging.Logger) -> Optional[TrackInfo]:
    if MutagenFile is None:
        logger.error("mutagen is required to read metadata: %s", path)
        return None
    try:
        audio = MutagenFile(path, easy=False)
    except Exception as exc:
        logger.error("Failed to read metadata: %s (%s)", path, exc)
        return None
    if audio is None:
        logger.error("Unsupported or corrupted file: %s", path)
        return None

    tags = getattr(audio, "tags", None)
    info = getattr(audio, "info", None)
    duration = float(getattr(info, "length", 0.0) or 0.0)
    bitrate = int(getattr(info, "bitrate", 0) or 0) // 1000
    sample_rate = int(getattr(info, "sample_rate", 0) or 0)
    vbr = None
    if hasattr(info, "bitrate_mode"):
        vbr = str(getattr(info, "bitrate_mode", "")).lower() == "vbr"

    title = tag_get(tags, ["TIT2", "TITLE", "title"])
    artist = tag_get(tags, ["TPE1", "ARTIST", "artist"])
    album = tag_get(tags, ["TALB", "ALBUM", "album"])
    album_artist = tag_get(tags, ["TPE2", "ALBUMARTIST", "albumartist"])
    track_number = tag_get(tags, ["TRCK", "TRACKNUMBER", "tracknumber"])
    disc_number = tag_get(tags, ["TPOS", "DISCNUMBER", "discnumber"])
    genre = tag_get(tags, ["TCON", "GENRE", "genre"])
    year = tag_get(tags, ["TDRC", "TYER", "DATE", "year"])

    has_art = False
    art_size = 0
    if config.get("use_album_art", True):
        try:
            for key in ("APIC:", "APIC"):
                if tags and key in tags:
                    has_art = True
                    art_size += len(tags[key].data)
        except Exception:
            has_art = False
            art_size = 0

    size = path.stat().st_size
    text_fields = [title, artist, album, album_artist, track_number, disc_number, genre, year]
    tag_fields = sum(1 for item in text_fields if item)

    norm_title = normalize_text(title)
    norm_artist = normalize_text(artist or album_artist)
    norm_album = normalize_text(album)
    norm_genre = normalize_genre(genre)
    duration_key = int(round(duration))
    track_key = f"{norm_artist}|{norm_title}|{duration_key}"

    return TrackInfo(
        path=str(path),
        title=title,
        artist=artist,
        album=album,
        album_artist=album_artist,
        track_number=track_number,
        disc_number=disc_number,
        genre=genre,
        year=year,
        duration=duration,
        bitrate=bitrate,
        sample_rate=sample_rate,
        size=size,
        vbr=vbr,
        has_art=has_art,
        art_size=art_size,
        tag_fields=tag_fields,
        norm_title=norm_title,
        norm_artist=norm_artist,
        norm_album=norm_album,
        norm_genre=norm_genre,
        track_key=track_key,
    )


def is_ignored(path: Path, config: Dict[str, object]) -> bool:
    ignore = config.get("ignore_patterns", []) or []
    whitelist = config.get("whitelist_patterns", []) or []
    path_str = str(path)
    for pattern in whitelist:
        if pattern and pattern in path_str:
            return False
    for pattern in ignore:
        if pattern and pattern in path_str:
            return True
    return False


def iter_mp3_files(root_dir: Path, include_subdirs: bool, follow_symlinks: bool, config: Dict[str, object]) -> Iterable[Path]:
    if include_subdirs:
        for base, dirs, files in os.walk(root_dir, followlinks=follow_symlinks):
            base_path = Path(base)
            if is_ignored(base_path, config):
                dirs[:] = []
                continue
            for filename in files:
                if filename.lower().endswith(".mp3"):
                    path = base_path / filename
                    if not is_ignored(path, config):
                        yield path
    else:
        for entry in os.scandir(root_dir):
            if entry.is_file() and entry.name.lower().endswith(".mp3"):
                path = Path(entry.path)
                if not is_ignored(path, config):
                    yield path


def load_cache(cache_path: Path, logger: logging.Logger) -> Dict[str, Dict[str, object]]:
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load cache: %s", exc)
    return {}


def save_cache(cache_path: Path, cache: Dict[str, Dict[str, object]], logger: logging.Logger) -> None:
    try:
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
    except Exception as exc:
        logger.warning("Failed to save cache: %s", exc)


def load_state(state_path: Path, logger: logging.Logger) -> Dict[str, object]:
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load state: %s", exc)
    return {}


def save_state(state_path: Path, state: Dict[str, object], logger: logging.Logger) -> None:
    try:
        with state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
    except Exception as exc:
        logger.warning("Failed to save state: %s", exc)


def build_track_from_cache(cache_entry: Dict[str, object]) -> TrackInfo:
    return TrackInfo(**cache_entry)


def file_cache_key(path: Path) -> str:
    stat = path.stat()
    return f"{path}|{stat.st_size}|{int(stat.st_mtime)}"


def scan_library(
    root_dir: Path,
    config: Dict[str, object],
    workers: int,
    include_subdirs: bool,
    follow_symlinks: bool,
    cache_path: Path,
    logger: logging.Logger,
    state: Dict[str, object],
) -> List[TrackInfo]:
    start_time = time.time()
    cache = load_cache(cache_path, logger)
    cached_keys = {entry.get("cache_key"): entry for entry in cache.values() if isinstance(entry, dict)}
    results: List[TrackInfo] = []
    files = list(iter_mp3_files(root_dir, include_subdirs, follow_symlinks, config))
    total = len(files)
    logger.info("Found %s mp3 files", total)
    processed_paths = set(state.get("processed", []))

    def process(path: Path) -> Optional[TrackInfo]:
        key = file_cache_key(path)
        if path.as_posix() in processed_paths and key in cached_keys:
            return build_track_from_cache(cached_keys[key])
        if key in cached_keys:
            return build_track_from_cache(cached_keys[key])
        return read_mp3_metadata(path, config, logger)

    completed = 0
    state.setdefault("processed", [])
    state.setdefault("errors", [])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(process, path): path for path in files}
        for future in as_completed(future_map):
            if stop_requested():
                logger.warning("Stop requested, saving progress")
                break
            path = future_map[future]
            try:
                track = future.result()
                if track:
                    results.append(track)
                    cache[file_cache_key(Path(track.path))] = track.to_dict()
                else:
                    state["errors"].append(str(path))
            except Exception as exc:
                logger.error("Failed to process %s (%s)", path, exc)
                state["errors"].append(str(path))
            completed += 1
            if completed % 200 == 0 or completed == total:
                elapsed = time.time() - start_time
                logger.info("Scanned %s/%s files (%.1f%%) in %.1fs", completed, total, (completed / total) * 100, elapsed)

    save_cache(cache_path, cache, logger)
    state["processed"] = [track.path for track in results]
    return results


def group_library(tracks: List[TrackInfo]) -> Dict[str, Dict[str, List[TrackInfo]]]:
    library: Dict[str, Dict[str, List[TrackInfo]]] = defaultdict(lambda: defaultdict(list))
    for track in tracks:
        artist_key = track.norm_artist or "unknown_artist"
        album_key = track.norm_album or "unknown_album"
        library[artist_key][album_key].append(track)
    return library


def group_candidates(tracks: List[TrackInfo], duration_tol: float, size_tol_kb: int) -> Dict[str, List[TrackInfo]]:
    groups: Dict[str, List[TrackInfo]] = defaultdict(list)
    for track in tracks:
        duration_bucket = int(track.duration // max(duration_tol, 1.0))
        size_bucket = int(track.size // (max(size_tol_kb, 1) * 1024))
        key = f"{duration_bucket}|{size_bucket}"
        groups[key].append(track)
    return {k: v for k, v in groups.items() if len(v) > 1}


def fuzzy_match(a: str, b: str, threshold: float) -> bool:
    if not a or not b:
        return False
    ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
    return ratio >= threshold


def stage2_duplicates(groups: Dict[str, List[TrackInfo]], threshold: float) -> List[List[TrackInfo]]:
    duplicates = []
    for candidates in groups.values():
        used = set()
        for i, base in enumerate(candidates):
            if i in used:
                continue
            group = [base]
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                other = candidates[j]
                if base.norm_artist == other.norm_artist and (
                    base.norm_title == other.norm_title or fuzzy_match(base.norm_title, other.norm_title, threshold)
                ):
                    group.append(other)
                    used.add(j)
            if len(group) > 1:
                duplicates.append(group)
    return duplicates


def fingerprint_available() -> bool:
    return shutil.which("fpcalc") is not None


def compute_fingerprint(path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["fpcalc", "-json", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        return data.get("fingerprint")
    except Exception:
        return None


def apply_fingerprints(
    groups: List[List[TrackInfo]],
    cache_path: Path,
    logger: logging.Logger,
    workers: int,
) -> List[List[TrackInfo]]:
    cache = load_cache(cache_path, logger)
    updated_groups = []
    fingerprint_cache = {entry.get("path"): entry.get("audio_fingerprint") for entry in cache.values() if isinstance(entry, dict)}

    def ensure_fingerprint(track: TrackInfo) -> Optional[str]:
        if track.path in fingerprint_cache and fingerprint_cache[track.path]:
            return fingerprint_cache[track.path]
        fingerprint = compute_fingerprint(track.path)
        if fingerprint:
            fingerprint_cache[track.path] = fingerprint
            track.audio_fingerprint = fingerprint
            cache_entry = track.to_dict()
            cache_entry["audio_fingerprint"] = fingerprint
            cache[file_cache_key(Path(track.path))] = cache_entry
        return fingerprint

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(ensure_fingerprint, track): track for group in groups for track in group}
        for future in as_completed(futures):
            track = futures[future]
            try:
                track.audio_fingerprint = future.result()
            except Exception as exc:
                logger.warning("Fingerprint failed for %s (%s)", track.path, exc)

    grouped: Dict[str, List[TrackInfo]] = defaultdict(list)
    for group in groups:
        for track in group:
            key = track.audio_fingerprint or track.track_key
            grouped[key].append(track)

    save_cache(cache_path, cache, logger)
    for bucket in grouped.values():
        if len(bucket) > 1:
            updated_groups.append(bucket)
    return updated_groups


def metadata_completeness(track: TrackInfo) -> float:
    return track.tag_fields / 8.0


def bitrate_score(track: TrackInfo, thresholds: List[int]) -> float:
    score = 0.0
    for threshold in thresholds:
        if track.bitrate >= threshold:
            score += 1
    return score / max(len(thresholds), 1)


def size_efficiency(track: TrackInfo) -> float:
    if track.duration <= 0:
        return 0.0
    kb_per_second = (track.size / 1024.0) / track.duration
    return min(kb_per_second / 40.0, 1.0)


def quality_score(track: TrackInfo, config: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
    weights = config.get("quality_weights", {})
    thresholds = config.get("bitrate_thresholds", [128, 192, 256, 320])
    bitrate_value = bitrate_score(track, thresholds)
    sample_rate_value = min(track.sample_rate / 48000.0, 1.0)
    metadata_value = metadata_completeness(track)
    artwork_value = 1.0 if track.has_art else 0.0
    efficiency_value = size_efficiency(track)
    score = (
        bitrate_value * weights.get("bitrate", 0.5)
        + sample_rate_value * weights.get("sample_rate", 0.1)
        + metadata_value * weights.get("metadata", 0.25)
        + artwork_value * weights.get("artwork", 0.1)
        + efficiency_value * weights.get("size_efficiency", 0.05)
    )
    if config.get("prefer_vbr") and track.vbr:
        score += 0.02
    return score, {
        "bitrate": bitrate_value,
        "sample_rate": sample_rate_value,
        "metadata": metadata_value,
        "artwork": artwork_value,
        "size_efficiency": efficiency_value,
    }


def recommend_track(
    group: List[TrackInfo],
    config: Dict[str, object],
) -> Tuple[TrackInfo, Dict[str, Dict[str, float]]]:
    scored = []
    score_details = {}
    for track in group:
        score, details = quality_score(track, config)
        scored.append((score, metadata_completeness(track), track))
        score_details[track.path] = details
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_score = scored[0][0]
    candidates = [item for item in scored if item[0] == top_score]
    if len(candidates) > 1 and config.get("tie_breaker") == "shortest_path":
        candidates.sort(key=lambda item: len(item[2].path))
    return candidates[0][2], score_details


def build_reasons(group: List[TrackInfo], recommended: TrackInfo) -> Dict[str, str]:
    if not group:
        return {}
    max_bitrate = max(track.bitrate for track in group)
    max_sample_rate = max(track.sample_rate for track in group)
    max_tags = max(track.tag_fields for track in group)
    reasons = {}
    for track in group:
        if track.path == recommended.path:
            keep_reasons = []
            if track.bitrate == max_bitrate:
                keep_reasons.append("highest bitrate")
            if track.sample_rate == max_sample_rate:
                keep_reasons.append("highest sample rate")
            if track.tag_fields == max_tags:
                keep_reasons.append("most complete tags")
            if track.has_art:
                keep_reasons.append("has artwork")
            if not keep_reasons:
                keep_reasons.append("highest overall score")
            reasons[track.path] = "; ".join(keep_reasons)
        else:
            reasons[track.path] = "lower quality score"
    return reasons


def interactive_choice(group: List[TrackInfo], recommended: TrackInfo) -> TrackInfo:
    print("\nDuplicate group:")
    for idx, track in enumerate(group, start=1):
        marker = "*" if track.path == recommended.path else " "
        print(f"{marker} {idx}. {track.path} ({track.bitrate} kbps, {track.sample_rate} Hz)")
    while True:
        choice = input("Select file to keep (Enter for recommended): ").strip()
        if not choice:
            return recommended
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(group):
                return group[index]
        print("Invalid selection, try again.")


def duplicate_detection(
    tracks: List[TrackInfo],
    config: Dict[str, object],
    use_fingerprint: bool,
    cache_path: Path,
    logger: logging.Logger,
    workers: int,
) -> List[List[TrackInfo]]:
    duration_tol = float(config.get("duration_tolerance", 2.0))
    size_tol = int(config.get("size_tolerance_kb", 5))
    stage1_groups = group_candidates(tracks, duration_tol, size_tol)
    stage2_groups = stage2_duplicates(stage1_groups, float(config.get("fuzzy_title_threshold", 0.86)))
    if use_fingerprint and fingerprint_available():
        stage2_groups = apply_fingerprints(stage2_groups, cache_path, logger, workers)
    return stage2_groups


def generate_reports(
    tracks: List[TrackInfo],
    duplicate_groups: List[List[TrackInfo]],
    recommendations: Dict[str, str],
    score_details: Dict[str, Dict[str, float]],
    reasons: Dict[str, str],
    report_dir: Path,
    config: Dict[str, object],
    logger: logging.Logger,
    write_html: bool,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "library_summary.json"
    summary_csv = report_dir / "library_summary.csv"
    dup_json = report_dir / "duplicates_report.json"
    dup_csv = report_dir / "duplicates_report.csv"
    quality_csv = report_dir / "quality_report.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump([track.to_dict() for track in tracks], handle, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tracks[0].to_dict().keys()) if tracks else [])
        writer.writeheader()
        for track in tracks:
            writer.writerow(track.to_dict())

    duplicates_payload = []
    for group_id, group in enumerate(duplicate_groups, start=1):
        for track in group:
            keep = track.path == recommendations.get(track.path)
            details = {
                "group_id": group_id,
                "path": track.path,
                "duration": track.duration,
                "bitrate": track.bitrate,
                "sample_rate": track.sample_rate,
                "size": track.size,
                "tag_fields": track.tag_fields,
                "quality_score": score_details.get(track.path, {}).get("overall", 0.0),
                "keep": keep,
                "reason": reasons.get(track.path, ""),
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
            }
            duplicates_payload.append(details)

    with dup_json.open("w", encoding="utf-8") as handle:
        json.dump(duplicates_payload, handle, indent=2)

    with dup_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(duplicates_payload[0].keys()) if duplicates_payload else [])
        writer.writeheader()
        for row in duplicates_payload:
            writer.writerow(row)

    low_quality = [track for track in tracks if track.bitrate and track.bitrate < 128]
    with quality_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tracks[0].to_dict().keys()) if tracks else [])
        writer.writeheader()
        for track in low_quality:
            writer.writerow(track.to_dict())

    if write_html:
        html_path = report_dir / "duplicates_report.html"
        with html_path.open("w", encoding="utf-8") as handle:
            handle.write("<html><head><meta charset='utf-8'><title>Duplicates Report</title></head><body>")
            handle.write("<h1>Duplicate Report</h1>")
            for group_id, group in enumerate(duplicate_groups, start=1):
                handle.write(f"<h2>Group {group_id}</h2>")
                handle.write("<table border='1' cellpadding='4' cellspacing='0'>")
                handle.write("<tr><th>Keep</th><th>Path</th><th>Bitrate</th><th>Sample Rate</th><th>Duration</th></tr>")
                for track in group:
                    keep = "yes" if track.path == recommendations.get(track.path) else "no"
                    handle.write(
                        f"<tr><td>{keep}</td><td>{track.path}</td>"
                        f"<td>{track.bitrate}</td><td>{track.sample_rate}</td><td>{track.duration:.1f}</td></tr>"
                    )
                handle.write("</table>")
            handle.write("</body></html>")

    logger.info("Reports written to %s", report_dir)


def generate_delete_script(report_dir: Path, duplicate_groups: List[List[TrackInfo]], recommendations: Dict[str, str]) -> Optional[Path]:
    if not duplicate_groups:
        return None
    is_windows = os.name == "nt"
    script_path = report_dir / ("delete_duplicates.bat" if is_windows else "delete_duplicates.sh")
    lines = []
    if not is_windows:
        lines.append("#!/usr/bin/env bash")
        lines.append("set -e")
    for group in duplicate_groups:
        for track in group:
            if track.path != recommendations.get(track.path):
                if is_windows:
                    lines.append(f'del "{track.path}"')
                else:
                    lines.append(f'rm -f "{track.path}"')
    with script_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    if not is_windows:
        script_path.chmod(0o755)
    return script_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a music library for duplicates and quality issues.",
    )
    parser.add_argument("--root-dir", required=True, help="Root folder to scan for mp3 files.")
    parser.add_argument("--report-dir", default="./reports", help="Directory for reports.")
    parser.add_argument("--log-file", default="./music_dedupe.log", help="Log file path.")
    parser.add_argument("--config", help="Optional JSON/YAML config file.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads.")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete files; only reports.")
    parser.add_argument("--include-subdirs", dest="include_subdirs", action="store_true", default=True)
    parser.add_argument("--no-subdirs", dest="include_subdirs", action="store_false")
    parser.add_argument("--fingerprint", action="store_true", help="Enable audio fingerprinting if available.")
    parser.add_argument("--interactive", action="store_true", help="Confirm recommendations interactively.")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state if available.")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML duplicates report.")
    parser.add_argument("--generate-delete-script", action="store_true", help="Generate delete_duplicates script.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING, ERROR).")
    parser.add_argument("--no-console-log", action="store_true", help="Disable console logging.")

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve()

    logger = setup_logging(str(log_file), args.log_level, console=not args.no_console_log)
    logger.info("Starting scan: %s", root_dir)

    config = load_config(args.config)
    follow_symlinks = bool(config.get("follow_symlinks", False))
    report_dir.mkdir(parents=True, exist_ok=True)
    state_path = report_dir / "state.json"
    cache_path = report_dir / "cache.json"

    signal.signal(signal.SIGINT, set_stop_requested)
    signal.signal(signal.SIGTERM, set_stop_requested)

    state = load_state(state_path, logger) if args.resume else {}

    tracks = scan_library(
        root_dir=root_dir,
        config=config,
        workers=args.workers,
        include_subdirs=args.include_subdirs,
        follow_symlinks=follow_symlinks,
        cache_path=cache_path,
        logger=logger,
        state=state,
    )

    if stop_requested():
        save_state(state_path, state, logger)
        logger.warning("Scan interrupted. State saved to %s", state_path)
        return 1

    logger.info("Grouping library")
    library = group_library(tracks)
    logger.info("Artists: %s", len(library))

    logger.info("Detecting duplicates")
    duplicate_groups = duplicate_detection(tracks, config, args.fingerprint, cache_path, logger, args.workers)

    recommendations: Dict[str, str] = {}
    score_details: Dict[str, Dict[str, float]] = {}
    reasons: Dict[str, str] = {}
    overrides: Dict[str, str] = {}
    for group in duplicate_groups:
        recommended, details = recommend_track(group, config)
        for track in group:
            detail_entry = details.get(track.path, {})
            score, _ = quality_score(track, config)
            detail_entry["overall"] = score
            score_details[track.path] = detail_entry
        if args.interactive and sys.stdin.isatty():
            chosen = interactive_choice(group, recommended)
            recommendations[chosen.path] = chosen.path
            if chosen.path != recommended.path:
                overrides[recommended.path] = chosen.path
            reasons.update(build_reasons(group, chosen))
        else:
            recommendations[recommended.path] = recommended.path
            reasons.update(build_reasons(group, recommended))

    generate_reports(
        tracks,
        duplicate_groups,
        recommendations,
        score_details,
        reasons,
        report_dir,
        config,
        logger,
        write_html=args.html_report,
    )

    if args.generate_delete_script:
        if args.dry_run:
            logger.info("Dry-run enabled; skipping delete script generation.")
        else:
            script_path = generate_delete_script(report_dir, duplicate_groups, recommendations)
            if script_path:
                logger.info("Delete script written to %s", script_path)

    if overrides:
        overrides_path = report_dir / "interactive_overrides.json"
        with overrides_path.open("w", encoding="utf-8") as handle:
            json.dump(overrides, handle, indent=2)
        logger.info("Interactive overrides saved to %s", overrides_path)

    summary = {
        "total_files": len(tracks),
        "duplicate_groups": len(duplicate_groups),
        "duplicates": sum(len(group) for group in duplicate_groups),
        "recommended_deletions": sum(len(group) - 1 for group in duplicate_groups),
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Done. Summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
