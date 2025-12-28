#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler

try:
    import yaml
except Exception:
    yaml = None

# Optional format-specific libraries.
try:
    from ebooklib import epub
except Exception:
    epub = None

try:
    import pypdf
except Exception:
    pypdf = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

DEFAULT_FORMAT_PRIORITY = ["epub", "azw3", "mobi", "pdf_text", "pdf_scan", "djvu", "fb2", "other"]
DEFAULT_EXCLUDES = ["/tmp", "/Trash", "/.trash", "/Calibre Library/Trash"]
DEFAULT_FORMATS = ["epub", "mobi", "azw3", "pdf", "djvu", "fb2"]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Ebook library deduper/organizer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--root-dir", required=True, help="Root directory to scan")
    parser.add_argument("--report-dir", default="./reports", help="Directory for reports")
    parser.add_argument("--log-file", default="./ebook_dedupe.log", help="Log file path")
    parser.add_argument("--config", help="Path to JSON or YAML config")
    parser.add_argument("--threads", type=int, default=8, help="Worker thread count")
    parser.add_argument("--dry-run", action="store_true", help="No moves/deletes, report only")
    parser.add_argument("--reindex", action="store_true", help="Force full rescan")
    parser.add_argument("--interactive", action="store_true", help="Allow user overrides")
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS), help="Comma list of formats")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks")
    parser.add_argument("--generate-delete-script", action="store_true", help="Create delete script")
    parser.add_argument("--delete", action="store_true", help="Delete duplicates directly")
    return parser.parse_args(argv)


def load_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        if path.lower().endswith(".json"):
            return json.load(fh)
        if path.lower().endswith((".yml", ".yaml")):
            if yaml is None:
                raise RuntimeError("PyYAML not installed; cannot read YAML config")
            return yaml.safe_load(fh)
    raise ValueError("Config must be .json or .yaml/.yml")


def setup_logging(log_file):
    logger = logging.getLogger("ebook_dedupe")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)

    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    rotating = RotatingFileHandler(log_file, maxBytes=2 * 1024 * 1024, backupCount=3)
    rotating.setLevel(logging.DEBUG)
    rotating.setFormatter(fmt)

    logger.addHandler(stream)
    logger.addHandler(rotating)
    return logger


def normalize_whitespace(value):
    return re.sub(r"\s+", " ", value).strip()


def normalize_author(author):
    author = normalize_whitespace(author)
    if "," in author:
        parts = [p.strip() for p in author.split(",")]
        if len(parts) >= 2:
            return f"{parts[1]} {parts[0]}".strip()
    return author


def normalize_title(title):
    title = normalize_whitespace(title)
    title = re.sub(r"\[[^\]]+\]", "", title)
    title = re.sub(r"\([^\)]+\)", "", title)
    title = re.sub(r"\b(kindle|edition|unabridged|abridged|sample|preview)\b", "", title, flags=re.I)
    return normalize_whitespace(title)


def slugify(value):
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_whitespace(value)


def parse_series_from_title(title):
    patterns = [
        r"^(?P<title>.+?)\s*#\s*(?P<index>\d+)\s*(?P<series>.+)?$",
        r"^(?P<title>.+?)\s*\(\s*book\s*(?P<index>\d+)\s*of\s*(?P<series>.+?)\s*\)$",
        r"^(?P<series>.+?)\s*book\s*(?P<index>\d+)\s*[:-]\s*(?P<title>.+)$",
    ]
    for pat in patterns:
        match = re.match(pat, title, flags=re.I)
        if match:
            group = match.groupdict()
            return (
                normalize_whitespace(group.get("series") or ""),
                group.get("index"),
                normalize_whitespace(group.get("title") or title),
            )
    return "", None, title


def extract_metadata_epub(path):
    meta = {}
    if epub is None:
        return meta
    try:
        book = epub.read_epub(path)
        meta["title"] = _first_meta(book, "DC", "title")
        meta["authors"] = book.get_metadata("DC", "creator")
        meta["language"] = _first_meta(book, "DC", "language")
        meta["publisher"] = _first_meta(book, "DC", "publisher")
        meta["date"] = _first_meta(book, "DC", "date")
        meta["year"] = parse_year(meta.get("date"))
        meta["subjects"] = [m[0] for m in book.get_metadata("DC", "subject")]
        meta["series"] = _first_meta(book, "calibre", "series")
        meta["series_index"] = _first_meta(book, "calibre", "series_index")
        meta["has_cover"] = bool(book.get_item_with_id("cover"))
    except Exception:
        meta["error"] = "epub_parse_failed"
    return meta


def _first_meta(book, namespace, name):
    try:
        meta = book.get_metadata(namespace, name)
        if meta:
            return meta[0][0]
    except Exception:
        return None
    return None


def extract_metadata_pdf(path):
    meta = {}
    reader = None
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(path)
        except Exception:
            reader = None
    if reader is None and PyPDF2 is not None:
        try:
            reader = PyPDF2.PdfReader(path)
        except Exception:
            reader = None
    if reader is None:
        return meta

    try:
        info = getattr(reader, "metadata", None) or {}
        meta["title"] = getattr(info, "title", None) or info.get("/Title")
        meta["author"] = getattr(info, "author", None) or info.get("/Author")
        meta["date"] = getattr(info, "creation_date", None) or info.get("/CreationDate")
        meta["year"] = parse_year(meta.get("date"))
        meta["page_count"] = len(reader.pages)
        meta["text_sample"] = _pdf_text_sample(reader)
        meta["pdf_text"] = bool(meta["text_sample"])
    except Exception:
        meta["error"] = "pdf_parse_failed"
    return meta


def _pdf_text_sample(reader, max_pages=3):
    samples = []
    try:
        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                samples.append(text[:2000])
        return "\n".join(samples).strip()
    except Exception:
        return ""


def fallback_metadata_from_filename(path):
    base = os.path.splitext(os.path.basename(path))[0]
    cleaned = re.sub(r"[_]+", " ", base)
    cleaned = re.sub(r"\s+-\s+", " - ", cleaned)
    parts = [p.strip() for p in cleaned.split(" - ")]
    meta = {}
    if len(parts) >= 2:
        meta["author"] = parts[0]
        meta["title"] = " - ".join(parts[1:])
    else:
        meta["title"] = cleaned
    return meta


def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).lower()
    return hashlib.sha1(text[:20000].encode("utf-8", errors="ignore")).hexdigest()


def extract_metadata(path, ext):
    meta = {}
    if ext == "epub":
        meta.update(extract_metadata_epub(path))
    elif ext == "pdf":
        meta.update(extract_metadata_pdf(path))
    else:
        meta["note"] = "metadata_not_supported"

    if "title" not in meta or not meta.get("title"):
        meta.update(fallback_metadata_from_filename(path))

    if "author" not in meta and meta.get("authors"):
        meta["author"] = meta["authors"][0][0]
    return meta


def normalize_metadata(meta):
    title = meta.get("title") or ""
    author = meta.get("author") or ""
    series = meta.get("series") or ""
    series_index = meta.get("series_index")

    series_from_title, idx_from_title, title = parse_series_from_title(title)
    if not series and series_from_title:
        series = series_from_title
    if not series_index and idx_from_title:
        series_index = idx_from_title

    norm = {
        "title": normalize_title(title),
        "author": normalize_author(author),
        "series": normalize_whitespace(series),
        "series_index": series_index,
        "language": (meta.get("language") or "").strip(),
        "publisher": (meta.get("publisher") or "").strip(),
        "subjects": meta.get("subjects") or [],
    }
    return norm


def parse_year(value):
    if not value:
        return None
    if hasattr(value, "year"):
        return value.year
    text = str(value)
    match = re.search(r"(19|20)\\d{2}", text)
    if match:
        return int(match.group(0))
    return None


def scan_files(root_dir, formats, excludes, follow_symlinks, logger):
    files = []
    excludes_norm = [e.rstrip(os.sep) for e in excludes]

    def _walk(dir_path):
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    try:
                        if entry.is_symlink() and not follow_symlinks:
                            continue
                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            if _is_excluded(entry.path, excludes_norm):
                                continue
                            _walk(entry.path)
                        elif entry.is_file(follow_symlinks=follow_symlinks):
                            ext = os.path.splitext(entry.name)[1].lower().lstrip(".")
                            if ext in formats:
                                files.append(entry.path)
                    except Exception:
                        logger.exception("Failed to scan entry: %s", entry.path)
        except Exception:
            logger.exception("Failed to scan directory: %s", dir_path)

    _walk(root_dir)
    return files


def _is_excluded(path, excludes):
    for item in excludes:
        if item and item in path:
            return True
    return False


def bucket_length(pages, size_bytes):
    if pages:
        return f"p{pages // 50}"
    return f"s{size_bytes // (1024 * 1024)}"


def compute_book_key(norm_meta, pages, size_bytes):
    return "|".join(
        [
            slugify(norm_meta.get("author", "")),
            slugify(norm_meta.get("title", "")),
            slugify(norm_meta.get("series", "")),
            str(norm_meta.get("series_index") or ""),
            bucket_length(pages, size_bytes),
        ]
    )


def similarity(a, b):
    if not a or not b:
        return 0.0
    a_set = set(slugify(a).split())
    b_set = set(slugify(b).split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def build_duplicates(candidates, allow_mixed_languages):
    groups = []
    for key, items in candidates.items():
        if len(items) <= 1:
            continue
        group = _cluster_group(items, allow_mixed_languages)
        groups.extend(group)
    return groups


def _cluster_group(items, allow_mixed_languages):
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _is_likely_duplicate(items[i], items[j], allow_mixed_languages):
                union(i, j)

    clusters = {}
    for i in range(len(items)):
        root = find(i)
        clusters.setdefault(root, []).append(items[i])

    return [c for c in clusters.values() if len(c) > 1]


def _is_likely_duplicate(a, b, allow_mixed_languages):
    if not allow_mixed_languages:
        if a.get("language") and b.get("language") and a.get("language") != b.get("language"):
            return False
    title_sim = similarity(a.get("norm_title"), b.get("norm_title"))
    author_sim = similarity(a.get("norm_author"), b.get("norm_author"))
    size_close = _size_close(a.get("size_bytes"), b.get("size_bytes"))

    if title_sim >= 0.7 and author_sim >= 0.5 and size_close:
        return True

    if a.get("text_fingerprint") and a.get("text_fingerprint") == b.get("text_fingerprint"):
        return True

    if a.get("sha256") and a.get("sha256") == b.get("sha256"):
        return True

    return False


def _size_close(a, b):
    if not a or not b:
        return False
    delta = abs(a - b) / max(a, b)
    return delta <= 0.25


def format_rank(item, priority_map):
    fmt = item.get("format")
    if fmt == "pdf":
        fmt = "pdf_text" if item.get("pdf_text") else "pdf_scan"
    return priority_map.get(fmt, priority_map.get("other", 999))


def quality_score(item, priority_map, weights):
    score = 0
    score += weights.get("format", 5) * (len(priority_map) - format_rank(item, priority_map))

    meta_fields = [
        item.get("title"),
        item.get("author"),
        item.get("series"),
        item.get("series_index"),
        item.get("language"),
        item.get("publisher"),
    ]
    completeness = sum(1 for v in meta_fields if v)
    score += weights.get("metadata", 2) * completeness

    if item.get("pdf_text"):
        score += weights.get("text", 2)

    if item.get("size_bytes"):
        score += weights.get("size", 1) * (item.get("size_bytes") / (1024 * 1024))

    if item.get("error"):
        score -= weights.get("errors", 5)

    return score


def recommend_keep(group, priority_map, weights):
    for item in group:
        item["quality_score"] = quality_score(item, priority_map, weights)
    ordered = sorted(
        group,
        key=lambda x: (x["quality_score"], -format_rank(x, priority_map), x.get("mtime", 0)),
        reverse=True,
    )
    keep = ordered[0]
    keep["recommendation"] = "keep"
    keep["reason"] = _build_reason(keep)
    for item in ordered[1:]:
        item["recommendation"] = "remove"
        item["reason"] = _build_reason(keep, alternate=item)
    return ordered


def _build_reason(kept, alternate=None):
    if alternate is None:
        return "Top quality score and preferred format"
    return "Kept best format/metadata score over alternate"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_index(report_dir, index):
    path = os.path.join(report_dir, "library_index.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    return path


def write_duplicates_report(report_dir, groups):
    json_path = os.path.join(report_dir, "duplicates_report.json")
    csv_path = os.path.join(report_dir, "duplicates_report.csv")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(groups, fh, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "group_id",
            "author",
            "title",
            "series",
            "series_index",
            "format",
            "language",
            "size_bytes",
            "page_count",
            "sha256",
            "quality_score",
            "recommendation",
            "reason",
            "path",
        ])
        for group in groups:
            for item in group["items"]:
                writer.writerow([
                    group["group_id"],
                    group["author"],
                    group["title"],
                    group["series"],
                    group["series_index"],
                    item.get("format"),
                    item.get("language"),
                    item.get("size_bytes"),
                    item.get("page_count"),
                    (item.get("sha256") or "")[:12],
                    item.get("quality_score"),
                    item.get("recommendation"),
                    item.get("reason"),
                    item.get("path"),
                ])
    return json_path, csv_path


def write_summary(report_dir, stats):
    path = os.path.join(report_dir, "summary.txt")
    lines = [
        f"Total books indexed: {stats['total_books']}",
        f"Duplicate groups found: {stats['duplicate_groups']}",
        f"Potential duplicate copies: {stats['duplicate_copies']}",
        f"Potential space savings (MB): {stats['potential_savings_mb']}",
        f"Top formats: {stats['top_formats']}",
        f"Top authors: {stats['top_authors']}",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def write_books_report(report_dir, index):
    path = os.path.join(report_dir, "books_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["author", "title", "series", "year", "path"])
        for item in index:
            writer.writerow([
                item.get("author"),
                item.get("title"),
                item.get("series"),
                item.get("year"),
                item.get("path"),
            ])
    return path


def generate_delete_script(report_dir, groups, shell="sh"):
    path = os.path.join(report_dir, "delete_duplicates.sh")
    lines = ["#!/bin/sh", "set -e"]
    for group in groups:
        for item in group["items"]:
            if item.get("recommendation") == "remove":
                lines.append(f"rm -f \"{item['path']}\"")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def interactive_override(groups):
    for group in groups:
        print("\nDuplicate group:")
        print(f"  {group['author']} - {group['title']}")
        for i, item in enumerate(group["items"], start=1):
            print(f"  [{i}] {item['format']} | {item['path']} | score={item.get('quality_score'):.2f}")
        choice = input("Keep which number? (enter to accept recommendation) ").strip()
        if not choice:
            continue
        try:
            keep_idx = int(choice) - 1
            for i, item in enumerate(group["items"]):
                if i == keep_idx:
                    item["recommendation"] = "keep"
                    item["reason"] = "User override"
                else:
                    item["recommendation"] = "remove"
                    item["reason"] = "User override"
        except Exception:
            print("Invalid choice; keeping original recommendation")


def index_worker(path, formats, config, existing_entry):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    size_bytes = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    meta = extract_metadata(path, ext)
    norm = normalize_metadata(meta)

    pdf_text = bool(meta.get("pdf_text"))
    sha256 = existing_entry.get("sha256") if existing_entry else None
    if not sha256 or existing_entry.get("size_bytes") != size_bytes:
        sha256 = compute_sha256(path)

    text_fingerprint = existing_entry.get("text_fingerprint") if existing_entry else ""
    if not text_fingerprint:
        if ext == "pdf" and meta.get("text_sample"):
            text_fingerprint = fingerprint_text(meta.get("text_sample"))
        elif ext == "epub":
            text_fingerprint = ""

    record = {
        "path": path,
        "format": ext,
        "title": meta.get("title"),
        "author": meta.get("author") or norm.get("author"),
        "series": norm.get("series"),
        "series_index": norm.get("series_index"),
        "language": norm.get("language"),
        "publisher": norm.get("publisher"),
        "subjects": norm.get("subjects"),
        "year": meta.get("year"),
        "page_count": meta.get("page_count"),
        "size_bytes": size_bytes,
        "mtime": mtime,
        "sha256": sha256,
        "text_fingerprint": text_fingerprint,
        "pdf_text": pdf_text,
        "error": meta.get("error"),
        "norm_title": norm.get("title"),
        "norm_author": norm.get("author"),
    }
    return record


def build_stats(index, groups):
    formats = {}
    authors = {}
    for item in index:
        formats[item.get("format")] = formats.get(item.get("format"), 0) + 1
        author = item.get("author") or "Unknown"
        authors[author] = authors.get(author, 0) + 1

    top_formats = sorted(formats.items(), key=lambda x: x[1], reverse=True)[:5]
    top_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]

    potential_savings = 0
    duplicate_copies = 0
    for group in groups:
        for item in group["items"]:
            if item.get("recommendation") == "remove":
                duplicate_copies += 1
                potential_savings += item.get("size_bytes") or 0

    return {
        "total_books": len(index),
        "duplicate_groups": len(groups),
        "duplicate_copies": duplicate_copies,
        "potential_savings_mb": round(potential_savings / (1024 * 1024), 2),
        "top_formats": top_formats,
        "top_authors": top_authors,
    }


def main(argv):
    args = parse_args(argv)
    config = load_config(args.config)

    logger = setup_logging(args.log_file)
    logger.info("Starting ebook dedupe run")

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    excludes = config.get("exclude_dirs", DEFAULT_EXCLUDES)
    follow_symlinks = config.get("follow_symlinks", args.follow_symlinks)

    report_dir = os.path.abspath(args.report_dir)
    ensure_dir(report_dir)

    index_path = os.path.join(report_dir, "library_index.json")
    existing_index = {}
    if os.path.exists(index_path) and not args.reindex:
        try:
            with open(index_path, "r", encoding="utf-8") as fh:
                for entry in json.load(fh):
                    existing_index[entry["path"]] = entry
        except Exception:
            logger.exception("Failed to load existing index; rebuilding")

    files = scan_files(args.root_dir, formats, excludes, follow_symlinks, logger)
    total = len(files)
    logger.info("Found %d files", total)

    index = []
    processed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {}
        for path in files:
            stat = os.stat(path)
            existing = existing_index.get(path)
            if (
                existing
                and existing.get("size_bytes") == stat.st_size
                and existing.get("mtime") == stat.st_mtime
            ):
                index.append(existing)
                processed += 1
                continue
            futures[executor.submit(index_worker, path, formats, config, existing)] = path

        for future in as_completed(futures):
            path = futures[future]
            try:
                record = future.result()
                index.append(record)
            except Exception:
                logger.exception("Failed processing file: %s", path)
                errors += 1
            processed += 1
            if processed % 100 == 0 or processed == total:
                pct = (processed / total * 100) if total else 100
                logger.info("Processed %d / %d (%.1f%%)", processed, total, pct)

    logger.info("Index built; errors=%d", errors)
    write_index(report_dir, index)
    write_books_report(report_dir, index)

    priority = config.get("format_priority", DEFAULT_FORMAT_PRIORITY)
    priority_map = {fmt: i for i, fmt in enumerate(priority)}
    weights = config.get("quality_weights", {"format": 5, "metadata": 2, "text": 2, "size": 0.1, "errors": 5})

    candidates = {}
    for item in index:
        book_key = compute_book_key(item, item.get("page_count"), item.get("size_bytes"))
        candidates.setdefault(book_key, []).append(item)

    allow_mixed_languages = config.get("allow_mixed_languages", False)
    dup_groups = build_duplicates(candidates, allow_mixed_languages)

    report_groups = []
    for group_id, items in enumerate(dup_groups, start=1):
        ordered = recommend_keep(items, priority_map, weights)
        group_meta = {
            "group_id": group_id,
            "author": ordered[0].get("author"),
            "title": ordered[0].get("title"),
            "series": ordered[0].get("series"),
            "series_index": ordered[0].get("series_index"),
            "items": ordered,
        }
        report_groups.append(group_meta)

    if args.interactive:
        interactive_override(report_groups)

    write_duplicates_report(report_dir, report_groups)

    stats = build_stats(index, report_groups)
    write_summary(report_dir, stats)

    if args.generate_delete_script:
        generate_delete_script(report_dir, report_groups)
        logger.info("Delete script generated")

    if args.delete:
        if args.dry_run:
            logger.info("Dry run: skipping deletions")
        else:
            for group in report_groups:
                for item in group["items"]:
                    if item.get("recommendation") == "remove":
                        try:
                            os.remove(item["path"])
                            logger.info("Deleted %s", item["path"])
                        except Exception:
                            logger.exception("Failed to delete %s", item["path"])

    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
