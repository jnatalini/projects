# Ebook Library Deduper/Organizer

A single-file Python tool that scans an ebook library, builds a reusable index, detects duplicates, scores quality, and generates reports.

## Features

- Recursively scans ebook folders (`.epub`, `.mobi`, `.azw3`, `.pdf`, `.djvu`, `.fb2`, configurable).
- Extracts metadata where possible (EPUB via `ebooklib`, PDF via `pypdf`/`PyPDF2`).
- Normalizes titles/authors and tries to infer series/volume from titles.
- Multi-stage duplicate detection (metadata similarity + exact hash).
- Ranks duplicates by configurable quality scoring and format preference.
- Generates JSON/CSV reports, a summary, and an optional delete script.
- Safe defaults: no deletions unless explicitly requested.

## Requirements

- Python 3.8+
- Optional libraries (install as needed):
  - `ebooklib` for EPUB metadata.
  - `pypdf` or `PyPDF2` for PDF metadata and text detection.
  - `PyYAML` for YAML config files.

Example install:

```bash
pip install ebooklib pypdf PyYAML
```

## Usage

Basic scan and report:

```bash
python ebook_dedupe.py --root-dir ~/Books --report-dir ./reports
```

Force full rescan and use more threads:

```bash
python ebook_dedupe.py --root-dir ~/Books --reindex --threads 8
```

Filter formats and generate delete script:

```bash
python ebook_dedupe.py --root-dir ~/Books --formats epub,mobi,pdf --generate-delete-script
```

Interactive override for duplicate recommendations:

```bash
python ebook_dedupe.py --root-dir ~/Books --interactive
```

## Output Files

- `reports/library_index.json`: master index of scanned files.
- `reports/duplicates_report.json` and `reports/duplicates_report.csv`: duplicate groups with recommendations.
- `reports/summary.txt`: totals and top authors/formats.
- `reports/delete_duplicates.sh`: optional delete script.
- `ebook_dedupe.log`: detailed logs (rotated).

## Sample Console Logs

```
2025-01-06 10:12:05 [INFO] Starting ebook dedupe run
2025-01-06 10:12:06 [INFO] Found 18420 files
2025-01-06 10:12:10 [INFO] Processed 100 / 18420 (0.5%)
2025-01-06 10:12:45 [INFO] Processed 1000 / 18420 (5.4%)
2025-01-06 10:15:05 [INFO] Processed 18420 / 18420 (100.0%)
2025-01-06 10:15:06 [INFO] Index built; errors=3
2025-01-06 10:15:07 [INFO] Done
```

## Config File (optional)

JSON or YAML. Example `config.json`:

```json
{
  "format_priority": ["epub", "azw3", "mobi", "pdf_text", "pdf_scan", "djvu", "fb2", "other"],
  "exclude_dirs": ["/tmp", "/Calibre Library/Trash"],
  "allow_mixed_languages": false,
  "follow_symlinks": false,
  "quality_weights": {
    "format": 5,
    "metadata": 2,
    "text": 2,
    "size": 0.1,
    "errors": 5
  }
}
```

Run with:

```bash
python ebook_dedupe.py --root-dir ~/Books --config config.json
```

## Notes

- The script prefers text-based PDFs when it can detect extractable text.
- If optional libraries are missing, the script falls back to filename parsing and logs a warning.
- For large libraries, the index file is reused when file size and mtime match (unless `--reindex`).

## Safety

- By default, the script does not delete or move any files.
- Use `--generate-delete-script` to review deletions manually.
- `--delete` is available, but should be used only after verifying reports and backups.
