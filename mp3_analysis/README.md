# MP3 Analysis: Music Deduplication Script

This project provides a single Python script that scans a local `.mp3` library, identifies duplicates, ranks copies by quality, and generates reports. It is designed for large libraries and avoids destructive changes by default.

## Requirements

- Python 3.8+
- `mutagen` for metadata parsing:
  - `pip install mutagen`
- Optional (for YAML config): `pyyaml`
  - `pip install pyyaml`
- Optional (for audio fingerprinting): `fpcalc` (Chromaprint)
  - macOS: `brew install chromaprint`
  - Ubuntu/Debian: `sudo apt-get install chromaprint`

## Usage

Basic scan with reports:

```bash
python music_dedupe.py --root-dir /path/to/music
```

Custom report directory and logging:

```bash
python music_dedupe.py --root-dir /path/to/music --report-dir ./reports --log-file ./music_dedupe.log
```

Enable fingerprinting and HTML report:

```bash
python music_dedupe.py --root-dir /path/to/music --fingerprint --html-report
```

Interactive mode to override recommendations:

```bash
python music_dedupe.py --root-dir /path/to/music --interactive
```

Generate a deletion script (never deletes directly):

```bash
python music_dedupe.py --root-dir /path/to/music --generate-delete-script
```

Note: `--dry-run` skips delete-script generation.

Resume after interruption:

```bash
python music_dedupe.py --root-dir /path/to/music --resume
```

## Command-line Options

- `--root-dir` (required): Root directory containing `.mp3` files.
- `--report-dir`: Report output directory (default: `./reports`).
- `--log-file`: Log file path (default: `./music_dedupe.log`).
- `--config`: JSON or YAML config file path.
- `--workers`: Number of worker threads (default: `8`).
- `--dry-run`: No deletions (reports only). The script never deletes files directly.
- `--include-subdirs/--no-subdirs`: Include subdirectories (default: include).
- `--fingerprint`: Use audio fingerprinting if `fpcalc` is installed.
- `--interactive`: Prompt for which duplicate to keep.
- `--resume`: Resume from the last saved state.
- `--html-report`: Generate `duplicates_report.html`.
- `--generate-delete-script`: Create `delete_duplicates.sh` or `.bat`.
- `--log-level`: Log level (`INFO`, `DEBUG`, `WARNING`, `ERROR`).
- `--no-console-log`: Disable console logging.

## Outputs

Reports are generated in the report directory (default `./reports`):

- `library_summary.json` and `library_summary.csv`: full track listing with normalized metadata.
- `duplicates_report.json` and `duplicates_report.csv`: duplicate groups and recommendations.
- `quality_report.csv`: low-quality tracks (bitrate < 128 kbps).
- `duplicates_report.html`: HTML summary (if `--html-report`).
- `summary.json`: totals for scanned files and duplicates.
- `state.json`: progress checkpoint for `--resume`.
- `cache.json`: metadata/fingerprint cache for faster re-runs.
- `delete_duplicates.sh` or `.bat`: optional deletion script.

## Config File

The config file lets you tune quality scoring, ignore patterns, and matching thresholds.

Example JSON:

```json
{
  "quality_weights": {
    "bitrate": 0.6,
    "sample_rate": 0.1,
    "metadata": 0.2,
    "artwork": 0.1,
    "size_efficiency": 0.0
  },
  "bitrate_thresholds": [128, 192, 256, 320],
  "prefer_vbr": true,
  "duration_tolerance": 2.0,
  "size_tolerance_kb": 5,
  "fuzzy_title_threshold": 0.86,
  "ignore_patterns": ["Downloads", "Temp"],
  "whitelist_patterns": [],
  "follow_symlinks": false,
  "use_album_art": true,
  "tie_breaker": "shortest_path"
}
```

YAML is also supported if `pyyaml` is installed.

## Notes

- Duplicate detection uses a multi-stage approach: duration/size grouping, metadata similarity, and optional audio fingerprints.
- The delete script is generated but never executed; review it before running.
- If interrupted (Ctrl+C), the script saves progress to `state.json`.

## Minimal Test Scenario

To validate behavior, create a small test folder with:

- Two identical songs with different filenames.
- One low-bitrate copy of a track.
- One corrupt file (empty `.mp3`).

Run:

```bash
python music_dedupe.py --root-dir ./test_music --html-report --generate-delete-script
```
