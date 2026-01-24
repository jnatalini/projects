# MP3 Organizer by Genre

Single-file Python script that reads MP3 metadata, infers genre locally (no APIs), updates ID3 tags, and organizes files into:

`<source-folder>/GENRE/ARTIST/ALBUM/FILENAME.mp3`

## Features
- Reads ID3 tags (artist, title, album, track, year, genre)
- Fills missing metadata by parsing filenames
- Infers genre locally using tags, filename/path hints, and optional mapping files
- Normalizes genres into 11 common buckets
- Writes updated tags to MP3s (ID3v2.3 or ID3v2.4)
- Moves or copies files into a clean library structure
- Handles collisions with skip/overwrite/rename policies
- Generates CSV report and JSON review list
- Local inference only (no network calls)

## Requirements
- Ubuntu server
- Python 3.8+
- No internet required

Python dependencies:
- `mutagen`

Install:
```
pip install mutagen
```

## Quick Start
Dry run:
```
python3 mp3_org_genre.py /path/to/music --dry-run
```

Move into `<source-folder>/Genre/Artist/Album`:
```
python3 mp3_org_genre.py /path/to/music --mode move
```

Copy instead of move:
```
python3 mp3_org_genre.py /path/to/music --mode copy
```

## How It Works
1) Discover `.mp3` files (recursive by default).
2) Read ID3 tags.
3) If tags are missing, try to parse `Artist - Title.mp3` or `01 - Title.mp3`.
4) If genre is missing (or confirmation is enabled), infer locally using:
   - existing tags
   - artist/album/title mappings (optional)
   - filename/path keywords
5) Normalize genre into common buckets.
6) Write updated tags.
7) Move/copy into `~/GENRE/ARTIST/ALBUM/FILENAME.mp3`.

## Genre Buckets
The script maps incoming genres into these 11 buckets:
- Pop
- Rock
- Hip-Hop
- Electronic
- R&B
- Country
- Jazz
- Classical
- Reggae
- Latin
- Middle Eastern / International

## Reports and Logs
- CSV report (default: `mp3_org_report.csv`):
  - original path, new path, old tags, new tags, source, confidence, notes
- JSON review queue (default: `mp3_org_review.json`):
  - mismatches or ambiguous cases
- Optional log file via `--log-file`.

## Mapping File Formats
CSV:
```
The Beatles,Rock
Daft Punk,Electronic
```

JSON:
```
{
  "the beatles": "Rock",
  "daft punk": "Electronic"
}
```

## Usage
```
python3 mp3_org_genre.py [inputs...] [options]
```

Inputs can be:
- File paths
- Folder paths (recursively scanned)

### Common Options
- `--target-root PATH`  
  Target library root (used when `--target-root-mode fixed`)

- `--target-root-mode source|fixed`  
  Use each file's source directory or a fixed `--target-root` (default `source`)

- `--mode move|copy`  
  Move or copy files (default `move`)

- `--conflict skip|overwrite|rename`  
  Collision policy when target exists (default `rename`)

- `--dry-run`  
  Show actions without writing or moving

- `--skip-hidden`  
  Ignore hidden folders/files

- `--min-confidence FLOAT`  
  Minimum confidence to accept inferred genre (default `0.6`)

- `--artist-genre-map PATH`  
  CSV/JSON map of `artist -> genre` (missing artists are appended with empty genre)

- `--album-genre-map PATH`  
  CSV/JSON map of `album -> genre`

- `--title-genre-map PATH`  
  CSV/JSON map of `title -> genre`

- `--keyword-genre-map PATH`  
  CSV/JSON map of `keyword -> genre` (matched against filename/path)

- `--learn-map PATH`  
  Append learned mappings interactively to a CSV file

- `--learn-scope artist|album|title`  
  Which field to learn when prompting (default `artist`)

- `--id3-version 3|4`  
  ID3 tag version to write (default `3`)

- `--rename-pattern original|track-title|artist-title`  
  Rename files based on metadata (default `original`)

- `--unknown-artist NAME`  
  Placeholder for missing artist (default `Unknown Artist`)

- `--unknown-album NAME`  
  Placeholder for missing album (default `Unknown Album`)

- `--unknown-genre NAME`  
  Placeholder for missing genre (default `Unknown Genre`)

- `--report-file PATH`  
  CSV report output path (default `mp3_org_report.csv`)

- `--review-file PATH`  
  JSON review output path (default `mp3_org_review.json`)

- `--max-workers N`  
  Worker threads for tag reading (default `8`)

- `--mismatch-policy keep|replace|both|flag`  
  What to do if local genre != inferred genre (default `flag`)

- `--confirm-existing` / `--no-confirm-existing`  
  Also infer genre even if a tag already exists (default `--confirm-existing`). Artist mappings are always honored when present.

## Examples
Organize and copy into a custom target root:
```
python3 mp3_org_genre.py /music --mode copy --target-root-mode fixed --target-root /library
```

Skip hidden folders:
```
python3 mp3_org_genre.py /music --skip-hidden
```

Rename files to `01 - Title.mp3`:
```
python3 mp3_org_genre.py /music --rename-pattern track-title
```

Keep both local and inferred genres on mismatch:
```
python3 mp3_org_genre.py /music --mismatch-policy both
```

Use a custom artist mapping:
```
python3 mp3_org_genre.py /music --artist-genre-map /home/jose/genre_maps/artist_genres.csv
```

Learn genres interactively and save to a CSV map:
```
python3 mp3_org_genre.py /music --learn-map /path/artist_genres.csv
```

## Notes and Limitations
- Inference is local and deterministic; accuracy improves with mapping files.
- For best results, ensure Artist and Title tags are accurate.
- Generic/placeholder genres like "Music" or "Unknown" are ignored.

## License
Use and modify freely for personal use.
