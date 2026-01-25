# Artist Genre CSV

This script reads a CSV where the first column is an artist or group name, looks up the artist on public websites (no APIs), maps the found genres to one of 10 target genres, and writes a new CSV with `artist,genre`.

## Target genres
- Pop
- Rock
- Hip-Hop / Rap
- Electronic / EDM
- R&B / Soul
- Country
- Jazz
- Classical
- Reggae
- Latin

## Requirements
- Python 3.9+
- `requests`
- `beautifulsoup4`

Install dependencies:
```bash
python -m pip install requests beautifulsoup4
```

## Usage
```bash
python artist_genre.py input.csv output.csv
```

Optional flags:
```bash
python artist_genre.py input.csv output.csv \
  --fallback "Pop" \
  --cache genre_cache.json \
  --pause 1.0 \
  --timeout 10 \
  --verbose
```

- `--fallback`: The genre to use if the script cannot find any genre online.
- `--cache`: JSON cache to avoid repeated lookups across runs.
- `--pause`: Sleep time between requests (polite crawling).
- `--no-header`: Skip writing the header row to the output file.
- `--verbose`: Print how each genre was determined to stderr.

## How it determines genres (no APIs)
The script uses multiple HTML sources and fallbacks to ensure every artist gets a genre:
1. **Wikipedia**: Searches the artist page and extracts “Genre(s)” from the infobox.
2. **AllMusic**: If Wikipedia fails, it parses the artist page for genre/style tags.
3. **Last.fm**: If AllMusic fails, it parses tags on the artist page.
4. **Name heuristic**: If no web genre is found, it infers from keywords (e.g., “DJ”, “Orchestra”).
5. **Fallback genre**: If still unknown, it uses the `--fallback` value (default: Pop).

All retrieved genre labels are mapped into the 10 target genres with keyword rules.

## Notes
- The script uses the Python `csv` module to correctly quote artist names with commas.
- Some websites may block heavy scraping. Use `--pause` to slow down requests.
- The output CSV always includes the artist from the input and exactly one mapped genre.

## Example input
```csv
Artist
AC/DC
Bad Bunny
"Earth, Wind & Fire"
```

## Example output
```csv
artist,genre
AC/DC,Rock
Bad Bunny,Latin
"Earth, Wind & Fire",R&B / Soul
```
