#!/usr/bin/env python3
"""Infer a single high-level genre for each artist in a CSV.

Reads a CSV where the first column is artist/group name, looks up genres on the
public web (no APIs), maps them to one of 11 target genres, and writes a new CSV
with artist and genre.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup

GENRES = [
    "Pop",
    "Rock",
    "Hip-Hop / Rap",
    "Electronic / EDM",
    "R&B / Soul",
    "Country",
    "Jazz",
    "Classical",
    "Reggae",
    "Latin",
]

DEFAULT_HEADERS = ["artist", "genre"]

USER_AGENT = (
    "artist-genre-script/1.0 (+https://example.com; "
    "contact: local-script)"
)


@dataclass
class LookupResult:
    source: str
    raw_genres: list[str]


class GenreLookup:
    def __init__(self, pause: float = 1.0, timeout: int = 10):
        self.pause = pause
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _sleep(self):
        if self.pause > 0:
            time.sleep(self.pause)

    def _get(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.text
        except requests.RequestException:
            return None
        return None

    def wikipedia(self, artist: str) -> Optional[LookupResult]:
        # Use search page to find a likely match, follow redirects.
        query = urllib.parse.quote(artist)
        search_url = f"https://en.wikipedia.org/w/index.php?search={query}"
        html = self._get(search_url)
        self._sleep()
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")

        # If search led directly to an article, look for infobox on this page.
        if soup.find("table", class_=re.compile("infobox")):
            genres = _extract_wikipedia_genres(soup)
            if genres:
                return LookupResult("wikipedia", genres)

        # Otherwise, follow first search result.
        result = soup.select_one(".mw-search-result-heading a")
        if not result or not result.get("href"):
            return None
        page_url = "https://en.wikipedia.org" + result["href"]
        html = self._get(page_url)
        self._sleep()
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        genres = _extract_wikipedia_genres(soup)
        if genres:
            return LookupResult("wikipedia", genres)
        return None

    def allmusic(self, artist: str) -> Optional[LookupResult]:
        # AllMusic HTML search, no API.
        query = urllib.parse.quote(artist)
        search_url = f"https://www.allmusic.com/search/artists/{query}"
        html = self._get(search_url)
        self._sleep()
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        result_link = soup.select_one(".search-results .name a")
        if not result_link or not result_link.get("href"):
            return None
        artist_url = result_link["href"]
        html = self._get(artist_url)
        self._sleep()
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        # Genres appear under a div with class "genre" or "styles".
        genre_tags = soup.select(".genre a")
        if not genre_tags:
            genre_tags = soup.select(".styles a")
        genres = [g.get_text(strip=True) for g in genre_tags if g.get_text(strip=True)]
        if genres:
            return LookupResult("allmusic", genres)
        return None

    def lastfm(self, artist: str) -> Optional[LookupResult]:
        # Last.fm HTML search, no API.
        query = urllib.parse.quote(artist)
        search_url = f"https://www.last.fm/search/artists?q={query}"
        html = self._get(search_url)
        self._sleep()
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        result_link = soup.select_one(".link-block-target")
        if not result_link or not result_link.get("href"):
            return None
        artist_url = "https://www.last.fm" + result_link["href"]
        html = self._get(artist_url)
        self._sleep()
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        tags = soup.select(".tag a, .tags-list a")
        genres = [t.get_text(strip=True) for t in tags if t.get_text(strip=True)]
        if genres:
            return LookupResult("lastfm", genres)
        return None


def _extract_wikipedia_genres(soup: BeautifulSoup) -> list[str]:
    infobox = soup.find("table", class_=re.compile("infobox"))
    if not infobox:
        return []
    # Find table header cells with text "Genre" or "Genres".
    for row in infobox.find_all("tr"):
        header = row.find("th")
        if not header:
            continue
        header_text = header.get_text(strip=True).lower()
        if header_text in {"genre", "genres"}:
            cell = row.find("td")
            if not cell:
                continue
            # Prefer linked genres, but also accept plain text.
            links = [a.get_text(strip=True) for a in cell.find_all("a")]
            if links:
                return [g for g in links if g]
            text = cell.get_text(" ", strip=True)
            if text:
                return [t.strip() for t in re.split(r",|/|;", text) if t.strip()]
    return []


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _map_to_target(genres: Iterable[str]) -> Optional[str]:
    if not genres:
        return None
    normalized = [_normalize(g) for g in genres]

    # Priority order matters for ambiguous tags.
    mappings = [
        ("classical", "Classical"),
        ("baroque", "Classical"),
        ("romantic", "Classical"),
        ("orchestral", "Classical"),
        ("symph", "Classical"),
        ("jazz", "Jazz"),
        ("bebop", "Jazz"),
        ("swing", "Jazz"),
        ("reggae", "Reggae"),
        ("dancehall", "Reggae"),
        ("ska", "Reggae"),
        ("hip hop", "Hip-Hop / Rap"),
        ("hip-hop", "Hip-Hop / Rap"),
        ("rap", "Hip-Hop / Rap"),
        ("trap", "Hip-Hop / Rap"),
        ("r&b", "R&B / Soul"),
        ("rb", "R&B / Soul"),
        ("soul", "R&B / Soul"),
        ("neo soul", "R&B / Soul"),
        ("country", "Country"),
        ("bluegrass", "Country"),
        ("americana", "Country"),
        ("latin", "Latin"),
        ("reggaeton", "Latin"),
        ("salsa", "Latin"),
        ("bachata", "Latin"),
        ("bossa", "Latin"),
        ("edm", "Electronic / EDM"),
        ("electronic", "Electronic / EDM"),
        ("techno", "Electronic / EDM"),
        ("house", "Electronic / EDM"),
        ("trance", "Electronic / EDM"),
        ("dubstep", "Electronic / EDM"),
        ("ambient", "Electronic / EDM"),
        ("rock", "Rock"),
        ("metal", "Rock"),
        ("punk", "Rock"),
        ("alternative", "Rock"),
        ("indie", "Rock"),
        ("pop", "Pop"),
    ]

    for token, target in mappings:
        for g in normalized:
            if token in g:
                return target
    return None


def _heuristic_from_name(name: str) -> Optional[str]:
    n = _normalize(name)
    if any(k in n for k in ["dj ", " dj", "producer", "remix"]):
        return "Electronic / EDM"
    if any(k in n for k in ["mc ", " mc", "rapper"]):
        return "Hip-Hop / Rap"
    if any(k in n for k in ["orchestra", "symphony", "philharmonic", "quartet", "ensemble"]):
        return "Classical"
    if any(k in n for k in ["jazz", "swing", "big band"]):
        return "Jazz"
    return None


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    try:
        path.write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding="utf-8")
    except OSError:
        pass


def infer_genre(
    artist: str,
    lookup: GenreLookup,
    fallback: str,
    cache: dict[str, str],
    verbose: bool,
) -> str:
    if artist in cache:
        return cache[artist]

    sources = [lookup.wikipedia, lookup.allmusic, lookup.lastfm]
    variations = [artist, f"{artist} band", f"{artist} musician"]

    for name in variations:
        for source in sources:
            result = source(name)
            if result and result.raw_genres:
                mapped = _map_to_target(result.raw_genres)
                if mapped:
                    cache[artist] = mapped
                    if verbose:
                        print(f"{artist}: {mapped} (via {result.source})", file=sys.stderr)
                    return mapped

    heuristic = _heuristic_from_name(artist)
    if heuristic:
        cache[artist] = heuristic
        if verbose:
            print(f"{artist}: {heuristic} (heuristic)", file=sys.stderr)
        return heuristic

    cache[artist] = fallback
    if verbose:
        print(f"{artist}: {fallback} (fallback)", file=sys.stderr)
    return fallback


def read_artists(path: Path) -> list[str]:
    artists: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if not name:
                continue
            # Skip a header row if it looks like one.
            if not artists and _normalize(name) in {"artist", "name", "group"}:
                continue
            # If a genre is already provided in column 2, skip this row.
            if len(row) > 1 and row[1].strip():
                continue
            artists.append(name)
    return artists


def write_output(path: Path, rows: Iterable[tuple[str, str]], include_header: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if include_header:
            writer.writerow(DEFAULT_HEADERS)
        for artist, genre in rows:
            writer.writerow([artist, genre])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infer a single genre for each artist in a CSV (first column)."
    )
    parser.add_argument("input_csv", type=Path, help="Path to input CSV")
    parser.add_argument("output_csv", type=Path, help="Path to output CSV")
    parser.add_argument(
        "--fallback",
        default="Pop",
        choices=GENRES,
        help="Genre to use if no match can be found (default: Pop)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("genre_cache.json"),
        help="Cache file to reuse lookups (default: genre_cache.json)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write header row to output",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to pause between web requests (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print lookup details to stderr",
    )

    args = parser.parse_args()
    lookup = GenreLookup(pause=args.pause, timeout=args.timeout)
    cache = _load_cache(args.cache)

    artists = read_artists(args.input_csv)
    rows: list[tuple[str, str]] = []
    for artist in artists:
        genre = infer_genre(artist, lookup, args.fallback, cache, args.verbose)
        rows.append((artist, genre))

    write_output(args.output_csv, rows, include_header=not args.no_header)
    _save_cache(args.cache, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
