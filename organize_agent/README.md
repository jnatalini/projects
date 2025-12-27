# Organize Agent

Always-on Ubuntu desktop agent that watches media folders, extracts metadata, indexes files,
and optionally organizes content. This repository is a single-file Python agent plus this
README.

## What it does
- Watches configured directories for new/changed/deleted files.
- Classifies files as pictures, music, ebooks, or other.
- Extracts metadata (EXIF, ID3, ebook metadata when libraries are installed).
- Indexes files in a local SQLite database for fast search and reports.
- Optionally moves/renames files into a structured library layout.
- Generates periodic reports (new content, storage, duplicates, missing metadata).

## Requirements
- Ubuntu desktop
- Python 3.x
- Optional libraries for richer metadata and filesystem events:
  - `pip install watchdog Pillow mutagen ebooklib pypdf pyyaml`

You can run without these libraries; the agent will log warnings and continue.

## Files
- `agent.py` - the single-file implementation.

## Quick start
1) Create a config file (YAML or JSON).
2) Run once to verify:
   - `python3 agent.py --config /path/to/config.yaml --once --log-level DEBUG`
3) Run continuously:
   - `python3 agent.py --config /path/to/config.yaml`

## Example config (YAML)
```yaml
pictures_dirs:
  - /home/jose/Pictures/Inbox
music_dirs:
  - /home/jose/Music/Inbox
ebooks_dirs:
  - /home/jose/Books/Inbox
other_dirs:
  - /home/jose/Downloads

extensions:
  pictures: [".jpg", ".jpeg", ".png", ".heic"]
  music: [".mp3", ".flac", ".m4a"]
  ebooks: [".epub", ".pdf", ".mobi"]

organize:
  auto_organize: false
  dry_run: true
  pictures_target: /home/jose/Pictures
  music_target: /home/jose/Music
  ebooks_target: /home/jose/Books
  pictures_template: "{year}/{month}/{day}/{name}"
  music_template: "{artist}/{album}/{track:02d} - {title}{ext}"
  ebooks_template: "{author}/{series_or_title}/{title}{ext}"
  normalize_filenames: true

index:
  path: /home/jose/.agent_index/agent.db

reports:
  dir: /home/jose/agent_reports
  daily_time: "02:00"

scan:
  interval_seconds: 300
  debounce_seconds: 5
  use_watchdog: true
  partial_hash_bytes: 0

resources:
  max_workers: 4

logging:
  level: INFO
  file: /home/jose/.agent_index/agent.log
  audit_file: /home/jose/.agent_index/audit.log
```

## Command-line usage
- `--config /path/to/config.yaml` Optional config file path
- `--once` Run one scan and exit
- `--reindex` Clear the index and rebuild
- `--log-file /path/to/log.log` Override log file path
- `--log-level INFO|DEBUG|...` Override log level
- `--report` Generate reports and exit
- `--list pictures --since 2025-01-01` List entries from the index
- `--search "artist=Beatles"` Search the index

## Systemd service (autostart)
Create `/etc/systemd/system/organize-agent.service`:
```ini
[Unit]
Description=Organize Agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/jose/Development/python/organize_agent/agent.py --config /home/jose/Development/python/organize_agent/config.yaml
Restart=on-failure
User=jose
WorkingDirectory=/home/jose/Development/python/organize_agent

[Install]
WantedBy=default.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable organize-agent.service
sudo systemctl start organize-agent.service
```

## Tmux alternative
```bash
tmux new -s organize-agent "python3 /home/jose/Development/python/organize_agent/agent.py --config /path/to/config.yaml"
```

## Notes
- The agent uses SQLite by default. The database and logs live under `~/.agent_index/`.
- Duplicates report is based on hashes; no automatic deletion is performed.
- File moves are disabled by default. Enable `organize.auto_organize` and disable `dry_run`
  only when ready.
