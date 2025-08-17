PgDump helper

This project contains `PgDump.py`, a helper script that exports PostgreSQL schema objects into per-object SQL files.

Repository status — Python dumper is final

- Current state: the Python dumper (`PgDump.py`) is considered final for this repository. The core functionality is implemented and stable for dumping schema objects into per-object SQL files, writing `.sha256` sidecars, and moving stale files to `_stale/`.
- Future changes: any future changes merged into this repository should be database-related only (SQL, schema, data changes, migrations, or supplemental SQL files). Do not expect Python code changes unless a critical bug is discovered. If you intend to propose a Python change, open an issue first to discuss necessity and scope.

Quick setup

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Provide database password securely:

- The script relies on libpq authentication. Keep a `pgpass.conf` in the project root (the same folder as `PgDump.py`). Set the environment variable `PGPASSFILE` to point to this file: `export PGPASSFILE=./pgpass.conf`
- Copy `pgpass.conf.example` and update host/port/database/username/password when creating the file.

Test connection:

```bash
psql -h 127.0.0.1 -U postgres -d postgres -c "SELECT version();"
```

pgpass.conf format (one line per entry):

```
hostname:port:database:username:password
```

Security

- Do NOT commit real `pgpass.conf` to source control. The repo includes `pgpass.conf.example` only.

Running the dumper

```bash
# basic run
python PgDump.py --dbname mydb --out ./sql-by-object --verbose

# dry-run (shows what would change)
python PgDump.py --dbname mydb --out ./sql-by-object --dry-run --verbose
```

Notes

- `pg_dump` (Postgres client tools) must be on PATH.
- The script focuses on schema object dumps; data-only dumps are out of scope.
- Per-object SQL files and `.sha256` sidecars are produced under `<out>/<dbname>/...`.
- Stale files are moved to a `_stale/` folder under the database output folder.

Database contribution guidance

- Place SQL-only changes under the appropriate folders (for example `HR/tables/public/` or `HR/sequences/public/`) and include a short description in the PR.
- If your change affects how dumps should be structured, open an issue describing the desired SQL/layout change so we can discuss without touching the Python code.

Changelog

- Updated: 2025-08-17 — README clarified: Python dumper marked final; future changes should be database-related only.
