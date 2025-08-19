"""PgDump.py

Dump Postgres schema objects into per-object files and only update files that changed.

Usage (example):
		python PgDump.py --dbname mydb --out d:/Projects/Database/Postgres/Database_Projects/sql-by-object

Requires:
	- Python 3.8+
	- psycopg2-binary
	- pg_dump on PATH
	- .pgpass or PGPASSWORD env to avoid interactive prompts
"""

from pathlib import Path
import os
import sys
import subprocess
import psycopg2
import getpass
import hashlib
import tempfile
import shutil
import re
import datetime

# Globals for dry-run / verbose behavior and tracking expected files
DRY_RUN = False
VERBOSE = False
EXPECTED_PATHS = set()
CURRENT_USER = getpass.getuser()


def set_pgpass_env():
	"""Locate a pgpass file and set PGPASSFILE env var so libpq and pg_dump pick it up."""
	# Respect existing env
	if os.environ.get("PGPASSFILE"):
		return
	# Look for pgpass.conf in project folder only
	proj_pg = str(Path(__file__).resolve().parent / 'pgpass.conf')
	
	if os.path.exists(proj_pg):
		os.environ['PGPASSFILE'] = proj_pg
		if VERBOSE:
			print(f"Using pgpass file: {proj_pg}")
		return
	
	# Error if pgpass.conf not found in project folder
	raise FileNotFoundError(f"pgpass.conf not found in project folder: {proj_pg}. Copy pgpass.conf.example and update with your credentials.")


def load_password_from_pgpass(host, port, dbname, user):
	"""Parse PGPASSFILE and set PGPASSWORD env var if a matching entry is found."""
	p = os.environ.get('PGPASSFILE')
	if not p or not os.path.exists(p):
		return None
	try:
		with open(p, 'r', encoding='utf8') as f:
			for ln in f:
				ln = ln.strip()
				if not ln or ln.startswith('#'):
					continue
				parts = ln.split(':')
				if len(parts) < 5:
					continue
				phost, pport, pdb, puser, ppassword = parts[0], parts[1], parts[2], parts[3], ':'.join(parts[4:])
				def match(field, val):
					if field == '*':
						return True
					if field == val:
						return True
					# treat localhost and loopback equivalents as matching
					hosts_loopback = {'127.0.0.1', '::1', 'localhost'}
					if field in hosts_loopback and val in hosts_loopback:
						return True
					return False
				if VERBOSE:
					print(f"pgpass entry: host={phost} port={pport} db={pdb} user={puser}")
				if match(phost, host) and match(pport, str(port)) and match(pdb, dbname or '*') and match(puser, user):
					# set env for libpq
					os.environ['PGPASSWORD'] = ppassword
					if VERBOSE:
						print(f"Loaded password from pgpass for {user}@{host}:{port}/{dbname}")
					return ppassword
	except Exception:
		return None
	return None


def compute_hash_bytes(data: bytes) -> str:
	h = hashlib.sha256()
	h.update(data)
	return h.hexdigest()


def compute_file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
	h = hashlib.sha256()
	with path.open("rb") as f:
		while True:
			chunk = f.read(chunk_size)
			if not chunk:
				break
			h.update(chunk)
	return h.hexdigest()


def normalize_sql(sql: str) -> str:
	"""Normalize SQL text to reduce non-semantic differences.

	- Remove SET search_path lines
	- Remove lines containing session-local options
	- Strip trailing spaces and collapse multiple blank lines
	- Normalize line endings
	"""
	# Normalize line endings
	text = sql.replace('\r\n', '\n').replace('\r', '\n')

	lines = []
	for ln in text.split('\n'):
		# remove search_path/settings lines (common noisy lines)
		if re.match(r"^\s*SET\s+search_path\b", ln, flags=re.I):
			continue
		if re.match(r"^\s*SET\s+\w+\s*=", ln, flags=re.I):
			# skip other SET lines which may vary per run
			continue
		# drop trailing whitespace
		ln = ln.rstrip()
		lines.append(ln)

	# collapse multiple blank lines to max 2
	out_lines = []
	blank_run = 0
	for ln in lines:
		if ln == "":
			blank_run += 1
			if blank_run <= 2:
				out_lines.append(ln)
		else:
			blank_run = 0
			out_lines.append(ln)

	return '\n'.join(out_lines).strip() + '\n'


def atomic_write(path: Path, data: bytes) -> None:
	# write to a temp file in same directory then move
	path.parent.mkdir(parents=True, exist_ok=True)
	fd, tmp = tempfile.mkstemp(prefix=".", dir=str(path.parent))
	try:
		with os.fdopen(fd, "wb") as f:
			f.write(data)
		# replace
		shutil.move(tmp, str(path))
	finally:
		if os.path.exists(tmp):
			try:
				os.remove(tmp)
			except Exception:
				pass


def _strip_leading_comment_block(text: str) -> str:
	"""Remove an initial SQL comment block (lines starting with --) and return the rest."""
	lines = text.splitlines()
	i = 0
	while i < len(lines) and lines[i].lstrip().startswith("--"):
		i += 1
	# skip following blank line if present
	if i < len(lines) and lines[i].strip() == "":
		i += 1
	return "\n".join(lines[i:]) + ("\n" if i < len(lines) else "")


def _extract_created_from_header(path: Path) -> str:
	try:
		with path.open("r", encoding="utf8") as f:
			for _ in range(30):
				ln = f.readline()
				if not ln:
					break
				m = re.search(r"Created:\s*(.+)", ln)
				if m:
					return m.group(1).strip()
	except Exception:
		pass
	return None


def _build_header(path: Path, metadata: dict, created: str, modified: str, modified_by: str) -> str:
	# metadata dict -> format as key: value pairs
	meta_parts = []
	for k, v in (metadata or {}).items():
		meta_parts.append(f"{k}={v}")
	meta_line = "; ".join(meta_parts) if meta_parts else ""
	header_lines = [
		"-- ------------------------------------------------------------",
		f"-- File: {path.name}",
		f"-- Created: {created}",
		f"-- Modified: {modified}",
		f"-- Modified-By: {modified_by}",
	]
	if meta_line:
		header_lines.append(f"-- Metadata: {meta_line}")
	header_lines.append("-- ------------------------------------------------------------")
	header_lines.append("")
	return "\n".join(header_lines)


def write_if_changed(path: Path, content_sql: str, metadata: dict = None) -> bool:
	"""Write SQL content to path only if the SQL content changed.

	Header (comment box) is prepended but not included in hash comparisons.
	Returns True if file was written/updated (or would be in dry-run), False if unchanged.
	"""
	# Register expected main file and sidecar (store sidecars in a sibling _sha256 folder)
	EXPECTED_PATHS.add(str(path.resolve()))
	sidecar_dir = path.parent / '_sha256'
	sidecar_dir.mkdir(parents=True, exist_ok=True)
	sidecar_path = sidecar_dir / (path.name + '.sha256')
	EXPECTED_PATHS.add(str(sidecar_path.resolve()))

	sql_data = content_sql if content_sql.endswith("\n") else content_sql + "\n"
	sql_bytes = sql_data.encode("utf8")
	new_hash = compute_hash_bytes(sql_bytes)
	# Write/read sidecar from the dedicated _sha256 folder
	hash_path = path.parent / '_sha256' / (path.name + '.sha256')

	# Quick sidecar check
	if hash_path.exists():
		try:
			stored = hash_path.read_text(encoding="utf8").strip()
			if stored == new_hash:
				if VERBOSE:
					print(f"Unchanged (sidecar): {path}")
				return False
		except Exception:
			pass

	# If file exists, compute existing SQL-only hash by stripping header
	if path.exists():
		try:
			existing_full = path.read_text(encoding="utf8")
			existing_sql = _strip_leading_comment_block(existing_full)
			existing_hash = compute_hash_bytes(existing_sql.encode("utf8"))
			if existing_hash == new_hash:
				# ensure sidecar exists for next runs
				try:
					hash_path.write_text(new_hash, encoding="utf8")
				except Exception:
					pass
				if VERBOSE:
					print(f"Unchanged (content): {path}")
				return False
		except Exception:
			pass

	# At this point content has changed (or we couldn't determine); write file
	created = _extract_created_from_header(path) or datetime.datetime.now(datetime.timezone.utc).isoformat()
	modified = datetime.datetime.now(datetime.timezone.utc).isoformat()
	modified_by = CURRENT_USER or os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"

	header = _build_header(path, metadata, created, modified, modified_by)
	final = header + sql_data

	if DRY_RUN:
		if VERBOSE:
			print(f"[DRY] Would write: {path}")
		return True

	atomic_write(path, final.encode("utf8"))
	try:
		hash_path.write_text(new_hash, encoding="utf8")
	except Exception:
		pass
	if VERBOSE:
		print(f"Wrote: {path}")
	return True


def run_pg_dump(conninfo: dict, extra_args: list) -> str:
	cmd = [
		"pg_dump",
		"-h", conninfo["host"],
		"-p", str(conninfo["port"]),
		"-U", conninfo["user"],
		"-d", conninfo["dbname"],
		"--format", "plain",
		"--schema-only",
		"--no-owner",
		"--no-privileges",
	] + extra_args

	env = os.environ.copy()
	# Set password from connection info if available
	if "password" in conninfo and conninfo["password"]:
		env["PGPASSWORD"] = conninfo["password"]
	
	proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
	if proc.returncode != 0:
		raise RuntimeError(proc.stderr.strip() or f"pg_dump failed: {proc.returncode}")
	return proc.stdout


def dump_tables(conn, conninfo: dict, outdir: Path, include_data: bool = False) -> int:
	cur = conn.cursor()
	cur.execute(
		"""
		SELECT table_schema, table_name
		FROM information_schema.tables
		WHERE table_schema NOT IN ('pg_catalog','information_schema','pg_toast')
		  AND table_type = 'BASE TABLE'
		ORDER BY table_schema, table_name
		"""
	)
	rows = cur.fetchall()
	changed = 0
	
	# Get the full schema dump for each schema
	schema_tables = {}
	schema_ddl = {}
	for schema, _ in rows:
		if schema not in schema_ddl:
			try:
				schema_ddl[schema] = run_pg_dump(conninfo, ["--schema-only", "--schema", schema])
			except Exception as e:
				print(f"Warning: failed to dump schema {schema}: {e}", file=sys.stderr)
				schema_ddl[schema] = ""
			schema_tables[schema] = parse_schema_for_tables(schema_ddl[schema])

	for schema, name in rows:
		fullname = f"{schema}.{name}"
		target = outdir / "tables" / schema / f"{name}.psql"

		# Extract this table's DDL from the schema dump for its schema
		tables_dict = schema_tables.get(schema, {})
		ddl = None
		# Try both qualified and unqualified names
		if name in tables_dict:
			ddl = tables_dict[name]
		elif f"{schema}.{name}" in tables_dict:
			ddl = tables_dict[f"{schema}.{name}"]
		if ddl:
			ddl = normalize_sql(ddl)
			if write_if_changed(target, ddl):
				changed += 1
				print(f"Updated: {target}")
		else:
			print(f"Warning: table {name} not found in schema dump for schema {schema}", file=sys.stderr)

		if include_data:
			data_target = outdir / "tables" / schema / f"{name}.data.psql"
			try:
				# For data, we can try the individual table approach
				data_sql = run_pg_dump(conninfo, ["--data-only", "--table", f'{schema}."{name}"'])
				data_sql = normalize_sql(data_sql)
				if write_if_changed(data_target, data_sql):
					changed += 1
					print(f"Updated: {data_target}")
			except Exception as e:
				print(f"Warning: failed to dump data for {fullname}: {e}", file=sys.stderr)
				continue

	cur.close()
	return changed


def parse_schema_for_tables(schema_ddl: str) -> dict:
	"""Parse full schema DDL and extract ALL code related to each table (CREATE TABLE, ALTER TABLE, constraints, etc.)"""
	tables = {}
	lines = schema_ddl.split('\n')
	current_table = None
	current_table_lines = []
	in_table = False
	current_schema = None
	
	for line in lines:
		# Look for CREATE TABLE statements
		m = re.match(r'CREATE TABLE (?:([\w]+)\.)?"?([^\"]+)"?', line.strip())
		if m:
			if current_table and current_table_lines:
				# Save previous table (remove trailing empty lines)
				while current_table_lines and not current_table_lines[-1].strip():
					current_table_lines.pop()
				# Clean up table name keys
				key = current_table.strip().rstrip(' (')
				tables[key] = '\n'.join(current_table_lines)
				if current_schema:
					tables[f"{current_schema}.{key}"] = '\n'.join(current_table_lines)
			current_schema = m.group(1) or 'public'
			current_table = m.group(2)
			current_table_lines = [line]
			in_table = True
		elif in_table:
			# Check if this line ends the current table (CREATE TABLE block ends with );)
			if line.strip().endswith(');') and 'CREATE TABLE' in '\n'.join(current_table_lines):
				current_table_lines.append(line)
				if current_table and current_table_lines:
					while current_table_lines and not current_table_lines[-1].strip():
						current_table_lines.pop()
					key = current_table.strip().rstrip(' (')
					tables[key] = '\n'.join(current_table_lines)
					if current_schema:
						tables[f"{current_schema}.{key}"] = '\n'.join(current_table_lines)
				current_table = None
				current_table_lines = []
				in_table = False
			elif line.strip().startswith('--') and ('Name:' in line and ('Type: TABLE' in line or 'Type: SEQUENCE' in line or 'Type: INDEX' in line)):
				if current_table and current_table_lines:
					while current_table_lines and not current_table_lines[-1].strip():
						current_table_lines.pop()
					key = current_table.strip().rstrip(' (')
					tables[key] = '\n'.join(current_table_lines)
					if current_schema:
						tables[f"{current_schema}.{key}"] = '\n'.join(current_table_lines)
				current_table = None
				current_table_lines = []
				in_table = False
			elif line.strip().startswith('CREATE '):
				if current_table and current_table_lines:
					while current_table_lines and not current_table_lines[-1].strip():
						current_table_lines.pop()
					key = current_table.strip().rstrip(' (')
					tables[key] = '\n'.join(current_table_lines)
					if current_schema:
						tables[f"{current_schema}.{key}"] = '\n'.join(current_table_lines)
				current_table = None
				current_table_lines = []
				in_table = False
			else:
				current_table_lines.append(line)
	
	# Save the last table
	if current_table and current_table_lines:
		while current_table_lines and not current_table_lines[-1].strip():
			current_table_lines.pop()
		key = current_table.strip().rstrip(' (')
		tables[key] = '\n'.join(current_table_lines)
		if current_schema:
			tables[f"{current_schema}.{key}"] = '\n'.join(current_table_lines)
	return tables


def dump_matviews(conn, conninfo: dict, outdir: Path) -> int:
	cur = conn.cursor()
	cur.execute(
		"""
		SELECT n.nspname, c.relname
		FROM pg_class c
		JOIN pg_namespace n ON c.relnamespace = n.oid
		WHERE c.relkind = 'm' AND n.nspname NOT IN ('pg_catalog','information_schema','pg_toast')
		ORDER BY n.nspname, c.relname
		"""
	)
	rows = cur.fetchall()
	changed = 0
	for schema, name in rows:
		fullname = f"{schema}.{name}"
		target = outdir / "matviews" / schema / f"{name}.psql"
		try:
			ddl = run_pg_dump(conninfo, ["--table", fullname])
		except Exception as e:
			print(f"Warning: failed to dump matview {fullname}: {e}", file=sys.stderr)
			continue
		if write_if_changed(target, ddl):
			changed += 1
			print(f"Updated: {target}")
	cur.close()
	return changed


def dump_sequences(conn, conninfo: dict, outdir: Path) -> int:
	cur = conn.cursor()
	cur.execute(
		"""
		SELECT n.nspname, c.relname
		FROM pg_class c
		JOIN pg_namespace n ON c.relnamespace = n.oid
		WHERE c.relkind = 'S' AND n.nspname NOT IN ('pg_catalog','information_schema','pg_toast')
		ORDER BY n.nspname, c.relname
		"""
	)
	rows = cur.fetchall()
	changed = 0
	
	# Get the full schema dump for each schema
	schema_sequences = {}
	schema_ddl = {}
	for schema, _ in rows:
		if schema not in schema_ddl:
			try:
				schema_ddl[schema] = run_pg_dump(conninfo, ["--schema-only", "--schema", schema])
			except Exception as e:
				print(f"Warning: failed to dump schema for sequences in {schema}: {e}", file=sys.stderr)
				schema_ddl[schema] = ""
			schema_sequences[schema] = parse_schema_for_sequences(schema_ddl[schema])

	for schema, name in rows:
		fullname = f"{schema}.{name}"
		target = outdir / "sequences" / schema / f"{name}.psql"

		# Extract this sequence's DDL from the schema dump for its schema
		sequences_dict = schema_sequences.get(schema, {})
		if name in sequences_dict:
			ddl = sequences_dict[name]
			ddl = normalize_sql(ddl)
			if write_if_changed(target, ddl):
				changed += 1
				print(f"Updated: {target}")
		else:
			print(f"Warning: sequence {name} not found in schema dump for schema {schema}", file=sys.stderr)

	cur.close()
	return changed


def parse_schema_for_sequences(schema_ddl: str) -> dict:
	"""Parse full schema DDL and extract individual sequence definitions"""
	sequences = {}
	lines = schema_ddl.split('\n')
	current_sequence = None
	current_sequence_lines = []
	in_sequence = False
	current_schema = None
    
	for line in lines:
		# Look for CREATE SEQUENCE statements
		m = re.match(r'CREATE SEQUENCE (?:([\w]+)\.)?"?([^"]+)"?', line.strip())
		if m:
			if current_sequence and current_sequence_lines:
				while current_sequence_lines and not current_sequence_lines[-1].strip():
					current_sequence_lines.pop()
				sequences[current_sequence] = '\n'.join(current_sequence_lines)
				if current_schema:
					sequences[f"{current_schema}.{current_sequence}"] = '\n'.join(current_sequence_lines)
			current_schema = m.group(1) or 'public'
			current_sequence = m.group(2)
			current_sequence_lines = [line]
			in_sequence = True
			continue
		elif in_sequence:
			current_sequence_lines.append(line)
			# Check if this line ends the current sequence
			if line.strip() == ';':
				# End of sequence definition
				if current_sequence and current_sequence_lines:
					while current_sequence_lines and not current_sequence_lines[-1].strip():
						current_sequence_lines.pop()
					sequences[current_sequence] = '\n'.join(current_sequence_lines)
					if current_schema:
						sequences[f"{current_schema}.{current_sequence}"] = '\n'.join(current_sequence_lines)
				current_sequence = None
				current_sequence_lines = []
				in_sequence = False
			elif line.strip().startswith('--') and ('Name:' in line):
				# New object comment - end sequence
				if current_sequence and current_sequence_lines:
					while current_sequence_lines and not current_sequence_lines[-1].strip():
						current_sequence_lines.pop()
					sequences[current_sequence] = '\n'.join(current_sequence_lines)
					if current_schema:
						sequences[f"{current_schema}.{current_sequence}"] = '\n'.join(current_sequence_lines)
				current_sequence = None
				current_sequence_lines = []
				in_sequence = False
	# Save the last sequence
	if current_sequence and current_sequence_lines:
		while current_sequence_lines and not current_sequence_lines[-1].strip():
			current_sequence_lines.pop()
		sequences[current_sequence] = '\n'.join(current_sequence_lines)
		if current_schema:
			sequences[f"{current_schema}.{current_sequence}"] = '\n'.join(current_sequence_lines)
	return sequences


def dump_functions_procedures(conn, outdir: Path) -> int:
	cur = conn.cursor()
	cur.execute(
		"""
		SELECT p.oid, n.nspname, p.proname, p.prokind, pg_get_functiondef(p.oid)
		FROM pg_proc p
		JOIN pg_namespace n ON p.pronamespace = n.oid
		WHERE n.nspname NOT IN ('pg_catalog','information_schema','pg_toast')
		  AND p.prokind IN ('f','p')
		ORDER BY n.nspname, p.proname
		"""
	)
	rows = cur.fetchall()
	changed = 0
	for oid, schema, name, prokind, definition in rows:
		folder = "procedures" if prokind == 'p' else "functions"
		target = outdir / folder / schema / f"{name}.psql"
		# Normalize and write the function/procedure DDL
		ddl = normalize_sql(definition)
		if write_if_changed(target, ddl):
			changed += 1
			print(f"Updated: {target}")
	cur.close()
	return changed


def dump_triggers(conn, outdir: Path) -> int:
	cur = conn.cursor()
	cur.execute(
		"""
		SELECT t.oid, n.nspname, c.relname, t.tgname, pg_get_triggerdef(t.oid)
		FROM pg_trigger t
		JOIN pg_class c ON t.tgrelid = c.oid
		JOIN pg_namespace n ON c.relnamespace = n.oid
		WHERE NOT t.tgisinternal AND n.nspname NOT IN ('pg_catalog','information_schema','pg_toast')
		ORDER BY n.nspname, c.relname, t.tgname
		"""
	)
	rows = cur.fetchall()
	changed = 0
	for oid, schema, table, tgname, definition in rows:
		fname = f"{table}__{tgname}.psql"
		target = outdir / "triggers" / schema / fname
		content = f"-- Trigger: {schema}.{table}.{tgname}\n\n{definition}\n"
		if write_if_changed(target, content):
			changed += 1
			if VERBOSE:
				print(f"Updated: {target}")
	cur.close()
	return changed


def get_user_from_pgpass(host: str, port: int, dbname: str) -> str:
    """Get the user from pgpass.conf for the given connection parameters"""
    proj_pg = str(Path(__file__).resolve().parent / 'pgpass.conf')
    if not os.path.isfile(proj_pg):
        raise FileNotFoundError(f"pgpass.conf not found in project folder: {proj_pg}. Copy pgpass.conf.example and update with your credentials.")
    hosts_to_try = [host]
    for h in ("127.0.0.1", "localhost"):
        if h not in hosts_to_try:
            hosts_to_try.append(h)
    try:
        with open(proj_pg, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':')
                if len(parts) != 5:
                    continue
                pg_host, pg_port, pg_db, pg_user, pg_pass = parts
                for host_try in hosts_to_try:
                    if ((pg_host == '*' or pg_host == host_try) and 
                        (pg_port == '*' or pg_port == str(port)) and 
                        (pg_db == '*' or pg_db == dbname)):
                        return pg_user
    except Exception as e:
        raise RuntimeError(f"Error reading pgpass.conf: {e}")
    raise ValueError(f"No matching entry found in pgpass.conf for host={host}, port={port}, dbname={dbname}")


def dump_views(conn, conninfo: dict, outdir: Path) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.views
        WHERE table_schema NOT IN ('pg_catalog','information_schema','pg_toast')
        ORDER BY table_schema, table_name
        """
    )
    rows = cur.fetchall()
    changed = 0
    for schema, name in rows:
        fullname = f"{schema}.{name}"
        target = outdir / "views" / schema / f"{name}.psql"
        try:
            ddl = run_pg_dump(conninfo, ["--table", fullname])
        except Exception as e:
            print(f"Warning: failed to dump view {fullname}: {e}", file=sys.stderr)
            continue
        if write_if_changed(target, ddl):
            changed += 1
            print(f"Updated: {target}")
    cur.close()
    return changed


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dump Postgres objects into per-object files; update only changed files.")
    ap.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    ap.add_argument("--port", default=int(os.environ.get("PGPORT", 5432)), type=int)
    ap.add_argument("--user", default=None, help="Database user (auto-detected from pgpass if not specified)")
    ap.add_argument("--dbname", required=True)
    ap.add_argument("--include-data", action="store_true", help="Also dump per-table data files (data-only)")
    ap.add_argument("--dry-run", action="store_true", help="Show changes but don't write files")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--out", default=None, help="Optional root output directory; defaults to script folder")
    args = ap.parse_args()

    # Auto-detect user from pgpass if not specified
    if not args.user:
        args.user = get_user_from_pgpass(args.host, args.port, args.dbname)
        if args.verbose:
            print(f"Auto-detected user from pgpass: {args.user}")

    conninfo = {"host": args.host, "port": args.port, "user": args.user, "dbname": args.dbname}

    # Determine output root: user-specified or script folder
    if args.out:
        out_root = Path(args.out).resolve()
    else:
        out_root = Path(__file__).resolve().parent

    outdir = out_root / args.dbname
    outdir.mkdir(parents=True, exist_ok=True)

    global DRY_RUN, VERBOSE
    DRY_RUN = args.dry_run
    VERBOSE = args.verbose

    # Ensure pgpass is used if present: prioritize PGPASSFILE env, then common locations
    set_pgpass_env()
    conninfo_try = {"host": args.host, "port": args.port, "user": args.user, "dbname": args.dbname}

    def connect_with_retries(conninfo):
        last_exc = None
        hosts_to_try = [conninfo["host"]]
        for h in ("127.0.0.1", "localhost"):
            if h not in hosts_to_try:
                hosts_to_try.append(h)
        for h in hosts_to_try:
            try:
                if VERBOSE:
                    print(f"Trying connect host={h}")
                # Ensure password is loaded from pgpass (if present) and pass it explicitly
                pword = load_password_from_pgpass(h, conninfo["port"], conninfo["dbname"], conninfo["user"])
                if pword:
                    # Store password in conninfo for pg_dump usage
                    conninfo["password"] = pword
                    return psycopg2.connect(host=h, port=conninfo["port"], user=conninfo["user"], dbname=conninfo["dbname"], password=pword)
                else:
                    return psycopg2.connect(host=h, port=conninfo["port"], user=conninfo["user"], dbname=conninfo["dbname"])
            except Exception as e:
                last_exc = e
                if VERBOSE:
                    print(f"Connect failed for host={h}: {e}")
                continue
        raise last_exc

    conn = connect_with_retries(conninfo_try)

    total_changed = 0
    try:
        total_changed += dump_tables(conn, conninfo, outdir, include_data=args.include_data)
        total_changed += dump_views(conn, conninfo, outdir)
        total_changed += dump_matviews(conn, conninfo, outdir)
        total_changed += dump_sequences(conn, conninfo, outdir)
        total_changed += dump_functions_procedures(conn, outdir)
        total_changed += dump_triggers(conn, outdir)
    finally:
        conn.close()

    print(f"Done. Total updated files: {total_changed}")

    # cleanup stale files: move anything not in EXPECTED_PATHS to _stale
    cleanup_root = outdir
    stale_root = cleanup_root / "_stale"
    moved = 0
    for p in cleanup_root.rglob("*"):
        if p.is_file():
            rp = str(p.resolve())
            if rp.endswith('.sha256'):
                mainp = rp[:-7]
                if mainp in EXPECTED_PATHS:
                    continue
            if rp not in EXPECTED_PATHS:
                rel = p.relative_to(cleanup_root)
                dest = stale_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if DRY_RUN:
                    if VERBOSE:
                        print(f"[DRY] Would move stale: {p} -> {dest}")
                    moved += 1
                else:
                    shutil.move(str(p), str(dest))
                    moved += 1
                    if VERBOSE:
                        print(f"Moved stale: {p} -> {dest}")
    if VERBOSE:
        print(f"Stale files moved: {moved}")

if __name__ == "__main__":
    main()
