import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(os.environ.get('FORGE_DATA_DIR', Path(__file__).resolve().parents[1] / '.forge')).resolve()


@contextmanager
def connection():
    ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ROOT / 'forge.sqlite3', timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize():
    with connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL, progress TEXT, result TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL)')


def save(kind, data, record_id=None):
    record_id = record_id or uuid.uuid4().hex
    with connection() as db:
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?)', (record_id, kind, json.dumps(data), time.time()))
    return {'id': record_id, **data}


def get(record_id, kind):
    with connection() as db:
        row = db.execute('SELECT * FROM records WHERE id=? AND kind=?', (record_id, kind)).fetchone()
    if not row:
        raise ValueError(f'{kind.capitalize()} not found.')
    return {'id': row['id'], **json.loads(row['data'])}


def list_records(kind):
    with connection() as db:
        rows = db.execute('SELECT * FROM records WHERE kind=? ORDER BY created DESC', (kind,)).fetchall()
    return [{'id': r['id'], **json.loads(r['data'])} for r in rows]


def create_job(kind, payload):
    with connection() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM jobs WHERE status IN ('queued','running','paused')").fetchone():
            raise RuntimeError('Another model job is already running.')
        job_id = uuid.uuid4().hex
        now = time.time()
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)', (job_id, kind, json.dumps(payload), 'queued', 'Queued', None, None, now, now,))
    return job_id


def job(job_id):
    with connection() as db:
        row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    if not row:
        raise ValueError('Job not found.')
    result = dict(row)
    result['payload'] = json.loads(result['payload'])
    result['result'] = json.loads(result['result']) if result['result'] else None
    return result


def update_job(job_id, status, progress=None, result=None, error=None):
    with connection() as db:
        db.execute("UPDATE jobs SET status=?,progress=?,result=?,error=?,updated=? WHERE id=? AND (status!='cancelled' OR ?='cancelled')", (status, progress, json.dumps(result) if result is not None else None, error, time.time(), job_id, status))


def active_job():
    with connection() as db:
        row = db.execute("SELECT id FROM jobs WHERE status IN ('queued','running','paused') ORDER BY created LIMIT 1").fetchone()
    return job(row['id']) if row else None


def discard_job_output(job_id):
    outputs = [(ROOT / 'outputs' / job_id).resolve()]
    try:
        current = job(job_id)
        if current['payload'].get('output_path'):
            outputs.append(Path(current['payload']['output_path']).resolve())
    except (KeyError, ValueError, OSError):
        pass
    output_root = (ROOT / 'outputs').resolve()
    for output in outputs:
        try:
            is_internal = output.parent == output_root
            is_marked = (output / '.forge-operation').is_file()
            if output.exists() and (is_internal or is_marked):
                shutil.rmtree(output, ignore_errors=True)
        except OSError:
            # Cleanup must not terminate the persistent worker.
            continue


def cancel_job(job_id, force=False):
    current = job(job_id)
    if force or current['status'] in ('queued', 'running', 'paused'):
        update_job(job_id, 'cancelled', 'Operation stopped', error='Stopped by user')
        if current['kind'] == 'optimize':
            discard_job_output(job_id)
    return job(job_id)


def recover_jobs():
    with connection() as db:
        db.execute("UPDATE jobs SET status='interrupted',error='Backend stopped before this job completed.',updated=? WHERE status IN ('queued','running','paused')", (time.time(),))


def pause_job(job_id):
    current = job(job_id)
    if current['status'] not in ('queued', 'running'):
        raise ValueError('Only an active operation can be paused.')
    progress = 'Queued operation paused' if current['status'] == 'queued' else 'Operation paused'
    update_job(job_id, 'paused', progress)
    return job(job_id)


def resume_job(job_id):
    current = job(job_id)
    if current['status'] != 'paused':
        raise ValueError('Only a paused operation can be resumed.')
    queued = current.get('progress') == 'Queued operation paused'
    update_job(job_id, 'queued' if queued else 'running', 'Queued' if queued else 'Resuming operation')
    return job(job_id)


def next_job():
    with connection() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if row:
            db.execute("UPDATE jobs SET status='running',progress='Starting',updated=? WHERE id=? AND status='queued'", (time.time(), row['id']))
    return job(row['id']) if row else None
