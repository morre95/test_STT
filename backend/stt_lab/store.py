import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .metrics import accuracy


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    reference TEXT, duration REAL NOT NULL DEFAULT 0,
                    complete INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, recording_id TEXT NOT NULL,
                    data TEXT NOT NULL);
            """)
            # A process crash must not leave completed-looking tests behind.
            for row in db.execute("SELECT id, data FROM runs").fetchall():
                data = json.loads(row["data"])
                if data["status"] == "running":
                    data.update(status="failed", error="Server restarted during this test")
                    db.execute("UPDATE runs SET data=? WHERE id=?", (json.dumps(data), row["id"]))

    def connect(self):
        db = sqlite3.connect(self.root / "lab.sqlite3")
        db.row_factory = sqlite3.Row
        return db

    def create_recording(self):
        ident = str(uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO recordings(id,created_at) VALUES(?,?)",
                       (ident, datetime.now(UTC).isoformat()))
        return ident

    def audio_path(self, ident):
        self.recording(ident)  # Only database-owned identifiers can resolve file paths.
        return self.root / f"{ident}.wav"

    def recording(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        return dict(row)

    def finish_recording(self, ident, duration, complete):
        with self.connect() as db:
            db.execute("UPDATE recordings SET duration=?, complete=? WHERE id=?",
                       (duration, int(complete), ident))

    def save_run(self, data):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?)",
                       (data["id"], data["recording_id"], json.dumps(data, ensure_ascii=False)))

    def run(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT data FROM runs WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        return json.loads(row["data"])

    def history(self):
        with self.connect() as db:
            recordings = [dict(row) for row in db.execute(
                "SELECT * FROM recordings ORDER BY created_at DESC")]
            runs = [json.loads(row[0]) for row in db.execute("SELECT data FROM runs")]
        for recording in recordings:
            recording["runs"] = [{**run, **accuracy(recording["reference"], run["text"])}
                                 for run in runs if run["recording_id"] == recording["id"]]
        return recordings

    def set_reference(self, ident, reference):
        self.recording(ident)
        with self.connect() as db:
            db.execute("UPDATE recordings SET reference=? WHERE id=?", (reference, ident))
