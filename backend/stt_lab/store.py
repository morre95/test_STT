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
                CREATE TABLE IF NOT EXISTS speaker_references (
                    recording_id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS speaker_samples (
                    id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
                    data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS speaker_jobs (
                    id TEXT PRIMARY KEY, recording_id TEXT NOT NULL,
                    data TEXT NOT NULL);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(recordings)")}
            if "metadata" not in columns:
                db.execute("ALTER TABLE recordings ADD COLUMN metadata TEXT")
            # A process crash must not leave completed-looking tests behind.
            for row in db.execute("SELECT id, data FROM runs").fetchall():
                data = json.loads(row["data"])
                if data["status"] == "running":
                    data.update(status="failed", error="Server restarted during this test")
                    db.execute("UPDATE runs SET data=? WHERE id=?", (json.dumps(data), row["id"]))
            for row in db.execute("SELECT id, data FROM speaker_jobs").fetchall():
                data = json.loads(row["data"])
                if data["status"] in {"queued", "running"}:
                    data.update(status="failed", error="Server restarted during this benchmark")
                    db.execute("UPDATE speaker_jobs SET data=? WHERE id=?",
                               (json.dumps(data), row["id"]))

    def connect(self):
        db = sqlite3.connect(self.root / "lab.sqlite3")
        db.row_factory = sqlite3.Row
        return db

    def create_recording(self, *, source="live", original_name=None):
        ident = str(uuid4())
        metadata = json.dumps({"source": source, "original_name": original_name},
                              ensure_ascii=False)
        with self.connect() as db:
            db.execute("INSERT INTO recordings(id,created_at,metadata) VALUES(?,?,?)",
                       (ident, datetime.now(UTC).isoformat(), metadata))
        return ident

    def audio_path(self, ident):
        self.recording(ident)  # Only database-owned identifiers can resolve file paths.
        return self.root / f"{ident}.wav"

    def recording(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        result = dict(row)
        result["metadata"] = json.loads(result.get("metadata") or "{}")
        return result

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
            recording["metadata"] = json.loads(recording.get("metadata") or "{}")
            recording["speaker_reference"] = self.speaker_reference(recording["id"])
            recording["runs"] = [{**run, **accuracy(recording["reference"], run["text"])}
                                 for run in runs if run["recording_id"] == recording["id"]]
        return recordings

    def set_reference(self, ident, reference):
        self.recording(ident)
        with self.connect() as db:
            db.execute("UPDATE recordings SET reference=? WHERE id=?", (reference, ident))

    def set_speaker_reference(self, ident, data):
        self.recording(ident)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO speaker_references VALUES(?,?)",
                       (ident, json.dumps(data, ensure_ascii=False)))

    def speaker_reference(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT data FROM speaker_references WHERE recording_id=?",
                             (ident,)).fetchone()
        return json.loads(row["data"]) if row else None

    def create_profile(self, name):
        ident = str(uuid4())
        data = {"id": ident, "name": name, "created_at": datetime.now(UTC).isoformat()}
        with self.connect() as db:
            db.execute("INSERT INTO speaker_profiles VALUES(?,?)",
                       (ident, json.dumps(data, ensure_ascii=False)))
        return data

    def profile(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT data FROM speaker_profiles WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        data = json.loads(row["data"])
        data["samples"] = self.profile_samples(ident)
        return data

    def profiles(self):
        with self.connect() as db:
            rows = db.execute("SELECT data FROM speaker_profiles").fetchall()
        values = [json.loads(row["data"]) for row in rows]
        for value in values:
            value["samples"] = self.profile_samples(value["id"])
        return sorted(values, key=lambda item: item["name"].casefold())

    def update_profile(self, ident, name):
        profile = self.profile(ident)
        profile.pop("samples", None)
        profile["name"] = name
        with self.connect() as db:
            db.execute("UPDATE speaker_profiles SET data=? WHERE id=?",
                       (json.dumps(profile, ensure_ascii=False), ident))
        return self.profile(ident)

    def delete_profile(self, ident):
        profile = self.profile(ident)
        with self.connect() as db:
            db.execute("DELETE FROM speaker_samples WHERE profile_id=?", (ident,))
            db.execute("DELETE FROM speaker_profiles WHERE id=?", (ident,))
        return profile

    def add_profile_sample(self, profile_id, duration, original_name=None):
        self.profile(profile_id)
        ident = str(uuid4())
        data = {"id": ident, "profile_id": profile_id, "duration": duration,
                "original_name": original_name, "created_at": datetime.now(UTC).isoformat()}
        with self.connect() as db:
            db.execute("INSERT INTO speaker_samples VALUES(?,?,?)",
                       (ident, profile_id, json.dumps(data, ensure_ascii=False)))
        return data

    def profile_samples(self, profile_id):
        with self.connect() as db:
            rows = db.execute("SELECT data FROM speaker_samples WHERE profile_id=?",
                              (profile_id,)).fetchall()
        return [json.loads(row["data"]) for row in rows]

    def profile_sample(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT data FROM speaker_samples WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        return json.loads(row["data"])

    def delete_profile_sample(self, ident):
        data = self.profile_sample(ident)
        with self.connect() as db:
            db.execute("DELETE FROM speaker_samples WHERE id=?", (ident,))
        return data

    def profile_sample_path(self, profile_id, sample_id):
        sample = self.profile_sample(sample_id)
        if sample["profile_id"] != profile_id:
            raise KeyError(sample_id)
        folder = self.root / "profiles" / profile_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{sample_id}.wav"

    def save_speaker_job(self, data):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO speaker_jobs VALUES(?,?,?)",
                       (data["id"], data["recording_id"],
                        json.dumps(data, ensure_ascii=False)))

    def speaker_job(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT data FROM speaker_jobs WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        return json.loads(row["data"])

    def speaker_jobs(self):
        with self.connect() as db:
            rows = db.execute("SELECT data FROM speaker_jobs").fetchall()
        return sorted((json.loads(row["data"]) for row in rows),
                      key=lambda item: item["created_at"], reverse=True)
