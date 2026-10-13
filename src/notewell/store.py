"""SQLite persistence: visits (encrypted), PHI vaults (encrypted), uploads (encrypted) and the audit log.

Only non-identifying index columns are stored in clear text (id, status, specialty, dates). The visit document (patient
demographics, transcript, note) and the vault are AES-GCM encrypted. The audit log holds who did what to which visit
and when, plus counts and hashes, never identifiers or clinical text.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from notewell.crypto import Cipher
from notewell.models import Visit
from notewell.phi.masker import Vault

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    specialty TEXT NOT NULL,
    visit_date TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    doc BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS vaults (visit_id TEXT PRIMARY KEY, blob BLOB NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS uploads (visit_id TEXT PRIMARY KEY, mime TEXT NOT NULL, blob BLOB NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    visit_id TEXT,
    detail TEXT NOT NULL DEFAULT '{}'
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path, cipher: Cipher) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.cipher = cipher
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ---- visits
    def save_visit(self, visit: Visit) -> None:
        doc = self.cipher.encrypt(visit.model_dump_json().encode("utf-8"), aad=f"visit:{visit.id}")
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO visits (id, status, specialty, visit_date, synthetic, updated_at, doc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                "specialty=excluded.specialty, visit_date=excluded.visit_date, updated_at=excluded.updated_at, "
                "doc=excluded.doc",
                (visit.id, visit.status, visit.specialty, visit.visit_date, int(visit.synthetic), _now(), doc),
            )

    def get_visit(self, visit_id: str) -> Visit | None:
        with self._lock:
            row = self._conn.execute("SELECT doc FROM visits WHERE id = ?", (visit_id,)).fetchone()
        if row is None:
            return None
        return Visit.model_validate_json(self.cipher.decrypt(row[0], aad=f"visit:{visit_id}"))

    def list_visits(self) -> list[Visit]:
        with self._lock:
            rows = self._conn.execute("SELECT id, doc FROM visits ORDER BY visit_date DESC, id").fetchall()
        return [Visit.model_validate_json(self.cipher.decrypt(doc, aad=f"visit:{vid}")) for vid, doc in rows]

    def visit_ids(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self._conn.execute("SELECT id FROM visits").fetchall()}

    def delete_visit(self, visit_id: str) -> None:
        with self._tx() as conn:
            for table, column in (("visits", "id"), ("vaults", "visit_id"), ("uploads", "visit_id")):
                conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (visit_id,))

    # ---- vaults
    def save_vault(self, visit_id: str, vault: Vault) -> None:
        blob = self.cipher.encrypt(json.dumps(vault.to_dict()).encode("utf-8"), aad=f"vault:{visit_id}")
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO vaults (visit_id, blob, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(visit_id) DO UPDATE SET blob=excluded.blob, updated_at=excluded.updated_at",
                (visit_id, blob, _now()),
            )

    def get_vault(self, visit_id: str) -> Vault | None:
        with self._lock:
            row = self._conn.execute("SELECT blob FROM vaults WHERE visit_id = ?", (visit_id,)).fetchone()
        if row is None:
            return None
        return Vault.from_dict(json.loads(self.cipher.decrypt(row[0], aad=f"vault:{visit_id}")))

    # ---- uploads (recorded audio)
    def save_upload(self, visit_id: str, data: bytes, mime: str) -> None:
        blob = self.cipher.encrypt(data, aad=f"upload:{visit_id}")
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO uploads (visit_id, mime, blob, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(visit_id) DO UPDATE SET mime=excluded.mime, blob=excluded.blob",
                (visit_id, mime, blob, _now()),
            )

    def get_upload(self, visit_id: str) -> tuple[bytes, str] | None:
        with self._lock:
            row = self._conn.execute("SELECT blob, mime FROM uploads WHERE visit_id = ?", (visit_id,)).fetchone()
        if row is None:
            return None
        return self.cipher.decrypt(row[0], aad=f"upload:{visit_id}"), row[1]

    # ---- audit
    def audit(self, actor: str, action: str, visit_id: str | None = None, **detail: Any) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO audit (ts, actor, action, visit_id, detail) VALUES (?, ?, ?, ?, ?)",
                (_now(), actor, action, visit_id, json.dumps(detail, sort_keys=True, default=str)),
            )

    def audit_log(self, *, visit_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT ts, actor, action, visit_id, detail FROM audit"
        params: tuple[Any, ...] = ()
        if visit_id:
            query += " WHERE visit_id = ?"
            params = (visit_id,)
        query += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(query, (*params, limit)).fetchall()
        return [
            {"ts": ts, "actor": actor, "action": action, "visit_id": vid, "detail": json.loads(detail)}
            for ts, actor, action, vid, detail in rows
        ]

    # ---- retention
    def purge(self, *, older_than_days: int, actor: str = "system") -> int:
        """Delete uploads and vaults of signed/exported visits older than the retention period; keep the note."""
        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM visits WHERE status IN ('signed', 'exported') AND updated_at < ?", (cutoff,)
            ).fetchall()
        count = 0
        for (visit_id,) in rows:
            with self._tx() as conn:
                count += conn.execute("DELETE FROM uploads WHERE visit_id = ?", (visit_id,)).rowcount
                conn.execute("DELETE FROM vaults WHERE visit_id = ?", (visit_id,))
            self.audit(actor, "retention.purge", visit_id, older_than_days=older_than_days)
        return count
