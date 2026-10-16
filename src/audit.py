"""Append-only audit log (JSON Lines: one JSON object per line).

Every decision, tool call, approval, rejection, error and simulated action of
the agent is recorded, so that a run can be reconstructed afterwards.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src import config


class AuditLog:
    def __init__(self, path=None):
        self.path = Path(path or config.AUDIT_LOG_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One identifier per agent run, to group the entries of that run.
        self.run_id = datetime.now(timezone.utc).strftime("RUN-%Y%m%dT%H%M%SZ")

    def record(self, event, po_id=None, **details):
        """Append one entry and return it."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "event": event,
            "po_id": po_id,
            "details": details,
        }
        # The file is only ever opened in append mode: entries are never rewritten.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry


def read_audit_log(path=None):
    """Return all entries of an audit log as a list of dictionaries."""
    path = Path(path or config.AUDIT_LOG_PATH)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
