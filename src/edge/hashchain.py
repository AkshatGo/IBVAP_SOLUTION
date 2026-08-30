"""
Hash Chain — Tamper-evident event logging with JSONL persistence.

Each event includes the SHA-256 hash of the previous event,
creating an immutable audit trail. Events are appended to a
JSONL file so they survive process restarts.
"""
import hashlib
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EventRecord:
    """A single event in the hash chain."""
    event_id: str
    timestamp: str
    event_type: str
    site_id: str
    camera_id: str
    severity: str
    payload: dict
    prev_hash: str = ""
    hash: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "site_id": self.site_id,
            "camera_id": self.camera_id,
            "severity": self.severity,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventRecord":
        return cls(
            event_id=d["event_id"],
            timestamp=d["timestamp"],
            event_type=d["event_type"],
            site_id=d["site_id"],
            camera_id=d["camera_id"],
            severity=d["severity"],
            payload=d["payload"],
            prev_hash=d.get("prev_hash", ""),
            hash=d.get("hash", ""),
        )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this record (excluding hash and prev_hash)."""
        d = self.to_dict()
        d.pop("hash", None)
        d.pop("prev_hash", None)
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


class HashChain:
    """
    Immutable, tamper-evident event log with JSONL persistence.

    Each record's hash includes the previous record's hash,
    so editing any past record breaks the chain. Events are
    appended to a JSONL file so they survive process restarts.
    """

    def __init__(self, path: str = "data/hashchain.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.chain: List[EventRecord] = []
        self._head_hash = "0" * 64  # genesis hash
        self._load()

    def _load(self):
        """Load existing chain from JSONL file."""
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self.chain.append(EventRecord.from_dict(d))
                except (json.JSONDecodeError, KeyError):
                    continue  # skip corrupted lines
        if self.chain:
            self._head_hash = self.chain[-1].hash

    def add_event(
        self,
        event_id: str,
        event_type: str,
        site_id: str,
        camera_id: str,
        severity: str,
        payload: dict,
        timestamp: Optional[str] = None,
    ) -> EventRecord:
        """Add a new event to the chain and persist to disk."""
        record = EventRecord(
            event_id=event_id,
            timestamp=timestamp or datetime.utcnow().isoformat() + "Z",
            event_type=event_type,
            site_id=site_id,
            camera_id=camera_id,
            severity=severity,
            payload=payload,
            prev_hash=self._head_hash,
        )
        record.hash = record.compute_hash()
        self._head_hash = record.hash
        self.chain.append(record)

        # Persist to JSONL
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

        return record

    def verify(self) -> Tuple[bool, Optional[int]]:
        """
        Verify the integrity of the entire chain.
        Returns (is_valid: bool, broken_at: Optional[int])
        """
        if not self.chain:
            return True, None

        # Check genesis
        if self.chain[0].prev_hash != "0" * 64:
            return False, 0

        for i in range(len(self.chain)):
            record = self.chain[i]

            # Verify hash
            expected_hash = record.compute_hash()
            if record.hash != expected_hash:
                return False, i

            # Verify chain link
            if i > 0:
                if record.prev_hash != self.chain[i - 1].hash:
                    return False, i

        return True, None

    def get_head_hash(self) -> str:
        """Get the current head hash."""
        return self._head_hash

    def get_records(self, last_n: int = 10) -> List[dict]:
        """Get the last N records as dicts."""
        return [r.to_dict() for r in self.chain[-last_n:]]

    def get_all_records(self) -> List[dict]:
        """Get all records as dicts."""
        return [r.to_dict() for r in self.chain]

    def get_stats(self) -> dict:
        """Get chain statistics."""
        is_valid, broken_at = self.verify()
        return {
            "total_events": len(self.chain),
            "is_valid": is_valid,
            "broken_at": broken_at,
            "head_hash": self._head_hash[:32] + "...",
        }

    def export_json(self) -> str:
        """Export entire chain as JSON."""
        return json.dumps(self.get_all_records(), indent=2, default=str)

    def __len__(self):
        return len(self.chain)

    def __getitem__(self, idx):
        return self.chain[idx]


# ── CLI entry point ──────────────────────────────────────────────
# Run: python -m src.edge.hashchain [verify|dump|corrupt]
def main():
    import sys

    usage = (
        "Usage: python -m src.edge.hashchain <command>\n"
        "  verify  — Verify chain integrity and print result\n"
        "  dump    — Print all events as JSON\n"
        "  corrupt — Intentionally corrupt one record to demo chain break\n"
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]
    chain = HashChain()

    if cmd == "verify":
        is_valid, broken_at = chain.verify()
        if is_valid:
            print(f"✅ Chain VALID — {len(chain)} events verified")
        else:
            print(f"❌ Chain BROKEN at record index {broken_at}")
            print(f"   Record: {json.dumps(chain[broken_at].to_dict(), indent=2)}")
        sys.exit(0 if is_valid else 1)

    elif cmd == "dump":
        print(chain.export_json())
        sys.exit(0)

    elif cmd == "corrupt":
        if len(chain) < 2:
            print("Chain too short to corrupt (need at least 2 events)")
            sys.exit(1)
        # Corrupt the second-to-last record's payload
        idx = len(chain) - 2
        chain.chain[idx].payload["TAMPERED"] = True
        chain.chain[idx].hash = chain.chain[idx].compute_hash()
        # Rewrite the file
        with open(chain.path, "w") as f:
            for rec in chain.chain:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        print(f"🔓 Corrupted record at index {idx} — run 'verify' to see the break")
        sys.exit(0)

    else:
        print(usage)
        sys.exit(1)


if __name__ == "__main__":
    main()
