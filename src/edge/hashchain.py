"""
Hash Chain — Tamper-evident event logging.
Each event includes the SHA-256 hash of the previous event,
creating an immutable audit trail.
"""
import hashlib
import json
from typing import List, Dict, Optional
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

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this record (excluding hash field)."""
        d = self.to_dict()
        d.pop("hash", None)
        d.pop("prev_hash", None)
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


class HashChain:
    """
    Immutable, tamper-evident event log.
    
    Each record's hash includes the previous record's hash,
    so editing any past record breaks the chain.
    """

    def __init__(self):
        self.chain: List[EventRecord] = []
        self._head_hash = "0" * 64  # genesis hash

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
        """Add a new event to the chain."""
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
        return record

    def verify(self) -> tuple:
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
