"""Hash chain — persistence, linkage, and tamper detection.

This is the audit-trail guarantee the whole evidentiary claim rests on, so
the tests check the property that matters: any edit to any record must make
verify() fail and point at the edited record.
"""
import json

import pytest

from src.edge.hashchain import HashChain


def add(chain, i, event_type="fence_intrusion", severity="high"):
    return chain.add_event(
        event_id=f"e{i:04d}",
        event_type=event_type,
        site_id="BOP-01",
        camera_id="CAM-01",
        severity=severity,
        payload={"index": i},
    )


def test_first_record_links_to_genesis(chain_path):
    chain = HashChain(path=chain_path)
    record = add(chain, 0)
    assert record.prev_hash == "0" * 64
    assert len(record.hash) == 64


def test_each_record_links_to_the_previous(chain_path):
    chain = HashChain(path=chain_path)
    records = [add(chain, i) for i in range(5)]
    for earlier, later in zip(records, records[1:]):
        assert later.prev_hash == earlier.hash


def test_chain_survives_restart(chain_path):
    """The reason this exists: an in-memory chain dies on reboot."""
    chain = HashChain(path=chain_path)
    for i in range(3):
        add(chain, i)
    head_before = chain.get_head_hash()

    reloaded = HashChain(path=chain_path)
    assert len(reloaded) == 3
    assert reloaded.get_head_hash() == head_before
    assert reloaded.verify() == (True, None)


def test_verify_passes_on_an_untouched_chain(chain_path):
    chain = HashChain(path=chain_path)
    for i in range(4):
        add(chain, i)
    assert chain.verify() == (True, None)


def test_verify_detects_a_tampered_payload(chain_path):
    chain = HashChain(path=chain_path)
    for i in range(4):
        add(chain, i)

    chain.chain[1].payload["index"] = 999

    is_valid, broken_at = chain.verify()
    assert is_valid is False
    assert broken_at == 1


def test_verify_detects_tampering_written_to_disk(chain_path):
    """Editing the JSONL directly must be caught on reload, not just in RAM."""
    chain = HashChain(path=chain_path)
    for i in range(4):
        add(chain, i)

    lines = [json.loads(line) for line in open(chain_path) if line.strip()]
    lines[2]["severity"] = "low"
    with open(chain_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    is_valid, broken_at = HashChain(path=chain_path).verify()
    assert is_valid is False
    assert broken_at == 2


def test_verify_detects_a_severed_link(chain_path):
    """Rehashing a record without fixing its successor still breaks the chain."""
    chain = HashChain(path=chain_path)
    for i in range(4):
        add(chain, i)

    chain.chain[1].payload["index"] = 999
    chain.chain[1].hash = chain.chain[1].compute_hash()  # self-consistent again

    is_valid, broken_at = chain.verify()
    assert is_valid is False
    assert broken_at == 2  # the link from 1 -> 2 is now wrong


def test_empty_chain_is_valid(chain_path):
    chain = HashChain(path=chain_path)
    assert len(chain) == 0
    assert chain.verify() == (True, None)


def test_stats_report_length_and_validity(chain_path):
    chain = HashChain(path=chain_path)
    for i in range(3):
        add(chain, i)
    stats = chain.get_stats()
    assert stats["total_events"] == 3
    assert stats["is_valid"] is True
    # get_stats truncates the hash for display; the full one comes from
    # get_head_hash().
    assert chain.get_head_hash().startswith(stats["head_hash"].removesuffix("..."))


@pytest.mark.parametrize("count", [1, 2, 10])
def test_length_matches_events_added(chain_path, count):
    chain = HashChain(path=chain_path)
    for i in range(count):
        add(chain, i)
    assert len(chain) == count
    assert len(chain.get_all_records()) == count
