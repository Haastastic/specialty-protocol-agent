"""Sanity tests for retrieval — pure logic, no API calls."""

from src.protocol_agent.retrieval import ProtocolIndex


def test_index_loads_both_specialties():
    index = ProtocolIndex("protocols")
    assert "dermatology" in index.specialties
    assert "cardiology" in index.specialties


def test_retrieve_returns_relevant_chunks():
    index = ProtocolIndex("protocols")
    results = index.retrieve("I have a suspicious mole that's changing color", specialty="dermatology")
    assert len(results) > 0
    # escalation triggers must always surface for a melanoma-adjacent query, regardless of ranking
    assert any(c.section == "escalation_triggers" for c in results)


def test_specialty_detection():
    index = ProtocolIndex("protocols")
    assert index.detect_specialty("I need a cardiology appointment") == "cardiology"
    assert index.detect_specialty("just a general question") is None
