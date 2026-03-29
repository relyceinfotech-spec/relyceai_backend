from app.reasoning.evidence_ranker import rank_evidence


def test_rank_evidence_prefers_supported_trusted_recent_facts():
    facts = [
        {"entity": "A", "attribute": "winner", "value": "Max", "time": "2026", "source": "https://formula1.com"},
        {"entity": "A", "attribute": "winner", "value": "Max", "time": "2026", "source": "https://fia.com"},
        {"entity": "A", "attribute": "winner", "value": "Other", "time": "2010", "source": "https://randomblog.example"},
    ]
    ranked = rank_evidence(facts, {"verified_claims": []})

    assert ranked[0]["value"] == "Max"
    assert ranked[0]["score"] >= ranked[-1]["score"]
