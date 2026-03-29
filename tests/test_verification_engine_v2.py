from app.reasoning.verification_engine import verify_claims


def test_verify_claims_detects_verified_uncertain_conflicted():
    claims = [
        {"claim_id": "1", "entity": "A", "attribute": "winner", "value": "X"},
        {"claim_id": "2", "entity": "B", "attribute": "winner", "value": "Y"},
        {"claim_id": "3", "entity": "C", "attribute": "winner", "value": "Z"},
    ]
    facts = [
        {"entity": "A", "attribute": "winner", "value": "X", "source": "s1"},
        {"entity": "A", "attribute": "winner", "value": "X", "source": "s2"},
        {"entity": "B", "attribute": "winner", "value": "Y", "source": "s1"},
        {"entity": "B", "attribute": "winner", "value": "Q", "source": "s2"},
    ]

    out = verify_claims(claims, facts)

    assert len(out["verified_claims"]) == 1
    assert len(out["conflicted_claims"]) == 1
    assert len(out["uncertain_claims"]) == 1
    assert 0.0 <= out["confidence"] <= 1.0
