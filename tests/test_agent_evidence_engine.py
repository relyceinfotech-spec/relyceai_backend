from app.agent.evidence_engine import (
    extract_facts_from_workspace,
    extract_claims_from_facts,
    verify_claims,
    rank_evidence,
)


class _Workspace:
    def __init__(self, facts):
        self.facts = facts


def test_extract_facts_supports_structured_value_shape():
    ws = _Workspace(
        [
            {
                "entity": "Max Verstappen",
                "attribute": "2024 F1 champion",
                "value": "Max Verstappen won 2024 F1 Drivers Championship",
                "time": "2024",
                "source": "https://formula1.com",
                "source_trust": 0.95,
                "evidence_span": "Official standings page",
                "confidence": "high",
            }
        ]
    )

    facts = extract_facts_from_workspace(ws)
    assert len(facts) == 1
    assert facts[0]["value"].startswith("Max Verstappen")
    assert facts[0]["source"] == "https://formula1.com"
    assert facts[0]["source_trust"] == 0.95


def test_ranking_penalizes_disagreement_with_agreement_ratio():
    facts = [
        {
            "entity": "2024 F1 championship winner",
            "attribute": "winner",
            "value": "Max Verstappen",
            "time": "2024",
            "source": "https://formula1.com",
            "source_trust": 0.95,
            "evidence_span": "official table",
        },
        {
            "entity": "2024 F1 championship winner",
            "attribute": "winner",
            "value": "Max Verstappen",
            "time": "2024",
            "source": "https://fia.com",
            "source_trust": 0.95,
            "evidence_span": "fia standings",
        },
        {
            "entity": "2024 F1 championship winner",
            "attribute": "winner",
            "value": "Someone Else",
            "time": "2024",
            "source": "https://random-blog.example",
            "source_trust": 0.5,
            "evidence_span": "blog post",
        },
    ]

    claims = extract_claims_from_facts(facts)
    verification = verify_claims(claims, facts)
    ranked = rank_evidence(facts, verification)

    assert ranked
    assert ranked[0]["value"] == "Max Verstappen"
    assert ranked[0]["agreement_ratio"] >= ranked[-1]["agreement_ratio"]
