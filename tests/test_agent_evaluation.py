from app.agent.evaluation import EvaluationMetrics, check_regression_gate


def test_regression_gate_passes_for_small_changes():
    baseline = EvaluationMetrics(0.90, 0.92, 0.88, 3200, 0.40, 0.06)
    candidate = EvaluationMetrics(0.89, 0.91, 0.85, 3500, 0.38, 0.08)

    result = check_regression_gate(baseline, candidate)
    assert result['passed'] is True


def test_regression_gate_fails_for_large_degradation():
    baseline = EvaluationMetrics(0.90, 0.92, 0.88, 3200, 0.40, 0.06)
    candidate = EvaluationMetrics(0.82, 0.85, 0.70, 4500, 0.20, 0.20)

    result = check_regression_gate(baseline, candidate)
    assert result['passed'] is False
    assert 'answer_accuracy' in result['failures']
    assert 'claim_accuracy' in result['failures']
