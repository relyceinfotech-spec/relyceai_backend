from app.security import alert_notifier


def test_alert_notifier_deduplicates_by_code_and_range(monkeypatch):
    calls = []

    class _DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req, timeout=0):
        calls.append((req.full_url, timeout))
        return _DummyResp()

    monkeypatch.setenv("HS_ALERT_WEBHOOK_URL", "https://example.test/webhook")
    monkeypatch.setenv("HS_ALERT_COOLDOWN_SECONDS", "600")
    monkeypatch.setattr(alert_notifier.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(alert_notifier.time, "time", lambda: 1000.0)
    alert_notifier._LAST_ALERT_SENT.clear()

    payload = {
        "range": "all",
        "alerts": [{"code": "LOW_CONFIDENCE_SPIKE"}],
        "summary": {"total": 12},
    }

    alert_notifier.send_admin_alert(payload)
    alert_notifier.send_admin_alert(payload)

    assert len(calls) == 1


def test_alert_notifier_sends_again_after_cooldown(monkeypatch):
    calls = []

    class _DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req, timeout=0):
        calls.append((req.full_url, timeout))
        return _DummyResp()

    monkeypatch.setenv("HS_ALERT_WEBHOOK_URL", "https://example.test/webhook")
    monkeypatch.setenv("HS_ALERT_COOLDOWN_SECONDS", "10")
    monkeypatch.setattr(alert_notifier.request, "urlopen", _fake_urlopen)
    alert_notifier._LAST_ALERT_SENT.clear()

    t = {"value": 100.0}
    monkeypatch.setattr(alert_notifier.time, "time", lambda: t["value"])

    payload = {
        "range": "7d",
        "alerts": [{"code": "SOURCE_FAIL_SPIKE"}],
        "summary": {"total": 99},
    }

    alert_notifier.send_admin_alert(payload)
    t["value"] = 105.0
    alert_notifier.send_admin_alert(payload)
    t["value"] = 111.0
    alert_notifier.send_admin_alert(payload)

    assert len(calls) == 2
