from app.telemetry.high_stakes_backfill import backfill_high_stakes_daily
from app.telemetry import high_stakes_backfill as backfill_module


class _FakeDocSnapshot:
    def __init__(self, data):
        self._data = dict(data)

    def to_dict(self):
        return dict(self._data)


class _FakeDocumentRef:
    def __init__(self, collection, doc_id):
        self._collection = collection
        self.id = doc_id

    def get(self):
        return _FakeDocSnapshot(self._collection._docs.get(self.id, {}))

    def set(self, data, merge=True):
        if merge and self.id in self._collection._docs:
            merged = dict(self._collection._docs[self.id])
            merged.update(dict(data or {}))
            self._collection._docs[self.id] = merged
        else:
            self._collection._docs[self.id] = dict(data or {})


class _FakeCollection:
    def __init__(self, docs=None):
        self._docs = docs or {}

    def stream(self):
        return [_FakeDocSnapshot(v) for v in self._docs.values()]

    def document(self, doc_id):
        return _FakeDocumentRef(self, doc_id)


class _FakeDb:
    def __init__(self):
        self._collections = {
            "high_stakes_metrics_events": _FakeCollection(),
            "high_stakes_metrics_daily": _FakeCollection(),
        }

    def collection(self, name):
        return self._collections[name]


def test_backfill_high_stakes_daily_rolls_up(monkeypatch):
    db = _FakeDb()
    db.collection("high_stakes_metrics_events")._docs = {
        "e1": {
            "ts_epoch": 1710000000.0,
            "domain": "finance",
            "source_requirement_met": True,
            "recency_requirement_met": False,
            "confidence_level": "LOW",
        },
        "e2": {
            "ts_epoch": 1710000100.0,
            "domain": "finance",
            "source_requirement_met": False,
            "recency_requirement_met": True,
            "confidence_level": "HIGH",
        },
    }

    monkeypatch.setattr(backfill_module, "get_firestore_db", lambda: db)

    out = backfill_high_stakes_daily(start_ts=1709999000.0, end_ts=1710001000.0, max_events=1000)

    assert out["success"] is True
    assert out["events_scanned"] == 2
    assert out["daily_docs_updated"] == 1

    daily_docs = db.collection("high_stakes_metrics_daily")._docs
    assert len(daily_docs) == 1
    doc = next(iter(daily_docs.values()))
    assert doc["total"] == 2
    assert doc["source_met"] == 1
    assert doc["source_failed"] == 1
    assert doc["recency_met"] == 1
    assert doc["recency_failed"] == 1
    assert doc["low"] == 1
    assert doc["high"] == 1
