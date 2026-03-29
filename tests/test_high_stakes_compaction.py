from datetime import datetime, timedelta, timezone

from app.telemetry.high_stakes_store import HighStakesStore
from app.telemetry import high_stakes_store as hs_module


class _FakeDocSnapshot:
    def __init__(self, data, ref):
        self._data = dict(data)
        self.reference = ref

    def to_dict(self):
        return dict(self._data)


class _FakeDocumentRef:
    def __init__(self, collection, doc_id):
        self._collection = collection
        self.id = doc_id

    def get(self):
        return _FakeDocSnapshot(self._collection._docs.get(self.id, {}), self)

    def set(self, data, merge=True):
        if merge and self.id in self._collection._docs:
            merged = dict(self._collection._docs[self.id])
            merged.update(dict(data or {}))
            self._collection._docs[self.id] = merged
        else:
            self._collection._docs[self.id] = dict(data or {})

    def delete(self):
        self._collection._docs.pop(self.id, None)


class _FakeQuery:
    def __init__(self, collection, field, op, value):
        self._collection = collection
        self._field = field
        self._op = op
        self._value = value

    def _matches(self, row):
        cur = row.get(self._field)
        if self._op == "<":
            return cur < self._value
        if self._op == ">=":
            return cur >= self._value
        return False

    def stream(self):
        out = []
        for doc_id, row in list(self._collection._docs.items()):
            if self._matches(row):
                out.append(_FakeDocSnapshot(row, _FakeDocumentRef(self._collection, doc_id)))
        return out


class _FakeCollection:
    def __init__(self):
        self._docs = {}

    def add(self, data):
        doc_id = f"doc_{len(self._docs) + 1}"
        self._docs[doc_id] = dict(data or {})
        return _FakeDocumentRef(self, doc_id)

    def document(self, doc_id):
        return _FakeDocumentRef(self, doc_id)

    def where(self, field, op, value):
        return _FakeQuery(self, field, op, value)


class _FakeDb:
    def __init__(self):
        self._collections = {}

    def collection(self, name):
        if name not in self._collections:
            self._collections[name] = _FakeCollection()
        return self._collections[name]


def _evt(ts_epoch, domain="finance", source_met=True, recency_met=True, conf="LOW"):
    return {
        "timestamp": datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat(),
        "ts_epoch": float(ts_epoch),
        "domain": domain,
        "strict_mode": True,
        "source_requirement_met": bool(source_met),
        "recency_requirement_met": bool(recency_met),
        "confidence_level": conf,
    }


def test_compaction_rollup_correctness(monkeypatch):
    fixed_now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
    old_ts = (fixed_now - timedelta(days=5)).timestamp()
    new_ts = (fixed_now - timedelta(hours=6)).timestamp()

    store = HighStakesStore(max_events=100)
    monkeypatch.setattr(store, "_now", lambda: fixed_now)

    store._events.append(_evt(old_ts + 10, source_met=True, recency_met=False, conf="LOW"))
    store._events.append(_evt(old_ts + 20, source_met=False, recency_met=True, conf="MODERATE"))
    store._events.append(_evt(old_ts + 30, source_met=False, recency_met=False, conf="HIGH"))
    store._events.append(_evt(new_ts, domain="legal", source_met=True, recency_met=True, conf="LOW"))

    fake_db = _FakeDb()
    monkeypatch.setattr(hs_module, "get_firestore_db", lambda: fake_db)

    result = store.compact_old_events(retention_days=2)

    assert result["local_compacted_events"] == 3
    assert result["retention_days"] == 2
    assert len(store._events) == 1

    daily = fake_db.collection("high_stakes_metrics_daily")._docs
    day = datetime.fromtimestamp(old_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    key = f"{day}_finance"
    assert key in daily
    assert daily[key]["total"] == 3
    assert daily[key]["source_met"] == 1
    assert daily[key]["source_failed"] == 2
    assert daily[key]["recency_met"] == 1
    assert daily[key]["recency_failed"] == 2
    assert daily[key]["low"] == 1
    assert daily[key]["moderate"] == 1
    assert daily[key]["high"] == 1


def test_compaction_retention_deletes_old_firestore_raw_events(monkeypatch):
    fixed_now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
    cutoff = (fixed_now - timedelta(days=2)).timestamp()

    store = HighStakesStore(max_events=20)
    monkeypatch.setattr(store, "_now", lambda: fixed_now)

    fake_db = _FakeDb()
    raw_events = fake_db.collection("high_stakes_metrics_events")
    raw_events._docs = {
        "old_1": _evt(cutoff - 10),
        "old_2": _evt(cutoff - 1000),
        "new_1": _evt(cutoff + 10),
    }
    monkeypatch.setattr(hs_module, "get_firestore_db", lambda: fake_db)

    result = store.compact_old_events(retention_days=2)

    assert result["raw_firestore_deleted"] == 2
    assert sorted(raw_events._docs.keys()) == ["new_1"]
