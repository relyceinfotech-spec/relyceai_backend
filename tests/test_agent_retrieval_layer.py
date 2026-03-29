from app.agent.retrieval_layer import RetrievalLayer


def test_topic_id_is_stable_and_normalized():
    r = RetrievalLayer()
    id1 = r._topic_id("Who won the 2024 F1 Championship?")
    id2 = r._topic_id("  who won the 2024 f1 championship?  ")

    assert id1 == id2
    assert id1.startswith("topic_")
