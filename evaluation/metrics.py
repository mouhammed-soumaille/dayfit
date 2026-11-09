"""Retrieval metrics for a single query. Averaging over queries is done in run_eval.py."""


def recall_at_k(ranked_item_ids, target_item_id, k):
    """1 if the exact target item is somewhere in the top k results, else 0."""
    return 1.0 if target_item_id in ranked_item_ids[:k] else 0.0


def precision_at_k(ranked_labels, query_label, k):
    """Fraction of the top k results whose label (e.g. article_type) matches the query's."""
    top = ranked_labels[:k]
    hits = sum(1 for label in top if label == query_label)
    return hits / k
