"""Toy example with 3 queries where the answers are worked out by hand."""
from evaluation.metrics import precision_at_k, recall_at_k


def test_recall_toy():
    # query 1: target "A" is ranked 1st -> R@1 = 1, R@5 = 1
    assert recall_at_k(["A", "B", "C", "D", "E"], "A", 1) == 1
    assert recall_at_k(["A", "B", "C", "D", "E"], "A", 5) == 1
    # query 2: target "C" is ranked 3rd -> R@1 = 0, R@5 = 1
    assert recall_at_k(["B", "D", "C", "E", "F"], "C", 1) == 0
    assert recall_at_k(["B", "D", "C", "E", "F"], "C", 5) == 1
    # query 3: target "Z" is not returned -> R@1 = 0, R@5 = 0
    assert recall_at_k(["B", "D", "C", "E", "F"], "Z", 5) == 0
    # averages over the 3 queries: R@1 = 1/3, R@5 = 2/3


def test_precision_toy():
    # 3 of the top 5 are Jeans -> 0.6
    labels = ["Jeans", "Shorts", "Jeans", "Jeans", "Tshirts"]
    assert precision_at_k(labels, "Jeans", 5) == 0.6
    # all 5 match -> 1.0 ; none match -> 0.0
    assert precision_at_k(["Heels"] * 5, "Heels", 5) == 1.0
    assert precision_at_k(["Heels"] * 5, "Jeans", 5) == 0.0
