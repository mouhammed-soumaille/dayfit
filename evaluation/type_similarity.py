"""T2: does fine-tuning push garments of the same type apart?

For each model we load the saved catalog embeddings (one row per catalog item,
same order as data/catalog.csv) and measure:
  within       = average cosine similarity between two DIFFERENT items of the SAME type
  between      = average cosine similarity between two items of DIFFERENT types
  gap          = within - between  (bigger = garment types are better separated)
  gap_std      = gap / spread of all similarities, so models whose scores are
                 spread differently can be compared fairly
  nn_same_type = fraction of each item's 5 nearest catalog neighbours that have
                 the same type (like Precision@5, but catalog-to-catalog, no photos)

Run from the repo root:
    python -m evaluation.type_similarity
"""
import argparse

import numpy as np
import pandas as pd

MODELS = ["fashion_siglip", "finetuned", "finetuned_random"]


def type_column(catalog):
    """Find the column that holds the garment type (name differs between versions)."""
    for name in ["article_type", "articleType", "type"]:
        if name in catalog.columns:
            return name
    raise ValueError(f"no article-type column found; columns are {list(catalog.columns)}")


def measure(emb, types):
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # unit length -> dot product = cosine
    sim = emb @ emb.T                                       # n x n similarity matrix
    same = types[:, None] == types[None, :]                 # True where two items share a type
    not_self = ~np.eye(len(types), dtype=bool)              # leave out each item paired with itself

    within = sim[same & not_self].mean()
    between = sim[~same].mean()
    spread = sim[not_self].std()

    # 5 nearest neighbours of each item, not counting the item itself
    np.fill_diagonal(sim, -np.inf)
    top5 = np.argsort(-sim, axis=1)[:, :5]
    nn_same = (types[top5] == types[:, None]).mean()

    return dict(within=within, between=between, gap=within - between,
                gap_std=(within - between) / spread, nn_same_type=nn_same)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.csv")
    parser.add_argument("--index_dir", default="models/index")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog, dtype={"item_id": str})
    types = catalog[type_column(catalog)].to_numpy()

    rows = []
    for model in MODELS:
        emb = np.load(f"{args.index_dir}/{model}.npy")
        assert len(emb) == len(catalog), f"{model}: {len(emb)} embeddings, {len(catalog)} catalog items"
        rows.append(dict(model=model, **measure(emb, types)))

    table = pd.DataFrame(rows).round(3)
    print(table.to_string(index=False))
    table.to_csv("results/type_similarity.csv", index=False)


if __name__ == "__main__":
    main()