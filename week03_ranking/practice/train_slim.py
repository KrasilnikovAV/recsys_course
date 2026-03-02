import polars as pl
import numpy as np
from tqdm.auto import tqdm
import click

from slim import SLIM


def recall_at_k(preds: list[int], positives: set[int], k: int) -> float:
    if not positives:
        return 0.0
    preds = preds[:k]
    return len(set(preds) & positives) / len(positives)


def ndcg_at_k(preds: list[int], positives: set[int], k: int) -> float:
    if not positives:
        return 0.0
    dcg = 0.0
    for rank, item_id in enumerate(preds[:k], start=1):
        if item_id in positives:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(recommend_fn, test_by_user: dict[int, set[int]], k_list=(10, 50)):
    max_k = max(k_list)
    metrics = {f"recall@{k}": [] for k in k_list}
    metrics.update({f"ndcg@{k}": [] for k in k_list})

    for user_id, positives in tqdm(test_by_user.items(), total=len(test_by_user)):
        preds = recommend_fn(user_id, max_k)
        for k in k_list:
            metrics[f"recall@{k}"].append(recall_at_k(preds, positives, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(preds, positives, k))

    return {name: float(np.mean(values)) for name, values in metrics.items()}



@click.command()
@click.option(
    "--path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to actions parquet file.",
)
def main(path: str):
    actions = pl.read_parquet(path)
    interactions = (
        actions
        .filter(pl.col("action_type") == "AT_CartUpdate")
        .select(
            pl.col("user_id"),
            pl.col("item_id"),
            pl.col("timestamp"),
        )
        .with_columns(pl.lit(1).alias("rating"))
        .sort("timestamp")
    )
    split_idx = int(len(interactions) * 0.8)
    train = interactions[:split_idx]
    test = interactions[split_idx:]
    train_users = train.select(pl.col("user_id").unique())
    test = test.join(train_users, on="user_id", how="inner")
    train_by_user = (
        train
        .group_by("user_id")
        .agg(pl.col("item_id").unique().alias("items"))
    )
    test_by_user = (
        test
        .group_by("user_id")
        .agg(pl.col("item_id").unique().alias("items"))
    )
    train_by_user = {row["user_id"]: set(row["items"]) for row in train_by_user.iter_rows(named=True)}
    test_by_user = {row["user_id"]: set(row["items"]) for row in test_by_user.iter_rows(named=True)}
    model = SLIM()
    model.fit(train)
    metrics = evaluate_recommender(
        lambda user_id, n: model.recommend(user_id, n),
        test_by_user,
        k_list=(10, 50),
    )
    print(metrics)


if __name__ == "__main__":
    main()
