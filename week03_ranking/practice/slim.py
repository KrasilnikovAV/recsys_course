
import multiprocessing as mp
from functools import partial
import polars as pl
import numpy as np
import scipy as sp
import sklearn.linear_model


def _solve_item(idx, R, alpha, l1_ratio, positive):
    y = R[:, idx].todense()
    solver = sklearn.linear_model.ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        positive=positive,
        copy_X=False,
        fit_intercept=False,
        precompute=True,
        selection="random",
        max_iter=100,
    )
    solver.fit(R, y)
    return solver.sparse_coef_


class SLIM:
    def __init__(self,
                 alpha: float = 0.1,
                 l1_ratio: float = 0.01,
                 positive: bool = False,
                 num_processes: int | None = None,
                 ):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.positive = positive
        self.num_processes = num_processes

    def fit(self, interactions: pl.DataFrame):
        self.R, (self.user_id2idx, self.item_id2idx, self.user_idx2id, self.item_idx2id) = (
            build_interaction_matrix(interactions)
        )
        self.num_items = self.R.shape[1]
        self._solve = partial(
            _solve_item,
            R=self.R,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            positive=self.positive
        )
        if self.num_processes == 1:
            self.results = [
                self._solve(i) for i in range(self.num_items)
            ]
        else:
            pool = mp.Pool(self.num_processes)
            self.results = pool.map(self._solve, range(self.num_items))
        self.B = sp.sparse.vstack(self.results).T

    def recommend(self, user_id: int, k: int):
        user_idx = self.user_id2idx.get(user_id)
        if user_idx is None:
            return []
        item_scores = (self.R[user_idx, :] @ self.B).todense()
        top_items = np.argsort(item_scores)[::-1][:k]
        results = [self.item_idx2id[item_idx] for item_idx in top_items]
        return results


def build_interaction_matrix(ratings: pl.DataFrame, additive: bool = False):
    users = ratings.select(pl.col("user_id").unique())
    items = ratings.select(pl.col("item_id").unique())
    num_users = len(users)
    num_items = len(items)

    user_id2idx = {row["user_id"]: i for i, row in enumerate(users.iter_rows(named=True))}
    item_id2idx = {row["item_id"]: i for i, row in enumerate(items.iter_rows(named=True))}
    user_idx2id = {v: k for k, v in user_id2idx.items()}
    item_idx2id = {v: k for k, v in item_id2idx.items()}

    R = sp.sparse.lil_array((num_users, num_items))
    for row in ratings.iter_rows(named=True):
        user_idx = user_id2idx[row["user_id"]]
        item_idx = item_id2idx[row["item_id"]]
        if additive:
            R[user_idx, item_idx] += row["rating"]
        else:
            R[user_idx, item_idx] = row["rating"]

    return R.tocsr(), (user_id2idx, item_id2idx, user_idx2id, item_idx2id)
