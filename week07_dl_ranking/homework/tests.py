import torch
import torch.nn as nn

import polars as pl
import numpy as np


def test_extract_preference_pairs(listens: pl.DataFrame) -> None:
    _EXPECTED_SHAPE = (6_885_472, 23)
    assert listens.shape == _EXPECTED_SHAPE
    print('All good! :)')


def test_join_item_artist_album(listens: pl.DataFrame) -> None:
    _EXPECTED_SHAPE = (6_885_472, 25)
    _EXPECTED_LIST_IDS = pl.List(pl.UInt32)
    assert listens.shape == _EXPECTED_SHAPE
    assert "artist_ids" in listens.columns
    assert "album_ids" in listens.columns
    assert listens["artist_ids"].dtype == _EXPECTED_LIST_IDS
    assert listens["album_ids"].dtype == _EXPECTED_LIST_IDS
    print('All good! :)')


def test_temporal_train_test_split(
    train_listens: pl.DataFrame,
    test_listens: pl.DataFrame,
) -> None:
    _EXPECTED_TRAIN_SHAPE = (5_955_066, 25)
    _EXPECTED_TEST_SHAPE = (930_406, 25)
    _EXPECTED_TRAIN_TS_MAX = 23_407_980
    _EXPECTED_TEST_TS_MIN = 23_408_005

    assert train_listens.shape == _EXPECTED_TRAIN_SHAPE
    assert test_listens.shape == _EXPECTED_TEST_SHAPE
    assert int(train_listens["timestamp"].max()) == _EXPECTED_TRAIN_TS_MAX
    assert int(test_listens["timestamp"].min()) == _EXPECTED_TEST_TS_MIN
    assert train_listens.columns == test_listens.columns
    print('All good! :)')


def test_pairwise_accuracy(compute_pairwise_accuracy):
    uids = np.array([1, 1])
    timestamps = np.array([0, 10])
    labels = np.array([0.0, 1.0], dtype=np.float32)
    probs = np.array([0.2, 0.8], dtype=np.float32)
    result = compute_pairwise_accuracy(uids, timestamps, labels, probs)
    assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

    probs = np.array([0.9, 0.1], dtype=np.float32)
    result = compute_pairwise_accuracy(uids, timestamps, labels, probs)
    assert np.isclose(result, 0.0), f"Expected 0.0, got {result}"

    probs = np.array([0.5, 0.5], dtype=np.float32)
    result = compute_pairwise_accuracy(uids, timestamps, labels, probs)
    assert np.isclose(result, 0.5), f"Expected 0.5, got {result}"

    labels_same = np.array([1.0, 1.0], dtype=np.float32)
    probs = np.array([0.1, 0.9], dtype=np.float32)
    try:
        result = compute_pairwise_accuracy(uids, timestamps, labels_same, probs)
        assert (
            np.isnan(result) or result == 0.0
        ), f"Expected NaN or 0.0 when there are no valid pairs, got {result}"
    except ZeroDivisionError:
        pass

    timestamps_far = np.array([0, 3600])
    labels = np.array([0.0, 1.0], dtype=np.float32)
    probs = np.array([0.1, 0.9], dtype=np.float32)
    try:
        result = compute_pairwise_accuracy(
            uids, timestamps_far, labels, probs, session_gap_seconds=15 * 60
        )
        assert (
            np.isnan(result) or result == 0.0
        ), f"Expected NaN or 0.0 when there are no valid pairs, got {result}"
    except ZeroDivisionError:
        pass

    uids = np.array([1, 1, 1])
    timestamps = np.array([0, 10, 20])
    labels = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    probs = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    result = compute_pairwise_accuracy(uids, timestamps, labels, probs)
    assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

    uids = np.array([1, 1, 2, 2])
    timestamps = np.array([0, 10, 0, 10])
    labels = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    probs = np.array([0.1, 0.9, 0.8, 0.3], dtype=np.float32)
    result = compute_pairwise_accuracy(uids, timestamps, labels, probs)
    assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

    uids = np.array([1, 1, 1, 2, 2])
    timestamps = np.array([0, 10, 20, 0, 10])
    labels = np.array([0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    probs = np.array([0.1, 0.9, 0.95, 0.4, 0.4], dtype=np.float32)
    result = compute_pairwise_accuracy(uids, timestamps, labels, probs)
    assert np.isclose(result, 0.5), f"Expected 0.5, got {result}"

    uids = np.array([1, 1, 1])
    timestamps = np.array([20, 0, 10])
    labels = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    probs = np.array([0.2, 0.1, 0.9], dtype=np.float32)
    result = compute_pairwise_accuracy(uids, timestamps, labels, probs)
    assert np.isclose(result, 1.0), f"Expected 1.0, got {result}"

    print('All good! :)')


def test_random_baseline_metrics(metrics):
    assert abs(metrics["pair_accuracy_like"] - 0.5) < 0.05
    assert abs(metrics["pair_accuracy_full_play"] - 0.5) < 0.05
    print('All good! :)')


def test_popularity_baseline_metrics(metrics):
    assert metrics["pair_accuracy_like"] > 0.48
    assert metrics["pair_accuracy_full_play"] > 0.52
    print('All good! :)')


def test_catboost_baseline_metrics(metrics):
    assert metrics["pair_accuracy_like"] > 0.48
    assert metrics["pair_accuracy_full_play"] > 0.57
    print('All good! :)')


def test_ranker_dataset(RankerDataset):
    df = pl.DataFrame(
        {
            "uid": [1, 2, 3],
            "item_id": [10, 20, 30],
            "timestamp": [100, 200, 300],
            "is_like": [1.0, 0.0, 1.0],
            "is_full_play": [0.0, 1.0, 1.0],
            "f1": [0.1, 0.2, 0.3],
            "f2": [1.0, 2.0, 3.0],
            "artist_ids": [[101, 102], [201], [301, 302, 303]],
            "album_ids": [[1001], [2001, 2002], [3001]],
        }
    )

    dataset = RankerDataset(
        df=df,
        transforms=[],
        label_columns=["is_like", "is_full_play"],
        dense_columns=["f1", "f2"],
        sparse_columns=["uid", "item_id"],
        multivalent_columns=["artist_ids", "album_ids"],
        batch_size=2,
    )

    assert len(dataset) == 2

    batch = dataset[0]
    assert set(batch.keys()) == {
        "labels",
        "dense_features",
        "sparse_features",
        "multivalent_features",
        "meta",
    }

    assert batch["dense_features"].shape == (2, 2)
    assert batch["dense_features"].dtype == torch.float32

    assert set(batch["labels"].keys()) == {"is_like", "is_full_play"}
    assert batch["labels"]["is_like"].shape == (2,)
    assert batch["labels"]["is_full_play"].shape == (2,)

    assert set(batch["sparse_features"].keys()) == {"uid", "item_id"}
    assert batch["sparse_features"]["uid"].shape == (2,)
    assert batch["sparse_features"]["item_id"].shape == (2,)

    assert set(batch["multivalent_features"].keys()) == {"artist_ids", "album_ids"}
    assert torch.equal(
        batch["multivalent_features"]["artist_ids"]["lengths"],
        torch.tensor([2, 1], dtype=torch.long),
    )
    assert torch.equal(
        batch["multivalent_features"]["artist_ids"]["values"],
        torch.tensor([101, 102, 201], dtype=torch.long),
    )

    assert set(batch["meta"].keys()) == {"timestamp", "uid", "item_id"}
    assert batch["meta"]["timestamp"].shape == (2,)
    print('All good! :)')


def test_multihash_transform(MultihashTransform):
    transform = MultihashTransform(
        sparse_features_config={"uid": [17, 29]},
        sparse_features_name="sparse_features",
        multivalent_features_config={"artist_ids": [11, 13, 17]},
        multivalent_features_name="multivalent_features",
        cardinality=1000,
    )

    sample = {
        "sparse_features": {
            "uid": torch.tensor([1, 2, 3], dtype=torch.long),
        },
        "multivalent_features": {
            "artist_ids": {
                "values": torch.tensor([10, 20, 30, 40], dtype=torch.long),
                "lengths": torch.tensor([2, 2], dtype=torch.long),
            }
        },
    }

    out = transform(sample)

    assert out["sparse_features"]["uid"].shape == (3, 2)
    assert out["multivalent_features"]["artist_ids"]["values"].shape == (4, 3)
    assert torch.equal(
        out["multivalent_features"]["artist_ids"]["lengths"],
        torch.tensor([2, 2], dtype=torch.long),
    )
    assert out["sparse_features"]["uid"].dtype == torch.long
    assert out["multivalent_features"]["artist_ids"]["values"].dtype == torch.long
    assert (out["sparse_features"]["uid"] >= 0).all() and (out["sparse_features"]["uid"] < 1000).all()
    assert (
        (out["multivalent_features"]["artist_ids"]["values"] >= 0)
        .all()
        and (out["multivalent_features"]["artist_ids"]["values"] < 1000).all()
    )
    print('All good! :)')


def test_categorical_encoder(CategoricalEncoder):
    embeddings = nn.Embedding(100, 8)
    encoder = CategoricalEncoder(embeddings)

    ids = torch.tensor([1, 5, 7], dtype=torch.long)
    out = encoder(ids)
    assert out.shape == (3, 8)

    ids2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    out2 = encoder(ids2)
    assert out2.shape == (2, 2, 8)

    print('All good! :)')


def test_multivalent_encoder(MultivalentEncoder):
    embeddings = nn.Embedding(100, 8)
    encoder = MultivalentEncoder(embeddings)

    ids = torch.tensor(
        [
            [1, 11],
            [2, 12],
            [3, 13],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([2, 1], dtype=torch.long)

    out = encoder(ids, lengths)

    assert out.shape == (2, 2, 8)
    assert out.dtype == embeddings.weight.dtype

    print('All good! :)')


def test_piecewise_linear_encoder(PiecewiseLinearEncoder):
        df = pl.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "f2": [10.0, 10.0, 10.0, 20.0, 20.0],
            }
        )

        encoder = PiecewiseLinearEncoder.from_dataset(df, n_bins=4)

        assert isinstance(encoder.n_bins, list)
        assert len(encoder.n_bins) == 2
        assert all(1 <= x <= 4 for x in encoder.n_bins)

        x = df.to_torch().to(torch.float32)
        out = encoder(x)

        assert out.shape[0] == len(df)
        assert out.ndim == 2
        assert out.dtype == torch.float32
        assert out.shape[1] == sum(encoder.n_bins)

        bins = PiecewiseLinearEncoder.compute_bins(x, n_bins=4)
        assert isinstance(bins, list)
        assert len(bins) == x.shape[1]
        assert all(isinstance(b, torch.Tensor) for b in bins)
        assert all(len(b) >= 2 for b in bins)

        df_single_bin = pl.DataFrame(
            {
                "f1": [0.0, 0.0, 1.0, 1.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
            }
        )
        encoder_single_bin = PiecewiseLinearEncoder.from_dataset(df_single_bin, n_bins=4)
        x_single_bin = df_single_bin.to_torch().to(torch.float32)
        out_single_bin = encoder_single_bin(x_single_bin)

        assert out_single_bin.shape[0] == len(df_single_bin)
        assert out_single_bin.ndim == 2
        assert out_single_bin.dtype == torch.float32
        assert out_single_bin.shape[1] == sum(encoder_single_bin.n_bins)

        print('All good! :)')


def test_deep_network(DeepNetwork):
    model = DeepNetwork(input_dim=4, hidden_units=[8, 16])
    x = torch.randn(3, 4)
    out = model(x)

    assert out.shape == (3, 16)

    model_empty = DeepNetwork(input_dim=4, hidden_units=[])
    out_empty = model_empty(x)

    assert out_empty.shape == (3, 4)
    print('All good! :)')


def test_model_concat_mlp_metrics(metrics):
    assert metrics["pair_accuracy_full_play"] > 0.585
    print('All good! :)')


def test_mixture_low_rank_cross_network(
        MixtureLowRankCrossLayer,
        MixtureLowRankCrossNetwork,
    ):
        batch_size = 4
        input_dim = 8
        num_experts = 3
        rank = 2
        num_layers = 2

        layer = MixtureLowRankCrossLayer(
            input_dim=input_dim,
            num_experts=num_experts,
            rank=rank,
        )
        x0 = torch.randn(batch_size, input_dim)
        xl = torch.randn(batch_size, input_dim)
        out = layer(x0, xl)

        assert out.shape == (batch_size, input_dim)
        assert torch.is_floating_point(out)

        model = MixtureLowRankCrossNetwork(
            input_dim=input_dim,
            num_layers=num_layers,
            num_experts=num_experts,
            rank=rank,
        )
        x = torch.randn(batch_size, input_dim)
        out = model(x)

        assert out.shape == (batch_size, input_dim)
        assert torch.is_floating_point(out)
        print('All good! :)')


def test_res_deep_network(ResidualMLPBlock, ResDeepNetwork):
    block_same = ResidualMLPBlock(8, 8)
    x = torch.randn(4, 8)
    out = block_same(x)
    assert out.shape == (4, 8)
    assert torch.is_floating_point(out)

    block_proj = ResidualMLPBlock(8, 16)
    out_proj = block_proj(x)
    assert out_proj.shape == (4, 16)

    model = ResDeepNetwork(input_dim=8, hidden_units=[16, 12])
    out_model = model(x)
    assert out_model.shape == (4, 12)
    assert torch.is_floating_point(out_model)
    print('All good! :)')


def test_dense_deep_network(DenseDeepNetwork):
    model = DenseDeepNetwork(input_dim=8, hidden_units=[16, 12, 10])
    x = torch.randn(4, 8)
    out = model(x)

    assert out.shape == (4, 10)
    assert torch.is_floating_point(out)
    print('All good! :)')


def test_dcnv2(DCNV2):
        dense_train_df = pl.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "f2": [10.0, 20.0, 10.0, 30.0],
            }
        )

        model = DCNV2(
            embedding_size=4,
            cross_layers=2,
            deep_units=[16, 8],
            input_size=36,
            dense_train_df=dense_train_df,
            n_bins=2,
            train_df_slice=4,
            cardinality=128,
            num_experts=3,
            low_rank=2,
            deep_network="mlp",
            output_size=2,
        )

        inputs = {
            "dense_features": torch.tensor(
                [[0.5, 10.0], [2.5, 20.0]], dtype=torch.float32
            ),
            "sparse_features": {
                "item_id": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                "uid": torch.tensor([[5, 6], [7, 8]], dtype=torch.long),
            },
            "multivalent_features": {
                "artist_ids": {
                    "values": torch.tensor(
                        [[9, 10], [11, 12], [13, 14]], dtype=torch.long
                    ),
                    "lengths": torch.tensor([2, 1], dtype=torch.long),
                },
                "album_ids": {
                    "values": torch.tensor(
                        [[15, 16], [17, 18], [19, 20]], dtype=torch.long
                    ),
                    "lengths": torch.tensor([1, 2], dtype=torch.long),
                },
            },
        }

        out = model(inputs)
        assert out.shape == (2, 2)
        assert torch.is_floating_point(out)
        print('All good! :)')


def test_model_dcnv2_mlp_metrics(metrics):
    assert metrics["pair_accuracy_full_play"] > 0.588
    print('All good! :)')


def test_model_multitask_dcnv2_mlp_metrics(metrics):
    assert metrics["pair_accuracy_like"] > 0.53
    assert metrics["pair_accuracy_full_play"] > 0.588
    print('All good! :)')


def test_model_multitask_dcnv2_mlp_combined_score_metrics(metrics):
    assert metrics["pair_accuracy_like"] > 0.52
    assert metrics["pair_accuracy_full_play"] > 0.54
    print('All good! :)')