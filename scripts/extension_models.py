"""Per-target fitting and partition-aware real-label augmentation."""
import hashlib
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def id_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def assign_membership(pool, partition, group):
    roles = pool.row_id.map(partition.set_index("row_id").subset).fillna("partial_train")
    partial = pool.record_origin.ne("paired_pilot")
    for subset in ["calibration", "test"]:
        held_out = set(partition.loc[partition.subset.eq(subset), group])
        roles.loc[partial & pool[group].isin(held_out)] = "partial_excluded_" + subset
    assert roles[~partial].isin(["train", "calibration", "test"]).all()
    assert not roles[partial].isin(["train", "calibration", "test"]).any()
    allowed = roles.isin(["train", "partial_train"])
    reserved = roles.isin(["calibration", "test"])
    assert not set(pool.loc[allowed, group]) & set(pool.loc[reserved, group])
    assert not set(pool.loc[allowed, "pair_id"]) & set(pool.loc[reserved, "pair_id"])
    return roles


def fit_target(pool, X, roles, target, spec, protocol, seed, evaluation_indices):
    allowed_roles = ["train", "partial_train"] if spec["partial_labels"] else ["train"]
    mask = roles.isin(allowed_roles) & pool[target].notna()
    train_idx = np.flatnonzero(mask)
    start = time.perf_counter()
    family = spec["family"]
    metadata = {"training_rows": len(train_idx), "training_row_id_sha256": id_hash(pool.loc[mask, "row_id"]),
                "added_partial_rows": int((roles.eq("partial_train") & mask).sum()),
                "training_chromophores": pool.loc[mask, "chromophore_smiles"].nunique(), "tree_nodes": 0}
    if family == "solvent_mean":
        response_column = "phi" if target == "logit_phi" else "log10_tau_ns"
        means = pool.loc[mask].groupby("solvent_smiles")[response_column].mean()
        fallback = float(pool.loc[mask, response_column].mean())
        predictions = {role: pool.iloc[idx].solvent_smiles.map(means).fillna(fallback).to_numpy(dtype=np.float64)
                       for role, idx in evaluation_indices.items()}
        metadata["solvent_means"] = means.to_dict()
        metadata["fallback_global_mean"] = fallback
    else:
        y = pool.loc[mask, target].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(y[:, None])
        normalized_y = scaler.transform(y[:, None]).ravel()
        if family == "extra_trees":
            estimator = ExtraTreesRegressor(n_estimators=spec["trees_per_target"], random_state=seed, **protocol["extra_trees"])
        elif family == "hist_gradient_boosting":
            estimator = HistGradientBoostingRegressor(random_state=seed, **protocol["hist_gradient_boosting"])
        else:
            raise ValueError(family)
        pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator)
        with threadpool_limits(limits=protocol["thread_limit"]):
            pipeline.fit(X[train_idx], normalized_y)
            raw = {role: np.asarray(pipeline.predict(X[idx]), dtype=np.float64) for role, idx in evaluation_indices.items()}
        predictions = {}
        for role, values in raw.items():
            values = scaler.inverse_transform(values[:, None]).ravel()
            predictions[role] = expit(values) if target == "logit_phi" else values
        metadata["target_mean"] = float(scaler.mean_[0])
        metadata["target_scale"] = float(scaler.scale_[0])
        if family == "extra_trees":
            metadata["tree_nodes"] = sum(tree.tree_.node_count for tree in estimator.estimators_)
            metadata["trees"] = len(estimator.estimators_)
        else:
            metadata["iterations"] = int(estimator.n_iter_)
        assert np.isfinite(pipeline.named_steps["simpleimputer"].statistics_).all()
    metadata["elapsed_seconds"] = time.perf_counter() - start
    assert all(np.isfinite(values).all() for values in predictions.values())
    if target == "logit_phi":
        assert all(((values >= 0) & (values <= 1)).all() for values in predictions.values())
    return predictions, metadata
