"""Fixed-partition, structure-only paired-target feasibility benchmark."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_matrix(data, model, protocol):
    key = "rate_targets" if model == "et_apparent_rates" else "logit_targets" if model == "et_logit" else "direct_targets"
    return data[protocol[key]].to_numpy(dtype=float)


def invert(raw, model):
    # Some regressors return float32 even with float64 targets.
    # Decode and calibrate in float64 so saved CSV values reproduce the limits.
    raw = np.asarray(raw, dtype=np.float64)
    if model == "et_apparent_rates":
        log_sum = logsumexp(raw * np.log(10.0), axis=1) / np.log(10.0)
        phi = expit((raw[:, 0] - raw[:, 1]) * np.log(10.0))
        log_tau = 9.0 - log_sum
    else:
        phi = expit(raw[:, 0]) if model == "et_logit" else np.clip(raw[:, 0], 0, 1)
        log_tau = raw[:, 1]
    output = np.column_stack([phi, log_tau])
    assert np.isfinite(output).all() and ((output[:, 0] >= 0) & (output[:, 0] <= 1)).all()
    return output


def calibrated_width(residual, nominal):
    rank = int(np.ceil((len(residual) + 1) * nominal))
    return float(np.sort(residual)[rank - 1]) if rank <= len(residual) else float("inf")


def metrics(prediction):
    d = prediction
    ae_phi = np.abs(d.phi - d.pred_phi)
    ae_tau = np.abs(d.log10_tau_ns - d.pred_log10_tau_ns)
    inside_phi = d.phi.between(d.phi_lower, d.phi_upper)
    inside_tau = d.log10_tau_ns.between(d.log10_tau_lower, d.log10_tau_upper)
    result = {"n": len(d), "n_chromophores": d.chromophore_smiles.nunique(), "n_scaffolds": d.scaffold_id.nunique(),
              "phi_mae": ae_phi.mean(), "phi_rmse": mean_squared_error(d.phi, d.pred_phi) ** .5, "phi_r2": r2_score(d.phi, d.pred_phi),
              "log10_tau_mae": ae_tau.mean(), "log10_tau_rmse": mean_squared_error(d.log10_tau_ns, d.pred_log10_tau_ns) ** .5,
              "log10_tau_r2": r2_score(d.log10_tau_ns, d.pred_log10_tau_ns),
              "tau_ns_mae": mean_absolute_error(d.tau_ns, 10.0 ** d.pred_log10_tau_ns),
              "tau_within_factor_2": np.mean(ae_tau <= np.log10(2)),
              "phi_coverage": inside_phi.mean(), "log10_tau_coverage": inside_tau.mean(),
              "simultaneous_coverage": (inside_phi & inside_tau).mean(),
              "phi_interval_width": (d.phi_upper - d.phi_lower).mean(), "log10_tau_interval_width": (d.log10_tau_upper - d.log10_tau_lower).mean()}
    for group, label in [("chromophore_smiles", "chromophore"), ("scaffold_id", "scaffold")]:
        result[f"{label}_macro_phi_mae"] = ae_phi.groupby(d[group]).mean().mean()
        result[f"{label}_macro_log10_tau_mae"] = ae_tau.groupby(d[group]).mean().mean()
    return {k: int(v) if k.startswith("n") else float(v) for k, v in result.items()}


def partitions(data, group_col, seed, fractions):
    groups = np.sort(data[group_col].unique())
    order = np.random.default_rng(seed).permutation(groups)
    n_test = int(np.ceil(len(groups) * fractions["test"]))
    n_cal = int(np.ceil(len(groups) * fractions["calibration"]))
    assignment = {g: "test" if i < n_test else "calibration" if i < n_test + n_cal else "train" for i, g in enumerate(order)}
    roles = data[group_col].map(assignment).to_numpy()
    for a, b in [("train", "test"), ("train", "calibration"), ("calibration", "test")]:
        assert not set(data.loc[roles == a, group_col]) & set(data.loc[roles == b, group_col])
        assert not set(data.loc[roles == a, "pair_id"]) & set(data.loc[roles == b, "pair_id"])
    return roles


def main():
    protocol_path = ROOT / "protocol_pilot_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    feature_report = json.loads((ROOT / "results/pilot_features.json").read_text(encoding="utf-8"))
    assert sha(protocol_path) == feature_report["protocol_sha256"]
    assert sha(ROOT / protocol["dataset"]) == feature_report["data_sha256"]
    feature_path = ROOT / "data/processed/pilot_features.npz"
    assert sha(feature_path) == feature_report["features_sha256"]
    data = pd.read_csv(ROOT / protocol["dataset"])
    assert data.row_id.is_unique and data.phi.between(0, 1, inclusive="neither").all() and data.tau_ns.gt(0).all()
    features = np.load(feature_path, allow_pickle=False)
    assert np.array_equal(data.row_id, features["row_id"])
    X = features["X"]
    out = ROOT / "results/pilot_v1"
    out.mkdir(parents=True, exist_ok=True)
    for folder in ["partitions", "predictions", "fits"]:
        (out / folder).mkdir(exist_ok=True)
    lock = {"protocol_sha256": sha(protocol_path), "data_sha256": sha(ROOT / protocol["dataset"]),
            "features_sha256": sha(feature_path), "runner_sha256": sha(Path(__file__)),
            "created_utc": datetime.now(timezone.utc).isoformat(), "versions": feature_report["versions"]}
    lock_path = out / "run_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        assert all(previous[k] == lock[k] for k in lock if k != "created_utc"), "A locked pilot run cannot silently change."
    else:
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    # Freeze every partition before any fit.
    jobs, split_audit = [], []
    for scenario, group_col in protocol["splits"].items():
        for seed in protocol["seeds"]:
            roles = partitions(data, group_col, seed, protocol["fractions_of_groups"])
            saved = data[["row_id", "pair_id", "chromophore_smiles", "scaffold_id", "reference_doi"]].copy()
            saved["subset"] = roles
            partition_path = out / "partitions" / f"{scenario}_{seed}.csv"
            saved.to_csv(partition_path, index=False)
            train = data[roles == "train"]
            audit = {"scenario": scenario, "seed": seed, "group_column": group_col, "partition_sha256": sha(partition_path)}
            for role in ["train", "calibration", "test"]:
                subset = data[roles == role]
                audit[f"{role}_rows"] = len(subset)
                audit[f"{role}_groups"] = subset[group_col].nunique()
                if role != "train":
                    for column in ["chromophore_smiles", "solvent_smiles", "scaffold_id", "reference_doi"]:
                        audit[f"{role}_fraction_seen_{column}"] = float(subset[column].isin(set(train[column])).mean())
            split_audit.append(audit)
            jobs.append((scenario, seed, roles))
    pd.DataFrame(split_audit).to_csv(out / "partition_audit.csv", index=False)
    rows = []
    for scenario, seed, roles in jobs:
        tr, ca, te = [np.flatnonzero(roles == role) for role in ["train", "calibration", "test"]]
        for model in protocol["models"]:
            result_path = out / "fits" / f"{scenario}_{seed}_{model}.json"
            if result_path.exists():
                record = json.loads(result_path.read_text(encoding="utf-8"))
                assert record["protocol_sha256"] == lock["protocol_sha256"] and record["runner_sha256"] == lock["runner_sha256"]
                assert all(sha(out / name) == digest for name, digest in record["saved_prediction_sha256"].items())
                rows.append(record["metrics"])
                continue
            start = time.perf_counter()
            Xm = X[:, :feature_report["chromophore_features"]] if model == "et_chromophore_only" else X
            raw_targets = target_matrix(data, model, protocol)
            target_scaler = StandardScaler().fit(raw_targets[tr])
            y = target_scaler.transform(raw_targets[tr])
            if model == "mean":
                estimator = DummyRegressor(strategy="mean")
                pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator)
            elif model == "ridge_direct":
                estimator = Ridge(**protocol["ridge"])
                pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator)
            else:
                estimator = ExtraTreesRegressor(**protocol["extra_trees"], random_state=seed)
                pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator)
            pipeline.fit(Xm[tr], y)
            transformed_cal = target_scaler.inverse_transform(pipeline.predict(Xm[ca]))
            transformed_test = target_scaler.inverse_transform(pipeline.predict(Xm[te]))
            cal, test = invert(transformed_cal, model), invert(transformed_test, model)
            observed_cal = data.iloc[ca][protocol["direct_targets"]].to_numpy()
            widths = [calibrated_width(np.abs(observed_cal[:, j] - cal[:, j]), protocol["intervals"]["nominal_marginal_coverage"]) for j in range(2)]
            saved_hash = {}
            for role, indices, pred in [("calibration", ca, cal), ("test", te, test)]:
                result = data.iloc[indices][["row_id", "pair_id", "chromophore_smiles", "solvent_smiles", "solvent_name", "scaffold_id", "reference_doi", "phi", "tau_ns", "log10_tau_ns"]].copy()
                result["pred_phi"], result["pred_log10_tau_ns"] = pred[:, 0], pred[:, 1]
                result["phi_lower"], result["phi_upper"] = np.clip(pred[:, 0] - widths[0], 0, 1), np.clip(pred[:, 0] + widths[0], 0, 1)
                result["log10_tau_lower"], result["log10_tau_upper"] = pred[:, 1] - widths[1], pred[:, 1] + widths[1]
                path = out / "predictions" / f"{scenario}_{seed}_{model}_{role}.csv.gz"
                result.to_csv(path, index=False, compression="gzip")
                saved_hash[path.relative_to(out).as_posix()] = sha(path)
                if role == "test":
                    scores = {"scenario": scenario, "seed": seed, "model": model, **metrics(result)}
            record = {"protocol_sha256": lock["protocol_sha256"], "runner_sha256": lock["runner_sha256"],
                      "feature_count": Xm.shape[1], "training_rows": len(tr), "calibration_rows": len(ca),
                      "target_mean": target_scaler.mean_.tolist(), "target_scale": target_scaler.scale_.tolist(),
                      "calibration_absolute_width_phi": widths[0], "calibration_absolute_width_log10_tau": widths[1],
                      "elapsed_seconds": time.perf_counter() - start, "metrics": scores, "saved_prediction_sha256": saved_hash}
            result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            rows.append(scores)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            print(f"{scenario} seed={seed} {model}: phi MAE={scores['phi_mae']:.4f}, log10(tau) MAE={scores['log10_tau_mae']:.4f}, {record['elapsed_seconds']:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    assert len(rows) == len(jobs) * len(protocol["models"])
    manifest = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed {len(rows)} model/partition evaluations.", flush=True)


if __name__ == "__main__":
    main()
