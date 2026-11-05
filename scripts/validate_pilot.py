"""Independently recompute saved scores, calibration limits and split exclusions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / "results/pilot_v1"
    protocol = json.loads((ROOT / "protocol_pilot_v1.json").read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / protocol["dataset"]).set_index("row_id")
    lock = json.loads((run / "run_lock.json").read_text(encoding="utf-8"))
    assert sha(ROOT / protocol["dataset"]) == lock["data_sha256"]
    assert sha(ROOT / "protocol_pilot_v1.json") == lock["protocol_sha256"]
    assert sha(ROOT / "scripts/run_pilot.py") == lock["runner_sha256"]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert all(sha(run / name) == digest for name, digest in manifest.items())
    metrics = pd.read_csv(run / "metrics.csv")
    assert len(metrics) == 54
    tested = 0
    for scenario, group_column in protocol["splits"].items():
        for seed in protocol["seeds"]:
            p = pd.read_csv(run / "partitions" / f"{scenario}_{seed}.csv").set_index("row_id")
            assert p.index.is_unique and set(p.index) == set(data.index)
            assert p.subset.value_counts().min() > 0 and set(p.subset) == {"train", "calibration", "test"}
            for key in ["pair_id", group_column]:
                assert p.groupby(key).subset.nunique().max() == 1
            if scenario in {"scaffold", "chromophore"}:
                assert p.groupby("chromophore_smiles").subset.nunique().max() == 1
            for model in protocol["models"]:
                cal = pd.read_csv(run / "predictions" / f"{scenario}_{seed}_{model}_calibration.csv.gz").set_index("row_id")
                test = pd.read_csv(run / "predictions" / f"{scenario}_{seed}_{model}_test.csv.gz").set_index("row_id")
                for role, frame in [("calibration", cal), ("test", test)]:
                    assert frame.index.is_unique and set(frame.index) == set(p[p.subset == role].index)
                    for target in ["phi", "tau_ns", "log10_tau_ns"]:
                        np.testing.assert_allclose(frame[target], data.loc[frame.index, target], rtol=1e-13, atol=1e-13)
                    assert frame.pred_phi.between(0, 1).all() and np.isfinite(frame.pred_log10_tau_ns).all()
                score = metrics[(metrics.scenario == scenario) & (metrics.seed == seed) & (metrics.model == model)].iloc[0]
                rank = int(np.ceil((len(cal) + 1) * .9))
                for target, pred, lower, upper, metric in [
                    ("phi", "pred_phi", "phi_lower", "phi_upper", "phi"),
                    ("log10_tau_ns", "pred_log10_tau_ns", "log10_tau_lower", "log10_tau_upper", "log10_tau"),
                ]:
                    residual = np.abs(cal[target].to_numpy() - cal[pred].to_numpy())
                    width = np.partition(residual, rank - 1)[rank - 1]
                    expected_lo, expected_hi = test[pred].to_numpy() - width, test[pred].to_numpy() + width
                    if target == "phi":
                        expected_lo, expected_hi = expected_lo.clip(0, 1), expected_hi.clip(0, 1)
                    np.testing.assert_allclose(test[lower], expected_lo, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(test[upper], expected_hi, rtol=1e-12, atol=1e-12)
                    mae = np.mean(np.abs(test[target].to_numpy() - test[pred].to_numpy()))
                    coverage = np.mean((test[target] >= test[lower]) & (test[target] <= test[upper]))
                    assert abs(mae - score[metric + "_mae"]) < 1e-12
                    assert abs(coverage - score[metric + "_coverage"]) < 1e-12
                if model == "mean":
                    train = data.loc[p[p.subset == "train"].index]
                    np.testing.assert_allclose(test.pred_phi, train.phi.mean(), atol=1e-12)
                    np.testing.assert_allclose(test.pred_log10_tau_ns, train.log10_tau_ns.mean(), atol=1e-12)
                if model == "et_chromophore_only":
                    assert test.groupby("chromophore_smiles").pred_phi.nunique().max() == 1
                    assert test.groupby("chromophore_smiles").pred_log10_tau_ns.nunique().max() == 1
                tested += 1
    # Verify every saved solvent contrast using the underlying experimental records.
    c = pd.read_csv(ROOT / "results/pilot_analysis/test_solvent_contrasts.csv.gz")
    a, b = data.loc[c.row_a], data.loc[c.row_b]
    assert np.array_equal(a.chromophore_smiles.to_numpy(), b.chromophore_smiles.to_numpy())
    assert np.array_equal(a.reference_doi.to_numpy(), b.reference_doi.to_numpy())
    assert (a.solvent_smiles.to_numpy() != b.solvent_smiles.to_numpy()).all()
    np.testing.assert_allclose(c.delta_phi, b.phi.to_numpy() - a.phi.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(c.delta_log10_tau, b.log10_tau_ns.to_numpy() - a.log10_tau_ns.to_numpy(), atol=1e-12)
    result = {"status": "passed", "model_partition_evaluations_checked": tested, "split_files_checked": 9,
              "prediction_files_checked": tested * 2, "saved_contrast_rows_checked": len(c),
              "scores_recomputed_from_predictions": True, "intervals_recomputed_from_calibration_only": True,
              "group_exclusions_verified": True, "record_identifiers_and_targets_preserved": True,
              "files_in_run_manifest_checked": len(manifest)}
    (ROOT / "results/pilot_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
