"""Run the fixed exploratory extension on unchanged pilot holdouts."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_pilot_features import sha
from extension_models import assign_membership, fit_target
from run_pilot import calibrated_width, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/extension_v2"


def main():
    protocol_path = ROOT / "protocol_extension_v2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "results/extension_data_audit.json").read_text(encoding="utf-8"))
    assert sha(protocol_path) == audit["protocol_sha256"]
    assert all(sha(ROOT / name) == digest for name, digest in audit["file_sha256"].items())
    pool = pd.read_csv(ROOT / "data/extension_v2/training_pool.csv")
    features = np.load(ROOT / "data/extension_v2/features.npz", allow_pickle=False)
    assert np.array_equal(features["row_id"], pool.row_id)
    X = features["X"]
    for directory in [OUT, OUT / "memberships", OUT / "fits", OUT / "predictions"]:
        directory.mkdir(exist_ok=True)
    base_manifest = json.loads((ROOT / "results/pilot_v1/manifest.json").read_text(encoding="utf-8"))
    assert all(sha(ROOT / "results/pilot_v1" / name) == digest for name, digest in base_manifest.items())
    locks = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(protocol_path),
             "input_sha256": audit["file_sha256"], "pilot_manifest_sha256": sha(ROOT / "results/pilot_v1/manifest.json"),
             "source_sha256": {name: sha(ROOT / "scripts" / name) for name in ["run_extension.py", "extension_models.py", "run_pilot.py", "prepare_extension.py", "build_pilot_features.py"]},
             "versions": {name: importlib.metadata.version(name) for name in ["scikit-learn", "numpy", "pandas", "scipy", "rdkit", "threadpoolctl"]}}
    lock_path = OUT / "run_lock.json"
    if lock_path.exists():
        prior = json.loads(lock_path.read_text(encoding="utf-8"))
        assert all(prior[k] == locks[k] for k in locks if k != "created_utc"), "Locked extension changed."
    else:
        lock_path.write_text(json.dumps(locks, indent=2), encoding="utf-8")
    jobs, membership_audit = [], []
    for scenario, group in protocol["scenarios"].items():
        for seed in protocol["seeds"]:
            partition = pd.read_csv(ROOT / protocol["base_partitions"] / f"{scenario}_{seed}.csv")
            roles = assign_membership(pool, partition, group)
            membership = pool[["row_id", "pair_id", "chromophore_smiles", "scaffold_id", "reference_doi", "record_origin"]].copy()
            membership["subset"] = roles
            membership["has_phi"], membership["has_tau"] = pool.phi.notna(), pool.tau_ns.notna()
            path = OUT / "memberships" / f"{scenario}_{seed}.csv.gz"
            membership.to_csv(path, index=False, compression="gzip")
            jobs.append((scenario, seed, roles))
            row = {"scenario": scenario, "seed": seed, "group": group, "membership_sha256": sha(path),
                   "base_partition_sha256": sha(ROOT / protocol["base_partitions"] / f"{scenario}_{seed}.csv")}
            row.update({str(k): int(v) for k, v in roles.value_counts().items()})
            for target, column in [("phi", "phi"), ("tau", "tau_ns")]:
                row[f"added_{target}_rows"] = int((roles.eq("partial_train") & pool[column].notna()).sum())
            membership_audit.append(row)
    pd.DataFrame(membership_audit).to_csv(OUT / "membership_audit.csv", index=False)
    all_scores = []
    for scenario, seed, roles in jobs:
        evaluation = {role: np.flatnonzero(roles.eq(role)) for role in ["calibration", "test"]}
        for model, specification in protocol["models"].items():
            path = OUT / "fits" / f"{scenario}_{seed}_{model}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                assert record["protocol_sha256"] == locks["protocol_sha256"]
                assert all(sha(OUT / name) == digest for name, digest in record["prediction_sha256"].items())
                all_scores.append(record["metrics"])
                continue
            predictions, target_records = {}, {}
            for target in protocol["targets"]:
                predictions[target], target_records[target] = fit_target(pool, X, roles, target, specification, protocol, seed, evaluation)
            observed = pool.iloc[evaluation["calibration"]]
            widths = [calibrated_width(np.abs(observed.phi.to_numpy() - predictions["logit_phi"]["calibration"]), .9),
                      calibrated_width(np.abs(observed.log10_tau_ns.to_numpy() - predictions["log10_tau_ns"]["calibration"]), .9)]
            prediction_hashes = {}
            for role, indices in evaluation.items():
                frame = pool.iloc[indices][["row_id", "pair_id", "chromophore_smiles", "solvent_smiles", "solvent_name", "scaffold_id", "reference_doi", "phi", "tau_ns", "log10_tau_ns"]].copy()
                p_phi, p_tau = predictions["logit_phi"][role], predictions["log10_tau_ns"][role]
                frame["pred_phi"], frame["pred_log10_tau_ns"] = p_phi, p_tau
                frame["phi_lower"], frame["phi_upper"] = np.clip(p_phi-widths[0],0,1), np.clip(p_phi+widths[0],0,1)
                frame["log10_tau_lower"], frame["log10_tau_upper"] = p_tau-widths[1], p_tau+widths[1]
                output = OUT / "predictions" / f"{scenario}_{seed}_{model}_{role}.csv.gz"
                frame.to_csv(output,index=False,compression="gzip")
                prediction_hashes[output.relative_to(OUT).as_posix()] = sha(output)
                if role == "test":
                    scores = {"scenario": scenario, "seed": seed, "model": model, **metrics(frame)}
            record = {"protocol_sha256": locks["protocol_sha256"], "targets": target_records, "metrics": scores,
                      "calibration_width_phi": widths[0], "calibration_width_log10_tau": widths[1], "prediction_sha256": prediction_hashes}
            path.write_text(json.dumps(record,indent=2),encoding="utf-8")
            all_scores.append(scores)
            pd.DataFrame(all_scores).to_csv(OUT/"metrics.csv",index=False)
            seconds = sum(x["elapsed_seconds"] for x in target_records.values())
            print(f"{scenario} {seed} {model}: phi={scores['phi_mae']:.4f}, log_tau={scores['log10_tau_mae']:.4f}, {seconds:.1f}s",flush=True)
    assert len(all_scores) == len(jobs)*len(protocol["models"])
    pd.DataFrame(all_scores).to_csv(OUT/"metrics.csv",index=False)
    assert all(sha(ROOT/"results/pilot_v1"/name)==digest for name,digest in base_manifest.items())
    manifest = {p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob("*") if p.is_file() and p.name!="manifest.json"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Completed {len(all_scores)} additional model/partition evaluations.",flush=True)


if __name__ == "__main__":
    main()
