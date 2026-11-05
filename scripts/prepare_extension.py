"""Audit single-label observations and build features without altering the pilot."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator

from build_pilot_features import features, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / "protocol_pilot_v1.json").read_text(encoding="utf-8"))
    ext_path = ROOT / "protocol_extension_v2.json"
    audit_path = ROOT / "data/processed/record_audit.csv"
    d = pd.read_csv(audit_path).fillna({"chromophore_smiles": "", "solvent_smiles": "", "reference_doi": ""})
    paired = pd.read_csv(ROOT / protocol["dataset"])
    out = ROOT / "data/extension_v2"
    out.mkdir(exist_ok=True)
    single = d.phi.notna() ^ d.tau_ns.notna()
    d["partial_status"] = "eligible"
    tests = [(~single, "not_exactly_one_reported_target"),
             (d.chromophore_smiles.eq("") | d.solvent_smiles.eq(""), "invalid_structure_or_medium"),
             (d.self_medium, "self_medium"), (~d.known_solvent_identity, "outside_solvent_catalog"),
             (d.chromophore_fragments.ne(1), "disconnected_chromophore"),
             (d.reference_doi.eq(""), "unresolved_reference"), (d.exact_duplicate, "exact_duplicate"),
             (d.phi.notna() & ~d.phi.between(0, 1, inclusive="neither"), "invalid_or_endpoint_phi"),
             (d.tau_ns.notna() & ~(d.tau_ns.gt(0) & np.isfinite(d.tau_ns)), "invalid_lifetime")]
    for mask, reason in tests:
        d.loc[d.partial_status.eq("eligible") & mask, "partial_status"] = reason
    duplicates = set()
    for target in ["phi", "tau_ns"]:
        candidates = d[d.partial_status.eq("eligible") & d[target].notna()].sort_values("row_id")
        combined = pd.concat([paired, candidates], ignore_index=True)
        repeated = combined.duplicated(["chromophore_smiles", "solvent_smiles", "reference_doi", target], keep="first")
        duplicates.update(combined.loc[repeated & (combined.index >= len(paired)), "row_id"])
    d.loc[d.row_id.isin(duplicates), "partial_status"] = "target_repeated_within_source"
    d[["row_id", "partial_status", "phi", "tau_ns", "pair_id", "chromophore_smiles", "scaffold_id", "reference_doi"]].to_csv(out / "partial_record_audit.csv", index=False)
    partial = d[d.partial_status.eq("eligible")].copy()
    partial["record_origin"] = np.where(partial.phi.notna(), "phi_only", "tau_only")
    assert (partial.phi.notna() ^ partial.tau_ns.notna()).all()
    assert not set(partial.row_id) & set(paired.row_id)
    partial.to_csv(out / "partial_label_pool.csv", index=False)
    paired["record_origin"] = "paired_pilot"
    columns = ["row_id", "pair_id", "chromophore_smiles", "solvent_smiles", "solvent_name", "scaffold_id", "reference_doi", "phi", "tau_ns", "record_origin"]
    pool = pd.concat([paired[columns], partial[columns]], ignore_index=True).sort_values("row_id").reset_index(drop=True)
    pool["logit_phi"] = np.log(pool.phi / (1 - pool.phi))
    pool["log10_tau_ns"] = np.log10(pool.tau_ns)
    assert pool.row_id.is_unique
    pool.to_csv(out / "training_pool.csv", index=False)
    fp = protocol["fingerprints"]
    matrices = []
    for role in ["chromophore", "solvent"]:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=fp["radius"], fpSize=fp[role + "_bits"], includeChirality=fp["include_chirality"])
        lookup = {s: features(s, protocol["descriptor_names"], generator) for s in sorted(pool[role + "_smiles"].unique())}
        matrices.append(np.stack(pool[role + "_smiles"].map(lookup)))
    X = np.concatenate(matrices, axis=1)
    X[~np.isfinite(X)] = np.nan
    old = np.load(ROOT / "data/processed/pilot_features.npz", allow_pickle=False)
    index = pd.Series(np.arange(len(pool)), index=pool.row_id)
    np.testing.assert_array_equal(X[index.loc[old["row_id"]].to_numpy()], old["X"])
    np.savez_compressed(out / "features.npz", X=X, row_id=pool.row_id.to_numpy(), feature_names=old["feature_names"])
    report = {"protocol_sha256": sha(ext_path), "base_protocol_sha256": sha(ROOT / "protocol_pilot_v1.json"),
              "audit_source_sha256": sha(audit_path), "preparation_script_sha256": sha(Path(__file__)),
              "paired_rows": len(paired), "phi_only_rows": int(partial.phi.notna().sum()), "tau_only_rows": int(partial.tau_ns.notna().sum()),
              "total_pool_rows": len(pool), "pool_chromophores": pool.chromophore_smiles.nunique(),
              "partial_chromophores": partial.chromophore_smiles.nunique(), "partial_source_dois": partial.reference_doi.nunique(),
              "partial_status_counts": d.partial_status.value_counts().to_dict(), "features": X.shape[1],
              "pilot_feature_values_identical": True, "nonfinite_feature_entries": int(np.isnan(X).sum()),
              "file_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in out.glob("*") if p.is_file()}}
    (ROOT / "results/extension_data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
