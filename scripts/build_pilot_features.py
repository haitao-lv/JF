"""Build deterministic structure-only features; do not fit data preprocessing here."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.*")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(smiles, names, generator):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    values = [float(getattr(Descriptors, name)(m)) for name in names]
    values += [float(Chem.GetFormalCharge(m)),
               sum(a.GetIsAromatic() for a in m.GetAtoms()) / max(m.GetNumAtoms(), 1),
               sum(b.GetIsConjugated() for b in m.GetBonds()) / max(m.GetNumBonds(), 1),
               sum(a.GetAtomicNum() in {9, 17, 35, 53} for a in m.GetAtoms())]
    bits = np.asarray(generator.GetFingerprintAsNumPy(m), dtype=np.float32)
    return np.concatenate([values, bits]).astype(np.float32)


def main():
    config_path = ROOT / "protocol_pilot_v1.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / protocol["dataset"]
    data = pd.read_csv(data_path)
    fp = protocol["fingerprints"]
    matrices, labels = [], []
    for role in ["chromophore", "solvent"]:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=fp["radius"], fpSize=fp[role + "_bits"], includeChirality=fp["include_chirality"])
        lookup = {s: features(s, protocol["descriptor_names"], generator) for s in sorted(data[role + "_smiles"].unique())}
        matrices.append(np.stack(data[role + "_smiles"].map(lookup)))
        labels += [role + "__" + name for name in protocol["descriptor_names"] + protocol["custom_descriptors"]]
        labels += [f"{role}__morgan_{i:03d}" for i in range(fp[role + "_bits"])]
    X = np.concatenate(matrices, axis=1)
    X[~np.isfinite(X)] = np.nan
    assert X.shape == (len(data), len(labels)) and len(set(labels)) == len(labels)
    path = ROOT / "data/processed/pilot_features.npz"
    np.savez_compressed(path, X=X, row_id=data.row_id.to_numpy(), feature_names=np.asarray(labels))
    report = {"data_sha256": sha(data_path), "protocol_sha256": sha(config_path), "features_sha256": sha(path),
              "rows": len(data), "features": len(labels), "chromophore_features": matrices[0].shape[1],
              "solvent_features": matrices[1].shape[1], "nonfinite_entries_for_train_fitted_imputation": int(np.isnan(X).sum()),
              "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "rdkit", "scikit-learn"]},
              "feature_names": labels}
    (ROOT / "results/pilot_features.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
