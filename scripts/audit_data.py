"""Audit raw records and build an explicitly delimited, row-paired pilot set."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.*")

# Identity-based scope, not inferred phase labels or a response-based filter.
# Actual pressure, temperature, oxygen and aggregation conditions remain unknown.
SOLVENTS = {
    "water": "O", "methanol": "CO", "ethanol": "CCO", "1-propanol": "CCCO",
    "2-propanol": "CC(C)O", "1-butanol": "CCCCO", "1-pentanol": "CCCCCO",
    "1-hexanol": "CCCCCCO", "1-octanol": "CCCCCCCCO", "ethylene glycol": "OCCO",
    "glycerol": "OCC(O)CO", "propylene glycol": "CC(O)CO", "2,2,2-trifluoroethanol": "OCC(F)(F)F",
    "acetonitrile": "CC#N", "benzonitrile": "N#Cc1ccccc1", "acetone": "CC(C)=O",
    "ethyl acetate": "CCOC(C)=O", "dimethyl sulfoxide": "CS(C)=O", "dimethylformamide": "CN(C)C=O",
    "dimethylacetamide": "CC(=O)N(C)C", "tetrahydrofuran": "C1CCOC1", "1,4-dioxane": "C1COCCO1",
    "diethyl ether": "CCOCC", "dibutyl ether": "CCCCOCCCC", "2-methyltetrahydrofuran": "CC1CCCO1",
    "dichloromethane": "ClCCl", "chloroform": "ClC(Cl)Cl", "carbon tetrachloride": "ClC(Cl)(Cl)Cl",
    "1,2-dichloroethane": "ClCCCl", "chlorobenzene": "Clc1ccccc1", "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1", "cyclohexane": "C1CCCCC1", "methylcyclohexane": "CC1CCCCC1",
    "n-hexane": "CCCCCC", "n-heptane": "CCCCCCC", "n-octane": "CCCCCCCC", "isooctane": "CC(C)CC(C)(C)C",
    "propylene carbonate": "CC1COC(=O)O1", "pyridine": "c1ccncc1", "nitromethane": "C[N+](=O)[O-]",
}


def canon(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else ""


def doi(value):
    match = re.search(r"10\.\d{4,9}/\S+", str(value))
    return match.group(0).rstrip(".,;").lower() if match else ""


def summary(frame):
    return {"rows": len(frame), "chromophores": int(frame.chromophore_smiles.nunique()),
            "solvents": int(frame.solvent_smiles.nunique()), "pairs": int(frame.pair_id.nunique()),
            "scaffolds": int(frame.scaffold_id.nunique()), "references": int(frame.reference_doi.nunique())}


def main():
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "data/source_lock.json").read_text())
    raw_path = ROOT / lock["files"][0]["file"]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == lock["files"][0]["sha256"]
    raw = pd.read_csv(raw_path)
    assert raw.Tag.is_unique and len(raw) == 20236
    structures = []
    for value in sorted(set(raw.Chromophore) | set(raw.Solvent)):
        m = Chem.MolFromSmiles(value)
        smi = Chem.MolToSmiles(m, isomericSmiles=True) if m else ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else ""
        structures.append({"raw_smiles": value, "canonical_smiles": smi, "parse_valid": m is not None,
                           "fragments": len(Chem.GetMolFrags(m)) if m else np.nan,
                           "charge": Chem.GetFormalCharge(m) if m else np.nan,
                           "molecular_weight": Descriptors.MolWt(m) if m else np.nan,
                           "scaffold_id": scaffold or ("acyclic:" + smi if smi else "invalid")})
    structures = pd.DataFrame(structures)
    structures.to_csv(out / "structure_audit.csv", index=False)
    lookup = structures.set_index("raw_smiles")
    catalog = pd.DataFrame([{"name": k, "smiles": canon(v)} for k, v in SOLVENTS.items()])
    assert catalog.smiles.is_unique
    catalog.to_csv(out / "solvent_scope.csv", index=False)
    names = catalog.set_index("smiles").name.to_dict()
    d = pd.DataFrame({"row_id": raw.Tag, "chromophore_raw": raw.Chromophore, "solvent_raw": raw.Solvent,
                      "reference_raw": raw.Reference, "phi": raw["Quantum yield"], "tau_ns": raw["Lifetime (ns)"],
                      "absorption_nm": raw["Absorption max (nm)"], "emission_nm": raw["Emission max (nm)"],
                      "log10_extinction": raw["log(e/mol-1 dm3 cm-1)"]})
    d["chromophore_smiles"] = d.chromophore_raw.map(lookup.canonical_smiles)
    d["solvent_smiles"] = d.solvent_raw.map(lookup.canonical_smiles)
    d["scaffold_id"] = d.chromophore_raw.map(lookup.scaffold_id)
    d["chromophore_charge"] = d.chromophore_raw.map(lookup.charge)
    d["chromophore_fragments"] = d.chromophore_raw.map(lookup.fragments)
    d["reference_doi"] = d.reference_raw.map(doi)
    d["pair_id"] = [hashlib.sha256((a + "|" + b).encode()).hexdigest()[:20] for a, b in zip(d.chromophore_smiles, d.solvent_smiles)]
    d["solvent_name"] = d.solvent_smiles.map(names).fillna("")
    d["self_medium"] = (d.chromophore_smiles == d.solvent_smiles) & d.chromophore_smiles.ne("")
    d["known_solvent_identity"] = d.solvent_smiles.isin(names)
    d["paired_reported"] = d.phi.notna() & d.tau_ns.notna()
    d["valid_targets"] = d.phi.between(0, 1) & d.tau_ns.gt(0) & np.isfinite(d.tau_ns)
    d["boundary_phi"] = d.phi.isin([0, 1])
    d["exact_duplicate"] = raw.drop(columns="Tag").duplicated()
    d["scope_status"] = "included"
    tests = [
        (~d.paired_reported, "missing_paired_target"),
        (~d.valid_targets, "invalid_target"),
        (d.chromophore_smiles.eq("") | d.solvent_smiles.eq(""), "invalid_structure_or_nonmolecular_medium"),
        (d.self_medium, "self_medium_solid_convention"),
        (~d.known_solvent_identity, "outside_explicit_solvent_scope"),
        (d.chromophore_fragments.ne(1), "disconnected_chromophore"),
        (d.reference_doi.eq(""), "unresolved_reference"),
        (d.exact_duplicate, "exact_duplicate"),
    ]
    for condition, reason in tests:
        d.loc[d.scope_status.eq("included") & condition, "scope_status"] = reason
    d["kinetic_interpretation_status"] = "unverified_decay_model_and_measurement_conditions"
    d.to_csv(out / "record_audit.csv", index=False)
    paired = d[d.scope_status.eq("included")].copy()
    paired.to_csv(out / "paired_solution_scope.csv", index=False)
    interior = paired[~paired.boundary_phi].copy()
    interior["log10_tau_ns"] = np.log10(interior.tau_ns)
    interior["logit_phi"] = np.log(interior.phi / (1 - interior.phi))
    interior["log10_kr_apparent_s-1"] = np.log10(interior.phi / interior.tau_ns) + 9
    interior["log10_knr_apparent_s-1"] = np.log10((1 - interior.phi) / interior.tau_ns) + 9
    # This is an invertible reparameterization, not independently measured rates.
    rate_sum = 10 ** interior["log10_kr_apparent_s-1"] + 10 ** interior["log10_knr_apparent_s-1"]
    assert np.allclose(1e9 / rate_sum, interior.tau_ns, rtol=1e-12)
    assert np.allclose(10 ** interior["log10_kr_apparent_s-1"] / rate_sum, interior.phi, rtol=1e-12)
    interior.to_csv(out / "pilot_interior_pairs.csv", index=False)
    media = d[d.paired_reported].groupby(["solvent_raw", "solvent_smiles", "solvent_name", "known_solvent_identity", "self_medium"], dropna=False).agg(
        rows=("row_id", "size"), chromophores=("chromophore_smiles", "nunique"), references=("reference_doi", "nunique")).reset_index().sort_values("rows", ascending=False)
    media.to_csv(out / "paired_environment_inventory.csv", index=False)
    repeats = paired.groupby(["pair_id", "reference_doi"], dropna=False).agg(
        records=("row_id", "size"), phi_min=("phi", "min"), phi_max=("phi", "max"),
        tau_min_ns=("tau_ns", "min"), tau_max_ns=("tau_ns", "max")).reset_index()
    repeats["phi_range"] = repeats.phi_max - repeats.phi_min
    repeats["tau_ratio"] = repeats.tau_max_ns / repeats.tau_min_ns
    repeats[repeats.records > 1].to_csv(out / "within_reference_repeats.csv", index=False)
    reference_summary = paired.groupby("reference_doi").agg(rows=("row_id", "size"), chromophores=("chromophore_smiles", "nunique"),
        solvents=("solvent_smiles", "nunique"), tau_max_ns=("tau_ns", "max"), phi_min=("phi", "min"), phi_max=("phi", "max")).reset_index().sort_values("rows", ascending=False)
    reference_summary.to_csv(out / "source_review_queue.csv", index=False)
    report = {"source_sha256": lock["files"][0]["sha256"], "raw_rows": len(raw),
              "raw_unique_chromophore_strings": int(raw.Chromophore.nunique()), "raw_unique_environment_strings": int(raw.Solvent.nunique()),
              "published_introductory_counts": {"chromophores": 7016, "solvents": 365, "solid_hosts": 17},
              "count_note": "The versioned CSV contains 6815 raw chromophore strings. Published molecule counts were not substituted for counts recomputed from this file. Environment strings include solid/self environments.",
              "target_availability": {"phi": int(d.phi.notna().sum()), "tau": int(d.tau_ns.notna().sum()), "both": int(d.paired_reported.sum())},
              "all_raw_pairs": summary(d[d.paired_reported]), "exclusion_accounting": d.scope_status.value_counts().to_dict(),
              "solution_scope": summary(paired), "interior_pilot": summary(interior),
              "boundary_phi_in_solution_scope": int(paired.boundary_phi.sum()),
              "charged_chromophore_records_retained": int(paired.chromophore_charge.ne(0).sum()),
              "within_reference_repeated_pair_groups": int((repeats.records > 1).sum()),
              "within_reference_phi_range_above_0_1": int((repeats.phi_range > .1).sum()),
              "within_reference_tau_ratio_above_2": int((repeats.tau_ratio > 2).sum()),
              "metadata_unavailable": ["temperature", "concentration", "pH", "oxygen_or_degassing", "single_vs_multiexponential", "prompt_vs_delayed", "measurement_level_uncertainty"],
              "verified_single_state_rate_records": 0,
              "pilot_phi_tau_spearman": float(interior.phi.corr(interior.log10_tau_ns, method="spearman")),
              "pilot_lifetime_ns_quantiles": {str(k): float(v) for k, v in interior.tau_ns.quantile([0,.01,.5,.95,.99,1]).items()},
              "all_exact_duplicate_rows": int(d.exact_duplicate.sum())}
    assert sum(report["exclusion_accounting"].values()) == len(raw)
    (ROOT / "results/data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
