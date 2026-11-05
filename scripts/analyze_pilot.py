"""Analyze frozen predictions and same-source, same-molecule solvent contrasts."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/pilot_v1"
OUT = ROOT / "results/pilot_analysis"


def contrast_rows(predictions):
    keys = ["chromophore_smiles", "reference_doi", "solvent_smiles"]
    clean = predictions[~predictions.duplicated(keys, keep=False)].copy()
    records = []
    for (chromophore, doi), group in clean.groupby(keys[:2], sort=True):
        for a, b in combinations(group.sort_values("solvent_smiles").itertuples(), 2):
            assert a.solvent_smiles != b.solvent_smiles
            records.append({"chromophore_smiles": chromophore, "reference_doi": doi,
                            "row_a": a.row_id, "row_b": b.row_id, "solvent_a": a.solvent_name, "solvent_b": b.solvent_name,
                            "delta_phi": b.phi - a.phi, "pred_delta_phi": b.pred_phi - a.pred_phi,
                            "delta_log10_tau": b.log10_tau_ns - a.log10_tau_ns,
                            "pred_delta_log10_tau": b.pred_log10_tau_ns - a.pred_log10_tau_ns})
    return pd.DataFrame(records), len(predictions) - len(clean)


def contrast_scores(c, protocol):
    phi_cut = protocol["direction_subset"]["absolute_delta_phi_above"]
    tau_cut = np.log10(protocol["direction_subset"]["lifetime_ratio_beyond"])
    eligible_phi = c.delta_phi.abs() > phi_cut
    eligible_tau = c.delta_log10_tau.abs() > tau_cut
    both = eligible_phi & eligible_tau
    correct_phi = np.sign(c.delta_phi) == np.sign(c.pred_delta_phi)
    correct_tau = np.sign(c.delta_log10_tau) == np.sign(c.pred_delta_log10_tau)
    result = {"contrasts": len(c), "chromophores": c.chromophore_smiles.nunique(), "references": c.reference_doi.nunique(),
              "phi_direction_n": int(eligible_phi.sum()), "tau_direction_n": int(eligible_tau.sum()), "joint_direction_n": int(both.sum())}
    for target in ["phi", "log10_tau"]:
        error = (c[f"delta_{target}"] - c[f"pred_delta_{target}"]).abs()
        result[f"delta_{target}_mae"] = float(error.mean())
        result[f"macro_delta_{target}_mae"] = float(error.groupby(c.chromophore_smiles).mean().mean())
    for label, eligible, correct in [("phi_direction", eligible_phi, correct_phi), ("tau_direction", eligible_tau, correct_tau), ("joint_direction", both, correct_phi & correct_tau)]:
        result[label + "_accuracy"] = float(correct[eligible].mean())
        result["macro_" + label + "_accuracy"] = float(correct[eligible].groupby(c.loc[eligible, "chromophore_smiles"]).mean().mean())
    result["opposite_observed_direction_fraction"] = float((np.sign(c.delta_phi[both]) != np.sign(c.delta_log10_tau[both])).mean())
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "protocol_pilot_v1.json").read_text(encoding="utf-8"))
    contrast_protocol = json.loads((ROOT / "contrast_protocol_v1.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RUN / "metrics.csv")
    assert len(metrics) == 54 and not metrics.duplicated(["scenario", "seed", "model"]).any()
    scores = [c for c in metrics.columns if c not in {"scenario", "seed", "model"}]
    grouped = metrics.groupby(["scenario", "model"])[scores].agg(["mean", "std"])
    grouped.columns = ["__".join(c) for c in grouped.columns]
    grouped.reset_index().to_csv(OUT / "benchmark_summary.csv", index=False)
    contrasts, contrast_summary = [], []
    subgroups = []
    for scenario in protocol["splits"]:
        for seed in protocol["seeds"]:
            for model in protocol["models"]:
                d = pd.read_csv(RUN / "predictions" / f"{scenario}_{seed}_{model}_test.csv.gz")
                c, exclusions = contrast_rows(d)
                if len(c):
                    score = {"scenario": scenario, "seed": seed, "model": model,
                             "ambiguous_repeat_rows_excluded": exclusions, **contrast_scores(c, contrast_protocol)}
                    contrast_summary.append(score)
                    c["scenario"], c["seed"], c["model"] = scenario, seed, model
                    contrasts.append(c)
                for name, g in d.groupby("solvent_name"):
                    subgroups.append({"scenario": scenario, "seed": seed, "model": model, "solvent": name, "n": len(g),
                                      "phi_mae": float((g.phi - g.pred_phi).abs().mean()),
                                      "log10_tau_mae": float((g.log10_tau_ns - g.pred_log10_tau_ns).abs().mean()),
                                      "phi_coverage": float(g.phi.between(g.phi_lower, g.phi_upper).mean()),
                                      "log10_tau_coverage": float(g.log10_tau_ns.between(g.log10_tau_lower, g.log10_tau_upper).mean())})
    combined = pd.concat(contrasts, ignore_index=True)
    combined.to_csv(OUT / "test_solvent_contrasts.csv.gz", index=False, compression="gzip")
    cs = pd.DataFrame(contrast_summary)
    cs.to_csv(OUT / "contrast_metrics.csv", index=False)
    pd.DataFrame(subgroups).to_csv(OUT / "solvent_subgroups.csv", index=False)
    cs.groupby(["scenario", "model"]).mean(numeric_only=True).reset_index().to_csv(OUT / "contrast_summary.csv", index=False)
    comparisons = []
    for alternative, reference, label in [("et_direct", "et_chromophore_only", "solvent_information"),
                                            ("et_logit", "et_direct", "logit_parameterization"),
                                            ("et_apparent_rates", "et_logit", "rate_vs_logit_parameterization")]:
        joined = metrics[metrics.model == alternative].merge(metrics[metrics.model == reference], on=["scenario", "seed"], suffixes=("_alternative", "_reference"))
        for _, row in joined.iterrows():
            record = {"comparison": label, "alternative": alternative, "reference": reference, "scenario": row.scenario, "seed": int(row.seed)}
            for target in ["phi_mae", "log10_tau_mae"]:
                record[target + "_difference"] = row[target + "_alternative"] - row[target + "_reference"]
            comparisons.append(record)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_model_comparisons.csv", index=False)
    full = pd.read_csv(ROOT / protocol["dataset"])
    # Coverage-selected example: no prediction error or target value enters selection.
    example = pd.read_csv(RUN / "predictions/scaffold_17_et_logit_test.csv.gz")
    example = example[~example.duplicated(["chromophore_smiles", "reference_doi", "solvent_smiles"], keep=False)]
    sizes = example.groupby(["chromophore_smiles", "reference_doi"]).size().reset_index(name="n")
    chosen = sizes.sort_values(["n", "chromophore_smiles", "reference_doi"], ascending=[False, True, True]).iloc[0]
    example = example[(example.chromophore_smiles == chosen.chromophore_smiles) & (example.reference_doi == chosen.reference_doi)].copy()
    counts = full.solvent_name.value_counts()
    example["global_solvent_frequency"] = example.solvent_name.map(counts)
    example = example.sort_values(["global_solvent_frequency", "solvent_name"], ascending=[False, True])
    example.to_csv(OUT / "coverage_selected_example.csv", index=False)
    report = {"models_and_partitions": len(metrics), "saved_calibration_and_test_prediction_files": 108,
              "example_selection": "Scaffold seed 17, ET logit, largest same-chromophore/same-DOI test series after repeated-solvent exclusions; canonical tie-break. No selection on model error.",
              "example_doi": chosen.reference_doi, "example_solvents": int(chosen.n),
              "rate_comparison_note": "The apparent-rate parameterization does not consistently improve predictions and must not be presented as experimentally validated kinetics.",
              "contrast_interpretation": contrast_protocol}
    (OUT / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(cs.groupby(["scenario", "model"])[["contrasts", "joint_direction_n", "phi_direction_accuracy", "tau_direction_accuracy", "joint_direction_accuracy"]].mean().round(4).to_string())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
