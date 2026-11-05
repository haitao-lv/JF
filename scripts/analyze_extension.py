"""Combine frozen pilot/extension predictions without changing either run."""
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_pilot import contrast_rows, contrast_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/extension_analysis"


def summarize(frame, keys):
    columns = [c for c in frame.select_dtypes(include=np.number) if c not in keys and c != "seed"]
    result = frame.groupby(keys)[columns].agg(["mean", "std"])
    result.columns = ["__".join(c) for c in result.columns]
    return result.reset_index()


def main():
    assert json.loads((ROOT / "results/extension_validation.json").read_text())["status"] == "passed"
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((ROOT / "protocol_extension_v2.json").read_text())
    cp = json.loads((ROOT / "contrast_protocol_v1.json").read_text())
    metrics = pd.concat([pd.read_csv(ROOT / f"results/{run}/metrics.csv", float_precision="round_trip")
                         for run in ["pilot_v1", "extension_v2"]], ignore_index=True)
    assert len(metrics) == 117 and not metrics.duplicated(["scenario", "seed", "model"]).any()
    metrics.to_csv(OUT / "all_metrics.csv", index=False)
    summarize(metrics, ["scenario", "model"]).to_csv(OUT / "benchmark_summary.csv", index=False)
    comparisons = []
    for alternative, reference, label in protocol["comparisons"]:
        a = metrics[metrics.model.eq(alternative)]
        b = metrics[metrics.model.eq(reference)]
        joined = a.merge(b, on=["scenario", "seed"], suffixes=("_a", "_b"), validate="one_to_one")
        for _, r in joined.iterrows():
            record = dict(comparison=label, alternative=alternative, reference=reference, scenario=r.scenario, seed=int(r.seed))
            for key in ["phi_mae", "log10_tau_mae", "chromophore_macro_phi_mae", "chromophore_macro_log10_tau_mae"]:
                record[key + "_difference"] = r[key + "_a"] - r[key + "_b"]
                record[key + "_relative_percent"] = 100 * (r[key + "_a"] / r[key + "_b"] - 1)
                record[key + "_improved"] = int(r[key + "_a"] < r[key + "_b"])
            comparisons.append(record)
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(OUT / "paired_comparisons.csv", index=False)
    summarize(comparisons, ["comparison", "scenario"]).to_csv(OUT / "comparison_summary.csv", index=False)

    all_contrasts, scores, sensitivity, training = [], [], [], []
    reference_contrasts = {}
    for r in metrics.itertuples():
        run = "extension_v2" if r.model in protocol["models"] else "pilot_v1"
        pred = pd.read_csv(ROOT / f"results/{run}/predictions/{r.scenario}_{r.seed}_{r.model}_test.csv.gz", float_precision="round_trip")
        c, removed = contrast_rows(pred)
        identity = c[["row_a", "row_b", "delta_phi", "delta_log10_tau"]]
        key = (r.scenario, r.seed)
        if key in reference_contrasts:
            pd.testing.assert_frame_equal(identity, reference_contrasts[key], check_exact=True)
        else:
            reference_contrasts[key] = identity
        meta = dict(scenario=r.scenario, seed=r.seed, model=r.model)
        scores.append({**meta, "ambiguous_repeat_rows_excluded": removed, **contrast_scores(c, cp)})
        for phi_cut, tau_factor in product([.025, .05, .10], [1.1, 1.25, 1.5]):
            threshold = {"direction_subset": {"absolute_delta_phi_above": phi_cut, "lifetime_ratio_beyond": tau_factor}}
            sensitivity.append({**meta, "phi_cut": phi_cut, "tau_factor": tau_factor, **contrast_scores(c, threshold)})
        for k, v in meta.items(): c[k] = v
        all_contrasts.append(c)
        if run == "extension_v2":
            fit = json.loads((ROOT / f"results/{run}/fits/{r.scenario}_{r.seed}_{r.model}.json").read_text())
            for target, info in fit["targets"].items():
                training.append({**meta, "target": target, **{k:v for k,v in info.items() if not isinstance(v, dict)}})
    cs = pd.DataFrame(scores)
    cs.to_csv(OUT / "contrast_metrics.csv", index=False)
    summarize(cs, ["scenario", "model"]).to_csv(OUT / "contrast_summary.csv", index=False)
    combined = pd.concat(all_contrasts, ignore_index=True)
    combined.to_csv(OUT / "test_solvent_contrasts.csv.gz", index=False, compression="gzip")
    sens = pd.DataFrame(sensitivity)
    sens.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    summarize(sens, ["scenario", "model", "phi_cut", "tau_factor"]).to_csv(OUT / "threshold_summary.csv", index=False)
    pd.DataFrame(training).to_csv(OUT / "training_counts_and_costs.csv", index=False)
    baseline = cs[cs.model.eq("solvent_mean")]
    delta = cs.merge(baseline, on=["scenario", "seed"], suffixes=("", "_baseline"), validate="many_to_one")
    keys = ["delta_phi_mae", "delta_log10_tau_mae", "phi_direction_accuracy", "tau_direction_accuracy", "joint_direction_accuracy",
            "macro_phi_direction_accuracy", "macro_tau_direction_accuracy", "macro_joint_direction_accuracy"]
    for key in keys: delta[key + "_difference"] = delta[key] - delta[key + "_baseline"]
    delta[["scenario", "seed", "model"] + [k + "_difference" for k in keys]].to_csv(OUT / "solvent_baseline_comparisons.csv", index=False)
    pool = pd.read_csv(ROOT / "data/extension_v2/training_pool.csv")
    distribution = []
    for origin, g in pool.groupby("record_origin"):
        for target in ["phi", "tau_ns", "log10_tau_ns"]:
            values = g[target].dropna()
            if len(values):
                distribution.append(dict(origin=origin, target=target, n=len(values), mean=values.mean(), median=values.median(),
                                         q25=values.quantile(.25), q75=values.quantile(.75)))
    pd.DataFrame(distribution).to_csv(OUT / "observed_label_distributions.csv", index=False)
    report = {"status": "completed exploratory extension", "model_partition_evaluations": len(metrics), "new_evaluations": 63,
              "models": metrics.model.nunique(), "partition_count": 9, "partial_phi_pool": int(pool.record_origin.eq("phi_only").sum()),
              "partial_tau_pool": int(pool.record_origin.eq("tau_only").sum()), "contrast_rows_with_model_and_seed_repetition": len(combined),
              "unique_observed_contrast_pairs_across_splits": len(combined[["row_a", "row_b"]].drop_duplicates()),
              "same_observed_contrasts_verified_across_all_models": True,
              "threshold_analysis": "Exploratory full 3 x 3 grid: |delta phi| > 0.025, 0.05, 0.10 and lifetime ratio > 1.1, 1.25, 1.5; all models and seeds retained.",
              "interpretation": "Old held-out observations were reused. Independent target models train only on real available labels. No unseen-solvent, prospective, or microscopic-rate validation is claimed.",
              "csv_reading": "Use float_precision='round_trip' for saved predictions: the default parser can round a boundary and alter exact interval coverage. No model rerun was required for this reader issue."}
    # Get origin names from actual data, rather than silently accepting a renamed origin.
    origin_counts = pool.record_origin.value_counts().to_dict()
    report["origin_counts"] = {k:int(v) for k,v in origin_counts.items()}
    report.pop("partial_phi_pool"); report.pop("partial_tau_pool")
    (OUT / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    selected = cs[cs.model.isin(["solvent_mean", "et_logit", "et_independent_128", "et_independent_partial", "hgb_independent"])]
    print(selected.groupby(["scenario", "model"])[["phi_direction_accuracy", "tau_direction_accuracy", "joint_direction_accuracy", "macro_joint_direction_accuracy"]].mean().round(4).to_string())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
