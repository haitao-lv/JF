"""Independently check real-label membership, stored scores and calibration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/extension_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol=json.loads((ROOT/"protocol_extension_v2.json").read_text(encoding="utf-8"))
    lock=json.loads((OUT/"run_lock.json").read_text(encoding="utf-8"))
    assert sha(ROOT/"protocol_extension_v2.json")==lock["protocol_sha256"]
    for name,digest in lock["source_sha256"].items(): assert sha(ROOT/"scripts"/name)==digest
    for name,digest in lock["input_sha256"].items(): assert sha(ROOT/name)==digest
    assert sha(ROOT/"results/pilot_v1/manifest.json")==lock["pilot_manifest_sha256"]
    for directory in [OUT,ROOT/"results/pilot_v1"]:
        manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        assert all(sha(directory/name)==digest for name,digest in manifest.items())
    pool=pd.read_csv(ROOT/"data/extension_v2/training_pool.csv").set_index("row_id")
    raw=pd.read_csv(ROOT/"data/processed/record_audit.csv").set_index("row_id")
    assert pool.index.is_unique
    for col in ["phi","tau_ns"]:
        np.testing.assert_allclose(pool[col],raw.loc[pool.index,col],equal_nan=True,atol=1e-14,rtol=1e-14)
    partial=pool[pool.record_origin.ne("paired_pilot")]
    assert (partial.phi.notna() ^ partial.tau_ns.notna()).all()
    assert partial.phi.dropna().between(0,1,inclusive="neither").all()
    assert partial.tau_ns.dropna().gt(0).all()
    assert raw.loc[partial.index,"known_solvent_identity"].all()
    assert not raw.loc[partial.index,"self_medium"].any()
    assert raw.loc[partial.index,"chromophore_fragments"].eq(1).all()
    for target in ["phi","tau_ns"]:
        paired=pool[pool.record_origin.eq("paired_pilot")]
        candidates=partial[partial[target].notna()].sort_index()
        ordered=pd.concat([paired,candidates])
        repeated=ordered.duplicated(["chromophore_smiles","solvent_smiles","reference_doi",target])
        assert not repeated.iloc[len(paired):].any()
    metrics=pd.read_csv(OUT/"metrics.csv")
    assert len(metrics)==63 and not metrics.duplicated(["scenario","seed","model"]).any()
    count=0
    for scenario,group in protocol["scenarios"].items():
        for seed in protocol["seeds"]:
            base=pd.read_csv(ROOT/protocol["base_partitions"]/f"{scenario}_{seed}.csv").set_index("row_id")
            members=pd.read_csv(OUT/"memberships"/f"{scenario}_{seed}.csv.gz").set_index("row_id")
            assert set(members.index)==set(pool.index)
            assert members.loc[base.index,"subset"].equals(base.subset)
            protected_ids=base[base.subset.isin(["calibration","test"])].index
            protected=set(pool.loc[protected_ids,group])
            paired_train=set(base[base.subset.eq("train")].index)
            expected_partial=set(partial[~partial[group].isin(protected)].index)
            actual_partial=set(members[members.subset.eq("partial_train")].index)
            assert expected_partial==actual_partial
            allowed=pool.loc[sorted(paired_train|expected_partial)]
            assert not set(allowed[group]) & protected
            assert not set(allowed.pair_id) & set(pool.loc[protected_ids,"pair_id"])
            for model,spec in protocol["models"].items():
                record=json.loads((OUT/"fits"/f"{scenario}_{seed}_{model}.json").read_text(encoding="utf-8"))
                allowed_ids=paired_train|(expected_partial if spec["partial_labels"] else set())
                train=pool.loc[sorted(allowed_ids)]
                cal=pd.read_csv(OUT/"predictions"/f"{scenario}_{seed}_{model}_calibration.csv.gz", float_precision="round_trip").set_index("row_id")
                test=pd.read_csv(OUT/"predictions"/f"{scenario}_{seed}_{model}_test.csv.gz", float_precision="round_trip").set_index("row_id")
                for role,frame in [("calibration",cal),("test",test)]:
                    assert frame.index.is_unique and set(frame.index)==set(base[base.subset.eq(role)].index)
                    for col in ["phi","tau_ns","log10_tau_ns"]:
                        np.testing.assert_allclose(frame[col],pool.loc[frame.index,col],rtol=1e-13,atol=1e-13)
                for target in protocol["targets"]:
                    available=train[train[target].notna()]
                    info=record["targets"][target]
                    assert len(available)==info["training_rows"]
                    assert info["training_row_id_sha256"]==hashlib.sha256(np.asarray(available.index,dtype="<i8").tobytes()).hexdigest()
                    if spec["family"]=="solvent_mean":
                        col="phi" if target=="logit_phi" else "log10_tau_ns"
                        pred_col="pred_phi" if target=="logit_phi" else "pred_log10_tau_ns"
                        means=available.groupby("solvent_smiles")[col].mean()
                        expected=test.solvent_smiles.map(means).fillna(available[col].mean())
                        np.testing.assert_allclose(test[pred_col],expected,rtol=1e-12,atol=1e-12)
                    else:
                        assert abs(available[target].mean()-info["target_mean"])<1e-12
                        assert abs(available[target].std(ddof=0)-info["target_scale"])<1e-12
                score=metrics[(metrics.scenario==scenario)&(metrics.seed==seed)&(metrics.model==model)].iloc[0]
                rank=int(np.ceil(.9*(len(cal)+1)))
                for target,pred,lo,hi,key in [("phi","pred_phi","phi_lower","phi_upper","phi"),
                                             ("log10_tau_ns","pred_log10_tau_ns","log10_tau_lower","log10_tau_upper","log10_tau")]:
                    width=np.partition(np.abs(cal[target].to_numpy()-cal[pred].to_numpy()),rank-1)[rank-1]
                    lower=test[pred]-width
                    upper=test[pred]+width
                    if target=="phi": lower,upper=lower.clip(0,1),upper.clip(0,1)
                    np.testing.assert_allclose(test[lo],lower,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(test[hi],upper,rtol=1e-12,atol=1e-12)
                    error=(test[target]-test[pred]).to_numpy()
                    assert abs(np.abs(error).mean()-score[key+"_mae"])<1e-12
                    assert abs(np.sqrt(np.mean(error**2))-score[key+"_rmse"])<1e-12
                    r2=1-np.sum(error**2)/np.sum((test[target]-test[target].mean())**2)
                    assert abs(r2-score[key+"_r2"])<1e-12
                    assert abs(test[target].between(test[lo],test[hi]).mean()-score[key+"_coverage"])<1e-12, (scenario, seed, model, target, test[target].between(test[lo],test[hi]).mean(), score[key+"_coverage"])
                    for col,label in [("chromophore_smiles","chromophore"),("scaffold_id","scaffold")]:
                        macro=(test[target]-test[pred]).abs().groupby(test[col]).mean().mean()
                        assert abs(macro-score[label+"_macro_"+key+"_mae"])<1e-12
                extra = {
                    "n": len(test), "n_chromophores": test.chromophore_smiles.nunique(), "n_scaffolds": test.scaffold_id.nunique(),
                    "tau_ns_mae": np.abs(test.tau_ns - 10.0 ** test.pred_log10_tau_ns).mean(),
                    "tau_within_factor_2": ((test.log10_tau_ns-test.pred_log10_tau_ns).abs() <= np.log10(2)).mean(),
                    "simultaneous_coverage": (test.phi.between(test.phi_lower,test.phi_upper) & test.log10_tau_ns.between(test.log10_tau_lower,test.log10_tau_upper)).mean(),
                    "phi_interval_width": (test.phi_upper-test.phi_lower).mean(),
                    "log10_tau_interval_width": (test.log10_tau_upper-test.log10_tau_lower).mean()
                }
                for name,value in extra.items():
                    np.testing.assert_allclose(value,score[name],rtol=1e-12,atol=1e-12,err_msg=name)
                count+=1
    result={"status":"passed","model_partition_evaluations_checked":count,"prediction_files_checked":count*2,
            "membership_files_checked":9,"partial_labels_are_original_observations":True,
            "target_repeated_added_rows_absent":True,"protected_groups_excluded_from_every_target_training_set":True,
            "training_ids_and_scaling_recomputed":True,"solvent_baselines_independently_recomputed":True,
            "scores_and_calibration_recomputed":True,"pilot_v1_bytes_preserved":True}
    (ROOT/"results/extension_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
