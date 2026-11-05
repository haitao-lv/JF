"""Frozen-prediction analyses for the reviewer-led manuscript revision."""
from pathlib import Path
from itertools import combinations
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision_v4'
READ={'float_precision':'round_trip'}
SEEDS=[17,41,73]
SCENARIOS=['pair','chromophore','scaffold']
MODELS=['et_logit','et_independent_128','solvent_mean']

def read_prediction(s,seed,m):
    run='pilot_v1' if m=='et_logit' else 'extension_v2'
    return pd.read_csv(ROOT/f'results/{run}/predictions/{s}_{seed}_{m}_test.csv.gz',**READ)

def bootstrap_mean(values,seed):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(values),np.ones(len(values))/len(values),size=4000)
    b=weights@values/len(values)
    return np.quantile(b,[.025,.975])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'version':'revision-v4','status':'Exploratory post-review analysis of previously inspected frozen test predictions; no model fitting.',
        'direction_thresholds':{'absolute_delta_phi':.05,'lifetime_ratio':1.25},'bootstrap_replicates':4000,'bootstrap_seed':20260912,
        'source_macro':'Average eligible-contrast correctness within each DOI, then equally over DOIs.',
        'direction_bootstrap':'Resample paired DOI-level model differences; conditional on existing fitted models and test DOIs.',
        'mae_bootstrap':'Resample molecular groups for pair/chromophore holdouts and framework groups for scaffold holdouts, retaining all paired rows; ratio of resampled error sums to resampled row counts.',
        'case_selection':'Scaffold seed 17, ET-L test records; remove every repeated chromophore-DOI-solvent group; rank chromophore-DOI series by solvent count descending then canonical SMILES and DOI ascending; select first three series with distinct DOIs and framework identifiers, each with at least 3 solvents. No prediction error or response value enters selection.',
        'interval_decision':'On all unambiguous test contrasts, declare a target direction only if its separately calibrated 90% marginal intervals do not overlap; declare both only when both targets qualify. No joint or selective coverage guarantee is claimed.',
        'raw_scope':'Use original frozen paired records and all model predictions; no target correction or removal.'}
    (OUT/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    c=pd.read_csv(ROOT/'results/extension_analysis/test_solvent_contrasts.csv.gz',**READ)
    c['eligible']=(c.delta_phi.abs()>.05)&(c.delta_log10_tau.abs()>np.log10(1.25))
    c['response_type']=np.where(c.delta_phi*c.delta_log10_tau>0,'Same','Opposite')
    c['correct']=(np.sign(c.delta_phi)==np.sign(c.pred_delta_phi))&(np.sign(c.delta_log10_tau)==np.sign(c.pred_delta_log10_tau))
    joint=c[c.eligible].copy()
    records=[];counts=[];source=[];source_boot=[];jack=[]
    for (s,seed,m),g in joint.groupby(['scenario','seed','model']):
        for kind in ['All','Same','Opposite']:
            z=g if kind=='All' else g[g.response_type.eq(kind)]
            records.append(dict(scenario=s,seed=seed,model=m,response_type=kind,n=len(z),molecules=z.chromophore_smiles.nunique(),dois=z.reference_doi.nunique(),
                accuracy=z.correct.mean(),molecule_macro=z.groupby('chromophore_smiles').correct.mean().mean(),source_macro=z.groupby('reference_doi').correct.mean().mean()))
            if m=='et_logit':
                vc=z.reference_doi.value_counts()
                counts.append(dict(scenario=s,seed=seed,response_type=kind,contrasts=len(z),molecules=z.chromophore_smiles.nunique(),dois=len(vc),
                    top_doi_fraction=vc.iloc[0]/len(z),top5_doi_fraction=vc.head(5).sum()/len(z)))
        for doi,z in g.groupby('reference_doi'):
            source.append(dict(scenario=s,seed=seed,model=m,reference_doi=doi,contrasts=len(z),accuracy=z.correct.mean()))
    response=pd.DataFrame(records);response.to_csv(OUT/'response_scores.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'effective_counts.csv',index=False)
    src=pd.DataFrame(source);src.to_csv(OUT/'source_scores.csv',index=False)
    counter=0
    for s in SCENARIOS:
        for seed in SEEDS:
            a=src[(src.scenario==s)&(src.seed==seed)&(src.model=='et_logit')]
            for m in ['et_independent_128','solvent_mean']:
                b=src[(src.scenario==s)&(src.seed==seed)&(src.model==m)]
                z=a.merge(b,on=['scenario','seed','reference_doi'],suffixes=('_shared','_reference'),validate='one_to_one')
                assert (z.contrasts_shared==z.contrasts_reference).all()
                diff=z.accuracy_shared-z.accuracy_reference
                lo,hi=bootstrap_mean(diff,20260912+counter);counter+=1
                source_boot.append(dict(scenario=s,seed=seed,reference_model=m,dois=len(z),difference=diff.mean(),ci_low=lo,ci_high=hi))
                for i in range(len(z)):
                    reduced=z.drop(index=z.index[i]);weights=reduced.contrasts_shared
                    delta=reduced.accuracy_shared-reduced.accuracy_reference
                    jack.append(dict(scenario=s,seed=seed,reference_model=m,omitted_doi=z.iloc[i].reference_doi,
                        contrast_difference=np.average(delta,weights=weights),source_macro_difference=delta.mean()))
    pd.DataFrame(source_boot).to_csv(OUT/'source_paired_bootstrap.csv',index=False)
    pd.DataFrame(jack).to_csv(OUT/'leave_one_source_out.csv',index=False)
    maes=[];decisions=[]
    for s in SCENARIOS:
        for seed in SEEDS:
            a=read_prediction(s,seed,'et_logit');b=read_prediction(s,seed,'et_independent_128')
            z=a.merge(b,on='row_id',suffixes=('_shared','_independent'),validate='one_to_one')
            assert len(z)==len(a)==len(b)
            group='scaffold_id_shared' if s=='scaffold' else 'chromophore_smiles_shared'
            z['phi_error_diff']=(z.phi_shared-z.pred_phi_shared).abs()-(z.phi_independent-z.pred_phi_independent).abs()
            z['tau_error_diff']=(z.log10_tau_ns_shared-z.pred_log10_tau_ns_shared).abs()-(z.log10_tau_ns_independent-z.pred_log10_tau_ns_independent).abs()
            g=z.groupby(group).agg(n=('row_id','size'),phi_sum=('phi_error_diff','sum'),tau_sum=('tau_error_diff','sum'))
            rng=np.random.default_rng(20260912+counter);counter+=1
            w=rng.multinomial(len(g),np.ones(len(g))/len(g),size=4000)
            result=dict(scenario=s,seed=seed,groups=len(g),rows=len(z),group_type='framework' if s=='scaffold' else 'chromophore')
            for target in ['phi','tau']:
                boot=(w@g[target+'_sum'].to_numpy())/(w@g.n.to_numpy())
                lo,hi=np.quantile(boot,[.025,.975]);result.update({target+'_difference':z[target+'_error_diff'].mean(),target+'_ci_low':lo,target+'_ci_high':hi})
            maes.append(result)
            for m in MODELS:
                p=read_prediction(s,seed,m).set_index('row_id')
                pairs=c[(c.scenario==s)&(c.seed==seed)&(c.model==m)]
                x=p.loc[pairs.row_a];y=p.loc[pairs.row_b]
                dphi=np.where(y.phi_lower.to_numpy()>x.phi_upper.to_numpy(),1,np.where(y.phi_upper.to_numpy()<x.phi_lower.to_numpy(),-1,0))
                dtau=np.where(y.log10_tau_lower.to_numpy()>x.log10_tau_upper.to_numpy(),1,np.where(y.log10_tau_upper.to_numpy()<x.log10_tau_lower.to_numpy(),-1,0))
                declared=(dphi!=0)&(dtau!=0)
                correct=(dphi==np.sign(pairs.delta_phi.to_numpy()))&(dtau==np.sign(pairs.delta_log10_tau.to_numpy()))
                decisions.append(dict(scenario=s,seed=seed,model=m,contrasts=len(pairs),phi_declared=int((dphi!=0).sum()),tau_declared=int((dtau!=0).sum()),
                    jointly_declared=int(declared.sum()),joint_declaration_fraction=declared.mean(),joint_accuracy=float(correct[declared].mean()) if declared.any() else np.nan))
    pd.DataFrame(maes).to_csv(OUT/'mae_paired_bootstrap.csv',index=False)
    pd.DataFrame(decisions).to_csv(OUT/'interval_decisions.csv',index=False)
    d=read_prediction('scaffold',17,'et_logit')
    d=d[~d.duplicated(['chromophore_smiles','reference_doi','solvent_smiles'],keep=False)]
    rank=d.groupby(['chromophore_smiles','reference_doi','scaffold_id']).size().reset_index(name='solvents').sort_values(['solvents','chromophore_smiles','reference_doi'],ascending=[False,True,True])
    rank.to_csv(OUT/'case_selection_ranking.csv',index=False)
    chosen=[];seen_doi=set();seen_scaffold=set()
    for r in rank.itertuples(index=False):
        if r.solvents<3:continue
        if r.reference_doi in seen_doi or r.scaffold_id in seen_scaffold:continue
        cid='ABC'[len(chosen)];chosen.append(dict(case=cid,chromophore_smiles=r.chromophore_smiles,reference_doi=r.reference_doi,scaffold_id=r.scaffold_id,solvents=int(r.solvents)))
        seen_doi.add(r.reference_doi);seen_scaffold.add(r.scaffold_id)
        if len(chosen)==3:break
    case_rows=[];case_scores=[]
    for case in chosen:
        q=d[(d.chromophore_smiles==case['chromophore_smiles'])&(d.reference_doi==case['reference_doi'])].sort_values('solvent_name').copy()
        q.insert(0,'case',case['case']);case_rows.append(q)
        t=joint[(joint.scenario=='scaffold')&(joint.seed==17)&(joint.model=='et_logit')&(joint.chromophore_smiles==case['chromophore_smiles'])&(joint.reference_doi==case['reference_doi'])]
        for kind in ['All','Same','Opposite']:
            v=t if kind=='All' else t[t.response_type==kind]
            case_scores.append(dict(case=case['case'],response_type=kind,n=len(v),correct=int(v.correct.sum()),accuracy=v.correct.mean()))
    pd.concat(case_rows).to_csv(OUT/'case_observations_predictions.csv',index=False)
    pd.DataFrame(case_scores).to_csv(OUT/'case_direction_scores.csv',index=False)
    (OUT/'selected_cases.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    raw=pd.read_csv(ROOT/'data/processed/record_audit.csv',**READ)
    paired=pd.read_csv(ROOT/'data/processed/pilot_interior_pairs.csv',**READ)
    tail=paired[(paired.tau_ns<.01)|(paired.tau_ns>100)].copy();tail.to_csv(OUT/'extreme_record_audit.csv',index=False)
    excluded=raw[raw.paired_reported&raw.scope_status.eq('outside_explicit_solvent_scope')]
    excluded.groupby(['solvent_raw','solvent_smiles']).agg(rows=('row_id','size'),chromophores=('chromophore_smiles','nunique')).reset_index().sort_values('rows',ascending=False).to_csv(OUT/'excluded_solvent_inventory.csv',index=False)
    audit={'status':'passed','models_retrained':0,'rows_relabelled':0,'case_selection_uses_errors':False,'conditional_bootstrap_replicates':4000,
        'shared_interval_declarations':pd.DataFrame(decisions).query("model == 'et_logit'")['jointly_declared'].tolist(),
        'case_counts':[x['solvents'] for x in chosen],'extreme_records':len(tail),
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}}
    (OUT/'analysis_manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(pd.DataFrame(source_boot).round(4).to_string(index=False))
    print(pd.DataFrame(maes).round(4).to_string(index=False))
    print(pd.DataFrame(decisions).to_string(index=False))
    print(pd.DataFrame(case_scores).to_string(index=False))

if __name__=='__main__':main()
