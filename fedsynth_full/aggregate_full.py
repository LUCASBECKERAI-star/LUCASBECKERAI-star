#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

RNG=np.random.default_rng(20261124)

def bootstrap_mean(x, n=3000):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): return {'mean':np.nan,'lo':np.nan,'hi':np.nan,'n':0}
    means=np.array([RNG.choice(x,len(x),replace=True).mean() for _ in range(n)])
    return {'mean':float(x.mean()),'lo':float(np.percentile(means,2.5)),'hi':float(np.percentile(means,97.5)),'n':int(len(x))}

def safe_spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return (np.nan,np.nan,int(m.sum()))
    r,p=spearmanr(x[m],y[m]); return (float(r),float(p),int(m.sum()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); (a.out/'figures').mkdir(exist_ok=True); (a.out/'tables').mkdir(exist_ok=True)
    metric_files=sorted(a.root.rglob('chunk_metrics_long.csv'))
    summary_files=sorted(a.root.rglob('chunk_case_summaries.jsonl'))
    if len(metric_files)!=30: print(f'WARNING metric chunks={len(metric_files)} expected=30')
    long=pd.concat([pd.read_csv(p) for p in metric_files],ignore_index=True)
    summaries=[]
    for p in summary_files:
        for line in p.read_text().splitlines():
            if line.strip(): summaries.append(json.loads(line))
    qa=pd.DataFrame([{
        'case_id':s['case_id'],'anonymous_center':s['anonymous_center'],
        'mean_range_hu':s['qa']['mean_range_hu'],'p95_range_hu':s['qa']['p95_range_hu'],
        'mean_mad_hu':s['qa']['mean_mad_hu'],'voxel_rho_range_error':s['qa']['spearman_range_vs_abs_error'],
        'inference_seconds_cpu':s['total_inference_seconds_cpu'],'hu_gate':s['hu_plausibility_gate']
    } for s in summaries])
    fused=long[long.prediction=='median_fusion'].copy()
    cases=fused.merge(qa,on=['case_id','anonymous_center'],how='left')
    long.to_csv(a.out/'tables'/'all_metrics_long.csv',index=False); cases.to_csv(a.out/'tables'/'case_level_fused_and_qa.csv',index=False)

    key=['full_valid_mae','soft_tissue_mae','skull_200_mae','dense_bone_500_mae','bone_air_boundary_mae','bone_dice_200','slice_mean_ssim_clipped','psnr_clipped_valid']
    records=[]
    for center in ['ALL','A','B','C']:
        sub=cases if center=='ALL' else cases[cases.anonymous_center==center]
        for metric in key:
            b=bootstrap_mean(sub[metric]); records.append({'center':center,'metric':metric,**b})
    pd.DataFrame(records).to_csv(a.out/'tables'/'bootstrap_key_metrics.csv',index=False)

    ab=[]
    for metric in ['full_valid_mae','soft_tissue_mae','skull_200_mae','bone_air_boundary_mae','bone_dice_200','slice_mean_ssim_clipped']:
        wide=long.pivot(index='case_id',columns='prediction',values=metric)
        for plane in ['plane0','plane1','plane2']:
            d=(wide['median_fusion']-wide[plane]).dropna()
            stat,p=wilcoxon(d) if len(d)>0 and np.any(d!=0) else (np.nan,np.nan)
            ab.append({'metric':metric,'comparison':f'median_fusion_minus_{plane}','n':len(d),'mean_difference':d.mean(),'median_difference':d.median(),'wilcoxon_p':p})
    pd.DataFrame(ab).to_csv(a.out/'tables'/'fusion_ablation.csv',index=False)

    qa_rows=[]
    for center in ['ALL','A','B','C']:
        sub=cases if center=='ALL' else cases[cases.anonymous_center==center]
        for feature in ['mean_range_hu','p95_range_hu','mean_mad_hu']:
            for outcome in ['full_valid_mae','skull_200_mae','bone_air_boundary_mae']:
                r,p,n=safe_spearman(sub[feature],sub[outcome]); qa_rows.append({'center':center,'feature':feature,'outcome':outcome,'rho':r,'p':p,'n':n})
    pd.DataFrame(qa_rows).to_csv(a.out/'tables'/'qa_correlations.csv',index=False)

    features=['mean_range_hu','p95_range_hu','mean_mad_hu']
    loco=[]
    for held in ['A','B','C']:
        tr=cases[cases.anonymous_center!=held].copy(); te=cases[cases.anonymous_center==held].copy()
        Xtr=np.log1p(tr[features].to_numpy(float)); Xte=np.log1p(te[features].to_numpy(float)); ytr=tr.full_valid_mae.to_numpy(float); yte=te.full_valid_mae.to_numpy(float)
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.0)); model.fit(Xtr,ytr); pred=model.predict(Xte)
        threshold=float(np.percentile(ytr,75)); truth=(yte>=threshold).astype(int)
        auc=roc_auc_score(truth,pred) if len(np.unique(truth))==2 else np.nan
        apv=average_precision_score(truth,pred) if truth.sum()>0 else np.nan
        r,p,n=safe_spearman(pred,yte)
        for cid,actual,pr,t in zip(te.case_id,yte,pred,truth): loco.append({'heldout_center':held,'case_id':cid,'actual_mae':actual,'predicted_mae':pr,'high_error_threshold_train':threshold,'high_error_truth':int(t)})
        qa_rows.append({'center':f'heldout_{held}','feature':'combined_ridge','outcome':'full_valid_mae','rho':r,'p':p,'n':n,'auroc':auc,'auprc':apv,'prediction_mae':float(np.mean(np.abs(pred-yte)))})
    pd.DataFrame(loco).to_csv(a.out/'tables'/'loco_predictions.csv',index=False)
    pd.DataFrame(qa_rows).to_csv(a.out/'tables'/'qa_correlations_with_loco.csv',index=False)

    coverage=[]
    ordered=cases.sort_values('p95_range_hu',ascending=True)
    for c in np.arange(0.5,1.001,0.05):
        n=max(1,int(np.floor(c*len(ordered)))); keep=ordered.iloc[:n]
        coverage.append({'coverage':float(c),'retained_n':n,'mean_full_mae':keep.full_valid_mae.mean(),'mean_skull_mae':keep.skull_200_mae.mean(),'p95_full_mae':np.percentile(keep.full_valid_mae,95)})
    cov=pd.DataFrame(coverage); cov.to_csv(a.out/'tables'/'risk_coverage.csv',index=False)

    fig,ax=plt.subplots(figsize=(8,5))
    for center,grp in cases.groupby('anonymous_center'): ax.scatter(grp.p95_range_hu,grp.full_valid_mae,label=f'Center {center}',alpha=.75)
    ax.set_xlabel('P95 cross-plane range (HU)'); ax.set_ylabel('Paired full-mask MAE (HU)'); ax.legend(); fig.tight_layout(); fig.savefig(a.out/'figures'/'qa_scatter.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot(cov.coverage,cov.mean_full_mae,marker='o',label='Full-mask MAE'); ax.plot(cov.coverage,cov.mean_skull_mae,marker='s',label='Skull MAE'); ax.set_xlabel('Automatic coverage'); ax.set_ylabel('Retained mean MAE (HU)'); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(a.out/'figures'/'risk_coverage.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); data=[long[long.prediction==p].groupby('case_id').full_valid_mae.first() for p in ['plane0','plane1','plane2','median_fusion']]; ax.boxplot(data,tick_labels=['Plane 0','Plane 1','Plane 2','Median']); ax.set_ylabel('Full-mask MAE (HU)'); fig.tight_layout(); fig.savefig(a.out/'figures'/'fusion_ablation.png',dpi=220); plt.close(fig)

    summary={'n_cases':int(cases.case_id.nunique()),'centers':cases.groupby('anonymous_center').case_id.nunique().to_dict(),'hu_gates_passed':int(cases.hu_gate.sum()),'key_metrics':{m:bootstrap_mean(cases[m]) for m in key},'qa':{'p95_range_vs_full_mae':safe_spearman(cases.p95_range_hu,cases.full_valid_mae),'p95_range_vs_skull_mae':safe_spearman(cases.p95_range_hu,cases.skull_200_mae)}}
    (a.out/'full_analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=True),encoding='utf-8')
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__': main()
