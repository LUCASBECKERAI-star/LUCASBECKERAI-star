#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled model-input stress tests for the FedSynthCT Li ONNX workflow.

Perturbations are applied after standardized N4/crop/area-resize/min-max
preprocessing. They are controlled technical stressors, not substitutes for
real scanner-domain external validation.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'fedsynth_meta'/'scripts'))
import pilot_official as po


def read_case(case_dir:Path):
    base=po.pa._pilot
    mr=sitk.ReadImage(str(case_dir/'mr.nii.gz'))
    ct=sitk.ReadImage(str(case_dir/'ct.nii.gz'))
    mask=sitk.ReadImage(str(case_dir/'mask.nii.gz'))
    if not base.same_geometry(ct,mr): ct=base.resample_to_reference(ct,mr,False,-1024.0)
    if not base.same_geometry(mask,mr): mask=base.resample_to_reference(mask,mr,True,0.0)
    corrected=base.n4_correct(mr,mask)
    mra=sitk.GetArrayFromImage(corrected).astype(np.float32)
    cta=sitk.GetArrayFromImage(ct).astype(np.float32)
    ma=sitk.GetArrayFromImage(mask).astype(np.float32)
    tfm=base.make_transform(tuple(int(x) for x in mra.shape))
    cube=base.normalize_mri(base.to_cube(mra,tfm,1,0.0))
    ctc=base.to_cube(cta,tfm,1,-1024.0).astype(np.float32)
    mask_float=base.to_cube(ma,tfm,0,0.0).astype(np.float32)
    return cube,ctc,mask_float


def conditions(base:np.ndarray,mask:np.ndarray,case_id:str):
    seed=int(hashlib.sha256(case_id.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    valid=mask!=0
    out={'baseline':base.copy()}
    for sigma in (0.03,0.06):
        arr=base.copy()
        arr[valid]=np.clip(arr[valid]+rng.normal(0,sigma,int(valid.sum())).astype(np.float32),0,1)
        out[f'noise_{sigma:.2f}']=arr
    blur=ndimage.gaussian_filter(base,sigma=1.0,mode='nearest').astype(np.float32)
    blur[~valid]=0
    out['blur_sigma_1.0']=blur
    small=ndimage.zoom(base,zoom=(0.5,0.5,0.5),order=1,mode='nearest',prefilter=False)
    low=ndimage.zoom(small,zoom=(2,2,2),order=1,mode='nearest',prefilter=False).astype(np.float32)[:256,:256,:256]
    low[~valid]=0
    out['downup_128']=low
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if po.file_sha256(args.model)!=po.MODEL_SHA256: raise SystemExit('Model hash mismatch')
    _,session=po.inspect_model(args.model)
    rows=[]
    for case_dir in sorted(p for p in args.data.iterdir() if p.is_dir() and (p/'mr.nii.gz').exists()):
        base,ct,mask_float=read_case(case_dir)
        valid=(mask_float!=0)&np.isfinite(ct)
        skull=valid&(ct>=200)
        for condition,mri in conditions(base,mask_float,case_dir.name).items():
            planes,timing=po.infer_planes_official(session,mri,mask_float)
            stack=np.stack([planes['plane0'],planes['plane1'],planes['plane2']])
            fused=np.median(stack,axis=0).astype(np.float32)
            fused[mask_float==0]=-1024
            disagreement=np.ptp(stack,axis=0)
            err=np.abs(fused-ct)
            r,p,n=po.pa._pilot.qa_correlation(disagreement,err,valid)
            rows.append({
                'case_id':case_dir.name,
                'anonymous_center':case_dir.name[2],
                'condition':condition,
                'full_valid_mae':float(np.mean(err[valid])),
                'soft_tissue_mae':float(np.mean(err[valid&(ct>=-200)&(ct<=200)])),
                'skull_200_mae':float(np.mean(err[skull])),
                'bone_dice_200':po.pa._pilot.dice(ct>=200,fused>=200),
                'p95_crossplane_range':float(np.percentile(disagreement[valid],95)),
                'mean_crossplane_range':float(np.mean(disagreement[valid])),
                'voxel_rho_range_error':r,
                'voxel_rho_p':p,
                'voxel_rho_n':n,
                'inference_seconds':float(sum(timing.values())),
                'analysis_version':po.ANALYSIS_VERSION,
            })
    df=pd.DataFrame(rows)
    df.to_csv(args.out/'stress_case_condition.csv',index=False)
    baseline=df[df.condition=='baseline'][['case_id','full_valid_mae','soft_tissue_mae','skull_200_mae','p95_crossplane_range']].set_index('case_id')
    delta=df.join(baseline,on='case_id',rsuffix='_baseline')
    for col in ['full_valid_mae','soft_tissue_mae','skull_200_mae','p95_crossplane_range']:
        delta[f'delta_{col}']=delta[col]-delta[f'{col}_baseline']
    delta.to_csv(args.out/'stress_case_condition_with_delta.csv',index=False)
    summary=delta.groupby('condition').agg(
        n=('case_id','count'),
        mean_full_mae=('full_valid_mae','mean'),
        mean_delta_full_mae=('delta_full_valid_mae','mean'),
        mean_soft_mae=('soft_tissue_mae','mean'),
        mean_delta_soft_mae=('delta_soft_tissue_mae','mean'),
        mean_skull_mae=('skull_200_mae','mean'),
        mean_delta_skull_mae=('delta_skull_200_mae','mean'),
        mean_p95_range=('p95_crossplane_range','mean'),
        mean_delta_p95_range=('delta_p95_crossplane_range','mean'),
        mean_voxel_rho=('voxel_rho_range_error','mean'),
    ).reset_index()
    summary.to_csv(args.out/'stress_summary.csv',index=False)
    (args.out/'stress_summary.json').write_text(json.dumps(summary.to_dict(orient='records'),indent=2),encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
