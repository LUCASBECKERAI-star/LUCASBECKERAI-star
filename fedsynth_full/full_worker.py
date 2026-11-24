#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'fedsynth_meta' / 'scripts'))
from pilot_official import MODEL_SHA256, ANALYSIS_VERSION, file_sha256, inspect_model, process_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--keep-first-figure', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summaries').mkdir(exist_ok=True)
    actual = file_sha256(args.model)
    if actual != MODEL_SHA256:
        raise SystemExit(f'Model hash mismatch: {actual}')
    model_info, session = inspect_model(args.model)
    model_info.update({'analysis_version': ANALYSIS_VERSION, 'model_sha256': actual, 'providers': session.get_providers()})
    (args.out/'summaries'/'environment_and_model.json').write_text(json.dumps(model_info, indent=2), encoding='utf-8')
    case_dirs = sorted(p for p in args.data.iterdir() if p.is_dir() and (p/'mr.nii.gz').exists())
    if not case_dirs:
        raise SystemExit('No cases found')
    rows=[]
    summaries=[]
    for index, case_dir in enumerate(case_dirs):
        summary, case_rows = process_case(case_dir, session, args.out)
        summaries.append(summary)
        rows.extend(case_rows)
        shutil.rmtree(args.out/'volumes'/case_dir.name, ignore_errors=True)
        if not args.keep_first_figure or index > 0:
            (args.out/'figures'/f'{case_dir.name}_pilot_panel.png').unlink(missing_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out/'summaries'/'chunk_metrics_long.csv', index=False)
    table[table.prediction=='median_fusion'].to_csv(args.out/'summaries'/'chunk_metrics_fused.csv', index=False)
    with (args.out/'summaries'/'chunk_case_summaries.jsonl').open('w', encoding='utf-8') as f:
        for item in summaries:
            f.write(json.dumps(item, allow_nan=True)+'\n')
    print(json.dumps({'n_cases':len(case_dirs),'case_ids':[p.name for p in case_dirs],'analysis_version':ANALYSIS_VERSION}, indent=2))

if __name__=='__main__':
    main()
