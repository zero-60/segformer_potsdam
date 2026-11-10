#!/usr/bin/env python3
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse
from pathlib import Path
import torch
from train_potsdam_cmx import parser,resolve_args,make_spec
from cmx_model import PotsdamCMX
from cmx_evaluate import evaluate

def main():
    p=parser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--save-predictions',action='store_true')
    a=resolve_args(p.parse_args());w=a.weights if a.weights.is_absolute() else a.project_root/a.weights
    ck=torch.load(w,map_location='cpu',weights_only=False);s=make_spec(a)
    if ck['protocol_sha256']!=s.hash:raise ValueError('Checkpoint and protocol differ: select original training protocol')
    torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1)
    s.prepare(a.split)
    cfg=ck['config'];device=torch.device(a.device)
    model=PotsdamCMX(a.pretrained,variant=cfg['variant']).to(device)
    model.load_state_dict(ck['model'],strict=True)
    # Dedicated evaluation output avoids writing into a training directory.
    out=a.project_root/f'outputs/evaluation/{w.parent.name}_{a.split}_{w.stem}'
    if any(x=='--out' or x.startswith('--out=') for x in __import__('sys').argv):out=a.out
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new --out for evaluation')
    r=evaluate(model,s,a.split,device,out,not a.no_amp,a.save_predictions)
    print(f'[DONE] split={a.split} mIoU={r["mIoU"]:.4f} OA={r["OA"]:.4f} vegetation_MAE_pp={r["vegetation_MAE_pp"]:.3f} out={out}')
if __name__=='__main__':main()
