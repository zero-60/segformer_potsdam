#!/usr/bin/env python3
"""Evaluate a full-protocol checkpoint; validation never needs pretrained files."""
import sys
from pathlib import Path
import torch
from train_potsdam_cmx_full import parser,resolve_args,make_spec,load_full_checkpoint,checkpoint_model,seed
from cmx_full_evaluate import evaluate
from cmx_full_console import disable_quick_edit
from cmx_full_runtime import heartbeat,write_json


def main():
    p=parser();p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--save-predictions',action='store_true')
    a=resolve_args(p.parse_args())
    if not a.keep_quick_edit:disable_quick_edit()
    torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1);seed(a.seed)
    weights=a.weights if a.weights.is_absolute() else a.project_root/a.weights
    ck=load_full_checkpoint(weights);spec=make_spec(a)
    if ck['protocol_sha256']!=spec.hash:raise ValueError('Checkpoint/protocol mismatch: use its original full protocol')
    explicit_out=any(v=='--out' or v.startswith('--out=') for v in sys.argv[1:])
    out=a.out if explicit_out else a.project_root/f'outputs/evaluation/{weights.parent.name}_{a.split}_{weights.stem}'
    if out.exists() and any(out.iterdir()):raise FileExistsError('Evaluation output is nonempty; choose a new --out')
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    spec.prepare(a.split)
    with heartbeat('loading checkpoint model'):model=checkpoint_model(ck).to(device)
    model.load_state_dict(ck['model'],strict=True)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'evaluation_config.json',{'weights':str(weights.resolve()),'checkpoint_epoch':ck['epoch'],
              'variant':ck['config']['variant'],'split':a.split,'protocol_sha256':spec.hash,'eval_batch_size':a.eval_batch_size,
              'no_amp':a.no_amp,'fusion_storage':a.fusion_storage})
    del ck
    result=evaluate(model,spec,a.split,device,out,not a.no_amp,a.save_predictions)
    print(f'[DONE] split={a.split} mIoU={result["mIoU"]:.6f} OA={result["OA"]:.6f} vegetation_MAE_pp={result["vegetation_MAE_pp"]:.4f} out={out}',flush=True)


if __name__=='__main__':main()
