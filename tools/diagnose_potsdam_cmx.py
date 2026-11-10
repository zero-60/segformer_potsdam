"""Inspect a trained model's gate/residual activity on one fixed training crop."""
import json
from pathlib import Path
import torch
from train_potsdam_cmx import parser,resolve_args,make_spec
from cmx_model import PotsdamCMX
from cmx_data import FastTrainSet

def main():
    p=parser();p.add_argument('--weights',type=Path,required=True)
    a=resolve_args(p.parse_args());torch.set_num_threads(a.cpu_threads)
    path=a.weights if a.weights.is_absolute() else a.project_root/a.weights
    ck=torch.load(path,map_location='cpu',weights_only=False)
    spec=make_spec(a)
    if ck['protocol_sha256']!=spec.hash:raise ValueError('Protocol differs from checkpoint')
    spec.prepare('train');spec.preload('train')
    model=PotsdamCMX(a.pretrained,variant=ck['config']['variant']).to(a.device)
    model.load_state_dict(ck['model']);model.eval()
    rgb,nir,_=FastTrainSet(spec)[0]
    with torch.inference_mode():
        _,rows=model(rgb[None].to(a.device),nir[None].to(a.device),return_diagnostics=True)
    result={'variant':model.variant,'checkpoint_epoch':ck['epoch'],'tile':spec.split['train'][0],
            'sampling_epoch':0,'sample_slot':0,'stages':rows,
            'note':'single training crop in eval mode; activity diagnostics, not proof of accuracy gain'}
    out=path.parent/'cmx_diagnostics.json';out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2));print('[SAVED]',out)
if __name__=='__main__':main()
