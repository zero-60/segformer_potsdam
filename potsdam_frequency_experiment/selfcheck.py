"""CPU synthetic checks, no network, data or pretrained weights required."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from transformers import SegformerConfig
from model import PotsdamModel,FrequencyResidual
from data import starts,randbelow,d4,encode
from train import seed,lovasz
from evaluate import evaluate,metrics

def main():
    torch.set_num_threads(2)
    cfg=SegformerConfig(num_labels=6,hidden_sizes=[8,16,32,64],depths=[1,1,1,1],num_attention_heads=[1,2,4,8],
         sr_ratios=[8,4,2,1],decoder_hidden_size=16,drop_path_rate=0.)
    seed(3);base=PotsdamModel(config=cfg,frequency=False)
    seed(3);freq=PotsdamModel(config=cfg,frequency=True)
    for k,v in base.state_dict().items():assert torch.equal(v,freq.state_dict()[k]),k
    r=torch.randn(1,3,64,64);n=torch.randn(1,1,64,64);lab=torch.randint(0,6,(1,64,64))
    for model in [base,freq]:
        logits=model(r,n,raw=True)
        loss=torch.nn.functional.cross_entropy(torch.nn.functional.interpolate(logits,size=(64,64),mode='bilinear',align_corners=False),lab)+.5*lovasz(logits,lab)
        loss.backward();assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for f in freq.frequency:
        assert f.out.weight.grad.abs().sum()>0
        assert f.gate.weight.grad.abs().sum()>0
        assert f.low.weight.grad.abs().sum()>0
    f=FrequencyResidual(8);x=torch.randn(2,8,17,19);lo,hi=f.decompose(x)
    assert torch.allclose(lo+hi,x,atol=1e-6)
    c=torch.ones_like(x);lo,hi=f.decompose(c);assert hi.abs().max()<1e-6
    assert len(starts(6000,512,384))==16
    for k in range(8):assert np.array_equal(d4(np.arange(9).reshape(3,3),k),d4(np.arange(9).reshape(3,3),k))
    colors=np.array([[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]],np.uint8)
    try:encode(np.zeros((2,2,3),np.uint8),colors);raise AssertionError('Unknown color accepted')
    except ValueError:pass
    class FakeSpec:
        hash='synthetic';split={'val':['fake']}
        p={'validation_test_inference':{'tile_size':[64,64],'crop_size':[16,16],'stride':[12,12],'overlap_fusion':'uniform mean-logit accumulation'}}
        def image(self,tid):return np.broadcast_to(np.array([0,0,255,1],np.uint8),(64,64,4))
        def gt(self,tid):return np.broadcast_to(np.array([0,0,255],np.uint8),(64,64,3))
        def normalize(self,a):return torch.zeros(3,*a.shape[:2]),torch.zeros(1,*a.shape[:2])
    s=FakeSpec();s.colors=colors
    class Constant(torch.nn.Module):
        def forward(self,r,n):
            z=torch.zeros(r.shape[0],6,*r.shape[-2:]);z[:,1]=3;return z
    with tempfile.TemporaryDirectory() as out:
        result=evaluate(Constant(),s,'val',torch.device('cpu'),out,False,True)
        assert result['OA']==1 and result['mIoU']==1
        assert result['confusion_matrix'][1][1]==4096
        assert result['vegetation_MAE_pp']==0
    # End-of-epoch training state serialization/load.
    with tempfile.TemporaryDirectory() as out:
        path=Path(out)/'weights.pt';torch.save(freq.state_dict(),path)
        other=PotsdamModel(config=cfg,frequency=True);other.load_state_dict(torch.load(path,weights_only=True))
        freq.eval();other.eval()
        with torch.no_grad():assert torch.equal(freq(r,n),other(r,n))
    print('PASS: shared baseline initialization; dual-encoder forward/backward; frequency gate/filter gradients; FFT reconstruction/DC; sliding fusion/metrics/exports; checkpoint reload')
if __name__=='__main__':main()
