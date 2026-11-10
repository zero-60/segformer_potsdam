"""CPU synthetic integration check; no imagery or pretrained weights required."""
import tempfile
from pathlib import Path
import torch
from transformers import SegformerConfig
from isd_base_model import PotsdamModel
from isd_raf_model import PotsdamRAF
from train_potsdam_isd_raf import seed,lovasz

def main():
    torch.set_num_threads(2)
    cfg=SegformerConfig(num_labels=6,hidden_sizes=[8,16,32,64],depths=[1,1,1,1],
        num_attention_heads=[1,2,4,8],sr_ratios=[8,4,2,1],decoder_hidden_size=16,drop_path_rate=0.)
    seed(3);old=PotsdamModel(config=cfg,frequency=False)
    seed(3);baseline=PotsdamRAF(config=cfg,use_raf=False)
    seed(3);raf=PotsdamRAF(config=cfg,use_raf=True)
    for k,v in old.state_dict().items():
        assert torch.equal(v,baseline.state_dict()[k]),k
        assert torch.equal(v,raf.state_dict()[k]),k
    rgb=torch.randn(1,3,64,64);nir=torch.randn(1,1,64,64);lab=torch.randint(0,6,(1,64,64))
    old.eval();baseline.eval()
    with torch.no_grad():assert torch.equal(old(rgb,nir),baseline(rgb,nir))
    opt=torch.optim.AdamW(raf.parameters(),lr=.001)
    for step in range(2):
        raf.train();opt.zero_grad();raw=raf(rgb,nir,raw=True)
        logits=torch.nn.functional.interpolate(raw,(64,64),mode='bilinear',align_corners=False)
        loss=torch.nn.functional.cross_entropy(logits,lab)+.5*lovasz(raw,lab)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in raf.parameters())
        assert raf.raf_output.weight.grad.abs().sum()>0
        assert raf.shallow_projection[0].weight.grad.abs().sum()>0
        assert raf.deep_projection[0].weight.grad.abs().sum()>0
        assert raf.raf.g1.grad.abs().sum()>0
        if step==1:assert raf.raf.spatial_mlp[0].weight.grad.abs().sum()>0
        opt.step()
    raf.eval()
    with torch.no_grad():
        pred,details=raf(rgb,nir,return_diagnostics=True)
        assert pred.shape==(1,6,64,64) and details['relative_residual_l1']>0
        paired=raf(rgb.repeat(2,1,1,1),nir.repeat(2,1,1,1))
        assert torch.allclose(pred,paired[:1],atol=1e-5,rtol=1e-4)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state.pt';torch.save(raf.state_dict(),path)
        restored=PotsdamRAF(config=cfg,use_raf=True);restored.load_state_dict(torch.load(path,weights_only=True));restored.eval()
        with torch.no_grad():assert torch.equal(pred,restored(rgb,nir))
    print('PASS: identical shared initialization/baseline logits; batch1 GN; output shape; RAF branch and relation-MLP gradients; batch inference consistency; checkpoint reload')
    print('DIAGNOSTICS',details)
if __name__=='__main__':main()
