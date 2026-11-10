"""Offline CPU tests: no real images, pretrained files or extra dependencies."""
import tempfile
from pathlib import Path
import torch
from transformers import SegformerConfig
from cmx_model import PotsdamCMX
from cmx_base_model import PotsdamModel
from cmx_modules import FeatureRectifyModule,FeatureFusionModule
from train_potsdam_cmx import seed,lovasz

def check_official_modules():
    # Load shipped reference after substituting only the initializer import.
    source=Path(__file__).resolve().parents[1]/'third_party/official_net_utils.py'
    if not source.exists():
        print('SKIP: reference equivalence (place third_party beside tools to run it)');return
    namespace={}
    exec(compile(source.read_text().replace('from timm.models.layers import trunc_normal_',
        'from torch.nn.init import trunc_normal_'),str(source),'exec'),namespace)
    x=torch.randn(1,16,8,8);y=torch.randn(1,16,8,8)
    for name,local in [('FeatureRectifyModule',FeatureRectifyModule),('FeatureFusionModule',FeatureFusionModule)]:
        kwargs={'dim':16}
        if name=='FeatureFusionModule':kwargs['num_heads']=4
        seed(4);reference=namespace[name](**kwargs).eval()
        seed(4);ported=local(**kwargs).eval()
        ported.load_state_dict(reference.state_dict())
        with torch.no_grad():a=reference(x,y);b=ported(x,y)
        if isinstance(a,tuple):
            assert all(torch.equal(u,v) for u,v in zip(a,b))
        else:assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    print('PASS: FRM/FFM FP32 outputs agree with official equations')

def main():
    torch.set_num_threads(2);check_official_modules()
    cfg=SegformerConfig(num_labels=6,hidden_sizes=[8,16,32,64],depths=[1,1,1,1],
        num_attention_heads=[1,2,4,8],sr_ratios=[8,4,2,1],decoder_hidden_size=16,drop_path_rate=0.)
    seed(3);base=PotsdamModel(config=cfg)
    models={}
    for variant in ['baseline','frm','cmx']:
        seed(3);m=PotsdamCMX(config=cfg,variant=variant)
        for k,v in base.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k
        models[variant]=m
    rgb=torch.randn(1,3,64,64);nir=torch.randn(1,1,64,64);lab=torch.randint(0,6,(1,64,64))
    base.eval();models['baseline'].eval()
    with torch.no_grad():assert torch.equal(base(rgb,nir),models['baseline'](rgb,nir))
    for variant,m in models.items():
        m.train();raw=m(rgb,nir,raw=True)
        up=torch.nn.functional.interpolate(raw,(64,64),mode='bilinear',align_corners=False)
        loss=torch.nn.functional.cross_entropy(up,lab)+.5*lovasz(raw,lab)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        if variant!='baseline':
            for frm in m.frms:
                assert sum(p.grad.abs().sum() for p in frm.parameters() if p.grad is not None)>0
        if variant=='cmx':
            for ffm in m.ffms:
                assert sum(p.grad.abs().sum() for p in ffm.parameters() if p.grad is not None)>0
            assert m.ffm_scales.grad.abs().sum()>0
        m.eval()
        with torch.no_grad():
            pred,details=m(rgb,nir,return_diagnostics=True)
            assert pred.shape==(1,6,64,64)
            batch=m(rgb.repeat(2,1,1,1),nir.repeat(2,1,1,1))
            assert torch.allclose(pred,batch[:1],atol=1e-5,rtol=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.pt';torch.save(m.state_dict(),path)
            restored=PotsdamCMX(config=cfg,variant=variant);restored.load_state_dict(torch.load(path,weights_only=True));restored.eval()
            with torch.no_grad():assert torch.equal(pred,restored(rgb,nir))
    # Stage1 token length of a real 512 crop: no NxN attention allocation.
    ffm=FeatureFusionModule(64,num_heads=1).eval()
    x=torch.randn(1,64,128,128);y=torch.randn_like(x)
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):out=ffm(x,y)
    assert out.shape==x.shape and torch.isfinite(out).all()
    print('PASS: baseline/shared initialization; three-mode forward/backward; per-stage FRM/FFM gradients; diagnostics; checkpoint round-trip; batch consistency; 16384-token mixed-precision FFM')
if __name__=='__main__':main()
