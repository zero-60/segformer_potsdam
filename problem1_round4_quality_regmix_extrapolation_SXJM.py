#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 - Round 4
Quality-augmented RegMix + scale-robust evaluation + A12-A15 extrapolation audit.

This script is deliberately based on the Round-3 conclusion:
- GradientBoosting is the strongest M0 family on A4+A5 CV.
- 1M absolute prediction is strong.
- 60M/1B preserve mixture ranking much better than absolute Loss level.
Therefore:
(1) Spearman/ranking is the PRIMARY cross-scale criterion.
(2) Raw RMSE/R2 across model sizes are SECONDARY because model-size scaling shifts
    the entire Loss level.
(3) Q is tested by strict ablation against the same M0 protocol.
(4) A12-A15 are used only for extrapolation robustness, never for fitting/tuning.

Expected local project:
<project>/
  real_attachments/A_data_value/
    regmix_tables/
    domain_mapping_guide.csv
  round2_outputs/
    tables/04_A1_domain_quality.csv
    tables/06_extended_quality_validation.csv
  round4_outputs/

Run:
  python problem1_round4_quality_regmix_extrapolation_SXJM.py

Dependencies:
  pip install numpy pandas scipy scikit-learn matplotlib tqdm
"""
from __future__ import annotations
import argparse, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

def args():
    p=argparse.ArgumentParser()
    here=Path(__file__).resolve().parent
    p.add_argument("--data-dir",type=Path,default=here/"real_attachments"/"A_data_value")
    p.add_argument("--round2-dir",type=Path,default=here/"round2_outputs")
    p.add_argument("--out-dir",type=Path,default=here/"round4_outputs")
    p.add_argument("--seed",type=int,default=20260923)
    p.add_argument("--cv-folds",type=int,default=5)
    p.add_argument("--eps",type=float,default=1e-6)
    p.add_argument("--bootstrap",type=int,default=1000)
    return p.parse_args()

def norm(s):
    s=str(s).lower()
    s=s.replace("train_the_pile_","").replace("metric/the_pile_","")
    s=s.replace("_val_loss","").replace("_backgrounds","").replace("_pg_19","")
    s=s.replace("_en","").replace("_irc","").replace("_emails","")
    s=re.sub(r"[^a-z0-9]+","_",s).strip("_")
    aliases={"free_law":"freelaw","dm_math":"dm_mathematics","pubmed_abstract":"pubmed_abstracts",
             "pilecc":"pile_cc","gutenberg_pg":"gutenberg","gutenberg_pg_19":"gutenberg",
             "uspto_background":"uspto","wikipedia_en":"wikipedia","ubuntu_irc":"ubuntu",
             "enron_emails":"enron"}
    return aliases.get(s,s)

def req(p):
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def read_pair(mixp,lossp):
    m=pd.read_csv(mixp); y=pd.read_csv(lossp)
    key="index" if "index" in m.columns and "index" in y.columns else None
    mc=[c for c in m.columns if c!=key]
    yc=[c for c in y.columns if c!=key]
    if key:
        d=m.merge(y,on=key,validate="one_to_one")
    else:
        if len(m)!=len(y): raise ValueError("Mixture/loss row counts differ.")
        d=pd.concat([m.reset_index(drop=True),y.reset_index(drop=True)],axis=1)
    P=d[mc].astype(float)
    if P.sum(axis=1).median()>10:P=P/100
    P=P.div(P.sum(axis=1),axis=0)
    d.loc[:,mc]=P
    return d,mc,yc

def helmert(D):
    H=np.zeros((D,D-1))
    for j in range(1,D):
        H[:j,j-1]=1/np.sqrt(j*(j+1)); H[j,j-1]=-j/np.sqrt(j*(j+1))
    return H

def ilr(P,eps,H):
    X=np.asarray(P,float)+eps
    X=X/X.sum(1,keepdims=True)
    L=np.log(X); clr=L-L.mean(1,keepdims=True)
    return clr@H

def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    return {"RMSE":np.sqrt(mean_squared_error(y,p)),
            "MAE":mean_absolute_error(y,p),"R2":r2_score(y,p),
            "Spearman":stats.spearmanr(y,p).statistic}

def gb(seed):
    # Freeze a conservative GradientBoosting specification for fair ablation.
    return GradientBoostingRegressor(n_estimators=250,learning_rate=.03,max_depth=2,
                                     min_samples_leaf=3,subsample=.8,random_state=seed)

def find_col(df,keys):
    for c in df.columns:
        n=norm(c)
        if any(k in n for k in keys): return c
    return None

def load_quality(round2):
    q=pd.read_csv(req(round2/"tables"/"04_A1_domain_quality.csv"))
    dcol=find_col(q,["source_domain","domain"])
    qcol=find_col(q,["q_mean"])
    if dcol is None or qcol is None:
        raise ValueError("Cannot detect domain/Q_mean in Round2 04_A1_domain_quality.csv")
    Q={norm(d):float(v) for d,v in zip(q[dcol],q[qcol]) if pd.notna(d) and pd.notna(v)}
    # Prefer full A2/A3 extended means for arxiv/github when available.
    ep=round2/"tables"/"06_extended_quality_validation.csv"
    if ep.exists():
        e=pd.read_csv(ep)
        dc=find_col(e,["domain"]); qc=find_col(e,["q_extended_mean"])
        if dc and qc:
            for d,v in zip(e[dc],e[qc]):
                if pd.notna(v): Q[norm(d)]=float(v)
    return Q,q

def parse_mapping(mapping_path,train_domains,Q):
    """
    Use A16 when machine-readable. If its column names vary, infer columns by overlap.
    No invented domain quality: unmatched domains shrink to observed global mean.
    """
    mp=pd.read_csv(req(mapping_path))
    # score each text column by overlap with normalized train domains / Q domains
    textcols=list(mp.columns)
    tnames={norm(c) for c in train_domains}
    qnames=set(Q)
    scores=[]
    for c in textcols:
        vals={norm(v) for v in mp[c].dropna().astype(str)}
        scores.append((c,len(vals&tnames),len(vals&qnames)))
    traincol=max(scores,key=lambda x:x[1])[0]
    qcol=max(scores,key=lambda x:x[2])[0]
    if traincol==qcol:
        # try second best Q overlap
        alt=sorted(scores,key=lambda x:x[2],reverse=True)
        qcol=next((x[0] for x in alt if x[0]!=traincol),qcol)
    confcol=find_col(mp,["confidence","mapping_type","match_type","relation","quality"])
    global_q=float(np.mean(list(Q.values())))
    rows=[]; q17={}
    for td in train_domains:
        tn=norm(td)
        hit=mp[mp[traincol].astype(str).map(norm)==tn]
        source=None; conf="unmatched"; rawq=np.nan
        if len(hit):
            source=norm(hit.iloc[0][qcol])
            rawq=Q.get(source,np.nan)
            if confcol and confcol in hit.columns: conf=str(hit.iloc[0][confcol])
        # direct name fallback
        if not np.isfinite(rawq) and tn in Q:
            source=tn;rawq=Q[tn];conf="direct_name"
        cs=norm(conf)
        if "direct" in cs and "near" not in cs: r=1.0
        elif "near" in cs: r=.75
        elif "infer" in cs: r=.50
        elif np.isfinite(rawq): r=.60
        else:r=0.0
        final=global_q if not np.isfinite(rawq) else r*rawq+(1-r)*global_q
        q17[td]=final
        rows.append({"train_domain":td,"normalized_train_domain":tn,
                     "mapped_quality_domain":source,"mapping_confidence":conf,
                     "shrinkage_r":r,"raw_Q":rawq,"mapped_Q":final})
    return q17,pd.DataFrame(rows),mp

def features(P,mc,H,eps,q17,mode):
    Z=ilr(P[mc],eps,H)
    if mode=="M0": return Z
    qvec=np.array([q17[c] for c in mc],float)
    Qmix=P[mc].to_numpy(float)@qvec
    if mode=="M1": return np.column_stack([Z,Qmix])
    # M2: limited interaction, not 16 unrestricted products; include Qmix*ILR.
    return np.column_stack([Z,Qmix,Z*Qmix[:,None]])

def cv_ablation(train,mc,yc,H,eps,q17,folds,seed):
    cv=KFold(folds,shuffle=True,random_state=seed)
    rows=[]
    for mode in ["M0","M1","M2"]:
        X=features(train,mc,H,eps,q17,mode)
        for target in tqdm(yc,desc=f"CV {mode}",unit="target"):
            y=train[target].to_numpy(float)
            pred=cross_val_predict(gb(seed),X,y,cv=cv,n_jobs=-1)
            rows.append({"model":mode,"target":target,**metric(y,pred)})
        # macro from independently cross-validated target predictions
        preds=[]
        for target in yc:
            y=train[target].to_numpy(float)
            preds.append(cross_val_predict(gb(seed),X,y,cv=cv,n_jobs=-1))
        pm=np.column_stack(preds).mean(1); ym=train[yc].mean(axis=1).to_numpy()
        rows.append({"model":mode,"target":"macro_loss",**metric(ym,pm)})
    return pd.DataFrame(rows)

def fit_predict(train,test,mc,yc,H,eps,q17,mode,seed):
    Xtr=features(train,mc,H,eps,q17,mode); Xte=features(test,mc,H,eps,q17,mode)
    rows=[]; preds=[]
    for target in yc:
        model=gb(seed); model.fit(Xtr,train[target])
        p=model.predict(Xte);preds.append(p)
        rows.append({"model":mode,"target":target,**metric(test[target],p)})
    pm=np.column_stack(preds).mean(1); ym=test[yc].mean(axis=1).to_numpy()
    rows.append({"model":mode,"target":"macro_loss",**metric(ym,pm)})
    return pd.DataFrame(rows),ym,pm

def bootstrap_spearman_delta(y,p0,p1,B,rng):
    y=np.asarray(y);p0=np.asarray(p0);p1=np.asarray(p1);n=len(y)
    vals=[]
    for _ in tqdm(range(B),desc="Bootstrap ΔSpearman",leave=False):
        ix=rng.integers(0,n,n)
        if len(np.unique(y[ix]))<3: continue
        vals.append(stats.spearmanr(y[ix],p1[ix]).statistic-
                    stats.spearmanr(y[ix],p0[ix]).statistic)
    return np.quantile(vals,[.025,.5,.975]) if vals else [np.nan]*3

def main():
    a=args(); root=a.data_dir.resolve(); r2=a.round2_dir.resolve(); out=a.out_dir.resolve()
    tab=out/"tables";fig=out/"figures"
    for p in [tab,fig]:p.mkdir(parents=True,exist_ok=True)
    rt=req(root/"regmix_tables")

    files={
      "train":("train_mixture_1m.csv","train_pile_loss_1m.csv"),
      "1M":("test_mixture_1m.csv","test_pile_loss_1m.csv"),
      "60M":("test_mixture_60m.csv","test_pile_loss_60m.csv"),
      "1B":("test_mixture_1B.csv","test_pile_loss_1B.csv"),
      "10B_est":("est_mixture_10b.csv","est_pile_loss_10b.csv"),
      "70B_est":("est_mixture_70b.csv","est_pile_loss_70b.csv")}
    data={}
    for k,(m,l) in files.items():
        data[k],mc,yc=read_pair(req(rt/m),req(rt/l))
    H=helmert(len(mc))

    Q,_=load_quality(r2)
    q17,maptab,_=parse_mapping(root/"domain_mapping_guide.csv",mc,Q)
    maptab.to_csv(tab/"01_quality_to_regmix_mapping.csv",index=False)
    pd.DataFrame({"train_domain":mc,"mapped_Q":[q17[c] for c in mc]}).to_csv(
        tab/"02_17domain_quality.csv",index=False)

    print("Running strict TRAIN-CV quality ablation ...")
    cvres=cv_ablation(data["train"],mc,yc,H,a.eps,q17,a.cv_folds,a.seed)
    cvres.to_csv(tab/"03_train_CV_quality_ablation.csv",index=False)

    # External tests + extrapolation tables.
    rows=[]; boot=[]; rng=np.random.default_rng(a.seed)
    for scale in ["1M","60M","1B","10B_est","70B_est"]:
        cache={}
        for mode in ["M0","M1","M2"]:
            me,y,p=fit_predict(data["train"],data[scale],mc,yc,H,a.eps,q17,mode,a.seed)
            me.insert(0,"scale",scale);rows.append(me);cache[mode]=(y,p)
        # Compare quality models against M0 on macro ranking.
        for mode in ["M1","M2"]:
            ci=bootstrap_spearman_delta(cache["M0"][0],cache["M0"][1],
                                       cache[mode][1],a.bootstrap,rng)
            boot.append({"scale":scale,"comparison":f"{mode}-M0",
                         "delta_spearman_CI_low":ci[0],
                         "delta_spearman_median":ci[1],
                         "delta_spearman_CI_high":ci[2]})
    ext=pd.concat(rows,ignore_index=True)
    ext.to_csv(tab/"04_external_quality_ablation.csv",index=False)
    boot=pd.DataFrame(boot);boot.to_csv(tab/"05_bootstrap_delta_spearman.csv",index=False)

    macro=ext[ext.target=="macro_loss"].copy()
    macro.to_csv(tab/"06_macro_ablation_summary.csv",index=False)

    # Scale-shift diagnosis: observed and M0 predicted macro mean/sd.
    shift=[]
    for scale in ["1M","60M","1B","10B_est","70B_est"]:
        me,y,p=fit_predict(data["train"],data[scale],mc,yc,H,a.eps,q17,"M0",a.seed)
        shift.append({"scale":scale,"observed_mean":np.mean(y),"predicted_mean":np.mean(p),
                      "mean_bias_pred_minus_obs":np.mean(p)-np.mean(y),
                      "observed_sd":np.std(y,ddof=1),"predicted_sd":np.std(p,ddof=1),
                      "spearman":stats.spearmanr(y,p).statistic})
    pd.DataFrame(shift).to_csv(tab/"07_scale_shift_diagnostics.csv",index=False)

    # Decision rule: quality accepted only if train-CV macro Spearman improves AND
    # median real-test (1M/60M/1B) macro Spearman improves.
    cvm=cvres[cvres.target=="macro_loss"].set_index("model")
    real=macro[macro.scale.isin(["1M","60M","1B"])]
    med=real.groupby("model").Spearman.median()
    decision=[]
    for mode in ["M1","M2"]:
        dcv=cvm.loc[mode,"Spearman"]-cvm.loc["M0","Spearman"]
        dext=med.loc[mode]-med.loc["M0"]
        accept=bool(dcv>0 and dext>0)
        decision.append({"model":mode,"delta_CV_macro_spearman":dcv,
                         "delta_median_realtest_spearman":dext,
                         "accept_quality_extension":accept})
    dec=pd.DataFrame(decision);dec.to_csv(tab/"08_quality_inclusion_decision.csv",index=False)

    # Figures.
    f,ax=plt.subplots(figsize=(8,5))
    piv=macro.pivot(index="scale",columns="model",values="Spearman")
    piv.plot(kind="bar",ax=ax)
    ax.set_ylabel("Macro-Loss Spearman");ax.set_xlabel("Scale")
    ax.set_title("Quality ablation across scales")
    ax.set_ylim(-1,1);f.tight_layout();f.savefig(fig/"01_quality_ablation_spearman.png",dpi=220);plt.close(f)

    sh=pd.read_csv(tab/"07_scale_shift_diagnostics.csv")
    f,ax=plt.subplots(figsize=(8,5))
    x=np.arange(len(sh));w=.35
    ax.bar(x-w/2,sh.observed_mean,w,label="Observed")
    ax.bar(x+w/2,sh.predicted_mean,w,label="M0 predicted")
    ax.set_xticks(x);ax.set_xticklabels(sh.scale)
    ax.set_ylabel("Macro Loss level");ax.set_xlabel("Scale")
    ax.set_title("Model-size Loss-level shift")
    ax.legend();f.tight_layout();f.savefig(fig/"02_scale_loss_shift.png",dpi=220);plt.close(f)

    f,ax=plt.subplots(figsize=(9,5))
    qd=pd.DataFrame({"domain":[norm(c) for c in mc],"Q":[q17[c] for c in mc]})
    ax.bar(qd.domain,qd.Q);ax.tick_params(axis="x",rotation=90)
    ax.set_ylabel("Mapped quality Q");ax.set_xlabel("RegMix training domain")
    ax.set_title("Quality mapped to 17 RegMix domains")
    f.tight_layout();f.savefig(fig/"03_mapped_domain_quality.png",dpi=220);plt.close(f)

    # Human-readable conclusion generated from actual experiment after execution.
    accepted=dec[dec.accept_quality_extension]
    if len(accepted):
        chosen=accepted.sort_values("delta_median_realtest_spearman",ascending=False).iloc[0].model
        qcon=f"Quality information passes the predefined ablation rule; retain {chosen} as the quality-enhanced candidate."
    else:
        chosen="M0"
        qcon=("Quality information does not pass the strict ablation rule. Retain M0 as the primary predictive model; "
              "report Q as an explanatory/diagnostic variable rather than claiming independent predictive gain.")
    realtxt=macro[macro.scale.isin(["1M","60M","1B"])][
        ["scale","model","RMSE","R2","Spearman"]].to_string(index=False)
    esttxt=macro[macro.scale.isin(["10B_est","70B_est"])][
        ["scale","model","RMSE","R2","Spearman"]].to_string(index=False)
    summary=f"""ROUND 4 - QUALITY REGMIX + EXTRAPOLATION

Pre-registered interpretation:
Cross-scale ranking (Spearman) is primary because Round 3 showed a large model-size
Loss-level shift: 1M absolute fit was strong, while 60M/1B retained ranking much
better than absolute calibration.

REAL TESTS
{realtxt}

A12-A15 EXTRAPOLATION TABLES
{esttxt}

QUALITY DECISION
{dec.to_string(index=False)}

Conclusion:
{qcon}

Important:
A12-A15 are extrapolated reference tables, not real large-model observations.
Their results support or weaken extrapolation robustness only; they do not constitute
independent 10B/70B experimental validation.
"""
    (out/"round4_summary.txt").write_text(summary,encoding="utf-8")
    (out/"run_config.json").write_text(json.dumps({
        "seed":a.seed,"eps":a.eps,"cv_folds":a.cv_folds,"bootstrap":a.bootstrap,
        "quality_decision":dec.to_dict("records"),"selected_model":chosen
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n"+summary)
    print("Done ->",out)

if __name__=="__main__":
    main()
