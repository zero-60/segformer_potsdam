#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 - Round 5 FINAL ANALYSIS
Corrected A16 mapping audit + quality re-ablation + domain main/interaction effects
+ support-constrained candidate mixture search.

Why this round exists
---------------------
Round 4 found a small real-test gain from Q, but its mapping table showed that only
6/17 RegMix domains received non-default Q and the exported "mapping_confidence"
field was clearly not parsed reliably (e.g. values such as "arxiv"). Therefore the
Round-4 M1 conclusion is treated as PROVISIONAL.

This script first reconstructs A16 mapping by ROW CONTENT rather than guessed column
names. It then reruns M0/M1/M2. Only after this audit does it estimate domain effects
and search candidate mixtures.

Important interpretation
------------------------
- Real tests: 1M / 60M / 1B.
- A12-A15 (10B/70B) are extrapolated reference tables only.
- Round 4 showed negative rank transfer to 10B/70B. Therefore this script DOES NOT
  claim that a 1M-trained optimum is optimal at 10B/70B.
- Candidate mixtures are support-constrained recommendations inside the observed
  training-mixture region, not unconstrained simplex optima.

Run:
    python problem1_round5_final_domain_effects_SXJM.py

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
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")

def parse_args():
    p=argparse.ArgumentParser()
    here=Path(__file__).resolve().parent
    p.add_argument("--data-dir",type=Path,default=here/"real_attachments"/"A_data_value")
    p.add_argument("--round2-dir",type=Path,default=here/"round2_outputs")
    p.add_argument("--out-dir",type=Path,default=here/"round5_outputs")
    p.add_argument("--seed",type=int,default=20260923)
    p.add_argument("--eps",type=float,default=1e-6)
    p.add_argument("--cv-folds",type=int,default=5)
    p.add_argument("--n-candidates",type=int,default=50000)
    p.add_argument("--top-k",type=int,default=20)
    p.add_argument("--perturb",type=float,default=.01,
                   help="1 percentage-point simplex perturbation for effects.")
    return p.parse_args()

def norm(x):
    s=str(x).strip().lower()
    for a in ["train_the_pile_","metric/the_pile_"]:
        s=s.replace(a,"")
    for a in ["_val_loss","_backgrounds","_pg_19","_emails","_irc","_en"]:
        s=s.replace(a,"")
    s=re.sub(r"[^a-z0-9]+","_",s).strip("_")
    alias={"free_law":"freelaw","dm_math":"dm_mathematics",
           "pubmed_abstract":"pubmed_abstracts","pilecc":"pile_cc",
           "common_crawl":"commoncrawl","books":"book"}
    return alias.get(s,s)

def req(p):
    if not p.exists(): raise FileNotFoundError(str(p))
    return p

def read_pair(mp,lp):
    m=pd.read_csv(mp);y=pd.read_csv(lp)
    key="index" if "index" in m.columns and "index" in y.columns else None
    mc=[c for c in m.columns if c!=key];yc=[c for c in y.columns if c!=key]
    if key:d=m.merge(y,on=key,validate="one_to_one")
    else:
        if len(m)!=len(y):raise ValueError("row mismatch")
        d=pd.concat([m.reset_index(drop=True),y.reset_index(drop=True)],axis=1)
    P=d[mc].astype(float)
    if P.sum(1).median()>10:P/=100
    P=P.div(P.sum(1),axis=0);d.loc[:,mc]=P
    return d,mc,yc

def helmert(D):
    H=np.zeros((D,D-1))
    for j in range(1,D):
        H[:j,j-1]=1/np.sqrt(j*(j+1));H[j,j-1]=-j/np.sqrt(j*(j+1))
    return H

def ilr(P,eps,H):
    X=np.asarray(P,float)+eps;X/=X.sum(1,keepdims=True)
    L=np.log(X);return (L-L.mean(1,keepdims=True))@H

def gb(seed):
    return GradientBoostingRegressor(n_estimators=250,learning_rate=.03,max_depth=2,
                                     min_samples_leaf=3,subsample=.8,random_state=seed)

def spearman(y,p): return float(stats.spearmanr(y,p).statistic)

def load_Q(r2):
    q=pd.read_csv(req(r2/"tables"/"04_A1_domain_quality.csv"))
    dc=next(c for c in q if "domain" in c.lower())
    qc=next(c for c in q if "q_mean" in c.lower())
    Q={norm(d):float(v) for d,v in zip(q[dc],q[qc]) if pd.notna(v)}
    ext=r2/"tables"/"06_extended_quality_validation.csv"
    if ext.exists():
        e=pd.read_csv(ext)
        dcs=[c for c in e if "domain" in c.lower()]
        qcs=[c for c in e if "extended" in c.lower() and "mean" in c.lower()]
        if dcs and qcs:
            for d,v in zip(e[dcs[0]],e[qcs[0]]):
                if pd.notna(v):Q[norm(d)]=float(v)
    return Q

def confidence_from_row(row):
    txt=" ".join(norm(v) for v in row if pd.notna(v))
    if "near_direct" in txt or ("near" in txt and "direct" in txt):return "near_direct",.75
    if "direct" in txt:return "direct",1.
    if "infer" in txt:return "inferred",.50
    return "unspecified",.60

def robust_A16_mapping(path,train_domains,Q):
    """
    Robust mapping:
    For each A16 row, search ALL cells for a RegMix-domain token and ALL OTHER cells
    for an available quality-domain token. This is orientation/column-name invariant.
    """
    mp=pd.read_csv(req(path))
    tnorm={norm(c):c for c in train_domains}
    qnames=set(Q)
    found={}
    audit=[]
    for ridx,row in mp.iterrows():
        vals=[norm(v) for v in row.tolist() if pd.notna(v)]
        ts=[v for v in vals if v in tnorm]
        qs=[v for v in vals if v in qnames]
        conf,r=confidence_from_row(row.tolist())
        for t in ts:
            # Prefer Q token different from t; direct same-name allowed.
            cand=[q for q in qs if q!=t] or ([t] if t in qs else [])
            if cand:
                q=cand[0]
                found[t]=(q,conf,r,ridx)
    globalQ=float(np.mean(list(Q.values())))
    rows=[];q17={}
    for td in train_domains:
        t=norm(td)
        if t in found:
            q,conf,r,ridx=found[t];raw=Q[q]
        elif t in Q:
            q=t;conf="direct_name";r=1.;ridx=np.nan;raw=Q[t]
        else:
            q=None;conf="unmatched";r=0.;ridx=np.nan;raw=np.nan
        mapped=globalQ if not np.isfinite(raw) else r*raw+(1-r)*globalQ
        q17[td]=mapped
        rows.append({"train_domain":td,"train_norm":t,"quality_domain":q,
                     "confidence":conf,"r":r,"A16_row":ridx,
                     "raw_Q":raw,"mapped_Q":mapped,
                     "is_informative":bool(r>0 and np.isfinite(raw))})
    return q17,pd.DataFrame(rows),mp

def feats(P,mc,H,eps,q17,mode):
    Z=ilr(P[mc],eps,H)
    if mode=="M0":return Z
    q=np.array([q17[c] for c in mc]);qm=P[mc].to_numpy()@q
    if mode=="M1":return np.c_[Z,qm]
    return np.c_[Z,qm,Z*qm[:,None]]

def cv_macro(train,mc,yc,H,eps,q17,mode,folds,seed):
    X=feats(train,mc,H,eps,q17,mode);cv=KFold(folds,shuffle=True,random_state=seed)
    ps=[]
    for ycol in tqdm(yc,desc=f"CV {mode}",leave=False):
        ps.append(cross_val_predict(gb(seed),X,train[ycol],cv=cv,n_jobs=-1))
    # ps is a list of 13 arrays, each shape (n_samples,).
    # np.c_[ps] treats each array as a ROW and yields (13, n_samples),
    # which caused the previous 512-vs-13 length mismatch.
    # column_stack correctly yields (n_samples, n_targets).
    p=np.column_stack(ps).mean(axis=1)
    y=train[yc].mean(axis=1).to_numpy()
    return {"model":mode,"RMSE":np.sqrt(mean_squared_error(y,p)),
            "Spearman":spearman(y,p)}

def fit_models(train,mc,yc,H,eps,q17,mode,seed):
    X=feats(train,mc,H,eps,q17,mode);mods=[]
    for c in yc:
        m=gb(seed);m.fit(X,train[c]);mods.append(m)
    return mods

def predict_macro(mods,P,mc,H,eps,q17,mode):
    X=feats(P,mc,H,eps,q17,mode)
    return np.column_stack([m.predict(X) for m in mods]).mean(1)

def external(train,test,mc,yc,H,eps,q17,mode,seed):
    mods=fit_models(train,mc,yc,H,eps,q17,mode,seed)
    p=predict_macro(mods,test,mc,H,eps,q17,mode)
    y=test[yc].mean(1).to_numpy()
    return {"model":mode,"RMSE":np.sqrt(mean_squared_error(y,p)),
            "Spearman":spearman(y,p)},mods

def simplex_perturb(P,j,delta):
    """Increase component j by delta, reduce all others proportionally."""
    X=np.asarray(P,float).copy()
    old=X[:,j].copy()
    add=np.minimum(delta,1-old-1e-10)
    others=1-old
    factor=np.where(others>0,(others-add)/others,1)
    X*=factor[:,None]
    X[:,j]=old+add
    X=np.maximum(X,1e-12);X/=X.sum(1,keepdims=True)
    return X

def effect_analysis(mods,train,mc,H,eps,q17,mode,delta):
    P=train[mc].to_numpy(float)
    base=predict_macro(mods,train,mc,H,eps,q17,mode)
    main=[]
    single_effects={}
    for j,c in enumerate(tqdm(mc,desc="Main domain effects")):
        Pj=simplex_perturb(P,j,delta)
        df=pd.DataFrame(Pj,columns=mc)
        pred=predict_macro(mods,df,mc,H,eps,q17,mode)
        e=pred-base;single_effects[j]=e
        main.append({"domain":c,"delta_share":delta,
                     "mean_delta_loss":e.mean(),"median_delta_loss":np.median(e),
                     "beneficial_fraction":float((e<0).mean())})
    inter=[]
    for j in tqdm(range(len(mc)),desc="Pair interactions"):
        for k in range(j+1,len(mc)):
            Pjk=simplex_perturb(simplex_perturb(P,j,delta),k,delta)
            pred=predict_macro(mods,pd.DataFrame(Pjk,columns=mc),mc,H,eps,q17,mode)
            interaction=(pred-base)-single_effects[j]-single_effects[k]
            inter.append({"domain_1":mc[j],"domain_2":mc[k],
                          "mean_interaction_delta_loss":interaction.mean(),
                          "median_interaction_delta_loss":np.median(interaction)})
    return pd.DataFrame(main),pd.DataFrame(inter)

def candidate_search(mods,train,mc,H,eps,q17,mode,n,topk,seed):
    """
    Sample candidates from a Dirichlet fitted to observed mean/concentration,
    then reject candidates outside the 95th percentile nearest-neighbor ILR support.
    """
    rng=np.random.default_rng(seed)
    P=train[mc].to_numpy(float)
    mean=P.mean(0)
    # Estimate moderate concentration from average marginal variance; clamp for stability.
    vals=[]
    for j in range(len(mc)):
        v=P[:,j].var(ddof=1)
        if v>0 and mean[j]*(1-mean[j])>v:
            vals.append(mean[j]*(1-mean[j])/v-1)
    alpha0=float(np.clip(np.median(vals) if vals else 50,10,500))
    alpha=np.maximum(mean*alpha0,.05)
    cand=rng.dirichlet(alpha,size=n)
    Ztr=ilr(P,eps,H);Zc=ilr(cand,eps,H)
    nn=NearestNeighbors(n_neighbors=2).fit(Ztr)
    dtrain=nn.kneighbors(Ztr)[0][:,1]
    threshold=float(np.quantile(dtrain,.95))
    dc=nn.kneighbors(Zc,n_neighbors=1)[0][:,0]
    keep=dc<=threshold
    if keep.sum()<topk:
        # fall back to closest candidates, still flag support status.
        idx=np.argsort(dc)[:max(topk,100)]
    else: idx=np.where(keep)[0]
    C=pd.DataFrame(cand[idx],columns=mc)
    pred=predict_macro(mods,C,mc,H,eps,q17,mode)
    C.insert(0,"predicted_macro_loss",pred)
    C.insert(1,"nearest_train_ILR_distance",dc[idx])
    C.insert(2,"within_95pct_training_support",dc[idx]<=threshold)
    C=C.sort_values("predicted_macro_loss").head(topk).reset_index(drop=True)
    return C,threshold,alpha0

def main():
    a=parse_args();root=a.data_dir.resolve();rt=req(root/"regmix_tables")
    out=a.out_dir.resolve();tab=out/"tables";fig=out/"figures"
    for p in [tab,fig]:p.mkdir(parents=True,exist_ok=True)

    files={"train":("train_mixture_1m.csv","train_pile_loss_1m.csv"),
           "1M":("test_mixture_1m.csv","test_pile_loss_1m.csv"),
           "60M":("test_mixture_60m.csv","test_pile_loss_60m.csv"),
           "1B":("test_mixture_1B.csv","test_pile_loss_1B.csv"),
           "10B_est":("est_mixture_10b.csv","est_pile_loss_10b.csv"),
           "70B_est":("est_mixture_70b.csv","est_pile_loss_70b.csv")}
    data={}
    for s,(m,l) in files.items():data[s],mc,yc=read_pair(req(rt/m),req(rt/l))
    H=helmert(17);Q=load_Q(a.round2_dir.resolve())
    q17,maptab,A16=robust_A16_mapping(root/"domain_mapping_guide.csv",mc,Q)
    maptab.to_csv(tab/"01_A16_mapping_audit.csv",index=False)
    pd.DataFrame({"A16_column":A16.columns}).to_csv(tab/"00_A16_columns.csv",index=False)

    informative=int(maptab.is_informative.sum())
    print(f"A16 audit: informative mapped RegMix domains = {informative}/17")
    print(maptab[["train_domain","quality_domain","confidence","mapped_Q"]].to_string(index=False))

    # Re-ablation after corrected mapping.
    cvrows=[cv_macro(data["train"],mc,yc,H,a.eps,q17,m,a.cv_folds,a.seed)
            for m in ["M0","M1","M2"]]
    cv=pd.DataFrame(cvrows);cv.to_csv(tab/"02_corrected_CV_ablation.csv",index=False)
    extrows=[];model_cache={}
    for s in ["1M","60M","1B","10B_est","70B_est"]:
        for m in ["M0","M1","M2"]:
            r,mods=external(data["train"],data[s],mc,yc,H,a.eps,q17,m,a.seed)
            r["scale"]=s;extrows.append(r)
            if s=="1M":model_cache[m]=mods
    ext=pd.DataFrame(extrows);ext.to_csv(tab/"03_corrected_external_ablation.csv",index=False)

    # Conservative final model rule: quality must improve CV and median REAL test ranking.
    cv0=cv.set_index("model").Spearman
    real=ext[ext.scale.isin(["1M","60M","1B"])].groupby("model").Spearman.median()
    decision=[]
    for m in ["M1","M2"]:
        decision.append({"model":m,"delta_CV_spearman":cv0[m]-cv0["M0"],
                         "delta_realtest_median_spearman":real[m]-real["M0"],
                         "pass":bool(cv0[m]>cv0["M0"] and real[m]>real["M0"])})
    dec=pd.DataFrame(decision);dec.to_csv(tab/"04_final_quality_decision.csv",index=False)
    passed=dec[dec["pass"]]
    final_mode=(passed.sort_values("delta_realtest_median_spearman",ascending=False).iloc[0].model
                if len(passed) else "M0")
    print("Final model for effect analysis:",final_mode)

    # Refit final model on all A4+A5.
    mods=fit_models(data["train"],mc,yc,H,a.eps,q17,final_mode,a.seed)

    main_eff,inter=effect_analysis(mods,data["train"],mc,H,a.eps,q17,final_mode,a.perturb)
    main_eff=main_eff.sort_values("mean_delta_loss")
    inter["interaction_type"]=np.where(inter.mean_interaction_delta_loss<0,
                                      "synergy","redundancy_or_antagonism")
    inter=inter.reindex(inter.mean_interaction_delta_loss.abs().sort_values(ascending=False).index)
    main_eff.to_csv(tab/"05_domain_main_effects.csv",index=False)
    inter.to_csv(tab/"06_pairwise_interactions.csv",index=False)

    cand,thr,a0=candidate_search(mods,data["train"],mc,H,a.eps,q17,final_mode,
                                 a.n_candidates,a.top_k,a.seed)
    cand.to_csv(tab/"07_support_constrained_candidate_mixtures.csv",index=False)

    # Compare top candidate to empirical best train recipe.
    ytrain=data["train"][yc].mean(1)
    bestidx=int(np.argmin(ytrain.to_numpy()))
    empirical=data["train"].iloc[bestidx][mc]
    compare=pd.DataFrame([
        {"recipe":"best_observed_train","observed_macro_loss":float(ytrain.iloc[bestidx]),
         "predicted_macro_loss":float(predict_macro(mods,
             pd.DataFrame([empirical.values],columns=mc),mc,H,a.eps,q17,final_mode)[0]),
         **{c:float(empirical[c]) for c in mc}},
        {"recipe":"best_support_constrained_candidate","observed_macro_loss":np.nan,
         "predicted_macro_loss":float(cand.iloc[0].predicted_macro_loss),
         **{c:float(cand.iloc[0][c]) for c in mc}}
    ])
    compare.to_csv(tab/"08_best_recipe_comparison.csv",index=False)

    # Figures.
    f,ax=plt.subplots(figsize=(9,6))
    plot=main_eff.sort_values("mean_delta_loss")
    ax.barh([norm(x) for x in plot.domain],plot.mean_delta_loss)
    ax.axvline(0,linewidth=.8);ax.set_xlabel(f"Predicted Δ macro Loss for +{a.perturb:.0%} share")
    ax.set_ylabel("Training domain");ax.set_title("Local domain effects on macro Loss")
    f.tight_layout();f.savefig(fig/"01_domain_main_effects.png",dpi=220);plt.close(f)

    mat=pd.DataFrame(np.nan,index=mc,columns=mc)
    for _,r in inter.iterrows():
        mat.loc[r.domain_1,r.domain_2]=r.mean_interaction_delta_loss
        mat.loc[r.domain_2,r.domain_1]=r.mean_interaction_delta_loss
    f,ax=plt.subplots(figsize=(10,8));im=ax.imshow(mat.to_numpy(),aspect="auto")
    ax.set_xticks(range(17));ax.set_xticklabels([norm(x) for x in mc],rotation=90,fontsize=7)
    ax.set_yticks(range(17));ax.set_yticklabels([norm(x) for x in mc],fontsize=7)
    ax.set_title("Pairwise interaction effects");f.colorbar(im,ax=ax,label="Interaction Δ Loss")
    f.tight_layout();f.savefig(fig/"02_pairwise_interactions.png",dpi=220);plt.close(f)

    # Extrapolation audit plot.
    p=ext.pivot(index="scale",columns="model",values="Spearman")
    f,ax=plt.subplots(figsize=(8,5));p.plot(kind="bar",ax=ax)
    ax.axhline(0,linewidth=.8);ax.set_ylim(-1,1)
    ax.set_ylabel("Macro-Loss Spearman");ax.set_xlabel("Scale")
    ax.set_title("Real-test and extrapolation rank transfer")
    f.tight_layout();f.savefig(fig/"03_scale_rank_transfer.png",dpi=220);plt.close(f)

    top_main=main_eff.head(5)[["domain","mean_delta_loss"]].to_string(index=False)
    top_inter=inter.head(8)[["domain_1","domain_2","mean_interaction_delta_loss",
                             "interaction_type"]].to_string(index=False)
    extrap=ext[(ext.model==final_mode)&ext.scale.isin(["10B_est","70B_est"])][
        ["scale","Spearman"]].to_string(index=False)
    summary=f"""ROUND 5 FINAL DOMAIN-EFFECT ANALYSIS

A16 mapping audit:
  informative quality mappings: {informative}/17
  unmatched domains are shrunk to the observed global Q mean; they are NOT assigned
  invented domain-specific quality.

Corrected quality decision:
{dec.to_string(index=False)}
Selected final model for interpretation: {final_mode}

Most beneficial local +{a.perturb:.0%} share perturbations
(negative ΔLoss = predicted improvement):
{top_main}

Largest pair interactions:
{top_inter}

Extrapolation rank audit for selected model:
{extrap}

Candidate search:
  {a.n_candidates} Dirichlet candidates generated around observed training composition.
  Only candidates within the 95th-percentile nearest-neighbor ILR support are preferred.
  Estimated Dirichlet concentration = {a0:.3f}
  ILR support-distance threshold = {thr:.6f}
  Top {a.top_k} candidates saved to tables/07_support_constrained_candidate_mixtures.csv.

Interpretation:
1. Domain main effects are LOCAL compositional perturbations, not causal effects.
2. Pair interactions describe model-implied non-additivity inside the observed support.
3. Candidate mixtures are support-constrained model recommendations, not proven global optima.
4. Negative 10B/70B rank transfer, if it remains present, means the large-scale extrapolation
   is NOT robust. Do not recommend the 1M optimum as a 10B/70B optimum.
"""
    (out/"round5_summary.txt").write_text(summary,encoding="utf-8")
    (out/"run_config.json").write_text(json.dumps(
        {"selected_model":final_mode,"informative_A16_mappings":informative,
         "perturb":a.perturb,"n_candidates":a.n_candidates,
         "support_threshold":thr,"dirichlet_concentration":a0},
        ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n"+summary)
    print("Done ->",out)

if __name__=="__main__":
    main()
