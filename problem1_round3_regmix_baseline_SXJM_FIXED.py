#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 - Round 3 (FIXED for the actual RegMix CSV schema)
17-domain compositional RegMix baseline + cross-scale validation.

Actual contest files expected under:
<project>/real_attachments/A_data_value/regmix_tables/

train_mixture_1m.csv
train_pile_loss_1m.csv
test_mixture_1m.csv
test_pile_loss_1m.csv
test_mixture_60m.csv
test_pile_loss_60m.csv
test_mixture_1B.csv
test_pile_loss_1B.csv

A12-A15 (est_*_10b/70b) are intentionally NOT used in Round 3.

Main fixes vs previous version
------------------------------
1. Uses the exact filenames shown in the user's directory instead of inferring
   tables from guessed column names.
2. Detects mixture columns from the actual mixture file: every numeric column
   except the index/key column. Thus it does not depend on guessed domain names.
3. Detects loss columns from the actual loss file in the same way.
4. Robustly detects index/id/idx; if absent, verifies equal row counts and joins
   by row order.
5. Audits the detected schema before modeling and saves it.
6. Keeps tqdm progress bars.
7. A4+A5 only for fitting/tuning; A6-A11 only for external testing.

Run:
    python problem1_round3_regmix_baseline_SXJM_FIXED.py

Dependencies:
    pip install numpy pandas scipy scikit-learn matplotlib tqdm
"""

from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def parse_args():
    p=argparse.ArgumentParser()
    here=Path(__file__).resolve().parent
    p.add_argument("--data-dir",type=Path,
                   default=here/"real_attachments"/"A_data_value"/"regmix_tables")
    p.add_argument("--out-dir",type=Path,default=here/"round3_outputs")
    p.add_argument("--seed",type=int,default=20260923)
    p.add_argument("--cv-folds",type=int,default=5)
    p.add_argument("--eps",type=float,default=1e-6)
    return p.parse_args()

def require(root,name):
    p=root/name
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    return p

def normalize_col(c):
    return str(c).strip()

def read_csv(path):
    print(f"Loading {path.name} ...")
    d=pd.read_csv(path)
    d.columns=[normalize_col(c) for c in d.columns]
    # Drop accidental unnamed CSV index columns only when they are a trivial 0..n-1 sequence.
    for c in list(d.columns):
        if c.lower().startswith("unnamed"):
            x=pd.to_numeric(d[c],errors="coerce")
            if x.notna().all() and np.array_equal(x.to_numpy(),np.arange(len(d))):
                d=d.drop(columns=c)
    return d

def numeric_columns(d):
    out=[]
    for c in d.columns:
        x=pd.to_numeric(d[c],errors="coerce")
        if x.notna().mean()>=0.99:
            out.append(c)
    return out

def detect_key(mix,loss):
    preferred=["index","id","idx","mixture_id","mixture_index","recipe_id"]
    lower_mix={c.lower():c for c in mix.columns}
    lower_loss={c.lower():c for c in loss.columns}
    for k in preferred:
        if k in lower_mix and k in lower_loss:
            cm,cl=lower_mix[k],lower_loss[k]
            if mix[cm].is_unique and loss[cl].is_unique:
                return cm,cl
    # Any common unique non-floating identifier.
    for cm in mix.columns:
        for cl in loss.columns:
            if cm.lower()==cl.lower() and mix[cm].is_unique and loss[cl].is_unique:
                return cm,cl
    return None,None

def detect_feature_columns(d,key=None):
    cols=numeric_columns(d)
    if key in cols: cols.remove(key)
    # Exclude obvious metadata/size columns if present.
    bad_tokens=("step","tokens","token_count","model_size","params","parameters","seed")
    clean=[c for c in cols if not any(t in c.lower() for t in bad_tokens)]
    return clean

def pair(mix_path,loss_path,label,expected_mix_cols=17):
    mix=read_csv(mix_path); loss=read_csv(loss_path)
    km,kl=detect_key(mix,loss)
    mix_cols=detect_feature_columns(mix,km)
    loss_cols=detect_feature_columns(loss,kl)

    # The mixture table should have exactly 17 numeric proportions after metadata removal.
    # If extra numeric metadata remains, identify proportions by values >=0 and row sum behavior.
    if len(mix_cols)!=expected_mix_cols:
        candidates=[]
        for c in mix_cols:
            x=pd.to_numeric(mix[c],errors="coerce")
            if x.notna().all() and (x>=0).all():
                candidates.append(c)
        # Search for columns whose row sums are close to 1 or 100.
        if len(candidates)>=expected_mix_cols:
            # Prefer columns bounded by 1; otherwise bounded by 100.
            bounded1=[c for c in candidates if pd.to_numeric(mix[c]).max()<=1.000001]
            if len(bounded1)==expected_mix_cols:
                mix_cols=bounded1
            else:
                bounded100=[c for c in candidates if pd.to_numeric(mix[c]).max()<=100.000001]
                if len(bounded100)==expected_mix_cols:
                    mix_cols=bounded100

    if len(mix_cols)!=expected_mix_cols:
        raise ValueError(
            f"{label}: expected 17 mixture columns but detected {len(mix_cols)}.\n"
            f"Mixture file columns: {list(mix.columns)}\n"
            f"Detected numeric candidates: {mix_cols}"
        )
    if len(loss_cols)<10:
        raise ValueError(
            f"{label}: expected about 13 loss columns but detected {len(loss_cols)}.\n"
            f"Loss file columns: {list(loss.columns)}"
        )

    # Join.
    if km is not None and kl is not None:
        left=mix.rename(columns={km:"__key__"})
        right=loss.rename(columns={kl:"__key__"})
        d=left.merge(right,on="__key__",how="inner",validate="one_to_one",
                     suffixes=("","__loss"))
        if len(d)!=len(mix) or len(d)!=len(loss):
            raise ValueError(f"{label}: key merge lost rows.")
        key="__key__"
    else:
        if len(mix)!=len(loss):
            raise ValueError(f"{label}: no shared key and row counts differ: {len(mix)} vs {len(loss)}")
        mix=mix.copy();loss=loss.copy()
        mix["__row_order__"]=np.arange(len(mix))
        loss["__row_order__"]=np.arange(len(loss))
        d=mix.merge(loss,on="__row_order__",validate="one_to_one",
                    suffixes=("","__loss"))
        key="__row_order__"

    # Convert feature columns.
    for c in mix_cols:
        d[c]=pd.to_numeric(d[c],errors="raise")
    # Loss names may collide with mixture names in pathological cases; resolve from merged table.
    resolved_loss=[]
    for c in loss_cols:
        cc=c
        if c in mix.columns and c!=kl and f"{c}__loss" in d.columns:
            cc=f"{c}__loss"
        d[cc]=pd.to_numeric(d[cc],errors="raise")
        resolved_loss.append(cc)

    P=d[mix_cols].astype(float).copy()
    sums=P.sum(axis=1)
    scale_factor=1.0
    if np.nanmedian(sums)>10:
        P=P/100.0
        scale_factor=100.0
    # Close small numerical deviations to exact simplex.
    if (P.sum(axis=1)<=0).any():
        raise ValueError(f"{label}: non-positive mixture row sum.")
    P=P.div(P.sum(axis=1),axis=0)
    d.loc[:,mix_cols]=P

    print(
        f"{label}: rows={len(d)}, mixture columns={len(mix_cols)}, "
        f"loss columns={len(resolved_loss)}, key={key}, "
        f"input scale={'percent' if scale_factor==100 else 'fraction'}"
    )
    return d,mix_cols,resolved_loss,key

def helmert_basis(D):
    H=np.zeros((D,D-1))
    for j in range(1,D):
        H[:j,j-1]=1/np.sqrt(j*(j+1))
        H[j,j-1]=-j/np.sqrt(j*(j+1))
    return H

def ilr(P,eps,basis):
    X=np.asarray(P,float)
    if np.any(X<0): raise ValueError("Negative mixture proportion found.")
    X=X+eps
    X=X/X.sum(axis=1,keepdims=True)
    logx=np.log(X)
    clr=logx-logx.mean(axis=1,keepdims=True)
    return clr@basis

def models(seed):
    return {
      "Ridge":(
        Pipeline([("scale",StandardScaler()),("model",Ridge())]),
        {"model__alpha":[1e-3,1e-2,1e-1,1,10,100]}
      ),
      "ElasticNet":(
        Pipeline([("scale",StandardScaler()),
                  ("model",ElasticNet(max_iter=20000,random_state=seed))]),
        {"model__alpha":[1e-4,1e-3,1e-2,1e-1],
         "model__l1_ratio":[.1,.3,.5,.7,.9]}
      ),
      "RandomForest":(
        RandomForestRegressor(random_state=seed,n_jobs=-1),
        {"n_estimators":[300],"max_depth":[None,6,10],
         "min_samples_leaf":[1,3,5],"max_features":["sqrt",.7]}
      ),
      "GradientBoosting":(
        GradientBoostingRegressor(random_state=seed),
        {"n_estimators":[100,250],"learning_rate":[.03,.07],
         "max_depth":[2,3],"min_samples_leaf":[3,8],"subsample":[.8,1.0]}
      )
    }

def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    ok=np.isfinite(y)&np.isfinite(p);y=y[ok];p=p[ok]
    return dict(
      RMSE=float(np.sqrt(mean_squared_error(y,p))),
      MAE=float(mean_absolute_error(y,p)),
      R2=float(r2_score(y,p)),
      Spearman=float(stats.spearmanr(y,p).statistic)
    )

def fit_all(X,Y,cv,seed):
    fitted={};rows=[]
    specs=models(seed)
    bar=tqdm(total=len(Y.columns)*len(specs),desc="CV model selection",unit="model-target")
    for target in Y.columns:
        fitted[target]={}
        for name,(est,grid) in specs.items():
            gs=GridSearchCV(est,grid,scoring="neg_root_mean_squared_error",
                            cv=cv,n_jobs=-1,refit=True)
            gs.fit(X,Y[target])
            fitted[target][name]=gs.best_estimator_
            rows.append({
              "target":target,"model":name,"CV_RMSE":float(-gs.best_score_),
              "best_params":json.dumps(gs.best_params_,ensure_ascii=False)
            })
            bar.update(1)
    bar.close()
    return fitted,pd.DataFrame(rows)

def predict(fitted,family,X,Y,scale):
    pred=pd.DataFrame(index=Y.index);rows=[]
    for target in tqdm(Y.columns,desc=f"Predicting {scale}",unit="target"):
        p=fitted[target][family].predict(X)
        pred[f"observed__{target}"]=Y[target].to_numpy()
        pred[f"predicted__{target}"]=p
        rows.append({"scale":scale,"target":target,"model":family,
                     **metric(Y[target],p)})
    obs=Y.mean(axis=1).to_numpy()
    pp=np.column_stack([pred[f"predicted__{c}"] for c in Y.columns]).mean(axis=1)
    pred["observed__macro_loss"]=obs
    pred["predicted__macro_loss"]=pp
    rows.append({"scale":scale,"target":"macro_loss","model":family,**metric(obs,pp)})
    return pred,pd.DataFrame(rows)

def main():
    a=parse_args()
    root=a.data_dir.resolve();out=a.out_dir.resolve()
    tab=out/"tables";fig=out/"figures";pro=out/"processed"
    for x in [tab,fig,pro]:x.mkdir(parents=True,exist_ok=True)

    print("=== Round 3 FIXED: Compositional RegMix baseline ===")
    print("RegMix dir:",root)

    names={
      "train_mix":"train_mixture_1m.csv",
      "train_loss":"train_pile_loss_1m.csv",
      "1M_mix":"test_mixture_1m.csv",
      "1M_loss":"test_pile_loss_1m.csv",
      "60M_mix":"test_mixture_60m.csv",
      "60M_loss":"test_pile_loss_60m.csv",
      "1B_mix":"test_mixture_1B.csv",
      "1B_loss":"test_pile_loss_1B.csv",
    }
    paths={k:require(root,v) for k,v in names.items()}
    pd.DataFrame([{"role":k,"file":v.name,"path":str(v)}
                  for k,v in paths.items()]).to_csv(
        tab/"01_detected_A4_A11_tables.csv",index=False)

    train,domains,loss_cols,key=pair(paths["train_mix"],paths["train_loss"],"A4+A5 train")
    tests={}
    for s in ["1M","60M","1B"]:
        d,dc,lc,k=pair(paths[f"{s}_mix"],paths[f"{s}_loss"],f"{s} test")
        if dc!=domains:
            # Match by names if order differs.
            if set(dc)==set(domains):
                d=d.copy()
            else:
                raise ValueError(f"{s}: mixture domain names differ from training.\ntrain={domains}\ntest={dc}")
        # Map test loss names to training loss names by exact names first, then position.
        if set(lc)==set(loss_cols):
            rename_loss={}
        elif len(lc)==len(loss_cols):
            rename_loss={old:new for old,new in zip(lc,loss_cols)}
            d=d.rename(columns=rename_loss)
        else:
            raise ValueError(f"{s}: loss target count differs: train={len(loss_cols)}, test={len(lc)}")
        tests[s]=d

    # Save exact schema so the experiment is auditable.
    schema_rows=[]
    for role,p in paths.items():
        d=pd.read_csv(p,nrows=3)
        schema_rows.append({"role":role,"file":p.name,"columns":" | ".join(map(str,d.columns))})
    pd.DataFrame(schema_rows).to_csv(tab/"00_actual_schema.csv",index=False)

    # Diagnostics.
    diag=[]
    for label,d in [("train",train)]+list(tests.items()):
        sums=d[domains].sum(axis=1)
        diag.append({
          "dataset":label,"n":len(d),"n_domains":len(domains),
          "min_sum":sums.min(),"mean_sum":sums.mean(),"max_sum":sums.max(),
          "max_abs_sum_error":(sums-1).abs().max(),
          "zero_fraction":float((d[domains].to_numpy()==0).mean())
        })
    pd.DataFrame(diag).to_csv(tab/"02_simplex_diagnostics.csv",index=False)

    basis=helmert_basis(17)
    ilr_cols=[f"ilr_{i+1:02d}" for i in range(16)]
    pd.DataFrame(basis,index=domains,columns=ilr_cols).to_csv(tab/"03_ILR_basis.csv")
    Xtr=ilr(train[domains],a.eps,basis)
    Xt={s:ilr(d[domains],a.eps,basis) for s,d in tests.items()}

    Ytr=train[loss_cols].astype(float).copy()
    # Fit individual targets + direct macro sensitivity target.
    Yfit=Ytr.copy();Yfit["macro_loss"]=Ytr.mean(axis=1)

    cv=KFold(a.cv_folds,shuffle=True,random_state=a.seed)
    fitted,cvtab=fit_all(Xtr,Yfit,cv,a.seed)
    cvtab.to_csv(tab/"04_CV_all_models_targets.csv",index=False)

    fam=(cvtab.groupby("model",as_index=False)
         .agg(mean_CV_RMSE=("CV_RMSE","mean"),
              median_CV_RMSE=("CV_RMSE","median"))
         .sort_values("mean_CV_RMSE"))
    fam.to_csv(tab/"05_CV_model_family_summary.csv",index=False)
    best=fam.iloc[0].model
    print("\nSelected primary M0 family from TRAIN CV only:",best)

    metrics_all=[];preds={}
    for s,d in tests.items():
        Y=d[loss_cols].astype(float)
        pr,me=predict(fitted,best,Xt[s],Y,s)
        # Direct macro target as sensitivity analysis.
        direct=fitted["macro_loss"][best].predict(Xt[s])
        pr["predicted__macro_loss_direct"]=direct
        me=pd.concat([me,pd.DataFrame([{
          "scale":s,"target":"macro_loss_direct","model":best,
          **metric(Y.mean(axis=1),direct)
        }])],ignore_index=True)
        pr.to_csv(pro/f"{s}_predictions.csv",index=False)
        preds[s]=pr;metrics_all.append(me)

    met=pd.concat(metrics_all,ignore_index=True)
    met.to_csv(tab/"06_external_test_metrics.csv",index=False)
    macro=met[met.target.isin(["macro_loss","macro_loss_direct"])]
    macro.to_csv(tab/"07_macro_cross_scale_metrics.csv",index=False)

    target_summary=(met[~met.target.str.contains("direct")]
                    .groupby("target",as_index=False)
                    .agg(mean_RMSE=("RMSE","mean"),mean_MAE=("MAE","mean"),
                         mean_R2=("R2","mean"),mean_Spearman=("Spearman","mean"),
                         min_Spearman=("Spearman","min")))
    target_summary.to_csv(tab/"08_target_robustness_summary.csv",index=False)

    wins=(cvtab.sort_values("CV_RMSE").groupby("target").first().reset_index()
          .model.value_counts().rename_axis("model").reset_index(name="target_wins"))
    wins.to_csv(tab/"09_CV_target_winner_counts.csv",index=False)

    # Figures: English labels by design.
    f,ax=plt.subplots(figsize=(8,5))
    q=fam.sort_values("mean_CV_RMSE")
    ax.bar(q.model,q.mean_CV_RMSE)
    ax.set_ylabel("Mean CV RMSE");ax.set_xlabel("Model family")
    ax.set_title("Training CV model comparison");ax.tick_params(axis="x",rotation=25)
    f.tight_layout();f.savefig(fig/"01_CV_model_comparison.png",dpi=220);plt.close(f)

    c=train[domains].corr(method="spearman")
    f,ax=plt.subplots(figsize=(10,8));im=ax.imshow(c,vmin=-1,vmax=1)
    ax.set_xticks(range(17));ax.set_xticklabels(domains,rotation=90,fontsize=7)
    ax.set_yticks(range(17));ax.set_yticklabels(domains,fontsize=7)
    ax.set_title("17-domain mixture Spearman correlation")
    f.colorbar(im,ax=ax,label="Spearman correlation")
    f.tight_layout();f.savefig(fig/"02_training_mixture_correlation.png",dpi=220);plt.close(f)

    for s,pr in preds.items():
        f,ax=plt.subplots(figsize=(6,6))
        x=pr["observed__macro_loss"];y=pr["predicted__macro_loss"]
        ax.scatter(x,y,s=18,alpha=.65)
        lo=min(x.min(),y.min());hi=max(x.max(),y.max())
        ax.plot([lo,hi],[lo,hi],"--",linewidth=1)
        ax.set_xlabel("Observed macro Loss");ax.set_ylabel("Predicted macro Loss")
        ax.set_title(f"{s}: predicted vs observed")
        f.tight_layout();f.savefig(fig/f"03_predicted_vs_observed_{s}.png",dpi=220);plt.close(f)

    mm=met[met.target=="macro_loss"].copy()
    f,ax=plt.subplots(figsize=(8,5))
    ax.bar(mm.scale,mm.Spearman)
    ax.set_ylim(-1,1);ax.set_ylabel("Spearman rank correlation")
    ax.set_xlabel("Real test scale");ax.set_title("Cross-scale mixture rank stability")
    ax.axhline(0,linewidth=.8)
    f.tight_layout();f.savefig(fig/"04_cross_scale_spearman.png",dpi=220);plt.close(f)

    heat=(met[met.target!="macro_loss_direct"]
          .pivot(index="target",columns="scale",values="Spearman"))
    f,ax=plt.subplots(figsize=(7,8));im=ax.imshow(heat.to_numpy(),vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(len(heat.columns)));ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)));ax.set_yticklabels(heat.index,fontsize=8)
    ax.set_title("Per-target cross-scale Spearman")
    f.colorbar(im,ax=ax,label="Spearman correlation")
    f.tight_layout();f.savefig(fig/"05_target_scale_spearman_heatmap.png",dpi=220);plt.close(f)

    config={"paths":{k:str(v) for k,v in paths.items()},"domains":domains,
            "loss_targets":loss_cols,"eps":a.eps,"cv_folds":a.cv_folds,
            "seed":a.seed,"primary_model_family":best}
    (out/"run_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),
                                      encoding="utf-8")

    summary=f"""ROUND 3 FIXED - COMPOSITIONAL REGMIX BASELINE

A4+A5 training rows: {len(train)}
Detected mixture domains: {len(domains)}
Detected validation Loss targets: {len(loss_cols)}
Transform: zero smoothing eps={a.eps} + ILR 17 -> 16
Model selection: {a.cv_folds}-fold CV on A4+A5 only
Selected M0 family: {best}

REAL EXTERNAL TEST METRICS (macro Loss)
{met[met.target=="macro_loss"][["scale","RMSE","MAE","R2","Spearman"]].to_string(index=False)}

A12-A15 were NOT used in this round.
They remain reserved for extrapolation robustness after the real-test analysis.
"""
    (out/"round3_summary.txt").write_text(summary,encoding="utf-8")
    print("\n"+summary)
    print("Finished ->",out)

if __name__=="__main__":
    main()
