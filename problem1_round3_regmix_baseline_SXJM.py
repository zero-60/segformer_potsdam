#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 - Round 3
17-domain compositional RegMix baseline + cross-scale validation.

Purpose
-------
Train ONLY on A4+A5 (512 mixtures), then test the frozen pipeline on:
- A6+A7: 1M real test
- A8+A9: 60M real test
- A10+A11: 1B real test

This round intentionally DOES NOT inject Q yet.
It first establishes M0: Loss = f(ILR(p)).
Round 4 will compare M0 against quality-enhanced M1/M2.

Key safeguards
--------------
1. Mixture proportions are compositional: use zero replacement + ILR, not raw 17 columns.
2. A4/A5 are joined by index; test mixture/loss tables are also joined by index.
3. Model selection uses CV only inside A4+A5.
4. A6-A11 are never used for fitting or hyperparameter selection.
5. 13 validation losses are modeled separately, plus macro Loss.
6. Reports RMSE, MAE, R2 and Spearman at 1M/60M/1B.
7. Includes tqdm progress bars and publication-ready figures.

Run:
    python problem1_round3_regmix_baseline_SXJM.py

Dependencies:
    pip install numpy pandas scipy scikit-learn matplotlib tqdm
"""

from __future__ import annotations
import argparse, json, warnings, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Expected 17 Pile training domains. We also normalize common naming variants.
EXPECTED_DOMAINS = [
    "arxiv","freelaw","nih_exporter","pubmed_central","wikipedia",
    "dm_mathematics","github","philpapers","stackexchange","enron",
    "gutenberg","pile_cc","ubuntu","europarl","hackernews",
    "pubmed_abstracts","uspto"
]

ALIASES = {
    "enron_emails": "enron",
    "dm_math": "dm_mathematics",
    "pubmed_abstract": "pubmed_abstracts",
    "pilecc": "pile_cc",
    "free_law": "freelaw",
}

def parse_args():
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument("--data-dir", type=Path,
                   default=here/"real_attachments"/"A_data_value")
    p.add_argument("--out-dir", type=Path,
                   default=here/"round3_outputs")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Zero-replacement smoothing before ILR.")
    return p.parse_args()

def norm_name(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return ALIASES.get(s, s)

def all_csv(root):
    return sorted(root.rglob("*.csv"))

def inspect_csv(path):
    try:
        d = pd.read_csv(path, nrows=5)
        return list(d.columns)
    except Exception:
        return []

def find_pair_candidates(root):
    """Inventory CSV files and infer mixture/loss tables by schema."""
    rows = []
    for f in tqdm(all_csv(root), desc="Scanning CSV schemas", unit="file"):
        cols = inspect_csv(f)
        ncols = len(cols)
        n_domains = sum(norm_name(c) in EXPECTED_DOMAINS for c in cols)
        loss_like = sum(("loss" in norm_name(c)) for c in cols)
        rows.append({
            "path": str(f), "file": f.name, "ncols": ncols,
            "domain_cols": n_domains, "loss_like_cols": loss_like,
            "columns": "|".join(map(str, cols))
        })
    return pd.DataFrame(rows)

def choose_tables(inv):
    """
    Locate A4-A11 robustly from schema and filenames.
    Mixture tables: index + ~17 domain columns.
    Loss tables: index + ~13 loss/validation columns.
    We then classify train/1M/60M/1B using filename/path tokens and row counts.
    """
    mix = inv[inv.domain_cols >= 15].copy()
    loss = inv[(inv.domain_cols < 10) & (inv.ncols >= 10) & (inv.ncols <= 20)].copy()
    if mix.empty:
        raise FileNotFoundError("No 17-domain mixture CSV found under data-dir.")
    if loss.empty:
        raise FileNotFoundError("No validation-loss CSV found under data-dir.")

    def load_rows(p):
        try: return len(pd.read_csv(p, usecols=[0]))
        except: return -1

    for df in [mix, loss]:
        df["rows"] = [load_rows(x) for x in tqdm(df.path, desc="Counting table rows", leave=False)]
        df["key"] = (df["path"].str.lower()
                     .str.replace("\\\\","/",regex=False))

    def classify(row):
        s = row["key"]
        n = row["rows"]
        if "70b" in s: return "70B"
        if "10b" in s: return "10B"
        if "60m" in s: return "60M"
        if re.search(r"(^|[^0-9])1b([^0-9]|$)", s): return "1B"
        if re.search(r"(^|[^0-9])1m([^0-9]|$)", s): return "1M"
        # Contest row-count fallback:
        if n == 512: return "train"
        if n == 64: return "1B"
        if n == 256:
            # Need filename distinction between 1M and 60M; leave unresolved.
            return "256_unknown"
        return "unknown"

    mix["scale"] = mix.apply(classify, axis=1)
    loss["scale"] = loss.apply(classify, axis=1)

    def pick(df, scale, kind):
        x = df[df.scale == scale]
        if len(x) == 1: return Path(x.iloc[0].path)
        if len(x) > 1:
            # Prefer paths under regmix_tables and names mentioning test/train appropriately.
            score = x.key.str.contains("regmix_tables").astype(int)
            score += x.key.str.contains(scale.lower()).astype(int) * 2
            return Path(x.iloc[int(np.argmax(score.to_numpy()))].path)
        return None

    out = {
        "train_mix": pick(mix, "train", "mix"),
        "train_loss": pick(loss, "train", "loss"),
        "1M_mix": pick(mix, "1M", "mix"),
        "1M_loss": pick(loss, "1M", "loss"),
        "60M_mix": pick(mix, "60M", "mix"),
        "60M_loss": pick(loss, "60M", "loss"),
        "1B_mix": pick(mix, "1B", "mix"),
        "1B_loss": pick(loss, "1B", "loss"),
    }

    # Resolve 256-row tables by filename hints if explicit classifier missed them.
    for scale in ["1M","60M"]:
        for typ,df in [("mix",mix),("loss",loss)]:
            key=f"{scale}_{typ}"
            if out[key] is None:
                cand=df[df.rows==256].copy()
                token=scale.lower()
                hit=cand[cand.key.str.contains(token,regex=False)]
                if len(hit)==1: out[key]=Path(hit.iloc[0].path)

    missing=[k for k,v in out.items() if v is None]
    if missing:
        detail = inv[["file","path","ncols","domain_cols","loss_like_cols"]].to_string(index=False)
        raise FileNotFoundError(
            "Could not uniquely identify these A4-A11 tables: "
            + ", ".join(missing)
            + "\nInspect round3_outputs/tables/00_csv_inventory.csv or filenames.\n"
            + detail
        )
    return out

def read_pair(mix_path, loss_path, label):
    mix = pd.read_csv(mix_path)
    loss = pd.read_csv(loss_path)
    mix.columns=[norm_name(c) for c in mix.columns]
    loss.columns=[norm_name(c) for c in loss.columns]

    # Detect join key.
    common=[c for c in mix.columns if c in loss.columns]
    join_candidates=[c for c in common if c in {"index","id","idx","mixture_id","mixture_index"}]
    if join_candidates:
        key=join_candidates[0]
        d=mix.merge(loss,on=key,how="inner",validate="one_to_one")
    else:
        # Safe fallback only if equal length; preserve explicit row-order audit.
        if len(mix)!=len(loss):
            raise ValueError(f"{label}: no common index and row counts differ.")
        key="_row_order"
        mix[key]=np.arange(len(mix)); loss[key]=np.arange(len(loss))
        d=mix.merge(loss,on=key,validate="one_to_one")

    domain_cols=[c for c in EXPECTED_DOMAINS if c in d.columns]
    if len(domain_cols)!=17:
        raise ValueError(f"{label}: expected 17 domains, found {len(domain_cols)}: {domain_cols}")

    # Loss columns = columns from original loss table except join key.
    loss_cols=[c for c in loss.columns if c != key and c in d.columns]
    loss_cols=[c for c in loss_cols if pd.api.types.is_numeric_dtype(d[c])]
    if len(loss_cols) < 10:
        raise ValueError(f"{label}: too few numeric loss columns: {loss_cols}")

    # Coerce and check simplex.
    P=d[domain_cols].apply(pd.to_numeric,errors="coerce")
    Y=d[loss_cols].apply(pd.to_numeric,errors="coerce")
    if P.isna().any().any() or Y.isna().any().any():
        raise ValueError(f"{label}: missing/non-numeric values after parsing.")
    sums=P.sum(axis=1)
    # Some files may store percentages rather than fractions.
    if np.nanmedian(sums) > 10:
        P=P/100.0
        sums=P.sum(axis=1)
    d.loc[:,domain_cols]=P
    simplex_err=(sums-1).abs()
    print(f"{label}: n={len(d)}, domains={len(domain_cols)}, losses={len(loss_cols)}, "
          f"median simplex sum={sums.median():.8f}, max |sum-1|={simplex_err.max():.3e}")
    return d,domain_cols,loss_cols,key

def helmert_basis(D):
    """Orthonormal Helmert sub-matrix basis: D x (D-1)."""
    H=np.zeros((D,D-1))
    for j in range(1,D):
        H[:j,j-1]=1/np.sqrt(j*(j+1))
        H[j,j-1]=-j/np.sqrt(j*(j+1))
    return H

def close_and_smooth(P, eps):
    X=np.asarray(P,dtype=float)
    if np.any(X<0): raise ValueError("Mixture contains negative proportions.")
    X=X+eps
    X=X/X.sum(axis=1,keepdims=True)
    return X

def ilr_transform(P, eps, basis):
    X=close_and_smooth(P,eps)
    logX=np.log(X)
    clr=logX-logX.mean(axis=1,keepdims=True)
    return clr @ basis

def make_models(seed):
    return {
      "Ridge": (
        Pipeline([("scale",StandardScaler()),("model",Ridge())]),
        {"model__alpha":[1e-3,1e-2,1e-1,1,10,100]}
      ),
      "ElasticNet": (
        Pipeline([("scale",StandardScaler()),
                  ("model",ElasticNet(max_iter=20000,random_state=seed))]),
        {"model__alpha":[1e-4,1e-3,1e-2,1e-1],
         "model__l1_ratio":[.1,.3,.5,.7,.9]}
      ),
      "RandomForest": (
        RandomForestRegressor(random_state=seed,n_jobs=-1),
        {"n_estimators":[300],
         "max_depth":[None,6,10],
         "min_samples_leaf":[1,3,5],
         "max_features":["sqrt",.7]}
      ),
      "GradientBoosting": (
        GradientBoostingRegressor(random_state=seed),
        {"n_estimators":[100,250],
         "learning_rate":[.03,.07],
         "max_depth":[2,3],
         "min_samples_leaf":[3,8],
         "subsample":[.8,1.0]}
      )
    }

def metrics(y,p):
    ok=np.isfinite(y)&np.isfinite(p)
    y=np.asarray(y)[ok]; p=np.asarray(p)[ok]
    rho=stats.spearmanr(y,p).statistic if len(y)>2 else np.nan
    return {
      "RMSE":float(np.sqrt(mean_squared_error(y,p))),
      "MAE":float(mean_absolute_error(y,p)),
      "R2":float(r2_score(y,p)),
      "Spearman":float(rho)
    }

def fit_select_models(X,Y,cv,seed):
    """
    Tune each model independently for each target using TRAIN CV only.
    Return fitted best estimators and CV table.
    """
    models=make_models(seed)
    fitted={}; rows=[]
    targets=list(Y.columns)
    total=len(targets)*len(models)
    bar=tqdm(total=total,desc="CV model selection",unit="fit")
    for target in targets:
        fitted[target]={}
        y=Y[target].to_numpy()
        for name,(est,param) in models.items():
            gs=GridSearchCV(est,param,cv=cv,scoring="neg_root_mean_squared_error",
                            n_jobs=-1,refit=True)
            gs.fit(X,y)
            fitted[target][name]=gs.best_estimator_
            rows.append({
              "target":target,"model":name,
              "CV_RMSE":-gs.best_score_,
              "best_params":json.dumps(gs.best_params_,ensure_ascii=False)
            })
            bar.update(1)
    bar.close()
    return fitted,pd.DataFrame(rows)

def select_global_model(cvtab):
    """
    Pick ONE model family for the primary M0 using mean CV RMSE across targets.
    This avoids cherry-picking a different family at each external scale.
    """
    s=(cvtab.groupby("model",as_index=False)
       .agg(mean_CV_RMSE=("CV_RMSE","mean"),
            median_CV_RMSE=("CV_RMSE","median")))
    s=s.sort_values("mean_CV_RMSE")
    return s.iloc[0]["model"],s

def predict_dataset(fitted,model_name,X,Y,label):
    rows=[]; pred=pd.DataFrame(index=Y.index)
    for target in tqdm(Y.columns,desc=f"Predicting {label}",unit="target"):
        est=fitted[target][model_name]
        p=est.predict(X)
        pred[f"observed__{target}"]=Y[target].to_numpy()
        pred[f"predicted__{target}"]=p
        m=metrics(Y[target].to_numpy(),p)
        rows.append({"scale":label,"target":target,"model":model_name,**m})
    # Macro loss: average observed/predicted across the 13 targets.
    obs_cols=[c for c in pred if c.startswith("observed__")]
    prd_cols=[c for c in pred if c.startswith("predicted__")]
    obs_macro=pred[obs_cols].mean(axis=1).to_numpy()
    pred_macro=pred[prd_cols].mean(axis=1).to_numpy()
    pred["observed__macro_loss"]=obs_macro
    pred["predicted__macro_loss"]=pred_macro
    rows.append({"scale":label,"target":"macro_loss","model":model_name,
                 **metrics(obs_macro,pred_macro)})
    return pred,pd.DataFrame(rows)

def save_cv_plot(summary,path):
    d=summary.sort_values("mean_CV_RMSE")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(d.model,d.mean_CV_RMSE)
    ax.set_ylabel("Mean CV RMSE across validation domains")
    ax.set_xlabel("Model family")
    ax.set_title("Training-set cross-validation model comparison")
    ax.tick_params(axis="x",rotation=25)
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def save_pred_plot(pred,scale,path):
    obs=pred["observed__macro_loss"]
    pr=pred["predicted__macro_loss"]
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(obs,pr,s=18,alpha=.65)
    lo=min(obs.min(),pr.min()); hi=max(obs.max(),pr.max())
    ax.plot([lo,hi],[lo,hi],"--",linewidth=1)
    ax.set_xlabel("Observed macro Loss")
    ax.set_ylabel("Predicted macro Loss")
    ax.set_title(f"{scale}: predicted vs observed")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def save_scale_metrics(met,path):
    m=met[met.target=="macro_loss"].copy()
    x=np.arange(len(m))
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x,m.Spearman)
    ax.set_xticks(x);ax.set_xticklabels(m.scale)
    ax.set_ylim(-1,1)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_xlabel("Real test scale")
    ax.set_title("Cross-scale mixture rank stability")
    ax.axhline(0,linewidth=.8)
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def save_domain_corr(train,domains,path):
    c=train[domains].corr(method="spearman")
    fig,ax=plt.subplots(figsize=(10,8))
    im=ax.imshow(c,vmin=-1,vmax=1)
    ax.set_xticks(range(len(domains)));ax.set_xticklabels(domains,rotation=90)
    ax.set_yticks(range(len(domains)));ax.set_yticklabels(domains)
    ax.set_title("17-domain mixture Spearman correlation")
    fig.colorbar(im,ax=ax,label="Spearman correlation")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def main():
    a=parse_args()
    root=a.data_dir.resolve(); out=a.out_dir.resolve()
    tab=out/"tables"; fig=out/"figures"; pro=out/"processed"
    for p in [tab,fig,pro]:p.mkdir(parents=True,exist_ok=True)

    print("=== Round 3: Compositional RegMix baseline ===")
    print("Data dir:",root)
    inv=find_pair_candidates(root)
    inv.to_csv(tab/"00_csv_inventory.csv",index=False)
    paths=choose_tables(inv)
    pd.DataFrame([{"role":k,"path":str(v)} for k,v in paths.items()]).to_csv(
        tab/"01_detected_A4_A11_tables.csv",index=False)
    print("\nDetected tables:")
    for k,v in paths.items(): print(f"  {k:12s} -> {v.name}")

    train,domains,loss_cols,key=read_pair(paths["train_mix"],paths["train_loss"],"A4+A5 train")
    tests={}
    for scale in ["1M","60M","1B"]:
        d,dom2,loss2,key2=read_pair(paths[f"{scale}_mix"],paths[f"{scale}_loss"],f"{scale} test")
        if dom2!=domains:
            raise ValueError(f"{scale}: domain columns differ from training.")
        # Align validation losses by normalized names; require same set.
        missing=[c for c in loss_cols if c not in loss2]
        if missing:
            raise ValueError(f"{scale}: missing training loss targets {missing}")
        tests[scale]=d

    # Simplex diagnostics.
    diag=[]
    for label,d in [("train",train)]+list(tests.items()):
        sums=d[domains].sum(axis=1)
        diag.append({
          "dataset":label,"n":len(d),"min_sum":sums.min(),"mean_sum":sums.mean(),
          "max_sum":sums.max(),"max_abs_sum_error":(sums-1).abs().max(),
          "zero_fraction":(d[domains].to_numpy()==0).mean()
        })
    pd.DataFrame(diag).to_csv(tab/"02_simplex_diagnostics.csv",index=False)

    # Fixed ILR basis learned by dimension, not by test outcomes.
    basis=helmert_basis(len(domains))
    Xtr=ilr_transform(train[domains],a.eps,basis)
    Xtests={s:ilr_transform(d[domains],a.eps,basis) for s,d in tests.items()}
    ilr_cols=[f"ilr_{i+1:02d}" for i in range(16)]
    pd.DataFrame(basis,index=domains,columns=ilr_cols).to_csv(
        tab/"03_ILR_basis.csv")

    Ytr=train[loss_cols].copy()
    # Include a separately fitted macro target during CV/model selection.
    Yfit=Ytr.copy()
    Yfit["macro_loss"]=Ytr.mean(axis=1)

    cv=KFold(n_splits=a.cv_folds,shuffle=True,random_state=a.seed)
    fitted,cvtab=fit_select_models(Xtr,Yfit,cv,a.seed)
    cvtab.to_csv(tab/"04_CV_all_models_targets.csv",index=False)
    best_family,cvsummary=select_global_model(cvtab)
    cvsummary.to_csv(tab/"05_CV_model_family_summary.csv",index=False)
    print("\nPrimary M0 model family selected by TRAIN CV:",best_family)

    # Refit/predict 13 individual losses on each test. Macro prediction is average
    # of 13 predicted losses; separately fitted macro model is retained as sensitivity.
    all_metrics=[]; pred_tables={}
    for scale,d in tests.items():
        Y=d[loss_cols].copy()
        pred,met=predict_dataset(fitted,best_family,Xtests[scale],Y,scale)
        # Separately fitted macro sensitivity.
        macro_obs=Y.mean(axis=1).to_numpy()
        macro_direct=fitted["macro_loss"][best_family].predict(Xtests[scale])
        mm=metrics(macro_obs,macro_direct)
        met=pd.concat([met,pd.DataFrame([{
          "scale":scale,"target":"macro_loss_direct","model":best_family,**mm
        }])],ignore_index=True)
        pred["predicted__macro_loss_direct"]=macro_direct
        pred_tables[scale]=pred
        all_metrics.append(met)
        pred.to_csv(pro/f"{scale}_predictions.csv",index=False)

    met=pd.concat(all_metrics,ignore_index=True)
    met.to_csv(tab/"06_external_test_metrics.csv",index=False)

    # Rank-invariance matrix: observed macro losses across common mixture indices
    # where possible, otherwise compare scale-level prediction ranking only.
    macro=met[met.target.isin(["macro_loss","macro_loss_direct"])].copy()
    macro.to_csv(tab/"07_macro_cross_scale_metrics.csv",index=False)

    # Target-level robustness summary.
    target_summary=(met[~met.target.str.contains("direct")]
                    .groupby("target",as_index=False)
                    .agg(mean_RMSE=("RMSE","mean"),
                         mean_MAE=("MAE","mean"),
                         mean_R2=("R2","mean"),
                         mean_Spearman=("Spearman","mean"),
                         min_Spearman=("Spearman","min")))
    target_summary.to_csv(tab/"08_target_robustness_summary.csv",index=False)

    # Model family win counts on train CV.
    winners=(cvtab.sort_values("CV_RMSE").groupby("target").first().reset_index())
    wins=winners.model.value_counts().rename_axis("model").reset_index(name="target_wins")
    wins.to_csv(tab/"09_CV_target_winner_counts.csv",index=False)

    # Figures.
    save_cv_plot(cvsummary,fig/"01_CV_model_comparison.png")
    save_domain_corr(train,domains,fig/"02_training_mixture_correlation.png")
    for scale,pred in pred_tables.items():
        save_pred_plot(pred,scale,fig/f"03_predicted_vs_observed_{scale}.png")
    save_scale_metrics(met,fig/"04_cross_scale_spearman.png")

    # Per-target heatmap of Spearman across scales.
    heat=(met[(met.target!="macro_loss_direct")]
          .pivot(index="target",columns="scale",values="Spearman"))
    fig0,ax=plt.subplots(figsize=(7,8))
    im=ax.imshow(heat.to_numpy(),vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(len(heat.columns)));ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)));ax.set_yticklabels(heat.index)
    ax.set_title("Per-target cross-scale Spearman")
    fig0.colorbar(im,ax=ax,label="Spearman correlation")
    fig0.tight_layout();fig0.savefig(fig/"05_target_scale_spearman_heatmap.png",dpi=220)
    plt.close(fig0)

    config={
      "data_dir":str(root),"paths":{k:str(v) for k,v in paths.items()},
      "domains":domains,"loss_targets":loss_cols,"eps":a.eps,
      "cv_folds":a.cv_folds,"seed":a.seed,
      "primary_model_family":best_family
    }
    (out/"run_config.json").write_text(
        json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")

    macro_print=met[met.target=="macro_loss"][
        ["scale","RMSE","MAE","R2","Spearman"]].to_string(index=False)
    summary=f"""ROUND 3 COMPOSITIONAL REGMIX BASELINE SUMMARY

Training:
  A4+A5 rows = {len(train)}
  mixture domains = {len(domains)}
  validation Loss targets = {len(loss_cols)}
  transform = zero replacement (eps={a.eps}) + ILR(17 -> 16)
  CV folds = {a.cv_folds}

Primary model family selected using TRAIN CV only:
  {best_family}

External real-test macro metrics:
{macro_print}

Interpretation rules:
- A6-A11 were not used for training/tuning.
- Spearman evaluates whether mixture ranking transfers across model scales.
- Do not call A12-A15 'real validation'; they belong to the later extrapolation round.
- Round 4 should freeze this M0 protocol and compare quality-enhanced M1/M2
  using the SAME train/test splits and metrics.
"""
    (out/"round3_summary.txt").write_text(summary,encoding="utf-8")
    print("\n"+summary)
    print("Done ->",out)

if __name__=="__main__":
    main()
