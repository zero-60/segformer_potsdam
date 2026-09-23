#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 - Round 2
Final quality index Q + conflict detection/correction + domain aggregation
+ A1 vs A2/A3 extended validation + raw-content audit sample.

IMPORTANT correction from Round 1
---------------------------------
Round 1 intentionally compared mean vs median scalarization of list-valued fields.
The real data show that several list-valued fields are multi-logit classifier outputs
(e.g. 6 logits for ModernBERT, 2 for fluency/ad). Their arithmetic mean is NOT a
meaningful quality score.

Round 2 therefore re-reads RAW A1/A2/A3 and uses:
- singleton list: scalar value
- 2 logits: softmax probability of class 1
- >=3 logits: softmax expected ordinal class, normalized to [0,1]
This is explicit and auditable in tables/01_signal_transform_policy.csv.

The script then:
1) builds direction-aware [0,1] indicators using A1 reference quantiles;
2) uses CRITIC objective weights for the main Q score;
3) keeps ambiguous document-statistics out of the main Q unless direction is defensible;
4) creates interpretable quality groups;
5) defines sample-level conflict as group dispersion/range;
6) selects conflict penalty gamma by stability criterion rather than arbitrary hard-coding;
7) outputs Q0 (before conflict correction) and Q (after correction);
8) aggregates Q to source domains;
9) validates arxiv/github sample vs extended-set Q using bootstrap CIs;
10) exports stratified raw A1 text samples for human audit;
11) produces figures and tables.

Run from the SXJM project root:
    python problem1_round2_quality_conflict_SXJM.py

Dependencies:
    pip install numpy pandas scipy scikit-learn matplotlib tqdm
"""

from __future__ import annotations
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

QUALITY_COLS = [
"fineweb_edu","fluency_en","modernbert_cleanliness","modernbert_readability",
"modernbert_reasoning","modernbert_professionalism","dsir_books","dsir_wiki",
"dsir_math","qurater","ad_en","rps_doc_word_count","rps_doc_num_sentences",
"rps_doc_unigram_entropy","rps_doc_frac_unique_words","rps_doc_frac_no_alph_words",
"rps_doc_frac_chars_top_2gram","rps_doc_frac_chars_top_3gram",
"rps_lines_uppercase_letter_fraction",
"rps_lines_ending_with_terminal_punctution_mark",
"rps_lines_numerical_chars_fraction","rps_doc_mean_word_length"]

ALIASES={"rps_lines_ending_with_terminal_punctuation_mark":
         "rps_lines_ending_with_terminal_punctution_mark"}

LIST_COLS=["fluency_en","modernbert_cleanliness","qurater","ad_en",
           "modernbert_reasoning","fineweb_edu","modernbert_professionalism",
           "modernbert_readability"]

# Main Q: only indicators with a defensible monotone quality interpretation.
# Ambiguous length/entropy/numeric/mean-word-length features remain diagnostics.
DIRECTION={
"fineweb_edu":1,"fluency_en":1,"modernbert_cleanliness":1,
"modernbert_readability":1,"modernbert_reasoning":1,
"modernbert_professionalism":1,"dsir_books":1,"dsir_wiki":1,"dsir_math":1,
"qurater":1,"ad_en":-1,
"rps_doc_word_count":0,"rps_doc_num_sentences":0,"rps_doc_unigram_entropy":0,
"rps_doc_frac_unique_words":1,
"rps_doc_frac_no_alph_words":-1,
"rps_doc_frac_chars_top_2gram":-1,"rps_doc_frac_chars_top_3gram":-1,
"rps_lines_uppercase_letter_fraction":-1,
"rps_lines_ending_with_terminal_punctution_mark":1,
"rps_lines_numerical_chars_fraction":0,"rps_doc_mean_word_length":0}

MAIN_FEATURES=[c for c in QUALITY_COLS if DIRECTION[c] != 0]

# Interpretable conflict groups. Each group uses available direction-normalized indicators.
GROUPS={
"semantic_value":["fineweb_edu","modernbert_reasoning","dsir_books","dsir_wiki","dsir_math"],
"language_readability":["fluency_en","modernbert_readability","modernbert_professionalism","qurater"],
"cleanliness":["modernbert_cleanliness","ad_en","rps_doc_frac_no_alph_words",
               "rps_lines_uppercase_letter_fraction"],
"structural_integrity":["rps_doc_frac_unique_words","rps_doc_frac_chars_top_2gram",
                        "rps_doc_frac_chars_top_3gram",
                        "rps_lines_ending_with_terminal_punctution_mark"]
}

def parse_args():
    p=argparse.ArgumentParser()
    here=Path(__file__).resolve().parent
    p.add_argument("--data-dir",type=Path,
                   default=here/"real_attachments"/"A_data_value")
    p.add_argument("--round1-dir",type=Path,default=here/"round1_outputs")
    p.add_argument("--out-dir",type=Path,default=here/"round2_outputs")
    p.add_argument("--winsor-low",type=float,default=.005)
    p.add_argument("--winsor-high",type=float,default=.995)
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--seed",type=int,default=20260922)
    p.add_argument("--human-audit-per-stratum",type=int,default=50)
    return p.parse_args()

def locate(root,name,pattern):
    p=root/name
    if p.exists(): return p
    hits=sorted(root.rglob(pattern))
    if len(hits)==1:return hits[0]
    if not hits: raise FileNotFoundError(f"Cannot find {name} under {root}")
    raise RuntimeError("Multiple matches:\n"+"\n".join(map(str,hits)))

def load(path,domain=None):
    print(f"Loading {path.name} ...")
    d=pd.read_json(path,lines=True,compression="xz")
    d=d.rename(columns={k:v for k,v in ALIASES.items() if k in d.columns})
    if "_source_domain" not in d and domain:
        d["_source_domain"]=domain
    miss=[c for c in QUALITY_COLS if c not in d]
    if miss: raise ValueError(f"{path.name}: missing columns {miss}")
    return d

def softmax(a):
    a=np.asarray(a,dtype=float)
    a=a-np.nanmax(a)
    e=np.exp(a)
    s=np.nansum(e)
    return e/s if s>0 else np.full_like(e,np.nan)

def transform_list(v):
    """
    Convert classifier outputs to one scalar.
    Assumption: vector positions are ordered class logits.
    - length 1: scalar
    - length 2: P(class=1)
    - length >=3: E[class index]/(K-1)
    """
    if not isinstance(v,(list,tuple,np.ndarray)):
        try:return float(v)
        except:return np.nan
    a=pd.to_numeric(pd.Series(v),errors="coerce").to_numpy(float)
    if len(a)==0 or np.all(~np.isfinite(a)):return np.nan
    if len(a)==1:return float(a[0])
    p=softmax(a)
    if len(a)==2:return float(p[1])
    return float(np.dot(p,np.arange(len(a)))/(len(a)-1))

def scalarize(raw,label):
    print(f"Scalarizing {label} ...")
    o=pd.DataFrame(index=raw.index)
    for c in tqdm(QUALITY_COLS,desc=f"{label}: 22 signals",unit="feature"):
        if c in LIST_COLS:
            o[c]=raw[c].map(transform_list)
        else:
            o[c]=pd.to_numeric(raw[c],errors="coerce")
    for c in ["id","sub_path","_source_domain"]:
        if c in raw:o[c]=raw[c].values
    return o

def fit_bounds(a1,loq,hiq):
    b={}
    for c in tqdm(QUALITY_COLS,desc="Fitting A1 reference bounds",unit="feature"):
        x=a1[c].replace([np.inf,-np.inf],np.nan)
        lo,hi=x.quantile([loq,hiq])
        b[c]=(float(lo),float(hi))
    return b

def normalize(d,bounds):
    z=pd.DataFrame(index=d.index)
    for c in tqdm(QUALITY_COLS,desc="Direction-aware normalization",unit="feature"):
        lo,hi=bounds[c]
        x=d[c].clip(lo,hi)
        if hi<=lo: zz=pd.Series(np.nan,index=d.index)
        else: zz=((x-lo)/(hi-lo)).clip(0,1)
        if DIRECTION[c]==-1: zz=1-zz
        z[c]=zz
    for c in ["id","sub_path","_source_domain"]:
        if c in d:z[c]=d[c].values
    return z

def critic_weights(z,features):
    """
    CRITIC:
    information_j = std_j * sum_k(1-|corr_jk|)
    using Spearman correlation for robustness.
    """
    X=z[features].copy()
    X=X.fillna(X.median())
    sd=X.std(ddof=1)
    corr=X.corr(method="spearman").abs()
    info=sd*(1-corr).sum(axis=1)
    if info.sum()<=0:
        w=pd.Series(1/len(features),index=features)
    else:w=info/info.sum()
    return pd.DataFrame({"feature":features,"std":sd[features],
                         "critic_information":info[features],
                         "weight":w[features]}).sort_values("weight",ascending=False)

def weighted_score(z,weights):
    feats=list(weights.index)
    X=z[feats].copy()
    # Row-wise renormalization handles rare missing signals without dropping records.
    mask=X.notna().astype(float)
    W=pd.Series(weights,index=feats)
    num=X.fillna(0).mul(W,axis=1).sum(axis=1)
    den=mask.mul(W,axis=1).sum(axis=1)
    return num/den.replace(0,np.nan)

def group_scores(z,weights):
    out=pd.DataFrame(index=z.index)
    for g,features in tqdm(GROUPS.items(),desc="Computing conflict groups",unit="group"):
        fs=[f for f in features if f in weights.index]
        w=weights.loc[fs]
        w=w/w.sum()
        out[g]=weighted_score(z,w)
    return out

def conflict_metrics(gs):
    # Range = intuitive "best dimension - worst dimension".
    c=pd.DataFrame(index=gs.index)
    c["conflict_range"]=gs.max(axis=1)-gs.min(axis=1)
    c["conflict_sd"]=gs.std(axis=1,ddof=0)
    c["weakest_group"]=gs.idxmin(axis=1)
    c["strongest_group"]=gs.idxmax(axis=1)
    return c

def choose_gamma(q0,conflict,groups):
    """
    Data-driven stability choice:
    Search gamma in [0,0.30]. Prefer the smallest gamma that creates a meaningful
    penalty for high-conflict samples while retaining >=0.95 Spearman rank
    correlation with Q0. This preserves the base quality ordering and avoids an
    arbitrary large correction.
    """
    rows=[]
    threshold=conflict.quantile(.90)
    high=conflict>=threshold
    low=conflict<=conflict.quantile(.50)
    for g in np.linspace(0,0.30,31):
        q=(q0-g*conflict).clip(0,1)
        rho=stats.spearmanr(q0,q,nan_policy="omit").statistic
        separation=(q0[high]-q[high]).mean()-(q0[low]-q[low]).mean()
        rows.append([g,rho,separation,q.mean(),q.std()])
    tab=pd.DataFrame(rows,columns=["gamma","spearman_Q0_Q","extra_penalty_high_vs_low",
                                   "Q_mean","Q_std"])
    feasible=tab[(tab.spearman_Q0_Q>=.95)&(tab.gamma>0)]
    if len(feasible):
        # Among stable candidates choose the one maximizing high-vs-low penalty.
        gamma=float(feasible.sort_values(["extra_penalty_high_vs_low","gamma"],
                                         ascending=[False,True]).iloc[0].gamma)
    else: gamma=.05
    return gamma,tab

def domain_summary(qdf):
    def one(g):
        x=g["Q"].dropna()
        return pd.Series({
            "n":len(g),"Q_mean":x.mean(),"Q_median":x.median(),"Q_std":x.std(),
            "Q_p10":x.quantile(.10),"Q_p90":x.quantile(.90),
            "conflict_rate":g["is_conflict"].mean(),
            "Q0_mean":g["Q0"].mean()
        })
    return qdf.groupby("_source_domain",dropna=False).apply(one).reset_index()

def bootstrap_diff(x,y,B,rng):
    """Difference y_mean - x_mean with percentile bootstrap CI."""
    x=np.asarray(pd.Series(x).dropna(),float)
    y=np.asarray(pd.Series(y).dropna(),float)
    obs=y.mean()-x.mean()
    vals=np.empty(B)
    for b in tqdm(range(B),desc="Bootstrap",leave=False,unit="rep"):
        vals[b]=rng.choice(y,len(y),replace=True).mean()-rng.choice(x,len(x),replace=True).mean()
    lo,hi=np.quantile(vals,[.025,.975])
    return obs,lo,hi

def save_hist(qdf,path):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(qdf["Q"].dropna(),bins=50)
    ax.set_xlabel("Quality score Q (0-100)")
    ax.set_ylabel("Documents")
    ax.set_title("A1 final quality-score distribution")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def save_domain_plot(ds,path):
    d=ds.sort_values("Q_mean")
    fig,ax=plt.subplots(figsize=(9,6))
    ax.errorbar(d.Q_mean,d._source_domain,xerr=d.Q_std,fmt="o",capsize=3)
    ax.set_xlabel("Domain quality Q: mean ± SD")
    ax.set_ylabel("Source domain")
    ax.set_title("A1 domain-level quality")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def save_conflict_plot(gs,path):
    c=gs.corr(method="spearman")
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(c,vmin=-1,vmax=1)
    ax.set_xticks(range(len(c)));ax.set_xticklabels(c.columns,rotation=45,ha="right")
    ax.set_yticks(range(len(c)));ax.set_yticklabels(c.index)
    ax.set_title("Quality-group Spearman correlations")
    fig.colorbar(im,ax=ax,label="Spearman correlation")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def audit_sample(raw,qdf,n_each,seed):
    """Stratified raw-content sample for blinded human validation."""
    if "content" not in raw:return pd.DataFrame()
    d=qdf[["Q","Q0","conflict_range","is_conflict"]].copy()
    d["content"]=raw["content"].values
    d["id"]=raw["id"].values if "id" in raw else np.arange(len(raw))
    d["domain"]=raw["_source_domain"].values
    q01,q10,q90,q99=d.Q.quantile([.01,.10,.90,.99])
    strata={
      "bottom_1pct":d[d.Q<=q01],
      "bottom_10pct":d[(d.Q>q01)&(d.Q<=q10)],
      "middle":d[(d.Q>=d.Q.quantile(.45))&(d.Q<=d.Q.quantile(.55))],
      "top_10pct":d[(d.Q>=q90)&(d.Q<q99)],
      "top_1pct":d[d.Q>=q99],
      "high_conflict":d[d.is_conflict]
    }
    parts=[]
    for name,x in strata.items():
        take=min(n_each,len(x))
        s=x.sample(take,random_state=seed).copy()
        s["stratum"]=name
        parts.append(s)
    o=pd.concat(parts,ignore_index=True)
    # Empty columns for blinded manual rating.
    o["human_information_value"]=""
    o["human_fluency"]=""
    o["human_cleanliness"]=""
    o["human_readability"]=""
    o["human_overall_1to5"]=""
    o["reviewer_notes"]=""
    return o

def main():
    a=parse_args()
    root=a.data_dir.resolve(); out=a.out_dir.resolve()
    tab=out/"tables";fig=out/"figures";pro=out/"processed";auditdir=out/"human_audit"
    for p in [tab,fig,pro,auditdir]:p.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed)

    # Check Round 1 result is present and summarize why Round 2 changes scalarization.
    r1p=a.round1_dir/"tables"/"04_scalarization_robustness.csv"
    if r1p.exists():
        r1=pd.read_csv(r1p)
        r1.to_csv(tab/"00_round1_scalarization_diagnostic.csv",index=False)
        print("Round 1 diagnostic loaded. Low/negative mean-vs-median correlations confirm")
        print("that multi-logit vectors should not be averaged as ordinary repeated scores.")

    paths={
      "A1":locate(root,"slimpajama_quality_signal_sample.jsonl.xz","*quality_signal_sample*.jsonl.xz"),
      "A2_arxiv":locate(root,"arxiv_part-6777d8857c6e-000486.jsonl.xz","arxiv*.jsonl.xz"),
      "A3_github":locate(root,"github_part-6777d8857c6e-000275.jsonl.xz","github*.jsonl.xz")}
    raw={"A1":load(paths["A1"]),"A2_arxiv":load(paths["A2_arxiv"],"arxiv"),
         "A3_github":load(paths["A3_github"],"github")}

    policy=[]
    for c in QUALITY_COLS:
        if c in LIST_COLS:
            # observed vector lengths
            vals=raw["A1"][c].dropna().head(1000)
            lens=sorted(set(len(v) for v in vals if isinstance(v,(list,tuple,np.ndarray))))
            transform="softmax P(class=1)" if lens==[2] else (
                "singleton scalar" if lens==[1] else "softmax expected ordinal class")
        else: transform="numeric scalar"
        policy.append([c,transform,DIRECTION[c],
                       {1:"higher-is-better",-1:"lower-is-better",0:"diagnostic/non-monotone"}[DIRECTION[c]],
                       c in MAIN_FEATURES])
    pd.DataFrame(policy,columns=["feature","scalar_transform","direction_code",
        "direction_interpretation","included_in_main_Q"]).to_csv(
        tab/"01_signal_transform_policy.csv",index=False)

    scalar={k:scalarize(v,k) for k,v in raw.items()}
    bounds=fit_bounds(scalar["A1"],a.winsor_low,a.winsor_high)
    z={k:normalize(v,bounds) for k,v in scalar.items()}

    print("Computing CRITIC weights ...")
    cw=critic_weights(z["A1"],MAIN_FEATURES)
    cw.to_csv(tab/"02_CRITIC_weights.csv",index=False)
    weights=cw.set_index("feature")["weight"]

    results={}; group_all={}
    # Compute base Q/group conflict for each dataset.
    for name in tqdm(list(z.keys()),desc="Quality scoring datasets",unit="dataset"):
        q0=weighted_score(z[name],weights)
        gs=group_scores(z[name],weights)
        cm=conflict_metrics(gs)
        group_all[name]=gs
        results[name]=pd.concat([q0.rename("Q0_unit"),gs,cm],axis=1)
        if "_source_domain" in z[name]:
            results[name]["_source_domain"]=z[name]["_source_domain"].values

    # Conflict threshold/gamma are FIT ON A1 ONLY, then frozen for A2/A3.
    conflict_threshold=float(results["A1"]["conflict_range"].quantile(.90))
    gamma,gamma_tab=choose_gamma(results["A1"]["Q0_unit"],
                                 results["A1"]["conflict_range"],
                                 group_all["A1"])
    gamma_tab.to_csv(tab/"03_gamma_sensitivity.csv",index=False)
    print(f"Selected conflict threshold (A1 P90): {conflict_threshold:.4f}")
    print(f"Selected gamma: {gamma:.3f}")

    for name,r in results.items():
        r["is_conflict"]=r["conflict_range"]>=conflict_threshold
        r["Q0"]=100*r["Q0_unit"]
        r["Q"]=100*(r["Q0_unit"]-gamma*r["conflict_range"]).clip(0,1)
        r.drop(columns=["Q0_unit"],inplace=True)

    # Domain aggregation on A1.
    ds=domain_summary(results["A1"])
    ds.to_csv(tab/"04_A1_domain_quality.csv",index=False)

    # Conflict type table.
    ctype=(results["A1"].loc[results["A1"].is_conflict]
           .groupby(["strongest_group","weakest_group"]).size()
           .reset_index(name="n").sort_values("n",ascending=False))
    ctype["fraction_of_conflicts"]=ctype.n/ctype.n.sum()
    ctype.to_csv(tab/"05_conflict_types.csv",index=False)

    # Extended validation: same fitted transform/bounds/weights/gamma.
    ext_rows=[]
    for domain,extname in [("arxiv","A2_arxiv"),("github","A3_github")]:
        sample=results["A1"][results["A1"]["_source_domain"].astype(str).str.lower()==domain]
        ext=results[extname]
        obs,lo,hi=bootstrap_diff(sample.Q,ext.Q,a.bootstrap,rng)
        ks=stats.ks_2samp(sample.Q.dropna(),ext.Q.dropna())
        ext_rows.append({
          "domain":domain,"n_A1_sample":len(sample),"n_extended":len(ext),
          "Q_A1_mean":sample.Q.mean(),"Q_extended_mean":ext.Q.mean(),
          "delta_extended_minus_sample":obs,"bootstrap_CI_low":lo,"bootstrap_CI_high":hi,
          "KS_statistic":ks.statistic,"KS_pvalue":ks.pvalue,
          "conflict_rate_A1":sample.is_conflict.mean(),
          "conflict_rate_extended":ext.is_conflict.mean()})
    extval=pd.DataFrame(ext_rows)
    extval.to_csv(tab/"06_extended_quality_validation.csv",index=False)

    # Sensitivity: CRITIC vs equal-weight score.
    eq=pd.Series(1/len(MAIN_FEATURES),index=MAIN_FEATURES)
    qeq=100*weighted_score(z["A1"],eq)
    sens=pd.DataFrame({
      "Q_CRITIC_before_conflict":results["A1"]["Q0"],
      "Q_equal_weight":qeq})
    rho=stats.spearmanr(sens.iloc[:,0],sens.iloc[:,1],nan_policy="omit").statistic
    sens.describe().to_csv(tab/"07_weighting_sensitivity_summary.csv")
    pd.DataFrame([{"spearman_CRITIC_vs_equal":rho}]).to_csv(
        tab/"08_weighting_rank_stability.csv",index=False)

    # Save compact per-document A1 scores (no content duplicated here).
    keep=["Q0","Q","conflict_range","conflict_sd","is_conflict",
          "strongest_group","weakest_group","_source_domain"]+list(GROUPS.keys())
    a1out=results["A1"][keep].copy()
    if "id" in raw["A1"]:a1out.insert(0,"id",raw["A1"]["id"].values)
    a1out.to_csv(pro/"A1_quality_scores.csv.gz",index=False,compression="gzip")

    # Extended domain scores.
    for name in ["A2_arxiv","A3_github"]:
        results[name][["Q0","Q","conflict_range","conflict_sd","is_conflict"]+
                      list(GROUPS.keys())].to_csv(
            pro/f"{name}_quality_scores.csv.gz",index=False,compression="gzip")

    # Human audit sample with raw content.
    hs=audit_sample(raw["A1"],results["A1"],a.human_audit_per_stratum,a.seed)
    hs.to_csv(auditdir/"A1_blind_human_audit_sample.csv",index=False,encoding="utf-8-sig")

    # Figures.
    save_hist(results["A1"],fig/"01_A1_Q_distribution.png")
    save_domain_plot(ds,fig/"02_A1_domain_quality.png")
    save_conflict_plot(group_all["A1"],fig/"03_quality_group_correlation.png")

    # Q0 vs Q.
    f,ax=plt.subplots(figsize=(6,6))
    ax.scatter(results["A1"].Q0,results["A1"].Q,s=4,alpha=.15)
    ax.set_xlabel("Base quality Q0");ax.set_ylabel("Conflict-corrected Q")
    ax.set_title(f"Conflict correction (gamma={gamma:.2f})")
    f.tight_layout();f.savefig(fig/"04_Q0_vs_Q.png",dpi=220);plt.close(f)

    # Extended comparison.
    f,ax=plt.subplots(figsize=(7,5))
    xx=np.arange(len(extval));w=.35
    ax.bar(xx-w/2,extval.Q_A1_mean,w,label="A1 sample")
    ax.bar(xx+w/2,extval.Q_extended_mean,w,label="Extended")
    ax.set_xticks(xx);ax.set_xticklabels(extval.domain)
    ax.set_ylabel("Mean Q");ax.set_title("Sample vs extended domain quality")
    ax.legend();f.tight_layout();f.savefig(fig/"05_sample_vs_extended_Q.png",dpi=220);plt.close(f)

    config={"data_files":{k:str(v) for k,v in paths.items()},
      "main_features":MAIN_FEATURES,"groups":GROUPS,
      "winsor":[a.winsor_low,a.winsor_high],"conflict_threshold":conflict_threshold,
      "gamma":gamma,"bootstrap":a.bootstrap,"seed":a.seed}
    (out/"run_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),
                                      encoding="utf-8")

    summary=f"""ROUND 2 QUALITY + CONFLICT SUMMARY

A1 rows: {len(raw["A1"]):,}
A2 arxiv rows: {len(raw["A2_arxiv"]):,}
A3 github rows: {len(raw["A3_github"]):,}

IMPORTANT:
Round 1 showed that arithmetic mean is not appropriate for several multi-logit
list fields. Round 2 re-reads raw data and converts logits via softmax-based scores.

Main Q features: {len(MAIN_FEATURES)}
CRITIC/equal-weight rank Spearman: {rho:.4f}
A1 P90 conflict threshold: {conflict_threshold:.4f}
Selected gamma: {gamma:.3f}
A1 conflict rate: {results["A1"].is_conflict.mean():.4%}

DOMAIN QUALITY
{ds.to_string(index=False)}

EXTENDED VALIDATION
{extval.to_string(index=False)}

NEXT:
1. Inspect 01_signal_transform_policy.csv.
2. Inspect 02_CRITIC_weights.csv and 03_gamma_sensitivity.csv.
3. Review human_audit/A1_blind_human_audit_sample.csv blindly.
4. After manual ratings are filled, calculate human-Q Spearman and inter-rater agreement.
5. Use A16 mapping to transfer domain Q into the 17-domain mixture model in Round 3.
"""
    (out/"round2_summary.txt").write_text(summary,encoding="utf-8")
    print("\n"+summary)
    print(f"\nDone. Outputs -> {out}")

if __name__=="__main__":
    main()
