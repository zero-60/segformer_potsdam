#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 - Round 1 complete experiment (A1/A2/A3)

Goals
-----
1. Load all A1/A2/A3 records.
2. Convert the 8 list-valued quality signals to scalar values (mean as main,
   median as robustness check).
3. Audit missingness, zeros, tails, and list lengths.
4. Winsorize using A1 reference quantiles and apply the same bounds to A2/A3.
5. Apply only defensible provisional quality directions. Ambiguous statistical
   features are marked "diagnose" rather than forced positive/negative.
6. Produce Spearman-correlation and PCA diagnostics.
7. Compare arxiv/github A1 samples against A2/A3 full extended sets with
   KS tests, standardized mean differences, and BH-FDR correction.
8. Save publication-ready tables/figures and processed data.

Run
---
python problem1_round1_experiment.py --data-dir /path/to/files

Dependencies
------------
numpy pandas scipy scikit-learn matplotlib
"""

from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# +1 positive; -1 negative; 0 deliberately left non-monotonic/diagnostic.
DIRECTION={
"fineweb_edu":1,"fluency_en":1,"modernbert_cleanliness":1,
"modernbert_readability":1,"modernbert_reasoning":1,
"modernbert_professionalism":1,"dsir_books":1,"dsir_wiki":1,"dsir_math":1,
"qurater":1,"ad_en":-1,"rps_doc_word_count":0,"rps_doc_num_sentences":0,
"rps_doc_unigram_entropy":0,"rps_doc_frac_unique_words":0,
"rps_doc_frac_no_alph_words":-1,"rps_doc_frac_chars_top_2gram":-1,
"rps_doc_frac_chars_top_3gram":-1,"rps_lines_uppercase_letter_fraction":-1,
"rps_lines_ending_with_terminal_punctution_mark":1,
"rps_lines_numerical_chars_fraction":0,"rps_doc_mean_word_length":0}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,default=Path("."))
    p.add_argument("--out-dir",type=Path,default=Path("round1_outputs"))
    p.add_argument("--winsor-low",type=float,default=.005)
    p.add_argument("--winsor-high",type=float,default=.995)
    return p.parse_args()

def locate(root,name,pattern):
    p=root/name
    if p.exists(): return p
    hits=sorted(root.glob(pattern))
    if len(hits)==1:return hits[0]
    raise FileNotFoundError(f"Cannot uniquely locate {name}; matches={hits}")

def load(path,domain=None):
    d=pd.read_json(path,lines=True,compression="xz")
    d=d.rename(columns={k:v for k,v in ALIASES.items() if k in d.columns})
    if "_source_domain" not in d and domain:d["_source_domain"]=domain
    miss=[c for c in QUALITY_COLS if c not in d]
    if miss:raise ValueError(f"{path.name}: missing {miss}")
    return d

def reduce_value(v,method="mean"):
    if isinstance(v,(list,tuple,np.ndarray)):
        a=pd.to_numeric(pd.Series(v),errors="coerce").dropna().to_numpy(float)
        if not len(a):return np.nan
        return float(np.mean(a) if method=="mean" else np.median(a))
    try:return float(v)
    except:return np.nan

def scalarize(d,method="mean"):
    o=pd.DataFrame(index=d.index)
    for c in QUALITY_COLS:
        if c in LIST_COLS:o[c]=d[c].map(lambda v:reduce_value(v,method))
        else:o[c]=pd.to_numeric(d[c],errors="coerce")
    for c in ["id","sub_path","_source_domain"]:
        if c in d:o[c]=d[c].values
    return o

def list_lengths(d,label):
    rows=[]
    for c in LIST_COLS:
        lens=d[c].map(lambda v:len(v) if isinstance(v,(list,tuple,np.ndarray)) else (0 if pd.isna(v) else 1))
        for ln,n in lens.value_counts(dropna=False).sort_index().items():
            rows.append([label,c,int(ln),int(n),float(n/len(d))])
    return pd.DataFrame(rows,columns=["dataset","feature","list_length","n","fraction"])

def audit(d,label):
    rows=[]
    for c in QUALITY_COLS:
        x=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        y=x.dropna()
        rows.append(dict(dataset=label,feature=c,n=len(x),n_valid=len(y),
            missing_rate=x.isna().mean(),zero_rate=(y==0).mean() if len(y) else np.nan,
            mean=y.mean(),std=y.std(),min=y.min(),p01=y.quantile(.01),
            p05=y.quantile(.05),median=y.median(),p95=y.quantile(.95),
            p99=y.quantile(.99),max=y.max(),n_unique=y.nunique(),
            direction={1:"positive",-1:"negative",0:"diagnose"}[DIRECTION[c]]))
    return pd.DataFrame(rows)

def winsor(train,allsets,loq,hiq):
    bounds={}; reports=[]
    out={k:v.copy() for k,v in allsets.items()}
    for c in QUALITY_COLS:
        lo,hi=train[c].quantile([loq,hiq])
        bounds[c]=(float(lo),float(hi))
        for name,d in out.items():
            x=d[c]
            reports.append([name,c,lo,hi,(x<lo).mean(),(x>hi).mean()])
            d[c]=x.clip(lo,hi)
    rep=pd.DataFrame(reports,columns=["dataset","feature","A1_lower","A1_upper",
                                      "frac_below","frac_above"])
    rep["frac_clipped"]=rep.frac_below+rep.frac_above
    return out,bounds,rep

def scale(allsets,bounds):
    out={}
    for name,d in allsets.items():
        z=pd.DataFrame(index=d.index)
        for c in QUALITY_COLS:
            lo,hi=bounds[c]
            v=((d[c]-lo)/(hi-lo)).clip(0,1) if hi>lo else np.nan
            if DIRECTION[c]==-1:v=1-v
            z[c]=v
        for c in ["id","sub_path","_source_domain"]:
            if c in d:z[c]=d[c].values
        out[name]=z
    return out

def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); r=p[order]
    q=np.minimum.accumulate((r*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    ans=np.empty_like(q);ans[order]=np.clip(q,0,1);return ans

def shift(a1,ext,domain):
    base=a1[a1["_source_domain"].astype(str).str.lower()==domain]
    rows=[]
    for c in QUALITY_COLS:
        x=base[c].dropna();y=ext[c].dropna()
        k=stats.ks_2samp(x,y)
        pooled=np.sqrt((x.var()+y.var())/2)
        rows.append([domain,c,len(x),len(y),x.mean(),y.mean(),
                     (y.mean()-x.mean())/pooled if pooled>0 else np.nan,
                     k.statistic,k.pvalue])
    o=pd.DataFrame(rows,columns=["domain","feature","n_sample","n_extended",
        "sample_mean","extended_mean","standardized_mean_difference",
        "ks_statistic","ks_pvalue"])
    o["ks_fdr_bh"]=bh(o.ks_pvalue)
    return o

def corrplot(corr,path):
    fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(corr,vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(22));ax.set_xticklabels(corr.columns,rotation=90,fontsize=7)
    ax.set_yticks(range(22));ax.set_yticklabels(corr.index,fontsize=7)
    ax.set_title("A1 quality signals: Spearman correlation")
    fig.colorbar(im,ax=ax,label="Spearman correlation")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def shiftplot(s,path,domain):
    q=s.sort_values("ks_statistic")
    fig,ax=plt.subplots(figsize=(9,8))
    ax.barh(q.feature,q.ks_statistic)
    ax.set_xlabel("KS statistic")
    ax.set_title(f"{domain}: A1 sample vs extended-set shift")
    ax.tick_params(axis="y",labelsize=7)
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)

def main():
    a=args(); root=a.data_dir.resolve();out=a.out_dir.resolve()
    tab=out/"tables";fig=out/"figures";pro=out/"processed"
    for x in [tab,fig,pro]:x.mkdir(parents=True,exist_ok=True)

    paths={
      "A1":locate(root,"slimpajama_quality_signal_sample.jsonl.xz","*quality_signal_sample*.jsonl.xz"),
      "A2_arxiv":locate(root,"arxiv_part-6777d8857c6e-000486.jsonl.xz","arxiv*.jsonl.xz"),
      "A3_github":locate(root,"github_part-6777d8857c6e-000275.jsonl.xz","github*.jsonl.xz")}
    raw={"A1":load(paths["A1"]),"A2_arxiv":load(paths["A2_arxiv"],"arxiv"),
         "A3_github":load(paths["A3_github"],"github")}

    schema=pd.DataFrame([dict(dataset=k,rows=len(v),columns=len(v.columns),
        has_content="content" in v,has_source_domain="_source_domain" in v)
        for k,v in raw.items()])
    schema.to_csv(tab/"01_schema_summary.csv",index=False)
    raw["A1"]["_source_domain"].value_counts().rename_axis("domain").reset_index(name="n").to_csv(
        tab/"02_A1_domain_counts.csv",index=False)

    pd.concat([list_lengths(v,k) for k,v in raw.items()]).to_csv(
        tab/"03_list_field_lengths.csv",index=False)

    mean={k:scalarize(v,"mean") for k,v in raw.items()}
    med={k:scalarize(v,"median") for k,v in raw.items()}
    rr=[]
    for k in mean:
        for c in LIST_COLS:
            ok=mean[k][c].notna()&med[k][c].notna()
            rho=stats.spearmanr(mean[k].loc[ok,c],med[k].loc[ok,c]).statistic if ok.sum()>2 else np.nan
            rr.append([k,c,ok.sum(),rho,(mean[k].loc[ok,c]-med[k].loc[ok,c]).abs().mean()])
    pd.DataFrame(rr,columns=["dataset","feature","n","spearman_mean_vs_median",
        "mean_absolute_difference"]).to_csv(tab/"04_scalarization_robustness.csv",index=False)

    audits=pd.concat([audit(v,k) for k,v in mean.items()])
    audits.to_csv(tab/"05_feature_audit.csv",index=False)

    clipped,bounds,rep=winsor(mean["A1"],mean,a.winsor_low,a.winsor_high)
    rep.to_csv(tab/"06_winsorization_report.csv",index=False)
    z=scale(clipped,bounds)

    policy=pd.DataFrame({"feature":QUALITY_COLS,"direction_code":[DIRECTION[c] for c in QUALITY_COLS]})
    policy["direction"]=policy.direction_code.map({1:"positive",-1:"negative",0:"diagnose"})
    policy.to_csv(tab/"07_direction_policy.csv",index=False)

    corr=z["A1"][QUALITY_COLS].corr(method="spearman")
    corr.to_csv(tab/"08_A1_spearman.csv");corrplot(corr,fig/"01_A1_correlation.png")

    X=z["A1"][QUALITY_COLS].copy()
    X=X.fillna(X.median())
    pc=PCA().fit(StandardScaler().fit_transform(X))
    ev=pc.explained_variance_ratio_
    evdf=pd.DataFrame({"PC":np.arange(1,23),"explained_variance_ratio":ev,
                       "cumulative":np.cumsum(ev)})
    evdf.to_csv(tab/"09_pca_explained_variance.csv",index=False)
    pd.DataFrame(pc.components_.T,index=QUALITY_COLS,
        columns=[f"PC{i}" for i in range(1,23)]).to_csv(tab/"10_pca_loadings.csv")
    f,a1=plt.subplots(figsize=(8,5));xx=np.arange(1,23)
    a1.plot(xx,ev,marker="o",label="Individual");a1.plot(xx,np.cumsum(ev),marker=".",label="Cumulative")
    a1.axhline(.8,ls="--");a1.axhline(.9,ls="--");a1.set_xlabel("Principal component")
    a1.set_ylabel("Explained variance ratio");a1.set_title("PCA diagnostic");a1.legend()
    f.tight_layout();f.savefig(fig/"02_PCA_scree.png",dpi=220);plt.close(f)

    sa=shift(mean["A1"],mean["A2_arxiv"],"arxiv")
    sg=shift(mean["A1"],mean["A3_github"],"github")
    sa.to_csv(tab/"11_arxiv_sample_vs_extended.csv",index=False)
    sg.to_csv(tab/"12_github_sample_vs_extended.csv",index=False)
    shiftplot(sa,fig/"03_arxiv_shift.png","arxiv")
    shiftplot(sg,fig/"04_github_shift.png","github")

    # Save only A1 processed matrices by default; avoids duplicating huge A3 in outputs.
    mean["A1"].to_csv(pro/"A1_scalar_mean.csv.gz",index=False,compression="gzip")
    z["A1"].to_csv(pro/"A1_scaled.csv.gz",index=False,compression="gzip")

    n80=int(np.argmax(np.cumsum(ev)>=.8)+1);n90=int(np.argmax(np.cumsum(ev)>=.9)+1)
    summary=[
      "# Round 1 experiment summary","",
      "## Dataset dimensions",schema.to_string(index=False),"",
      f"## PCA diagnostic\nComponents for >=80% variance: {n80}\nComponents for >=90% variance: {n90}","",
      "## Largest arxiv shifts",
      sa.nlargest(5,"ks_statistic")[["feature","ks_statistic","standardized_mean_difference","ks_fdr_bh"]].to_string(index=False),"",
      "## Largest github shifts",
      sg.nlargest(5,"ks_statistic")[["feature","ks_statistic","standardized_mean_difference","ks_fdr_bh"]].to_string(index=False),"",
      "## Interpretation rule",
      "PCA and direction assignments here are diagnostics, not the final Q model. "
      "Ambiguous statistical indicators remain marked 'diagnose'. The next experiment "
      "should use these results to choose the Q weighting scheme and conflict groups."
    ]
    (out/"round1_summary.txt").write_text("\n".join(summary),encoding="utf-8")
    (out/"run_config.json").write_text(json.dumps({
        "files":{k:str(v) for k,v in paths.items()},"winsor":[a.winsor_low,a.winsor_high],
        "quality_columns":QUALITY_COLS,"list_columns":LIST_COLS,"direction":DIRECTION
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(schema.to_string(index=False))
    print(f"\nFinished. Outputs: {out}")

if __name__=="__main__":main()
