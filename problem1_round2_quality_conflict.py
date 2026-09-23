#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 1 - Round 2 experiment
Quality Index Q + Conflict Detection/Resolution + Extended-set validation + Human review sheet

This script is designed for the user's project layout:

SXJM/
├─ problem1_round2_quality_conflict.py
├─ round1_outputs/
└─ real_attachments/
   └─ A_data_value/
      ├─ slimpajama_quality_signal_sample.jsonl.xz
      └─ slimpajama_quality_extended/
         ├─ arxiv*.jsonl.xz
         └─ github*.jsonl.xz

Round-1 evidence used by this design
------------------------------------
1) PCA needed 9 PCs for >=80% and 12 PCs for >=90% variance: the 22 signals are
   multi-dimensional, so a single first-PC score is not appropriate.
2) dsir_books/wiki/math are almost perfectly correlated in A1: naive equal weighting
   would triple-count one latent signal. CRITIC redundancy correction is therefore useful.
3) A1 vs A2/A3 distribution shifts were small after FDR correction, supporting use of
   A1-derived preprocessing/weights on the extended sets.
4) Mean-vs-median scalarization was unstable for several ModernBERT list fields.
   Therefore this script keeps the Round-1 main convention (mean) and reports a
   median-based sensitivity Q rather than silently replacing the main convention.

Main model
----------
Step A. Explicit semantic signals form an anchor quality score.
Step B. Ambiguous statistical features are converted to "desirability" using the
        empirical distribution of high-anchor-quality A1 texts. This uses all 22 signals
        without pretending that length/entropy/etc. are globally monotonic.
Step C. CRITIC objective weights account for variability and redundancy.
Step D. Group balancing prevents one highly correlated signal family from dominating.
Step E. Conflict is defined between interpretable quality dimensions.
Step F. Final Q = base quality - conflict penalty - severe-shortfall penalty.
Step G. Apply the A1-fitted model unchanged to A2/A3 and compare sample vs extended domains.

Outputs
-------
round2_outputs/
  tables/
  figures/
  processed/
  human_review/
  round2_summary.txt
  round2_model.json

Dependencies
------------
numpy pandas scipy matplotlib

Run
---
python problem1_round2_quality_conflict.py

Optional:
python problem1_round2_quality_conflict.py --data-dir ".../A_data_value" \
    --round1-dir ".../round1_outputs" --out-dir ".../round2_outputs"
"""

from __future__ import annotations
import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm            # <--- 新增
tqdm.pandas()

warnings.filterwarnings("ignore", category=RuntimeWarning)

QUALITY_COLS = [
    "fineweb_edu","fluency_en","modernbert_cleanliness","modernbert_readability",
    "modernbert_reasoning","modernbert_professionalism","dsir_books","dsir_wiki",
    "dsir_math","qurater","ad_en","rps_doc_word_count","rps_doc_num_sentences",
    "rps_doc_unigram_entropy","rps_doc_frac_unique_words","rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram","rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction","rps_doc_mean_word_length"
]

LIST_COLS = [
    "fluency_en","modernbert_cleanliness","qurater","ad_en",
    "modernbert_reasoning","fineweb_edu","modernbert_professionalism",
    "modernbert_readability"
]

# Clear monotonic semantics. These create the anchor and remain monotonic in final Q.
POSITIVE = [
    "fineweb_edu","fluency_en","modernbert_cleanliness","modernbert_readability",
    "modernbert_reasoning","modernbert_professionalism",
    "dsir_books","dsir_wiki","dsir_math","qurater",
    "rps_doc_frac_unique_words",
    "rps_lines_ending_with_terminal_punctution_mark"
]
NEGATIVE = [
    "ad_en","rps_doc_frac_no_alph_words","rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram","rps_lines_uppercase_letter_fraction"
]
# These are not assumed to be "the larger/smaller the better".
AMBIGUOUS = [
    "rps_doc_word_count","rps_doc_num_sentences","rps_doc_unigram_entropy",
    "rps_lines_numerical_chars_fraction","rps_doc_mean_word_length"
]

# Interpretable conflict dimensions. Every signal is used exactly once here.
GROUPS = {
    "semantic_value": [
        "fineweb_edu","modernbert_reasoning","modernbert_professionalism",
        "dsir_books","dsir_wiki","dsir_math","qurater"
    ],
    "language_readability": [
        "fluency_en","modernbert_readability",
        "rps_lines_ending_with_terminal_punctution_mark",
        "rps_doc_mean_word_length"
    ],
    "cleanliness": [
        "modernbert_cleanliness","ad_en","rps_doc_frac_no_alph_words",
        "rps_doc_frac_chars_top_2gram","rps_doc_frac_chars_top_3gram",
        "rps_lines_uppercase_letter_fraction"
    ],
    "information_structure": [
        "rps_doc_word_count","rps_doc_num_sentences","rps_doc_unigram_entropy",
        "rps_doc_frac_unique_words","rps_lines_numerical_chars_fraction"
    ],
}

ALIASES = {
    "rps_lines_ending_with_terminal_punctuation_mark":
        "rps_lines_ending_with_terminal_punctution_mark"
}


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=here / "real_attachments" / "A_data_value")
    p.add_argument("--round1-dir", type=Path, default=here / "round1_outputs")
    p.add_argument("--out-dir", type=Path, default=here / "round2_outputs")
    p.add_argument("--anchor-top", type=float, default=0.20,
                   help="Top fraction of A1 anchor score used to learn desirable ranges.")
    p.add_argument("--conflict-quantile", type=float, default=0.90)
    p.add_argument("--gamma", type=float, default=0.15,
                   help="Conflict penalty strength.")
    p.add_argument("--eta", type=float, default=0.10,
                   help="Severe group-shortfall penalty strength.")
    p.add_argument("--shortfall-threshold", type=float, default=0.30)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def locate(root: Path, exact: str, pattern: str) -> Path:
    p = root / exact
    if p.exists():
        return p
    hits = sorted(root.rglob(pattern))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"Cannot find {exact} under {root}; pattern={pattern}")
    raise RuntimeError("Multiple matches:\n" + "\n".join(map(str, hits)))


def load_json(path: Path, domain=None):
    d = pd.read_json(path, lines=True, compression="xz")
    d = d.rename(columns={k:v for k,v in ALIASES.items() if k in d.columns})
    if "_source_domain" not in d and domain:
        d["_source_domain"] = domain
    miss = [c for c in QUALITY_COLS if c not in d]
    if miss:
        raise ValueError(f"{path.name} missing quality columns: {miss}")
    return d


def scalar(v, method="mean"):
    if isinstance(v, (list, tuple, np.ndarray)):
        a = pd.to_numeric(pd.Series(v), errors="coerce").dropna().to_numpy(float)
        if not len(a):
            return np.nan
        return float(np.mean(a) if method == "mean" else np.median(a))
    try:
        return float(v)
    except Exception:
        return np.nan


# 将原来的 scalarize 定义修改为如下代码：
def scalarize(d, name="Dataset", method="mean"):
    o = pd.DataFrame(index=d.index)
    for c in QUALITY_COLS:
        if c in LIST_COLS:
            # 原来的 d[c].map 替换为 d[c].progress_map，并增加 desc 描述提示
            o[c] = d[c].progress_map(lambda v: scalar(v, method), desc=f"{name} - {c}")
        else:
            o[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["id","sub_path","_source_domain"]:
        if c in d:
            o[c] = d[c].values
    return o


def load_round1_bounds(round1_dir: Path):
    p = round1_dir / "tables" / "06_winsorization_report.csv"
    if not p.exists():
        raise FileNotFoundError(f"Round-1 file missing: {p}")
    d = pd.read_csv(p)
    # One row per dataset/feature; use the A1 row where possible.
    if "dataset" in d and (d["dataset"] == "A1").any():
        d = d[d["dataset"] == "A1"]
    d = d.drop_duplicates("feature")
    return {
        r.feature: (float(r.A1_lower), float(r.A1_upper))
        for r in d.itertuples()
    }


def scale_monotonic(d: pd.DataFrame, bounds: dict):
    """A1-fitted robust min-max. Ambiguous features are only scaled, not directed."""
    z = pd.DataFrame(index=d.index)
    for c in QUALITY_COLS:
        lo, hi = bounds[c]
        x = pd.to_numeric(d[c], errors="coerce").clip(lo, hi)
        v = ((x - lo) / (hi - lo)).clip(0, 1) if hi > lo else pd.Series(np.nan, index=d.index)
        if c in NEGATIVE:
            v = 1 - v
        z[c] = v
    return z


def initial_anchor(z: pd.DataFrame):
    """
    Anchor only from explicit monotonic signals. Equal-weight at this stage is intentional:
    it only identifies a high-quality reference subset; final Q uses CRITIC/group balancing.
    """
    cols = POSITIVE + NEGATIVE
    return z[cols].mean(axis=1, skipna=True)


def fit_ambiguous_desirability(raw_a1: pd.DataFrame, anchor: pd.Series, top_frac=.20):
    """
    Learn the desirable interval for ambiguous statistics from high-anchor A1 texts.
    We use Q25/Q50/Q75 of the top-anchor subset and a robust Tukey-style scale.
    Desirability is 1 near the reference median and smoothly decreases outside.
    """
    cutoff = anchor.quantile(1 - top_frac)
    ref = raw_a1.loc[anchor >= cutoff]
    model = {}
    for c in AMBIGUOUS:
        x = pd.to_numeric(ref[c], errors="coerce").dropna()
        q25, med, q75 = x.quantile([.25,.50,.75])
        iqr = max(float(q75-q25), 1e-12)
        model[c] = {
            "q25": float(q25), "median": float(med), "q75": float(q75),
            "iqr": float(iqr),
            # scale chosen so values one IQR from median remain moderately desirable
            "scale": float(iqr)
        }
    return model, float(cutoff)


def apply_desirability(z: pd.DataFrame, raw: pd.DataFrame, model: dict):
    """
    Replace ambiguous raw scaled values with two-sided Gaussian desirability:
      exp(-0.5 * ((x-median)/IQR)^2)
    Monotonic features retain their direction-aware robust min-max values.
    """
    out = z.copy()
    for c, m in model.items():
        x = pd.to_numeric(raw[c], errors="coerce")
        s = max(m["scale"], 1e-12)
        out[c] = np.exp(-0.5 * ((x - m["median"]) / s) ** 2)
    return out.clip(0, 1)


def impute_from_a1(a1: pd.DataFrame, datasets: dict):
    med = a1[QUALITY_COLS].median()
    return {k: v[QUALITY_COLS].fillna(med) for k,v in datasets.items()}, med


def critic_weights(X: pd.DataFrame):
    """
    CRITIC:
      C_j = sigma_j * sum_k(1 - |rho_jk|)
    Spearman correlation is used because the signals are non-Gaussian and bounded.
    """
    sd = X.std(ddof=1)
    corr = X.corr(method="spearman").abs()
    info = sd * (1 - corr).sum(axis=1)
    if info.sum() <= 0:
        return pd.Series(1/len(info), index=info.index)
    return info / info.sum()


def group_balanced_weights(critic: pd.Series):
    """
    Equal total mass per conceptual group; CRITIC allocates weight within each group.
    This avoids the near-duplicate DSIR family dominating the final Q.
    """
    w = pd.Series(0.0, index=QUALITY_COLS)
    group_mass = 1.0 / len(GROUPS)
    for g, cols in GROUPS.items():
        local = critic[cols].copy()
        if local.sum() <= 0:
            local[:] = 1 / len(cols)
        else:
            local /= local.sum()
        w[cols] = group_mass * local
    return w / w.sum()


def score_components(X: pd.DataFrame, weights: pd.Series):
    base = X[QUALITY_COLS].mul(weights, axis=1).sum(axis=1)
    gs = pd.DataFrame(index=X.index)
    for g, cols in GROUPS.items():
        local = weights[cols] / weights[cols].sum()
        gs[g] = X[cols].mul(local, axis=1).sum(axis=1)
    return base, gs


def conflict_metrics(gs: pd.DataFrame):
    # Range captures "one dimension high while another is low".
    c_range = gs.max(axis=1) - gs.min(axis=1)
    # Dispersion is a complementary smooth conflict measure.
    c_sd = gs.std(axis=1, ddof=0)
    # Main conflict score normalized to [0,1] by theoretical range.
    conflict = c_range.clip(0, 1)
    return conflict, c_sd


def final_quality(base, gs, gamma, eta, threshold):
    conflict, _ = conflict_metrics(gs)
    shortfall = (threshold - gs.min(axis=1)).clip(lower=0)
    raw = base - gamma * conflict - eta * shortfall
    # Keep a fixed interpretable 0-100 scale rather than re-minmaxing each dataset.
    q = (100 * raw.clip(0, 1)).rename("Q")
    return q, conflict.rename("conflict_score"), shortfall.rename("shortfall_penalty_basis")


def bootstrap_ci(x, B=1000, seed=2026):
    x = pd.Series(x).dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = rng.choice(x, size=len(x), replace=True).mean()
    return tuple(np.quantile(vals, [.025,.975]))


def domain_summary(meta: pd.DataFrame, q: pd.Series, conflict: pd.Series,
                   threshold: float, B: int, seed: int):
    d = pd.DataFrame({
        "domain": meta["_source_domain"].astype(str).values,
        "Q": q.values,
        "conflict": conflict.values
    })
    rows = []
    for domain, g in d.groupby("domain"):
        lo, hi = bootstrap_ci(g.Q, B, seed)
        rows.append({
            "domain": domain, "n": len(g),
            "Q_mean": g.Q.mean(), "Q_median": g.Q.median(),
            "Q_std": g.Q.std(), "Q_p10": g.Q.quantile(.10),
            "Q_p90": g.Q.quantile(.90),
            "Q_mean_ci_low": lo, "Q_mean_ci_high": hi,
            "conflict_mean": g.conflict.mean(),
            "conflict_rate": (g.conflict > threshold).mean()
        })
    return pd.DataFrame(rows).sort_values("Q_mean", ascending=False)


def sample_vs_extended(a1_meta, a1_q, a1_conf, ext_meta, ext_q, ext_conf,
                       domain, conflict_threshold, B, seed):
    mask = a1_meta["_source_domain"].astype(str).str.lower() == domain.lower()
    a = pd.DataFrame({"Q":a1_q[mask].values, "conflict":a1_conf[mask].values})
    e = pd.DataFrame({"Q":ext_q.values, "conflict":ext_conf.values})
    ks = stats.ks_2samp(a.Q.dropna(), e.Q.dropna())
    delta = e.Q.mean() - a.Q.mean()

    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    av=a.Q.dropna().to_numpy(); ev=e.Q.dropna().to_numpy()
    for b in range(B):
        boot[b] = rng.choice(ev, len(ev), True).mean() - rng.choice(av, len(av), True).mean()

    return {
        "domain": domain,
        "n_A1_sample": len(a), "n_extended": len(e),
        "Q_A1_mean": a.Q.mean(), "Q_extended_mean": e.Q.mean(),
        "delta_Q_extended_minus_A1": delta,
        "delta_Q_ci_low": np.quantile(boot,.025),
        "delta_Q_ci_high": np.quantile(boot,.975),
        "KS_Q": ks.statistic, "KS_Q_pvalue": ks.pvalue,
        "conflict_rate_A1": (a.conflict > conflict_threshold).mean(),
        "conflict_rate_extended": (e.conflict > conflict_threshold).mean(),
    }


def plot_weights(weights, path):
    s=weights.sort_values()
    fig,ax=plt.subplots(figsize=(9,8))
    ax.barh(s.index,s.values)
    ax.set_xlabel("Final group-balanced CRITIC weight")
    ax.set_title("Quality-signal weights")
    ax.tick_params(axis="y",labelsize=7)
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def plot_domain(summary, path):
    s=summary.sort_values("Q_mean")
    xerr=np.vstack([s.Q_mean-s.Q_mean_ci_low,s.Q_mean_ci_high-s.Q_mean])
    fig,ax=plt.subplots(figsize=(8,5))
    ax.errorbar(s.Q_mean,s.domain,xerr=xerr,fmt="o",capsize=3)
    ax.set_xlabel("Mean Q (95% bootstrap CI)")
    ax.set_title("A1 domain quality")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def plot_conflict(gs, conflict, threshold, path):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(conflict,bins=50)
    ax.axvline(threshold,ls="--",label=f"P90 threshold={threshold:.3f}")
    ax.set_xlabel("Conflict score")
    ax.set_ylabel("Count")
    ax.set_title("A1 quality-conflict distribution")
    ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def plot_group_corr(gs, path):
    c=gs.corr(method="spearman")
    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(c,vmin=-1,vmax=1)
    ax.set_xticks(range(len(c)));ax.set_xticklabels(c.columns,rotation=45,ha="right")
    ax.set_yticks(range(len(c)));ax.set_yticklabels(c.index)
    fig.colorbar(im,ax=ax,label="Spearman correlation")
    ax.set_title("Quality-dimension correlations")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def make_human_review(raw_a1, q, conflict, gs, out_path, seed):
    """
    Blind-review sheet. Q/conflict are hidden from the reviewer columns but a stable
    review_id is kept. The answer key is saved separately by caller.
    """
    rng=np.random.default_rng(seed)
    work=pd.DataFrame({
        "row_id":np.arange(len(raw_a1)),
        "Q":q.values,"conflict":conflict.values,
        "content":raw_a1["content"].astype(str).values,
        "domain":raw_a1["_source_domain"].astype(str).values
    })
    bins=[
        ("bottom_1pct", work.Q <= work.Q.quantile(.01)),
        ("bottom_10pct", (work.Q > work.Q.quantile(.01)) & (work.Q <= work.Q.quantile(.10))),
        ("middle", (work.Q >= work.Q.quantile(.45)) & (work.Q <= work.Q.quantile(.55))),
        ("top_10pct", (work.Q >= work.Q.quantile(.90)) & (work.Q < work.Q.quantile(.99))),
        ("top_1pct", work.Q >= work.Q.quantile(.99)),
        ("high_conflict", work.conflict >= work.conflict.quantile(.95))
    ]
    picks=[]
    for label,mask in bins:
        pool=work[mask]
        n=min(30,len(pool))
        if n:
            ix=rng.choice(pool.index,n,replace=False)
            x=pool.loc[ix].copy();x["stratum"]=label;picks.append(x)
    review=pd.concat(picks).drop_duplicates("row_id").sample(frac=1,random_state=seed)
    review["review_id"]=np.arange(1,len(review)+1)
    blind=review[["review_id","content","domain"]].copy()
    blind["human_information_value_1to5"]=""
    blind["human_fluency_1to5"]=""
    blind["human_cleanliness_1to5"]=""
    blind["human_readability_1to5"]=""
    blind["human_overall_quality_1to5"]=""
    blind["notes"]=""
    blind.to_csv(out_path,index=False,encoding="utf-8-sig")
    key=review[["review_id","row_id","stratum","Q","conflict"]].copy()
    for c in gs.columns:
        key[c]=gs.iloc[review.row_id.values][c].values
    return key


def main():
    a=parse_args()
    out=a.out_dir.resolve()
    tab=out/"tables";fig=out/"figures";pro=out/"processed";human=out/"human_review"
    for p in [tab,fig,pro,human]: p.mkdir(parents=True,exist_ok=True)

    # ---------- Load ----------
    a1p=locate(a.data_dir,"slimpajama_quality_signal_sample.jsonl.xz","*quality_signal_sample*.jsonl.xz")
    a2p=locate(a.data_dir,"arxiv_part-6777d8857c6e-000486.jsonl.xz","arxiv*.jsonl.xz")
    a3p=locate(a.data_dir,"github_part-6777d8857c6e-000275.jsonl.xz","github*.jsonl.xz")
    print(">>> 正在加载原始 JSONL 压缩文件...")
    raw = {}
    for name, path, domain in [("A1", a1p, None), ("A2_arxiv", a2p, "arxiv"), ("A3_github", a3p, "github")]:
        print(f"加载 {name} ...")
        raw[name] = load_json(path, domain)

    bounds = load_round1_bounds(a.round1_dir)

    print("\n>>> 开始第一轮主标量化 (均值)...")
    # 传入 k 作为 name 参数，方便进度条显示
    mean = {k: scalarize(v, name=k, method="mean") for k, v in raw.items()}
    z0={k:scale_monotonic(v,bounds) for k,v in mean.items()}

    # ---------- Anchor and non-monotonic desirability ----------
    anchor=initial_anchor(z0["A1"])
    desirability_model,anchor_cut=fit_ambiguous_desirability(
        mean["A1"],anchor,a.anchor_top)
    X={k:apply_desirability(z0[k],mean[k],desirability_model) for k in mean}
    X,impute_medians=impute_from_a1(X["A1"],X)

    # ---------- CRITIC + group balancing ----------
    critic=critic_weights(X["A1"])
    weights=group_balanced_weights(critic)
    weight_table=pd.DataFrame({
        "feature":QUALITY_COLS,
        "critic_raw":[critic[c] for c in QUALITY_COLS],
        "final_weight":[weights[c] for c in QUALITY_COLS],
        "group":[next(g for g,cols in GROUPS.items() if c in cols) for c in QUALITY_COLS],
        "type":["positive" if c in POSITIVE else "negative" if c in NEGATIVE else "nonmonotonic"
                for c in QUALITY_COLS]
    }).sort_values("final_weight",ascending=False)
    weight_table.to_csv(tab/"01_quality_weights.csv",index=False)

    pd.DataFrame([
        {"feature":c,**m} for c,m in desirability_model.items()
    ]).to_csv(tab/"02_nonmonotonic_desirability_model.csv",index=False)

    # ---------- Q and conflicts ----------
    results={}
    for name in X:
        base,gs=score_components(X[name],weights)
        conf,conf_sd=conflict_metrics(gs)
        results[name]={"base":base,"groups":gs,"conflict":conf,"conflict_sd":conf_sd}

    conflict_threshold=float(results["A1"]["conflict"].quantile(a.conflict_quantile))
    for name,r in results.items():
        q,conf,short=final_quality(r["base"],r["groups"],a.gamma,a.eta,a.shortfall_threshold)
        r["Q"]=q;r["shortfall"]=short

    # ---------- Sensitivity: median scalarization ----------
    # Refit the whole Q construction using median list aggregation but keep the conceptual model.
    print("\n>>> 开始敏感性分析标量化 (中位数)...")
    # 原代码: med={k:scalarize(v,"median") for k,v in raw.items()}
    med = {k: scalarize(v, name=k, method="median") for k, v in raw.items()}
    med_z0={k:scale_monotonic(v,bounds) for k,v in med.items()}
    med_anchor=initial_anchor(med_z0["A1"])
    med_dm,_=fit_ambiguous_desirability(med["A1"],med_anchor,a.anchor_top)
    medX={k:apply_desirability(med_z0[k],med[k],med_dm) for k in med}
    medX,_=impute_from_a1(medX["A1"],medX)
    medcritic=critic_weights(medX["A1"])
    medw=group_balanced_weights(medcritic)
    mb,mgs=score_components(medX["A1"],medw)
    mq,mc,_=final_quality(mb,mgs,a.gamma,a.eta,a.shortfall_threshold)
    rho_q=stats.spearmanr(results["A1"]["Q"],mq).statistic
    pd.DataFrame([{
        "comparison":"mean_scalarization_vs_median_scalarization",
        "spearman_Q":rho_q,
        "mean_absolute_Q_difference":np.mean(np.abs(results["A1"]["Q"]-mq))
    }]).to_csv(tab/"03_Q_scalarization_sensitivity.csv",index=False)

    # ---------- Domain summaries ----------
    ds=domain_summary(mean["A1"],results["A1"]["Q"],results["A1"]["conflict"],
                      conflict_threshold,a.bootstrap,a.seed)
    ds.to_csv(tab/"04_A1_domain_quality.csv",index=False)

    extrows=[
        sample_vs_extended(mean["A1"],results["A1"]["Q"],results["A1"]["conflict"],
                           mean["A2_arxiv"],results["A2_arxiv"]["Q"],results["A2_arxiv"]["conflict"],
                           "arxiv",conflict_threshold,a.bootstrap,a.seed),
        sample_vs_extended(mean["A1"],results["A1"]["Q"],results["A1"]["conflict"],
                           mean["A3_github"],results["A3_github"]["Q"],results["A3_github"]["conflict"],
                           "github",conflict_threshold,a.bootstrap,a.seed)
    ]
    pd.DataFrame(extrows).to_csv(tab/"05_sample_vs_extended_Q.csv",index=False)

    # Conflict pattern summaries
    gs=results["A1"]["groups"]
    conflict_flag=results["A1"]["conflict"]>conflict_threshold
    ctab=gs.assign(conflict_score=results["A1"]["conflict"],is_conflict=conflict_flag)
    ctab["_source_domain"]=mean["A1"]["_source_domain"].values
    ctab.groupby("_source_domain").agg(
        n=("is_conflict","size"),
        conflict_rate=("is_conflict","mean"),
        mean_conflict=("conflict_score","mean"),
        semantic_value=("semantic_value","mean"),
        language_readability=("language_readability","mean"),
        cleanliness=("cleanliness","mean"),
        information_structure=("information_structure","mean")
    ).reset_index().to_csv(tab/"06_conflict_by_domain.csv",index=False)

    # Identify which group is highest/lowest for conflict cases.
    cg=gs.loc[conflict_flag]
    pattern=pd.DataFrame({
        "highest_group":cg.idxmax(axis=1),
        "lowest_group":cg.idxmin(axis=1)
    }).value_counts().rename("n").reset_index()
    pattern["fraction"]=pattern["n"]/pattern["n"].sum()
    pattern.to_csv(tab/"07_conflict_patterns.csv",index=False)

    # ---------- Processed outputs ----------
    a1out=pd.DataFrame({
        "id":mean["A1"]["id"] if "id" in mean["A1"] else np.arange(len(mean["A1"])),
        "domain":mean["A1"]["_source_domain"],
        "Q_base":results["A1"]["base"],
        "Q":results["A1"]["Q"],
        "conflict_score":results["A1"]["conflict"],
        "is_significant_conflict":conflict_flag,
        "shortfall_basis":results["A1"]["shortfall"]
    })
    for c in gs.columns:a1out[c]=gs[c]
    a1out.to_csv(pro/"A1_quality_scores.csv.gz",index=False,compression="gzip")

    for name in ["A2_arxiv","A3_github"]:
        r=results[name]
        eo=pd.DataFrame({
            "id":mean[name]["id"] if "id" in mean[name] else np.arange(len(mean[name])),
            "domain":mean[name]["_source_domain"],
            "Q_base":r["base"],"Q":r["Q"],"conflict_score":r["conflict"],
            "is_significant_conflict":r["conflict"]>conflict_threshold
        })
        eo.to_csv(pro/f"{name}_quality_scores.csv.gz",index=False,compression="gzip")

    # ---------- Human validation sheet ----------
    key=make_human_review(raw["A1"],results["A1"]["Q"],results["A1"]["conflict"],
                          gs,human/"A1_blind_human_review.csv",a.seed)
    key.to_csv(human/"A1_human_review_answer_key.csv",index=False,encoding="utf-8-sig")

    # ---------- Figures ----------
    plot_weights(weights,fig/"01_quality_weights.png")
    plot_domain(ds,fig/"02_A1_domain_quality.png")
    plot_conflict(gs,results["A1"]["conflict"],conflict_threshold,
                  fig/"03_conflict_distribution.png")
    plot_group_corr(gs,fig/"04_quality_dimension_correlation.png")

    # Q distributions by domain
    qplot=pd.DataFrame({"Q":results["A1"]["Q"],"domain":mean["A1"]["_source_domain"]})
    domains=list(ds.sort_values("Q_mean",ascending=False).domain)
    vals=[qplot.loc[qplot.domain==d,"Q"].dropna().values for d in domains]
    f,ax=plt.subplots(figsize=(10,6))
    ax.boxplot(vals,tick_labels=domains,showfliers=False)
    ax.set_ylabel("Q");ax.set_title("A1 quality-score distribution by domain")
    ax.tick_params(axis="x",rotation=35)
    f.tight_layout();f.savefig(fig/"05_A1_domain_Q_boxplot.png",dpi=220);plt.close(f)

    # Extended comparison
    er=pd.DataFrame(extrows)
    f,ax=plt.subplots(figsize=(7,5))
    x=np.arange(len(er))
    ax.bar(x-.18,er.Q_A1_mean,width=.36,label="A1 sample")
    ax.bar(x+.18,er.Q_extended_mean,width=.36,label="Extended")
    ax.set_xticks(x);ax.set_xticklabels(er.domain)
    ax.set_ylabel("Mean Q");ax.set_title("Sample vs extended quality")
    ax.legend();f.tight_layout();f.savefig(fig/"06_sample_vs_extended_Q.png",dpi=220);plt.close(f)

    # ---------- Summary/model ----------
    summary = f"""ROUND 2 QUALITY & CONFLICT EXPERIMENT

A1 rows: {len(raw['A1'])}
A2 arxiv rows: {len(raw['A2_arxiv'])}
A3 github rows: {len(raw['A3_github'])}

Round-1-driven decisions:
- PCA was not used as a one-dimensional Q because Round 1 required 9 PCs for 80% variance.
- CRITIC redundancy correction is used because several A1 signals were highly correlated.
- A1-fitted preprocessing is transferred unchanged to A2/A3.
- Mean list scalarization remains the main convention; median is a sensitivity analysis.

Q construction:
1. Direction-aware robust scaling using Round-1 A1 winsor bounds.
2. Five ambiguous statistical signals use two-sided desirability learned from the
   top {a.anchor_top:.0%} of A1's explicit-signal anchor score.
3. CRITIC objective information weights.
4. Equal total weight across four conceptual quality dimensions.
5. Base Q is penalized for cross-dimension conflict and severe minimum-dimension shortfall.

Parameters:
gamma = {a.gamma}
eta = {a.eta}
shortfall threshold = {a.shortfall_threshold}
significant conflict threshold = A1 P{int(a.conflict_quantile*100)} = {conflict_threshold:.6f}

Mean-vs-median Q sensitivity:
Spearman rho = {rho_q:.6f}

A1 domain quality:
{ds.to_string(index=False)}

Sample vs extended:
{pd.DataFrame(extrows).to_string(index=False)}

Important:
The blind human-review CSV is intentionally separated from the answer key.
Reviewers should score the blind sheet first; only then merge with the answer key
and calculate correlation between human overall quality and model Q.
"""
    (out/"round2_summary.txt").write_text(summary,encoding="utf-8")

    model={
        "quality_columns":QUALITY_COLS,
        "positive":POSITIVE,"negative":NEGATIVE,"ambiguous":AMBIGUOUS,
        "groups":GROUPS,
        "round1_winsor_bounds":{k:list(v) for k,v in bounds.items()},
        "anchor_top_fraction":a.anchor_top,
        "anchor_cutoff":anchor_cut,
        "ambiguous_desirability":desirability_model,
        "critic_raw":critic.to_dict(),
        "final_weights":weights.to_dict(),
        "imputation_medians":impute_medians.to_dict(),
        "gamma":a.gamma,"eta":a.eta,
        "shortfall_threshold":a.shortfall_threshold,
        "conflict_quantile":a.conflict_quantile,
        "conflict_threshold":conflict_threshold,
        "Q_formula":"100*clip(base - gamma*conflict - eta*max(0,tau-min_group),0,1)"
    }
    (out/"round2_model.json").write_text(
        json.dumps(model,ensure_ascii=False,indent=2),encoding="utf-8")

    print("Round 2 finished.")
    print(f"Output directory: {out}")
    print(f"Mean-vs-median Q Spearman: {rho_q:.4f}")
    print(f"Conflict threshold: {conflict_threshold:.4f}")
    print("\nA1 domain quality:")
    print(ds[["domain","n","Q_mean","Q_mean_ci_low","Q_mean_ci_high","conflict_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
