#!/usr/bin/env python3
"""Selective-prediction evidence for BlockTrain.

Computes, from saved prediction CSVs only (numpy + pandas; matplotlib for figures):
  * risk-coverage curves for several rejection scores
        var      : predictive variance sigma_p^2 (MC dropout)
        entropy  : predictive entropy H(p_bar) of the binary mean prediction
        maxprob  : 1 - max(p_bar, 1 - p_bar)   (max-probability confidence)
        random   : mean over 200 random permutations
  * AURC, oracle AURC, excess AURC (= AURC - oracle AURC)
  * selective risk at coverage 0.5 / 0.8 / 0.9
  * error rate by quintile of sigma_p^2 (with mean sigma_p^2 per quintile)
  * error-detection AUROC of every score (probability that a misclassified window has a
    higher score than a correctly classified one; 0.5 = uninformative, <0.5 = anti-informative)
  * per-seed results and mean +- SD (sample SD, ddof = 1) across seeds

Definitions
  prediction  : yhat = 1[prob >= threshold]; error = 1[yhat != y_true]
  ordering    : windows are retained in ascending order of the score (most uncertain rejected first)
  risk R(c)   : error rate among the round(c*n) retained windows (eq. selective_risk of the paper)
  AURC        : (1/n) sum_{k=1..n} R(k/n)   (full-resolution discrete AURC, = area under R(c))
  oracle      : retains all correct windows first -> R_oracle(k/n) = max(0, k-(n-E))/k, E = #errors
  quintiles   : equal-count bins (np.array_split on the stable ascending sort of sigma_p^2)
  ties        : stable (merge) sort; entropy and maxprob are both monotone in |p_bar-0.5| and so
                give the same ordering up to floating-point ties.

Usage
  python3 selective.py                       # authors' earlier validation predictions in valpreds/
  python3 selective.py --runs runs/main      # predictions saved by pipeline.py (test; and val if present)
  python3 selective.py --figs-only           # redraw figures from saved result CSVs
Outputs: runs/selective/<tag>/*.csv, figs/fig_risk_coverage*.{pdf,png},
         figs/fig_uncertainty_quantile*.{pdf,png}
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
COV_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
KEY_COV = (0.5, 0.8, 0.9)
N_PERM = 200
N_Q = 5
SCORE_LABEL = {
    "var": r"$\sigma_p^2$ (variance)",
    "entropy": r"$H(\bar p)$ (entropy)",
    "maxprob": r"$1-\max(\bar p,1-\bar p)$",
    "random": "random",
}
SCORE_ORDER = ["var", "entropy", "maxprob", "random"]


# --------------------------------------------------------------------------- loading
def load_preds(path, default_threshold=0.5):
    """Return dict(y, p, var, ent, thr, thr_source, ent_source)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    pcol = cols.get("prob") or cols.get("probability")
    vcol = cols.get("pred_var") or cols.get("predictive_variance")
    ecol = cols.get("pred_entropy")
    tcol = cols.get("threshold")
    y = df[cols["y_true"]].to_numpy().astype(int)
    p = df[pcol].to_numpy().astype(float)
    var = df[vcol].to_numpy().astype(float) if vcol else np.full(len(df), np.nan)
    if tcol and df[tcol].notna().all():
        thr = float(df[tcol].iloc[0])
        thr_src = "file column 'threshold'"
        if df[tcol].nunique() != 1:
            raise ValueError(f"{path}: threshold column is not constant")
    else:
        thr = default_threshold
        thr_src = f"default {default_threshold} (no threshold column)"
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    ent_calc = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))
    if ecol and df[ecol].notna().all():
        ent, ent_src = df[ecol].to_numpy().astype(float), "file column 'pred_entropy'"
    else:
        ent, ent_src = ent_calc, "computed from prob (nats)"
    atype = df[cols["anomaly_type"]].astype(str).to_numpy() if "anomaly_type" in cols else None
    return dict(atype=atype, y=y, p=p, var=var, ent=ent, ent_calc=ent_calc, thr=thr,
                thr_source=thr_src, ent_source=ent_src, n=len(df))


# --------------------------------------------------------------------------- core metrics
def rc_curve_from_order(err, order):
    """risk of the first k windows in `order`, k = 1..n"""
    e = err[order].astype(float)
    return np.cumsum(e) / np.arange(1, len(e) + 1)


def oracle_curve(err):
    n = len(err)
    E = int(err.sum())
    k = np.arange(1, n + 1)
    return np.maximum(0, k - (n - E)) / k


def grid_values(curve, n):
    idx = np.round(COV_GRID * n).astype(int)
    return curve[idx - 1]


def error_detection_auroc(score, err):
    """P(score_error > score_correct) + 0.5 P(tie), via average ranks."""
    s = pd.Series(score)
    r = s.rank(method="average").to_numpy()
    n1 = int(err.sum())
    n0 = len(err) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[err == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate_one(d, seed, scores_avail, rng_seed):
    """Returns (summary rows, curve rows, quintile rows) for one prediction file."""
    y, p = d["y"], d["p"]
    thr = d["thr"]
    yhat = (p >= thr).astype(int)
    err = (yhat != y).astype(int)
    n = len(err)
    score_vals = {
        "var": d["var"],
        "entropy": d["ent"],
        "maxprob": 1.0 - np.maximum(p, 1.0 - p),
    }
    orac = oracle_curve(err)
    aurc_oracle = float(orac.mean())
    summ, curves = [], []
    rng = np.random.default_rng(rng_seed)
    results = {}
    for name in SCORE_ORDER:
        if name == "random":
            acc = np.zeros(n)
            for _ in range(N_PERM):
                acc += rc_curve_from_order(err, rng.permutation(n))
            curve = acc / N_PERM
            auroc = 0.5
        else:
            if name not in scores_avail:
                continue
            s = score_vals[name]
            order = np.argsort(s, kind="mergesort")
            curve = rc_curve_from_order(err, order)
            auroc = error_detection_auroc(s, err)
        results[name] = curve
        row = dict(seed=seed, score=name, n=n, threshold=thr, error_rate=err.mean(),
                   n_errors=int(err.sum()), aurc=float(curve.mean()), aurc_oracle=aurc_oracle,
                   excess_aurc=float(curve.mean() - aurc_oracle),
                   error_detection_auroc=auroc)
        for c in KEY_COV:
            row[f"risk_at_{c:.1f}"] = float(curve[int(round(c * n)) - 1])
        summ.append(row)
        gv = grid_values(curve, n)
        for c, r in zip(COV_GRID, gv):
            curves.append(dict(seed=seed, score=name, coverage=c, risk=float(r)))
    gv = grid_values(orac, n)
    for c, r in zip(COV_GRID, gv):
        curves.append(dict(seed=seed, score="oracle", coverage=c, risk=float(r)))
    # quintiles of sigma_p^2
    quint = []
    if "var" in scores_avail:
        order = np.argsort(d["var"], kind="mergesort")
        for q, idx in enumerate(np.array_split(order, N_Q), start=1):
            quint.append(dict(seed=seed, quintile=q, n=len(idx),
                              mean_var=float(d["var"][idx].mean()),
                              min_var=float(d["var"][idx].min()), max_var=float(d["var"][idx].max()),
                              error_rate=float(err[idx].mean()),
                              mean_prob=float(p[idx].mean())))
        # consistency check: risk at coverage 0.2 (var order) == Q1 error when n divisible by 5
        if n % N_Q == 0:
            k = n // N_Q
            chk = float(results["var"][k - 1])
            assert abs(chk - quint[0]["error_rate"]) < 1e-12, "risk@0.2 != Q1 error"
    # diagnostics (saved to diagnostics_per_seed.csv by run_set)
    diag = dict(seed=seed, n=n, threshold=thr, error_rate=err.mean(), mean_prob=p.mean())
    if "var" in scores_avail:
        sv = pd.Series(d["var"])
        diag["spearman_var_entropy"] = sv.corr(pd.Series(d["ent"]), method="spearman")
        diag["spearman_var_abs_p_minus_thr"] = sv.corr(pd.Series(np.abs(p - thr)), method="spearman")
        diag["spearman_var_error"] = sv.corr(pd.Series(err), method="spearman")
        diag["spearman_entropy_error"] = pd.Series(d["ent"]).corr(pd.Series(err), method="spearman")
        cv = results["var"]
        g = grid_values(cv, n)
        diag["grid_cov_where_var_risk_above_error_rate"] = ";".join(
            f"{c:.2f}" for c, r in zip(COV_GRID, g) if c < 1.0 and r > err.mean())
        diag["max_var_risk_on_grid"] = float(g.max())
        diag["coverage_at_max_var_risk"] = float(COV_GRID[int(np.argmax(g))])
    tab = []
    if d.get("atype") is not None:
        hasvar = np.isfinite(d["var"]).any()
        for t in sorted(set(d["atype"])):
            m = d["atype"] == t
            tab.append(dict(seed=seed, anomaly_type=t, n=int(m.sum()), error_rate=float(err[m].mean()),
                            mean_prob=float(p[m].mean()),
                            mean_var=float(np.nanmean(d["var"][m])) if hasvar else np.nan,
                            median_var=float(np.nanmedian(d["var"][m])) if hasvar else np.nan,
                            mean_entropy=float(d["ent"][m].mean())))
    return summ, curves, quint, diag, tab


def mean_sd(df, by, cols):
    g = df.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = m.add_suffix("_mean").join(s.add_suffix("_sd"))
    out["n_seeds"] = g.size()
    return out.reset_index()


def paired_vs(summ, ref, other):
    """Per-seed differences in AURC: other - ref (negative => `other` better than `ref`)."""
    a = summ[summ.score == ref].set_index("seed")["aurc"]
    b = summ[summ.score == other].set_index("seed")["aurc"]
    j = a.to_frame("ref").join(b.to_frame("oth")).dropna()
    diff = j.oth - j.ref
    return dict(ref=ref, other=other, mean_diff=diff.mean(),
                sd_diff=diff.std(ddof=1) if len(diff) > 1 else np.nan,
                n_other_better=int((diff < 0).sum()), n_seeds=len(diff))


def run_set(files_by_seed, outdir, label, scores_avail=("var", "entropy", "maxprob"),
            default_threshold=0.5, extra_meta=None):
    os.makedirs(outdir, exist_ok=True)
    S, C, Q, META, DG, TB = [], [], [], [], [], []
    for seed in sorted(files_by_seed):
        path = files_by_seed[seed]
        d = load_preds(path, default_threshold)
        avail = [s for s in scores_avail
                 if (s != "var" or np.isfinite(d["var"]).all())]
        s_, c_, q_, dg_, tb_ = evaluate_one(d, seed, avail, rng_seed=int(seed))
        S += s_; C += c_; Q += q_; DG.append(dg_); TB += tb_
        META.append(dict(seed=seed, file=os.path.relpath(path, ROOT), n=d["n"],
                         threshold=d["thr"], threshold_source=d["thr_source"],
                         entropy_source=d["ent_source"],
                         max_abs_entropy_vs_prob=float(np.max(np.abs(d["ent"] - d["ent_calc"])))))
    S, C = pd.DataFrame(S), pd.DataFrame(C)
    Q = pd.DataFrame(Q)
    S.to_csv(f"{outdir}/per_seed_summary.csv", index=False)
    cols = ["error_rate", "aurc", "aurc_oracle", "excess_aurc", "error_detection_auroc"] + \
           [f"risk_at_{c:.1f}" for c in KEY_COV]
    mean_sd(S, "score", cols).to_csv(f"{outdir}/summary_mean_sd.csv", index=False)
    mean_sd(C, ["score", "coverage"], ["risk"]).to_csv(f"{outdir}/rc_curves_summary.csv", index=False)
    C.to_csv(f"{outdir}/rc_curves_per_seed.csv", index=False)
    if len(Q):
        Q.to_csv(f"{outdir}/quintiles_per_seed.csv", index=False)
        mean_sd(Q, "quintile", ["mean_var", "error_rate", "mean_prob", "n"]).to_csv(
            f"{outdir}/quintiles_summary.csv", index=False)
    pairs = []
    if "var" in set(S.score):
        for o in ("entropy", "maxprob", "random"):
            if o in set(S.score):
                pairs.append(paired_vs(S, "var", o))
    pd.DataFrame(pairs).to_csv(f"{outdir}/paired_aurc_vs_variance.csv", index=False)
    pd.DataFrame(META).to_csv(f"{outdir}/input_files.csv", index=False)
    pd.DataFrame(DG).to_csv(f"{outdir}/diagnostics_per_seed.csv", index=False)
    if TB:
        TB = pd.DataFrame(TB)
        TB.to_csv(f"{outdir}/by_anomaly_type_per_seed.csv", index=False)
        mean_sd(TB, "anomaly_type", ["n", "error_rate", "mean_prob", "mean_var", "median_var",
                                     "mean_entropy"]).to_csv(f"{outdir}/by_anomaly_type_summary.csv",
                                                              index=False)
    print(f"[{label}] {len(files_by_seed)} seeds -> {outdir}")
    print(pd.read_csv(f"{outdir}/summary_mean_sd.csv").round(4).to_string(index=False))
    return S


# --------------------------------------------------------------------------- figures (saved CSVs only)
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "font.family": "serif",
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42})
    return plt


STYLE = {  # colour-blind-safe (Okabe-Ito), distinct line styles
    "var": dict(color="#D55E00", ls="-", marker="o"),
    "entropy": dict(color="#0072B2", ls="--", marker="s"),
    "maxprob": dict(color="#009E73", ls="-.", marker="^"),
    "random": dict(color="#555555", ls=":", marker=None),
    "oracle": dict(color="#999999", ls=(0, (1, 1)), marker=None),
}


def fig_risk_coverage(resdir, stem):
    plt = _mpl()
    curves = pd.read_csv(f"{resdir}/rc_curves_summary.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for name in SCORE_ORDER + ["oracle"]:
        c = curves[curves.score == name].sort_values("coverage")
        if c.empty:
            continue
        st = STYLE[name]
        lab = SCORE_LABEL.get(name, "oracle")
        ax.plot(c.coverage, c.risk_mean, color=st["color"], ls=st["ls"], lw=1.5, label=lab,
                marker=st["marker"], ms=3.5 if st["marker"] else 0)
        if name != "oracle" and c.risk_sd.notna().any():
            ax.fill_between(c.coverage, c.risk_mean - c.risk_sd, c.risk_mean + c.risk_sd,
                            color=st["color"], alpha=0.15, lw=0)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def fig_quantile(resdir, stem):
    plt = _mpl()
    q = pd.read_csv(f"{resdir}/quintiles_summary.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    x = q.quintile.to_numpy()
    a1.errorbar(x, q.error_rate_mean, yerr=q.error_rate_sd, color="#D55E00", marker="o", ms=5,
                lw=1.5, capsize=3)
    for xi, yi in zip(x, q.error_rate_mean):
        a1.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(7, 5),
                    ha="left", fontsize=8, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    a1.set_xlim(0.6, 5.9)
    a1.set_xlabel(r"$\sigma_p^2$ quintile (Q1 = lowest)")
    a1.set_ylabel("Error rate")
    a1.set_xticks(x)
    a1.set_xticklabels([f"Q{i}" for i in x])
    lo = min(0.0, float((q.error_rate_mean - q.error_rate_sd.fillna(0)).min()))
    a1.set_ylim(lo, float((q.error_rate_mean + q.error_rate_sd.fillna(0)).max()) * 1.12)
    a1.grid(alpha=0.3, lw=0.5)
    a1.text(0.02, 0.97, "(a)", transform=a1.transAxes, va="top", fontweight="bold")
    a2.errorbar(x, q.mean_var_mean, yerr=q.mean_var_sd, color="#0072B2", marker="s", ms=5, lw=1.5,
                capsize=3)
    a2.set_yscale("log")
    for xi, yi in zip(x, q.mean_var_mean):
        a2.annotate(f"{yi:.2e}", (xi, yi), textcoords="offset points", xytext=(8, -13),
                    ha="left", fontsize=8)
    a2.set_xlim(0.6, 6.3)
    a2.set_xlabel(r"$\sigma_p^2$ quintile (Q1 = lowest)")
    a2.set_ylabel(r"Mean $\sigma_p^2$")
    a2.set_xticks(x)
    a2.set_xticklabels([f"Q{i}" for i in x])
    a2.set_ylim(top=float(q.mean_var_mean.max()) * 6)
    a2.grid(alpha=0.3, lw=0.5, which="both")
    a2.text(0.02, 0.97, "(b)", transform=a2.transAxes, va="top", fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def make_figs(primary_dir, extra_dirs):
    os.makedirs(f"{ROOT}/figs", exist_ok=True)
    fig_risk_coverage(primary_dir, f"{ROOT}/figs/fig_risk_coverage")
    fig_quantile(primary_dir, f"{ROOT}/figs/fig_uncertainty_quantile")
    for tag, dd in extra_dirs.items():
        if os.path.exists(f"{dd}/rc_curves_summary.csv"):
            fig_risk_coverage(dd, f"{ROOT}/figs/fig_risk_coverage_{tag}")
            if os.path.exists(f"{dd}/quintiles_summary.csv"):
                fig_quantile(dd, f"{ROOT}/figs/fig_uncertainty_quantile_{tag}")
    print("figures written; primary =", os.path.relpath(primary_dir, ROOT))


# --------------------------------------------------------------------------- main
def seed_files(pattern, regex):
    out = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(regex, f)
        if m:
            out[int(m.group(1))] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valpreds", default=f"{ROOT}/valpreds",
                    help="authors' earlier validation predictions")
    ap.add_argument("--runs", default=None, help="e.g. runs/main (output of pipeline.py)")
    ap.add_argument("--figs-only", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    sel = f"{ROOT}/runs/selective"
    d_val = f"{sel}/valpreds_old"
    d_test = f"{sel}/main_test"
    d_det = f"{sel}/main_test_cnn_bigru_att"
    d_newval = f"{sel}/main_val"
    if not a.figs_only:
        vf = seed_files(f"{a.valpreds}/BlockTrain_validation_predictions_seed_*.csv", r"seed_(\d+)\.csv")
        if vf:
            run_set(vf, d_val, "valpreds_old (threshold 0.5 assumed)")
        if a.runs:
            runs = a.runs if os.path.isabs(a.runs) else os.path.join(ROOT, a.runs)
            for split, outd in (("test", d_test), ("val", d_newval)):
                f = seed_files(f"{runs}/seed*/preds_blocktrain_{split}.csv", r"seed(\d+)/")
                if f:
                    run_set(f, outd, f"main {split} BlockTrain")
            f = seed_files(f"{runs}/seed*/preds_cnn_bigru_att_test.csv", r"seed(\d+)/")
            if f:
                run_set(f, d_det, "main test CNN-BiGRU-Att (deterministic; maxprob only)",
                        scores_avail=("maxprob",))
    if not a.no_figs:
        primary = d_test if os.path.exists(f"{d_test}/rc_curves_summary.csv") else d_val
        if not os.path.exists(f"{primary}/rc_curves_summary.csv"):
            sys.exit("no saved results to plot")
        make_figs(primary, {"valpreds_old": d_val, "main_test": d_test})


if __name__ == "__main__":
    main()
