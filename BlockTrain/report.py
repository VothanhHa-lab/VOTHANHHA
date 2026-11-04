#!/usr/bin/env python3
"""Regenerate main tables and figures from SAVED files only (runs/main/seed*/preds_*.csv, mc_sensitivity.csv).
Outputs: runs/main/metrics_per_seed.csv, metrics_summary.csv, metrics_by_anomaly.csv, metrics_val_per_seed.csv,
         runs/main/figdata_*.csv, figs/fig_{roc_mean,pr_mean,reliability,overall,mc_sensitivity}.{pdf,png}
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "runs", "main")
FIG = os.path.join(ROOT, "figs")
SEEDS = [int(x) for x in os.environ.get("REPORT_SEEDS", "11,22,33,44,55").split(",")]
MODELS = ["cnn", "bigru", "cnn_bigru", "cnn_bigru_att", "blocktrain"]
LABEL = {"cnn": "CNN", "bigru": "BiGRU", "cnn_bigru": "CNN–BiGRU", "cnn_bigru_att": "CNN–BiGRU–Attention",
         "blocktrain": "BlockTrain"}
METRICS = ["f1", "auroc", "auprc", "brier", "nll", "ece"]
MLABEL = {"f1": "F1", "auroc": "AUROC", "auprc": "AUPRC", "brier": "Brier", "nll": "NLL", "ece": "ECE"}
EPS = 1e-7
B_BOOT = int(os.environ.get("REPORT_B", "1000"))

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42})
COL = {"cnn": "#4C78A8", "bigru": "#F58518", "cnn_bigru": "#54A24B", "cnn_bigru_att": "#B279A2", "blocktrain": "#E45756"}
LS = {"cnn": "-", "bigru": "--", "cnn_bigru": "-.", "cnn_bigru_att": ":", "blocktrain": "-"}


def ece(y, p, nb=10):
    edges = np.linspace(0, 1, nb + 1)
    t = 0.0
    for i in range(nb):
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if i == nb - 1 else (p < edges[i + 1]))
        if m.any():
            t += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(t)


def all_metrics(y, p, thr):
    pc = np.clip(p, EPS, 1 - EPS)
    return dict(f1=f1_score(y, (p >= thr).astype(int), zero_division=0), auroc=roc_auc_score(y, p),
                auprc=average_precision_score(y, p), brier=float(np.mean((p - y) ** 2)),
                nll=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), ece=ece(y, p))


def load_preds(split):
    D = {}
    for s in SEEDS:
        for m in MODELS:
            f = os.path.join(RUN, f"seed{s}", f"preds_{m}_{split}.csv")
            D[(s, m)] = pd.read_csv(f)
    return D


def savefig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    T = load_preds("test")
    V = load_preds("val")
    # ---------- per-seed metrics ----------
    rows = []
    for (s, m), df in T.items():
        thr = float(df["threshold"].iloc[0])
        r = all_metrics(df.y_true.values, df.prob.values, thr)
        r.update(seed=s, model=m, threshold=thr, n_test=len(df), prevalence=float(df.y_true.mean()))
        rows.append(r)
    per = pd.DataFrame(rows)[["seed", "model"] + METRICS + ["threshold", "n_test", "prevalence"]]
    per.to_csv(os.path.join(RUN, "metrics_per_seed.csv"), index=False)
    rv = []
    for (s, m), df in V.items():
        thr = float(df["threshold"].iloc[0])
        r = all_metrics(df.y_true.values, df.prob.values, thr)
        r.update(seed=s, model=m, threshold=thr, n_val=len(df))
        rv.append(r)
    pd.DataFrame(rv)[["seed", "model"] + METRICS + ["threshold", "n_val"]].to_csv(
        os.path.join(RUN, "metrics_val_per_seed.csv"), index=False)

    # ---------- bootstrap CI over test windows (resample windows within seed, same indices for all models) ----------
    rng = np.random.default_rng(20260101)
    boot = {m: {k: np.zeros((B_BOOT, len(SEEDS))) for k in METRICS} for m in MODELS}
    for si, s in enumerate(SEEDS):
        n = len(T[(s, MODELS[0])])
        for b in range(B_BOOT):
            idx = rng.integers(0, n, n)
            for m in MODELS:
                df = T[(s, m)]
                y, p = df.y_true.values[idx], df.prob.values[idx]
                if y.min() == y.max():
                    continue
                r = all_metrics(y, p, float(df["threshold"].iloc[0]))
                for k in METRICS:
                    boot[m][k][b, si] = r[k]
    # ---------- summary ----------
    from scipy import stats
    tcrit = stats.t.ppf(0.975, len(SEEDS) - 1)
    srows = []
    for m in MODELS:
        g = per[per.model == m]
        for k in METRICS:
            v = g[k].values
            bm = boot[m][k].mean(1)
            srows.append(dict(model=m, metric=k, mean=v.mean(), sd=v.std(ddof=1), n_seeds=len(v),
                              ci_boot_windows_lo=np.percentile(bm, 2.5), ci_boot_windows_hi=np.percentile(bm, 97.5),
                              ci_t_seeds_lo=v.mean() - tcrit * v.std(ddof=1) / np.sqrt(len(v)),
                              ci_t_seeds_hi=v.mean() + tcrit * v.std(ddof=1) / np.sqrt(len(v)),
                              min_seed=v.min(), max_seed=v.max()))
    # trivial baselines, from the saved test labels
    prev = per[per.model == "blocktrain"].prevalence.values
    f1_pos = 2 * prev / (1 + prev)
    base = {
        "all_positive": dict(f1=f1_pos, auroc=np.full(len(SEEDS), 0.5), auprc=prev),
        "constant_prevalence": dict(brier=prev * (1 - prev),
                                    nll=-(prev * np.log(prev) + (1 - prev) * np.log(1 - prev)), ece=np.zeros(len(SEEDS)),
                                    auroc=np.full(len(SEEDS), 0.5), auprc=prev),
    }
    for m, d in base.items():
        for k, v in d.items():
            srows.append(dict(model=m, metric=k, mean=v.mean(), sd=v.std(ddof=1), n_seeds=len(v),
                              ci_boot_windows_lo=np.nan, ci_boot_windows_hi=np.nan, ci_t_seeds_lo=np.nan,
                              ci_t_seeds_hi=np.nan, min_seed=v.min(), max_seed=v.max()))
    summ = pd.DataFrame(srows)
    summ.to_csv(os.path.join(RUN, "metrics_summary.csv"), index=False)

    # ---------- by anomaly type: recall at the validation-chosen threshold ----------
    arows = []
    for (s, m), df in T.items():
        thr = float(df["threshold"].iloc[0])
        pred = df.prob.values >= thr
        for t in ["clean", "value_noise", "missing_block"]:
            mk = (df.anomaly_type == t).values
            arows.append(dict(seed=s, model=m, anomaly_type=t, n=int(mk.sum()),
                              frac_predicted_positive=float(pred[mk].mean()), mean_prob=float(df.prob.values[mk].mean())))
        for t in ["value_noise", "missing_block"]:
            mk = (df.anomaly_type.isin([t, "clean"])).values
            arows[-1 if False else len(arows) - 1]  # no-op
            arows.append(dict(seed=s, model=m, anomaly_type=f"auroc_{t}_vs_clean", n=int(mk.sum()),
                              frac_predicted_positive=np.nan,
                              mean_prob=float(roc_auc_score((df.anomaly_type[mk] != "clean").astype(int), df.prob.values[mk]))))
    pd.DataFrame(arows).to_csv(os.path.join(RUN, "metrics_by_anomaly.csv"), index=False)

    # ---------- figures ----------
    mean_tab = summ.pivot(index="model", columns="metric", values="mean")
    sd_tab = summ.pivot(index="model", columns="metric", values="sd")

    # ROC
    fpr_g = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    figdata = []
    for m in MODELS:
        tprs = []
        for s in SEEDS:
            df = T[(s, m)]
            f, t, _ = roc_curve(df.y_true, df.prob)
            tprs.append(np.interp(fpr_g, f, t))
        tprs = np.array(tprs)
        mu, sd = tprs.mean(0), tprs.std(0, ddof=1)
        ax.plot(fpr_g, mu, color=COL[m], ls=LS[m], lw=1.6,
                label=f"{LABEL[m]} (AUROC {mean_tab.loc[m, 'auroc']:.3f})")
        ax.fill_between(fpr_g, mu - sd, mu + sd, color=COL[m], alpha=0.10, lw=0)
        figdata.append(pd.DataFrame(dict(model=m, fpr=fpr_g, tpr_mean=mu, tpr_sd=sd)))
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=(0, (2, 2)), label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_aspect("equal"); ax.legend(loc="lower right", frameon=False)
    savefig(fig, "fig_roc_mean")
    pd.concat(figdata).to_csv(os.path.join(RUN, "figdata_roc.csv"), index=False)

    # PR
    rec_g = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    figdata = []
    for m in MODELS:
        precs = []
        for s in SEEDS:
            df = T[(s, m)]
            p_, r_, _ = precision_recall_curve(df.y_true, df.prob)
            order = np.argsort(r_)                       # recall ascending (precision_recall_curve returns descending)
            # precision envelope: highest precision attainable at recall >= r
            pe = np.maximum.accumulate(p_[order][::-1])[::-1]
            precs.append(np.interp(rec_g, r_[order], pe))
        precs = np.array(precs)
        mu, sd = precs.mean(0), precs.std(0, ddof=1)
        ax.plot(rec_g, mu, color=COL[m], ls=LS[m], lw=1.6,
                label=f"{LABEL[m]} (AUPRC {mean_tab.loc[m, 'auprc']:.3f})")
        ax.fill_between(rec_g, mu - sd, mu + sd, color=COL[m], alpha=0.10, lw=0)
        figdata.append(pd.DataFrame(dict(model=m, recall=rec_g, precision_mean=mu, precision_sd=sd)))
    prevm = float(prev.mean())
    ax.axhline(prevm, color="gray", lw=0.8, ls=(0, (2, 2)), label=f"Prevalence ({prevm:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower left", frameon=False)
    savefig(fig, "fig_pr_mean")
    pd.concat(figdata).to_csv(os.path.join(RUN, "figdata_pr.csv"), index=False)

    # Reliability (10 equal-width bins; per-seed bin stats, averaged over seeds with a non-empty bin)
    edges = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 5, figsize=(9.6, 2.5), sharex=True, sharey=True)
    figdata = []
    for ax, m in zip(axes, MODELS):
        conf = np.full((len(SEEDS), 10), np.nan); acc = np.full((len(SEEDS), 10), np.nan); cnt = np.zeros((len(SEEDS), 10))
        for si, s in enumerate(SEEDS):
            df = T[(s, m)]
            y, p = df.y_true.values, df.prob.values
            for i in range(10):
                mk = (p >= edges[i]) & ((p <= edges[i + 1]) if i == 9 else (p < edges[i + 1]))
                cnt[si, i] = mk.sum()
                if mk.any():
                    conf[si, i], acc[si, i] = p[mk].mean(), y[mk].mean()
        cm, am = np.nanmean(conf, 0), np.nanmean(acc, 0)
        asd = np.nanstd(acc, 0, ddof=1)
        ok = np.isfinite(cm) & (np.sum(np.isfinite(conf), 0) >= min(3, len(SEEDS)))
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=(0, (2, 2)))
        ax.errorbar(cm[ok], am[ok], yerr=np.nan_to_num(asd[ok]), color=COL[m], marker="o", ms=3.5, lw=1.2, capsize=2)
        ax.set_title(LABEL[m] if m != "cnn_bigru_att" else "CNN–BiGRU–Att.", fontsize=9)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        figdata.append(pd.DataFrame(dict(model=m, bin=np.arange(10), conf_mean=cm, acc_mean=am, acc_sd=asd,
                                         n_seeds_nonempty=np.sum(np.isfinite(conf), 0), mean_count=cnt.mean(0))))
    axes[0].set_ylabel("Observed frequency")
    fig.supxlabel("Mean predicted probability", fontsize=10)
    fig.tight_layout(w_pad=0.6)
    savefig(fig, "fig_reliability")
    pd.concat(figdata).to_csv(os.path.join(RUN, "figdata_reliability.csv"), index=False)

    # Overall comparison: bars at their table values
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.4))
    plotted = []
    for ax, k in zip(axes.ravel(), METRICS):
        mu = np.array([mean_tab.loc[m, k] for m in MODELS]); sd = np.array([sd_tab.loc[m, k] for m in MODELS])
        bars = ax.bar(np.arange(5), mu, yerr=sd, capsize=3, color=[COL[m] for m in MODELS], width=0.7,
                      error_kw=dict(lw=1, capthick=1))
        for b_, m_ in zip(bars, MODELS):
            plotted.append(dict(metric=k, model=m_, bar_height=b_.get_height()))
        top = (mu + sd).max()
        ax.set_ylim(0, top * 1.22)
        for i, (v, e) in enumerate(zip(mu, sd)):
            ax.text(i, v + e + top * 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(np.arange(5)); ax.set_xticklabels(["CNN", "BiGRU", "CNN–\nBiGRU", "+Att.", "Block-\nTrain"], fontsize=8)
        arrow = r"$\uparrow$" if k in ("f1", "auroc", "auprc") else r"$\downarrow$"
        ax.set_ylabel(f"{MLABEL[k]} {arrow}")
        if k == "f1":
            ax.axhline(summ[(summ.model == "all_positive") & (summ.metric == "f1")]["mean"].iloc[0], color="gray", lw=0.8, ls=(0, (2, 2)))
        if k == "auprc":
            ax.axhline(prevm, color="gray", lw=0.8, ls=(0, (2, 2)))
        if k == "auroc":
            ax.axhline(0.5, color="gray", lw=0.8, ls=(0, (2, 2)))
    fig.tight_layout()
    savefig(fig, "fig_overall")
    pl = pd.DataFrame(plotted)
    pl["table_mean"] = [mean_tab.loc[r.model, r.metric] for r in pl.itertuples()]
    pl["abs_diff"] = (pl.bar_height - pl.table_mean).abs()
    pl.to_csv(os.path.join(RUN, "figdata_overall_check.csv"), index=False)
    assert pl.abs_diff.max() < 1e-12, "plotted bars differ from table"
    print("overall bars equal table means: max |diff| =", pl.abs_diff.max())

    # MC sensitivity
    f = os.path.join(RUN, "mc_sensitivity.csv")
    if os.path.exists(f):
        mc = pd.read_csv(f)
        g = mc.groupby("K").agg(["mean", "std"])
        Ks = g.index.values
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
        def line(ax, cols, labs):
            for c, l in zip(cols, labs):
                ax.errorbar(Ks, g[(c, "mean")], yerr=g[(c, "std")], marker="o", ms=4, capsize=2, lw=1.2, label=l)
            ax.set_xticks(Ks); ax.set_xlabel("MC samples $K$")
        line(axes[0, 0], ["auroc", "auprc", "f1"], ["AUROC", "AUPRC", "F1"]); axes[0, 0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3); axes[0, 0].set_ylabel("Discrimination")
        line(axes[0, 1], ["brier", "ece"], ["Brier", "ECE"]); axes[0, 1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); axes[0, 1].set_ylabel("Calibration (lower is better)")
        line(axes[1, 0], ["mean_pred_var"], ["Mean predictive variance"]); axes[1, 0].set_ylabel("Mean predictive variance")
        line(axes[1, 1], ["test_infer_seconds"], ["Test inference time"]); axes[1, 1].set_ylabel("Test inference time (s)")
        fig.tight_layout()
        savefig(fig, "fig_mc_sensitivity")
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g.reset_index().to_csv(os.path.join(RUN, "mc_sensitivity_summary.csv"), index=False)
    print(summ[summ.model.isin(MODELS)].pivot(index="model", columns="metric", values="mean").loc[MODELS, METRICS].round(4))
    print(summ[summ.model.isin(MODELS)].pivot(index="model", columns="metric", values="sd").loc[MODELS, METRICS].round(4))


if __name__ == "__main__":
    main()
