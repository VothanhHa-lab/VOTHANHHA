#!/usr/bin/env python3
"""
U-Attention ablation on the SAME trained CNN-BiGRU-Attention checkpoints.

Variants (all evaluated on identical val/test windows of the same checkpoint):
  V0   deterministic: dropout OFF, single pass  (= CNN-BiGRU-Attention)
  V1   MC dropout, K = 1,  u_t = 0              (old "BlockTrain-NoUncertainty")
  V2   MC dropout, K = 10, u_t = 0              (MC averaging only)
  V3   MC dropout, K = 10, U-Attention on       (= BlockTrain, protocol item 11)
  exploratory (not part of the protocol):
  V4b1 / V4b3  K = 10, soft gate alpha_t * exp(-beta u_t), beta = 1 / 3
  V5   K = 10, U-Attention with u_t randomly permuted over time inside each window (control)

Why a re-implementation instead of calling pipeline.mc_uattention_predict for every variant:
  that function draws fresh dropout masks on every call, so V2 and V3 would see different masks.  Dropout in the
  model sits directly after the BatchNorm of each branch (cnn_bn2 -> cnn_mc_dropout, gru_bn -> gru_mc_dropout)
  and on the head hidden layer.  The branch pre-dropout features are deterministic, so one forward pass gives them
  and the K dropout draws are applied explicitly (inverted dropout: keep/(1-rate)).  All variants are then computed
  from the SAME K masks (V1 uses draw 0 only), so V2 vs V3 differs ONLY through u_t.  `check_equivalence` verifies,
  by injecting the same masks into pipeline.mc_uattention_predict, that this reproduces protocol item 11 exactly.

CLI:
  python ablation.py seed 11 [--root runs/main] [--out runs/ablation]
  python ablation.py aggregate [--out runs/ablation]
  python ablation.py smoke
"""
import os, sys, json, time, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import pipeline as pl
import tensorflow as tf
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

_THREADS = int(os.environ.get("ABL_THREADS", "1"))
try:
    tf.config.threading.set_intra_op_parallelism_threads(_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(_THREADS)
except RuntimeError:
    pass

EPS = pl.EPS
K_DEFAULT = pl.MC_K
R_DEFAULT = 5                                   # independent mask replicates (rep 0 = primary)
BETAS = (1.0, 3.0)
VARIANTS = ["V0", "V1", "V2", "V3", "V4b1", "V4b3", "V5"]
CORE = ["V0", "V1", "V2", "V3"]
LABEL = {"V0": "V0 Deterministic (dropout off)",
         "V1": "V1 MC K=1, u=0 (old NoUncertainty)",
         "V2": "V2 MC K=10, u=0 (MC averaging only)",
         "V3": "V3 MC K=10 + U-Attention (BlockTrain)",
         "V4b1": "V4 soft gate exp(-1.0 u) [exploratory]",
         "V4b3": "V4 soft gate exp(-3.0 u) [exploratory]",
         "V5": "V5 U-Attention, u permuted over time [control]"}
METRICS = ["f1", "auroc", "auprc", "brier", "nll", "ece"]
HIGHER_BETTER = {"f1": True, "auroc": True, "auprc": True, "brier": False, "nll": False, "ece": False}
SEEDS = pl.SEEDS


# --------------------------------------------------------------------------------------------------
# model parts
# --------------------------------------------------------------------------------------------------
class Parts:
    """Sub-graphs of a trained cnn_bigru_att model (pipeline.build_model layer names)."""

    def __init__(self, model):
        self.model = model
        pl.set_mc(model, False)
        self.feat = tf.keras.Model(model.input, [model.get_layer("cnn_bn2").output, model.get_layer("gru_bn").output])
        self.att = model.get_layer("attention")
        self.hd = model.get_layer("head_dense")
        self.hout = model.get_layer("probability")
        self.r_c = float(model.get_layer("cnn_mc_dropout").rate)
        self.r_g = float(model.get_layer("gru_mc_dropout").rate)
        self.r_h = float(model.get_layer("head_dropout").rate)


def _renorm(at):
    return at / (tf.reduce_sum(at, 1, keepdims=True) + EPS)


def _head(parts, c, mask_h):
    h = parts.hd(c)
    if mask_h is not None:
        h = h * mask_h / (1.0 - parts.r_h)
    return parts.hout(h)[..., 0]


def mc_core(parts, hc, hg, mk_c, mk_g, mk_h, rng_shuf=None, betas=BETAS, with_arrays=False):
    """One batch.  hc,hg: deterministic pre-dropout branch features [B,T,D].  mk_*: boolean keep-masks
    [K,B,T,D] / [K,B,H].  Returns dict variant -> (prob [B], var [B] or None) and optionally u/alpha/alpha_hat of V3."""
    K = mk_c.shape[0]
    zs = tf.concat([hc[None] * tf.cast(mk_c, tf.float32) / (1.0 - parts.r_c),
                    hg[None] * tf.cast(mk_g, tf.float32) / (1.0 - parts.r_g)], -1)       # [K,B,T,2D]
    mh = tf.cast(mk_h, tf.float32)
    out = {}
    # V1: K=1, u=0 -> first draw only
    z1 = zs[0]
    a1 = parts.att.scores(z1)
    c1 = tf.reduce_sum(z1 * a1[..., None], 1)
    out["V1"] = (_head(parts, c1, mh[0]), None)
    # shared quantities for the K draws (protocol item 11)
    zbar = tf.reduce_mean(zs, 0)
    u_raw = tf.reduce_sum(tf.reduce_sum((zs - zbar[None]) ** 2, -1), 0) / (K - 1)
    umin = tf.reduce_min(u_raw, 1, keepdims=True)
    umax = tf.reduce_max(u_raw, 1, keepdims=True)
    u = (u_raw - umin) / (umax - umin + EPS)
    alpha = parts.att.scores(zbar)

    def run(at):
        ah = _renorm(at)
        c = tf.reduce_sum(zbar * ah[..., None], 1)
        ps = tf.stack([_head(parts, c, mh[k]) for k in range(K)], 0)               # [K,B]
        pbar = tf.reduce_mean(ps, 0)
        var = tf.reduce_sum((ps - pbar[None]) ** 2, 0) / (K - 1)
        return pbar, var, ah

    p2, v2, _ = run(alpha)
    out["V2"] = (p2, v2)
    p3, v3, ah3 = run(alpha * (1.0 - u))
    out["V3"] = (p3, v3)
    for b in betas:
        pb, vb, _ = run(alpha * tf.exp(-b * u))
        out["V4b%d" % int(b)] = (pb, vb)
    if rng_shuf is not None:
        perm = np.argsort(rng_shuf.random(tuple(u.shape)), axis=1)
        u_perm = tf.gather(u, perm, axis=1, batch_dims=1)
        p5, v5, _ = run(alpha * (1.0 - u_perm))
        out["V5"] = (p5, v5)
    arrays = dict(u=u.numpy(), alpha=alpha.numpy(), alpha_hat=ah3.numpy()) if with_arrays else None
    return out, arrays


def run_variants(model, X, K=K_DEFAULT, seed=0, split_id=0, R=R_DEFAULT, batch=250, keep_arrays=False):
    """All variants on X.  Dropout masks: numpy Generator seeded with (seed, split_id, rep); identical for every
    variant.  Returns dict with prob[V] ([n] for V0, [R,n] otherwise), var[V], and (rep 0) u/alpha/alpha_hat."""
    parts = Parts(model)
    n, T = len(X), X.shape[1]
    Dc, Dg = int(parts.feat.output[0].shape[-1]), int(parts.feat.output[1].shape[-1])
    H = int(parts.hd.units)
    rngs = [np.random.default_rng(np.random.SeedSequence([int(seed), int(split_id), r, 7])) for r in range(R)]
    shufs = [np.random.default_rng(np.random.SeedSequence([int(seed), int(split_id), r, 99])) for r in range(R)]
    prob = {v: np.zeros((R, n)) for v in VARIANTS if v != "V0"}
    var = {v: np.zeros((R, n)) for v in VARIANTS if v not in ("V0", "V1")}
    prob["V0"] = pl.predict_deterministic(model, X)
    arr = {k: np.zeros((n, T), np.float32) for k in ("u", "alpha", "alpha_hat")} if keep_arrays else None
    for i in range(0, n, batch):
        xb = tf.constant(X[i:i + batch])
        B = int(xb.shape[0])
        hc, hg = parts.feat(xb, training=False)
        for r in range(R):
            g = rngs[r]
            mk_c = g.random((K, B, T, Dc), dtype=np.float32) >= parts.r_c
            mk_g = g.random((K, B, T, Dg), dtype=np.float32) >= parts.r_g
            mk_h = g.random((K, B, H), dtype=np.float32) >= parts.r_h
            out, a = mc_core(parts, hc, hg, mk_c, mk_g, mk_h, rng_shuf=shufs[r], with_arrays=(keep_arrays and r == 0))
            for v, (p, vv) in out.items():
                prob[v][r, i:i + B] = p.numpy()
                if vv is not None:
                    var[v][r, i:i + B] = vv.numpy()
            if a is not None:
                for k in arr:
                    arr[k][i:i + B] = a[k]
    return dict(prob=prob, var=var, arrays=arr)


# --------------------------------------------------------------------------------------------------
# equivalence check against pipeline.mc_uattention_predict (inject identical masks)
# --------------------------------------------------------------------------------------------------
def check_equivalence(model, X, K=K_DEFAULT, seed=123, tol=2e-5):
    """Replace MCDropout.call by a mask-injecting version, run pipeline.mc_uattention_predict (use_u True/False)
    and compare with mc_core on the same masks.  Needs len(X) <= pipeline batch (1000)."""
    parts = Parts(model)
    n, T = len(X), X.shape[1]
    Dc, Dg = int(parts.feat.output[0].shape[-1]), int(parts.feat.output[1].shape[-1]); H = int(parts.hd.units)
    g = np.random.default_rng(seed)
    mk_c = g.random((K, n, T, Dc), dtype=np.float32) >= parts.r_c
    mk_g = g.random((K, n, T, Dg), dtype=np.float32) >= parts.r_g
    mk_h = g.random((K, n, H), dtype=np.float32) >= parts.r_h
    src = {"cnn_mc_dropout": mk_c, "gru_mc_dropout": mk_g, "head_dropout": mk_h}
    cnt = {k: 0 for k in src}
    orig = pl.MCDropout.call

    def patched(self, inputs, training=None):
        if not self.mc:
            return inputs
        m = src[self.name]
        k = cnt[self.name]
        cnt[self.name] += 1
        return inputs * tf.cast(m[k], inputs.dtype) / (1.0 - self.rate)

    pl.MCDropout.call = patched
    try:
        res = {}
        for use_u in (True, False):
            for kk in cnt:
                cnt[kk] = 0
            res[use_u] = pl.mc_uattention_predict(model, X, K=K, use_u=use_u, batch=max(n, 1))
            res[use_u]["calls"] = dict(cnt)
    finally:
        pl.MCDropout.call = orig
        pl.set_mc(model, False)
    hc, hg = parts.feat(tf.constant(X), training=False)
    out, arrs = mc_core(parts, hc, hg, mk_c, mk_g, mk_h, with_arrays=True)
    d3 = float(np.max(np.abs(out["V3"][0].numpy() - res[True]["prob"])))
    d2 = float(np.max(np.abs(out["V2"][0].numpy() - res[False]["prob"])))
    dv3 = float(np.max(np.abs(out["V3"][1].numpy() - res[True]["var"])))
    du = float(np.max(np.abs(arrs["u"] - res[True]["u"])))
    dah = float(np.max(np.abs(arrs["alpha_hat"] - res[True]["alpha_hat"])))
    ok = max(d3, d2, dv3, du, dah) < tol
    return dict(ok=bool(ok), max_abs_diff_prob_V3=d3, max_abs_diff_prob_V2=d2, max_abs_diff_var_V3=dv3,
                max_abs_diff_u=du, max_abs_diff_alpha_hat=dah, tol=tol, mask_calls_V3=res[True]["calls"])


def check_no_dropout(model, X, K=K_DEFAULT):
    """With rate 0 (all keep) every MC variant must equal the deterministic pass."""
    parts = Parts(model)
    n, T = len(X), X.shape[1]
    Dc, Dg = int(parts.feat.output[0].shape[-1]), int(parts.feat.output[1].shape[-1]); H = int(parts.hd.units)
    rc, rg, rh = parts.r_c, parts.r_g, parts.r_h
    parts.r_c = parts.r_g = parts.r_h = 0.0
    hc, hg = parts.feat(tf.constant(X), training=False)
    out, _ = mc_core(parts, hc, hg, np.ones((K, n, T, Dc), bool), np.ones((K, n, T, Dg), bool), np.ones((K, n, H), bool))
    p0 = pl.predict_deterministic(model, X)
    parts.r_c, parts.r_g, parts.r_h = rc, rg, rh
    return {v: float(np.max(np.abs(out[v][0].numpy() - p0))) for v in ("V1", "V2", "V3")}


# --------------------------------------------------------------------------------------------------
# per-seed processing
# --------------------------------------------------------------------------------------------------
def process_seed(seed, main_root, out_root, K=K_DEFAULT, R=R_DEFAULT, equiv=True):
    sd = os.path.join(main_root, "seed%d" % seed)
    od = os.path.join(out_root, "seed%d" % seed)
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    model = pl.load_trained(os.path.join(sd, "checkpoints", "cnn_bigru_att.keras"))
    Xv = np.load(os.path.join(sd, "val_X.npy")); yv = np.load(os.path.join(sd, "val_y.npy")).astype(int)
    Xt = np.load(os.path.join(sd, "test_X.npy"))
    mt = pd.read_csv(os.path.join(sd, "test_meta.csv")); yt = mt["y_true"].values.astype(int)
    log = dict(seed=seed, K=K, R=R, n_val=len(Xv), n_test=len(Xt))
    if equiv:
        log["equivalence_vs_pipeline"] = check_equivalence(model, Xv[:200], K=K)
        log["no_dropout_equals_deterministic"] = check_no_dropout(model, Xv[:200], K=K)
    for si, (sp, X, y) in enumerate([("val", Xv, yv), ("test", Xt, yt)]):
        res = run_variants(model, X, K=K, seed=seed, split_id=si, R=R, keep_arrays=(sp == "test"))
        d = {"y": y}
        for v in VARIANTS:
            d["p_" + v] = res["prob"][v]
        for v in ("V2", "V3"):
            d["var_" + v] = res["var"][v]
        np.savez_compressed(os.path.join(od, "preds_%s.npz" % sp), **d)
        if sp == "test":
            np.savez_compressed(os.path.join(od, "mech_test.npz"), **res["arrays"])
    # cross-check against the saved BlockTrain predictions of pipeline.py (informational)
    chk = {}
    for nm, v in (("cnn_bigru_att", "V0"), ("blocktrain", "V3")):
        f = os.path.join(sd, "preds_%s_test.csv" % nm)
        if os.path.exists(f):
            a = pd.read_csv(f)["prob"].values
            mine = np.load(os.path.join(od, "preds_test.npz"))["p_" + v]
            mine = mine if mine.ndim == 1 else mine[0]
            chk[nm] = dict(max_abs_diff=float(np.max(np.abs(a - mine))), mean_abs_diff=float(np.mean(np.abs(a - mine))),
                           corr=float(np.corrcoef(a, mine)[0, 1]))
    log["vs_main_preds_test"] = chk
    log["seconds"] = time.time() - t0
    json.dump(log, open(os.path.join(od, "seed_log.json"), "w"), indent=1)
    open(os.path.join(od, "SEED_OK"), "w").close()
    print("seed", seed, "done", round(log["seconds"]), "s", json.dumps(log.get("equivalence_vs_pipeline", {})), flush=True)
    return log


# --------------------------------------------------------------------------------------------------
# metrics + bootstrap
# --------------------------------------------------------------------------------------------------
def fast_metrics(y, p, thr):
    """Same definitions as pipeline.metrics (F1 at fixed thr, AUROC, AUPRC, Brier, NLL, ECE-10); faster."""
    pred = p >= thr
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum((~pred) & (y == 1))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    n1 = y.sum(); n0 = len(y) - n1
    r = rankdata(p)
    auroc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    auprc = average_precision_score(y, p)
    pc = np.clip(p, EPS, 1 - EPS)
    return np.array([f1, auroc, auprc, np.mean((p - y) ** 2),
                     -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)), pl.ece(y, p)])


def load_seed(out_root, seed):
    od = os.path.join(out_root, "seed%d" % seed)
    return dict(np.load(os.path.join(od, "preds_val.npz"))), dict(np.load(os.path.join(od, "preds_test.npz"))), \
        dict(np.load(os.path.join(od, "mech_test.npz")))


def prob_rep(d, v, r=0):
    a = d["p_" + v]
    return a if a.ndim == 1 else a[r]


def point_metrics(val, test, v, r=0):
    pv, pt = prob_rep(val, v, r), prob_rep(test, v, r)
    thr = pl.best_threshold(val["y"], pv)
    return pl.metrics(test["y"], pt, thr)


def available_seeds(out_root):
    return [s for s in SEEDS if os.path.exists(os.path.join(out_root, "seed%d" % s, "SEED_OK"))]


def mechanism_stats(mech, steps):
    """Per-window mechanism statistics.  steps: [n,T] bool (True = corrupted step).  The post-CNN temporal length is
    60 (all convolutions use padding='same'), so step t of u/alpha maps 1:1 onto step t of the input window."""
    u, alpha, ah = mech["u"].astype(np.float64), mech["alpha"].astype(np.float64), mech["alpha_hat"].astype(np.float64)
    assert u.shape == steps.shape, (u.shape, steps.shape)
    n, T = steps.shape
    npos = steps.sum(1); nneg = T - npos
    valid = (npos > 0) & (nneg > 0)

    def auc_rows(score):
        r = rankdata(score, axis=1)
        a = (np.where(steps, r, 0).sum(1) - npos * (npos + 1) / 2) / np.maximum(npos * nneg, 1)
        return np.where(valid, a, np.nan)

    ratio = ah / alpha

    def mean_on(x, msk):
        return np.where(msk.sum(1) > 0, (x * msk).sum(1) / np.maximum(msk.sum(1), 1), np.nan)

    def v(x):
        return np.where(valid, x, np.nan)
    rc, rk = v(mean_on(ratio, steps)), v(mean_on(ratio, ~steps))
    return dict(auc_u=auc_rows(u), auc_alpha=auc_rows(alpha), auc_alpha_hat=auc_rows(ah),
                u_corr=v(mean_on(u, steps)), u_clean=v(mean_on(u, ~steps)),
                ratio_corr=rc, ratio_clean=rk, ratio_diff=rc - rk,
                alpha_corr=v(mean_on(alpha, steps)), alpha_clean=v(mean_on(alpha, ~steps)),
                alpha_hat_corr=v(mean_on(ah, steps)), alpha_hat_clean=v(mean_on(ah, ~steps)),
                frac_corr_steps=npos / T)


def summarise_mech(res, atype):
    out = {}
    for grp, msk in (("all_corrupted", np.isin(atype, ["value_noise", "missing_block"])),
                     ("value_noise", atype == "value_noise"), ("missing_block", atype == "missing_block")):
        msk = msk & ~np.isnan(res["auc_u"])
        if msk.sum() < 2:
            continue
        r = dict(n_windows=int(msk.sum()))
        for k in ("auc_u", "auc_alpha", "auc_alpha_hat", "u_corr", "u_clean", "ratio_corr", "ratio_clean", "ratio_diff",
                  "alpha_corr", "alpha_clean", "alpha_hat_corr", "alpha_hat_clean", "frac_corr_steps"):
            r[k + "_mean"] = float(np.mean(res[k][msk]))
        r["auc_u_median"] = float(np.median(res["auc_u"][msk]))
        r["auc_u_gt_0.5_frac"] = float(np.mean(res["auc_u"][msk] > 0.5))
        r["auc_u_sem"] = float(np.std(res["auc_u"][msk], ddof=1) / np.sqrt(msk.sum()))
        r["ratio_corr_lt_clean_frac"] = float(np.mean(res["ratio_corr"][msk] < res["ratio_clean"][msk]))
        r["ratio_diff_sem"] = float(np.std(res["ratio_diff"][msk], ddof=1) / np.sqrt(msk.sum()))
        out[grp] = r
    return out


def aggregate(out_root, main_root, B=2000, variants=VARIANTS, boot_seed=2026):
    seeds = available_seeds(out_root)
    if not seeds:
        raise SystemExit("no finished seeds in " + out_root)
    t0 = time.time()
    rows, store = [], {}
    for s in seeds:
        val, test, mech = load_seed(out_root, s)
        store[s] = (val, test, mech)
        for v in variants:
            rows.append(dict(seed=s, variant=v, label=LABEL[v], **point_metrics(val, test, v, 0)))
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(out_root, "ablation_per_seed.csv"), index=False)
    srows = []
    for v in variants:
        sub = per_seed[per_seed.variant == v]
        r = dict(variant=v, label=LABEL[v], n_seeds=len(sub))
        for m in METRICS + ["threshold"]:
            r[m + "_mean"] = sub[m].mean(); r[m + "_sd"] = sub[m].std(ddof=1) if len(sub) > 1 else np.nan
        srows.append(r)
    pd.DataFrame(srows).to_csv(os.path.join(out_root, "ablation_summary.csv"), index=False)
    # ---- mask-replicate robustness ----
    rrows = []
    for s in seeds:
        val, test, _ = store[s]
        R = test["p_V3"].shape[0]
        for r in range(R):
            for v in [x for x in variants if x != "V0"]:
                rrows.append(dict(seed=s, rep=r, variant=v, **point_metrics(val, test, v, r)))
    rep = pd.DataFrame(rrows)
    rep.to_csv(os.path.join(out_root, "mc_replicates.csv"), index=False)
    # ---- paired bootstrap over test windows ----
    rng = np.random.default_rng(np.random.SeedSequence([boot_seed, 5]))
    boot = {}
    for s in seeds:
        val, test, _ = store[s]
        y = test["y"].astype(int); n = len(y)
        thr = {v: pl.best_threshold(val["y"], prob_rep(val, v, 0)) for v in variants}
        P = {v: prob_rep(test, v, 0) for v in variants}
        for v in variants:
            fm = fast_metrics(y, P[v], thr[v]); pm = pl.metrics(y, P[v], thr[v])
            assert np.allclose(fm, [pm[k] for k in METRICS], atol=1e-9), (s, v, fm, pm)
        arr = np.zeros((B, len(variants), len(METRICS)))
        for b in range(B):
            idx = rng.integers(0, n, n)
            yb = y[idx]
            for vi_, v in enumerate(variants):
                arr[b, vi_] = fast_metrics(yb, P[v][idx], thr[v])
        boot[s] = arr
    vi = {v: i for i, v in enumerate(variants)}
    comps = [("V3", "V2"), ("V3", "V0"), ("V2", "V1"), ("V3", "V1"), ("V4b1", "V2"), ("V4b3", "V2"), ("V3", "V5")]
    comps = [(a, b) for a, b in comps if a in vi and b in vi]
    prow, psrow = [], []
    scheme = ("test windows resampled with replacement independently within each seed (B draws, paired across variants, "
              "thresholds fixed from validation); per draw the seed-level differences are averaged; percentile 95% CI")
    for a, b in comps:
        for mi, m in enumerate(METRICS):
            sgn = 1 if HIGHER_BETTER[m] else -1
            d_seed = {}
            for s in seeds:
                pa = per_seed[(per_seed.seed == s) & (per_seed.variant == a)][m].values[0]
                pb = per_seed[(per_seed.seed == s) & (per_seed.variant == b)][m].values[0]
                d_seed[s] = pa - pb
                db = boot[s][:, vi[a], mi] - boot[s][:, vi[b], mi]
                lo, hi = np.percentile(db, [2.5, 97.5])
                psrow.append(dict(comparison=f"{a}-{b}", metric=m, seed=s, diff=pa - pb, ci_lo=lo, ci_hi=hi,
                                  ci_excludes_0=bool(lo > 0 or hi < 0), first_better=bool(sgn * (pa - pb) > 0)))
            dm = float(np.mean(list(d_seed.values())))
            pooled = np.mean([boot[s][:, vi[a], mi] - boot[s][:, vi[b], mi] for s in seeds], axis=0)
            lo, hi = np.percentile(pooled, [2.5, 97.5])
            row = dict(comparison=f"{a}-{b}", metric=m, n_seeds=len(seeds))
            for s in seeds:
                row[f"diff_seed{s}"] = d_seed[s]
            row.update(mean_diff=dm, sd_seed_diff=float(np.std(list(d_seed.values()), ddof=1)) if len(seeds) > 1 else np.nan,
                       boot_ci_lo=float(lo), boot_ci_hi=float(hi), ci_excludes_0=bool(lo > 0 or hi < 0),
                       n_seeds_first_better=int(sum(sgn * d > 0 for d in d_seed.values())),
                       n_seeds_second_better=int(sum(sgn * d < 0 for d in d_seed.values())),
                       n_seeds_tied=int(sum(d == 0 for d in d_seed.values())), boot_B=B, boot_scheme=scheme)
            if a != "V0" and b != "V0":
                rd = []
                for r in sorted(rep.rep.unique()):
                    dd = []
                    for s in seeds:
                        x = rep[(rep.seed == s) & (rep.rep == r)]
                        dd.append(x[x.variant == a][m].values[0] - x[x.variant == b][m].values[0])
                    rd.append(float(np.mean(dd)))
                row.update(rep_mean_diff_min=min(rd), rep_mean_diff_max=max(rd), n_reps=len(rd))
            prow.append(row)
    pd.DataFrame(prow).to_csv(os.path.join(out_root, "paired_differences.csv"), index=False)
    pd.DataFrame(psrow).to_csv(os.path.join(out_root, "paired_differences_per_seed.csv"), index=False)
    # ---- mechanism ----
    mech_rows, allw, types = [], {}, []
    for s in seeds:
        _, _, mech = store[s]
        meta = pd.read_csv(os.path.join(main_root, "seed%d" % s, "test_meta.csv"))
        steps = np.load(os.path.join(main_root, "seed%d" % s, "test_corrupted_steps.npy")).astype(bool)
        res = mechanism_stats(mech, steps)
        for k, v in res.items():
            allw.setdefault(k, []).append(v)
        types.append(meta["anomaly_type"].values)
        for grp, r in summarise_mech(res, meta["anomaly_type"].values).items():
            mech_rows.append(dict(seed=s, group=grp, **r))
    pooled_res = {k: np.concatenate(v) for k, v in allw.items()}
    for grp, r in summarise_mech(pooled_res, np.concatenate(types)).items():
        mech_rows.append(dict(seed="pooled", group=grp, **r))
    pd.DataFrame(mech_rows).to_csv(os.path.join(out_root, "mechanism_check.csv"), index=False)
    print("aggregate done", seeds, round(time.time() - t0), "s")
    return seeds


# --------------------------------------------------------------------------------------------------
# figure + LaTeX tables (numbers read from the CSVs only)
# --------------------------------------------------------------------------------------------------
METRIC_NAME = {"f1": "F1", "auroc": "AUROC", "auprc": "AUPRC", "brier": "Brier", "nll": "NLL", "ece": "ECE"}
SHORT = {"V0": "V0", "V1": "V1", "V2": "V2", "V3": "V3"}


def make_figure(out_root, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    ps = pd.read_csv(os.path.join(out_root, "ablation_per_seed.csv"))
    sm = pd.read_csv(os.path.join(out_root, "ablation_summary.csv")).set_index("variant")
    cols = {"V0": "#6b7280", "V1": "#b45309", "V2": "#2563eb", "V3": "#047857"}
    arrow = {m: ("higher is better" if HIGHER_BETTER[m] else "lower is better") for m in METRICS}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    xs = np.arange(len(CORE))
    for ax, m in zip(axes.ravel(), METRICS):
        for i, v in enumerate(CORE):
            sub = ps[ps.variant == v][m].values
            jit = np.linspace(-0.12, 0.12, len(sub)) if len(sub) > 1 else np.zeros(1)
            ax.scatter(i + jit, sub, s=9, color=cols[v], alpha=0.35, linewidths=0, zorder=2)
            ax.errorbar(i, sm.loc[v, m + "_mean"], yerr=sm.loc[v, m + "_sd"], fmt="o", ms=5.5, color=cols[v],
                        ecolor=cols[v], elinewidth=1.4, capsize=3, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels(CORE)
        ax.set_ylabel(METRIC_NAME[m] + (r" $\uparrow$" if HIGHER_BETTER[m] else r" $\downarrow$"))
        ax.grid(axis="y", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(-0.6, len(CORE) - 0.4)
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "fig_ablation.pdf"))
    fig.savefig(os.path.join(fig_dir, "fig_ablation.png"), dpi=300)
    plt.close(fig)


def _f(x, d=3):
    return ("%." + str(d) + "f") % x


def _tex_num(x, d=3):
    s = _f(x, d)
    return s.replace("-", "$-$") if s.startswith("-") else s


def _signed(x, d=3):
    s = _f(x, d)
    if s.startswith("-"):
        return "$-$" + s[1:]
    return "$+$" + s


def build_tex_tables(out_root):
    sm = pd.read_csv(os.path.join(out_root, "ablation_summary.csv")).set_index("variant")
    pdiff = pd.read_csv(os.path.join(out_root, "paired_differences.csv"))
    nseed = int(sm.loc["V3", "n_seeds"])
    t1 = [r"\begin{table}[t]", r"\centering",
          r"\caption{Isolating the effect of U-Attention on the same trained CNN--BiGRU--Attention checkpoints "
          r"(test set, mean $\pm$ SD over %d seeds). V0: dropout off, single pass. V1: MC dropout $K{=}1$, $u_t{=}0$ "
          r"(the former ``NoUncertainty'' setting). V2: MC dropout $K{=}10$, $u_t{=}0$ (MC averaging only). V3: MC dropout $K{=}10$ with U-Attention "
          r"(BlockTrain). V1--V3 share the same dropout masks. F1 uses a threshold fixed on the validation predictions of each variant. "
          r"ECE uses 10 bins.}" % nseed,
          r"\label{tab:ablation_uattention}", r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
          r"Variant & F1 $\uparrow$ & AUROC $\uparrow$ & AUPRC $\uparrow$ & Brier $\downarrow$ & NLL $\downarrow$ & ECE $\downarrow$ \\", r"\midrule"]
    names = {"V0": r"V0 Deterministic", "V1": r"V1 MC $K{=}1$, $u_t{=}0$", "V2": r"V2 MC $K{=}10$, $u_t{=}0$",
             "V3": r"V3 MC $K{=}10$ + U-Attention"}
    for v in CORE:
        cells = [r"%s$\pm$%s" % (_f(sm.loc[v, m + "_mean"]), _f(sm.loc[v, m + "_sd"])) for m in METRICS]
        t1.append(names[v] + " & " + " & ".join(cells) + r" \\")
    t1 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    t2 = [r"\begin{table}[t]", r"\centering",
          r"\caption{Paired differences between inference variants on identical checkpoints and test windows. Each cell: mean over %d seeds of the per-seed "
          r"difference (first minus second variant), 95\%% bootstrap CI over test windows (windows resampled independently within each seed, "
          r"seed-level differences averaged per draw, %d draws), and in the last column the number of seeds in which the first variant is better "
          r"(higher F1/AUROC/AUPRC, lower Brier/NLL/ECE).}" % (nseed, int(pdiff.boot_B.iloc[0])),
          r"\label{tab:ablation_paired}", r"\small", r"\begin{tabular}{llccc}", r"\toprule",
          r"Comparison & Metric & Mean diff. & 95\% CI & Seeds favouring first \\", r"\midrule"]
    lab = {"V3-V2": r"V3 $-$ V2 (U-Attention)", "V3-V0": r"V3 $-$ V0 (BlockTrain vs deterministic)", "V2-V1": r"V2 $-$ V1 (MC averaging)"}
    for comp in ("V3-V2", "V3-V0", "V2-V1"):
        sub = pdiff[pdiff.comparison == comp].set_index("metric")
        for i, m in enumerate(METRICS):
            r = sub.loc[m]
            t2.append((lab[comp] if i == 0 else "") + " & " + METRIC_NAME[m] + " & " + _signed(r.mean_diff, 4) + " & [" +
                      _signed(r.boot_ci_lo, 4) + ", " + _signed(r.boot_ci_hi, 4) + "] & %d/%d \\\\" % (r.n_seeds_first_better, r.n_seeds))
        if comp != "V2-V1":
            t2.append(r"\midrule")
    t2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t1), "\n".join(t2)


def build_tex_exploratory(out_root):
    pdiff = pd.read_csv(os.path.join(out_root, "paired_differences.csv"))
    nseed = int(pdiff.n_seeds.iloc[0])
    t = [r"\begin{table}[t]", r"\centering",
         r"\caption{Exploratory variants (not part of the protocol), difference to V2 (same masks, $K{=}10$). V4: soft gate $\alpha_t\exp(-\beta u_t)$. "
         r"V5 (control): U-Attention with $u_t$ randomly permuted over time inside each window (V3 $-$ V5). Mean difference over %d seeds with 95\%% bootstrap CI and seeds favouring the first variant.}" % nseed,
         r"\label{tab:ablation_exploratory}", r"\small", r"\begin{tabular}{llccc}", r"\toprule",
         r"Comparison & Metric & Mean diff. & 95\% CI & Seeds favouring first \\", r"\midrule"]
    lab = {"V4b1-V2": r"V4 $\beta{=}1$ $-$ V2", "V4b3-V2": r"V4 $\beta{=}3$ $-$ V2", "V3-V5": r"V3 $-$ V5 (permuted $u_t$)"}
    for comp in ("V4b1-V2", "V4b3-V2", "V3-V5"):
        sub = pdiff[pdiff.comparison == comp].set_index("metric")
        for i, m in enumerate(METRICS):
            r = sub.loc[m]
            t.append((lab[comp] if i == 0 else "") + " & " + METRIC_NAME[m] + " & " + _signed(r.mean_diff, 4) + " & [" +
                     _signed(r.boot_ci_lo, 4) + ", " + _signed(r.boot_ci_hi, 4) + "] & %d/%d \\\\" % (r.n_seeds_first_better, r.n_seeds))
        if comp != "V3-V5":
            t.append(r"\midrule")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t)


def build_tex_mechanism(out_root):
    mc = pd.read_csv(os.path.join(out_root, "mechanism_check.csv"))
    mc = mc[mc.seed.astype(str) == "pooled"].set_index("group")
    t = [r"\begin{table}[t]", r"\centering",
         r"\caption{Mechanism check on corrupted test windows (pooled over seeds). For each corrupted window: AUROC of $u_t$ for separating corrupted from clean time steps; "
         r"mean of $\hat\alpha_t/\alpha_t$ on corrupted and on clean steps (a value below 1 means that the step is down-weighted by U-Attention). Entries are means over windows.}",
         r"\label{tab:ablation_mechanism}", r"\small", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Corruption & Windows & AUROC of $u_t$ & Fraction AUROC $>0.5$ & $\hat\alpha_t/\alpha_t$ corrupted & $\hat\alpha_t/\alpha_t$ clean \\", r"\midrule"]
    for g, nm in (("value_noise", "Value noise"), ("missing_block", "Missing block"), ("all_corrupted", "All corrupted")):
        if g in mc.index:
            r = mc.loc[g]
            t.append("%s & %d & %s & %s & %s & %s \\\\" % (nm, r.n_windows, _f(r.auc_u_mean), _f(r["auc_u_gt_0.5_frac"]),
                                                          _f(r.ratio_corr_mean), _f(r.ratio_clean_mean)))
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t)


def _classify(pdiff, comp):
    """Per metric: 'better' / 'worse' (95% CI excludes 0, from the point of view of the FIRST variant) or 'n.s.'."""
    out = {}
    sub = pdiff[pdiff.comparison == comp].set_index("metric")
    for m in METRICS:
        r = sub.loc[m]
        sgn = 1 if HIGHER_BETTER[m] else -1
        if r.boot_ci_lo > 0 or r.boot_ci_hi < 0:
            out[m] = "better" if sgn * r.mean_diff > 0 else "worse"
        else:
            out[m] = "n.s."
    return out


def _dtxt(pdiff, comp, m, d=4):
    r = pdiff[(pdiff.comparison == comp) & (pdiff.metric == m)].iloc[0]
    return "%s (95\\%% CI %s to %s)" % (_signed(r.mean_diff, d), _signed(r.boot_ci_lo, d), _signed(r.boot_ci_hi, d))


def _sign_txt(pdiff, comp, metrics):
    sub = pdiff[pdiff.comparison == comp].set_index("metric")
    return ", ".join("%s %d/%d" % (METRIC_NAME[m], sub.loc[m, "n_seeds_first_better"], sub.loc[m, "n_seeds"]) for m in metrics)


def build_paragraph(out_root):
    pdiff = pd.read_csv(os.path.join(out_root, "paired_differences.csv"))
    mc = pd.read_csv(os.path.join(out_root, "mechanism_check.csv"))
    mc = mc[mc.seed.astype(str) == "pooled"].set_index("group")
    ns = int(pdiff.n_seeds.iloc[0])
    c32, c21, c30 = _classify(pdiff, "V3-V2"), _classify(pdiff, "V2-V1"), _classify(pdiff, "V3-V0")
    rows = pdiff[pdiff.comparison == "V3-V2"].set_index("metric")
    p = []
    p.append(r"\paragraph{Isolating the effect of U-Attention.} "
             r"The earlier ``NoUncertainty'' ablation used $K{=}1$ and therefore changed two things at once: the number of Monte Carlo samples and the "
             r"U-Attention modulation. We therefore re-evaluated the same trained CNN--BiGRU--Attention checkpoint of every seed under four inference variants "
             r"(Table~\ref{tab:ablation_uattention}, Fig.~\ref{fig:ablation_uattention}): V0, dropout off; V1, MC dropout with $K{=}1$ and $u_t{=}0$; "
             r"V2, $K{=}10$ with $u_t{=}0$; V3, $K{=}10$ with U-Attention (BlockTrain). V1--V3 use the same dropout masks, so V2 and V3 differ only through $u_t$. "
             r"Paired differences are given in Table~\ref{tab:ablation_paired} with 95\%% bootstrap intervals over test windows and the number of seeds out of %d favouring the first variant." % ns)
    # MC averaging
    nb = [m for m in METRICS if c21[m] == "better"]; nw = [m for m in METRICS if c21[m] == "worse"]
    if nb and not nw:
        s21 = (r"Averaging $K{=}10$ samples without U-Attention (V2 $-$ V1) improved %s with intervals excluding zero (e.g.\ AUROC %s)."
               % (", ".join(METRIC_NAME[m] for m in nb), _dtxt(pdiff, "V2-V1", "auroc")))
    elif not nb and not nw:
        s21 = r"Averaging $K{=}10$ samples without U-Attention (V2 $-$ V1) changed no metric beyond sampling uncertainty (AUROC %s)." % _dtxt(pdiff, "V2-V1", "auroc")
    else:
        s21 = (r"Averaging $K{=}10$ samples without U-Attention (V2 $-$ V1) gave mixed results: better %s, worse %s (AUROC %s)."
               % (", ".join(METRIC_NAME[m] for m in nb) or "none", ", ".join(METRIC_NAME[m] for m in nw) or "none", _dtxt(pdiff, "V2-V1", "auroc")))
    p.append(s21)
    # U-Attention effect
    nb = [m for m in METRICS if c32[m] == "better"]; nw = [m for m in METRICS if c32[m] == "worse"]
    base = (r"Adding U-Attention on top of MC averaging (V3 $-$ V2) gave AUROC %s, AUPRC %s, F1 %s, Brier %s, NLL %s and ECE %s; the number of seeds favouring V3 was %s."
            % (_dtxt(pdiff, "V3-V2", "auroc"), _dtxt(pdiff, "V3-V2", "auprc"), _dtxt(pdiff, "V3-V2", "f1"),
               _dtxt(pdiff, "V3-V2", "brier"), _dtxt(pdiff, "V3-V2", "nll"), _dtxt(pdiff, "V3-V2", "ece"), _sign_txt(pdiff, "V3-V2", METRICS)))
    p.append(base)
    if len(nw) == len(METRICS):
        concl = r"In this experiment U-Attention therefore did not improve discrimination or calibration beyond MC averaging; the pooled intervals favour V2 on all six metrics."
        mixed = [m for m in METRICS if rows.loc[m, "n_seeds_first_better"] > 0]
        if mixed:
            concl += (r" The direction was not consistent across seeds for %s (seeds favouring V3: %s), and the corresponding differences are small."
                      % (", ".join(METRIC_NAME[m] for m in mixed), ", ".join("%s %d/%d" % (METRIC_NAME[m], rows.loc[m, "n_seeds_first_better"], ns) for m in mixed)))
    elif len(nb) == len(METRICS):
        concl = r"In this experiment U-Attention therefore improved all six metrics beyond MC averaging, with intervals excluding zero."
    elif not nb and not nw:
        concl = r"In this experiment U-Attention therefore produced no effect on discrimination or calibration that can be distinguished from zero beyond MC averaging."
    else:
        concl = (r"In this experiment the effect of U-Attention beyond MC averaging was mixed: better on %s, worse on %s, indistinguishable from zero on the rest."
                 % (", ".join(METRIC_NAME[m] for m in nb) or "no metric", ", ".join(METRIC_NAME[m] for m in nw) or "no metric"))
    p.append(concl)
    # replicate robustness
    r = rows.loc["auroc"]
    if "rep_mean_diff_min" in rows.columns:
        p.append(r"Repeating the dropout draws %d times changed the mean AUROC difference only within [%s, %s], so the result is not an artefact of one set of masks."
                 % (int(r.n_reps), _signed(r.rep_mean_diff_min, 4), _signed(r.rep_mean_diff_max, 4)))
    # V3 vs V0
    p.append(r"Relative to the deterministic model (V3 $-$ V0), AUROC changed by %s and NLL by %s; the seeds favouring V3 were %s."
             % (_dtxt(pdiff, "V3-V0", "auroc"), _dtxt(pdiff, "V3-V0", "nll"), _sign_txt(pdiff, "V3-V0", ["auroc", "nll"])))
    # mechanism
    if "all_corrupted" in mc.index:
        a = mc.loc["all_corrupted"]
        t_u = ("so $u_t$ is on average higher on corrupted steps, as intended" if a.auc_u_mean > 0.5 else
               "so $u_t$ did not mark corrupted steps")
        t_a = (r"The unmodulated attention weights themselves separated corrupted from clean steps with a mean AUROC of %s, i.e.\ the trained classifier already attends preferentially to the corrupted steps"
               % _f(a.auc_alpha_mean)) if a.auc_alpha_mean > 0.5 else (r"The unmodulated attention weights separated corrupted from clean steps with a mean AUROC of %s" % _f(a.auc_alpha_mean))
        t_r = (", and U-Attention reduces the relative weight of these steps" if a.ratio_corr_mean < a.ratio_clean_mean
               else ", and U-Attention does not reduce the relative weight of these steps")
        p.append(r"A direct check of the mechanism (Table~\ref{tab:ablation_mechanism}) used the stored corrupted time steps; the post-CNN length equals the window length (60), so steps map one to one. "
                 r"Within corrupted windows, $u_t$ separated corrupted from clean steps with a mean AUROC of %s (fraction of windows above 0.5: %s), %s. "
                 r"The ratio $\hat\alpha_t/\alpha_t$ averaged %s on corrupted and %s on clean steps (values above 1 arise from renormalisation of the weights). %s%s."
                 % (_f(a.auc_u_mean), _f(a["auc_u_gt_0.5_frac"], 2), t_u, _f(a.ratio_corr_mean), _f(a.ratio_clean_mean), t_a, t_r))
    # control
    cc = _classify(pdiff, "V3-V5")
    p.append(r"As a control, replacing $u_t$ by a random permutation of the same values over time within each window (V5) gave AUROC %s relative to V3 (V3 $-$ V5), "
             r"and a soft gate $\alpha_t e^{-\beta u_t}$ with $\beta{=}1$ and $\beta{=}3$ gave AUROC differences to V2 of %s and %s (Table~\ref{tab:ablation_exploratory}; exploratory, not part of the protocol)."
             % (_dtxt(pdiff, "V3-V5", "auroc"), _dtxt(pdiff, "V4b1-V2", "auroc"), _dtxt(pdiff, "V4b3-V2", "auroc")))
    p.append(r"These results come from one data set, one corruption protocol and %d training seeds; the bootstrap intervals capture test-window sampling only, whereas variation between seeds is shown by the per-seed differences and the seed counts." % ns)
    return " ".join(p)


def write_tex(out_root, tex_path):
    t1, t2 = build_tex_tables(out_root)
    t3 = build_tex_exploratory(out_root)
    t4 = build_tex_mechanism(out_root)
    fig = "\n".join([r"\begin{figure}[t]", r"\centering", r"\includegraphics[width=\linewidth]{figs/fig_ablation.pdf}",
                     r"\caption{Test-set metrics of the inference variants V0--V3 on the same trained CNN--BiGRU--Attention checkpoints. Large markers: mean over seeds; "
                     r"error bars: SD over seeds; small markers: individual seeds. Values are those of Table~\ref{tab:ablation_uattention}.}",
                     r"\label{fig:ablation_uattention}", r"\end{figure}"])
    txt = ["% Auto-generated by ablation.py (write_tex) from runs/ablation/*.csv -- do not edit numbers by hand.",
           "% ---- Section 5.4: Isolating the effect of U-Attention ----", build_paragraph(out_root), "", t1, "", fig, "", t2, "", t4, "",
           "% ---- exploratory variants (may go to an appendix) ----", t3, ""]
    open(tex_path, "w").write("\n".join(txt))


# --------------------------------------------------------------------------------------------------
# smoke test with a tiny model
# --------------------------------------------------------------------------------------------------
def smoke(out_base):
    t0 = time.time()
    root = os.path.join(out_base, "_smoke")
    main = os.path.join(root, "main"); out = os.path.join(root, "out")
    os.makedirs(os.path.join(main, "seed11", "checkpoints"), exist_ok=True)
    S = pl.make_split(11, 600, 300, 400)
    tf.keras.utils.set_random_seed(11)
    m = pl.build_model("cnn_bigru_att")
    pl.train_model(m, S["X_train"], S["y_train"], S["X_val"], S["y_val"], batch=64, epochs=1, patience=3)
    sd = os.path.join(main, "seed11")
    m.save(os.path.join(sd, "checkpoints", "cnn_bigru_att.keras"))
    np.save(os.path.join(sd, "val_X.npy"), S["X_val"]); np.save(os.path.join(sd, "val_y.npy"), S["y_val"])
    np.save(os.path.join(sd, "test_X.npy"), S["X_test"])
    np.save(os.path.join(sd, "test_corrupted_steps.npy"), S["steps_test"])
    S["meta_test"].to_csv(os.path.join(sd, "test_meta.csv"), index=False)
    print("smoke model ready", round(time.time() - t0), "s", flush=True)
    log = process_seed(11, main, out, K=10, R=2)
    print(json.dumps(log, indent=1))
    aggregate(out, main, B=50)
    pd.set_option("display.width", 250)
    print(pd.read_csv(os.path.join(out, "ablation_per_seed.csv")).drop(columns="label").round(4).to_string())
    print(pd.read_csv(os.path.join(out, "paired_differences.csv")).iloc[:, :12].round(4).to_string())
    print(pd.read_csv(os.path.join(out, "mechanism_check.csv")).round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed", "aggregate", "report", "smoke"])
    ap.add_argument("seeds", nargs="*", type=int)
    ap.add_argument("--root", default=os.path.join(ROOT, "runs", "main"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "ablation"))
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--R", type=int, default=R_DEFAULT)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--no_equiv", action="store_true")
    a = ap.parse_args()
    if a.cmd == "seed":
        for s in a.seeds:
            process_seed(s, a.root, a.out, K=a.K, R=a.R, equiv=not a.no_equiv)
    elif a.cmd == "aggregate":
        aggregate(a.out, a.root, B=a.B)
    elif a.cmd == "report":
        make_figure(a.out, os.path.join(ROOT, "figs"))
        write_tex(a.out, os.path.join(ROOT, "text", "B_ablation.tex"))
    else:
        smoke(a.out)
