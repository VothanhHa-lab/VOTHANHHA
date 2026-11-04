#!/usr/bin/env python3
"""
BlockTrain re-run: one clean pipeline that implements Sections 3-4 of the manuscript (see README.md)
(which follows manuscript Sections 3-4).

Importable API (names are stable):
    load_data()                         -> dict with per-participant 3-min grids
    make_participant_split()            -> dict(train=[...], val=[...], test=[...], rng_seed=..)
    make_split(seed, n_train, n_val, n_test) -> dict with X/y/meta/corrupted_steps per split
    corrupt_window(...)                 -> corruption of ONE raw window (documented in the function)
    Preprocessor                        -> train-only median imputation + z-score
    build_model(name)                   -> Keras model, name in MODEL_NAMES
    load_trained(path)                  -> load a saved checkpoint (custom layers registered)
    set_mc(model, on)                   -> switch MCDropout layers between deterministic / stochastic
    predict_deterministic(model, X)     -> dropout OFF, single pass
    mc_uattention_predict(model, X, K=10, use_u=True) -> dict(prob, var, entropy, u, alpha, alpha_hat)
    metrics(y, p, thr), best_threshold(y, p), ece(y, p)

CLI:
    python pipeline.py run     [--seeds 11 22 ..] [--n_train N] [--batch B] [--epochs E] [--patience P]
    python pipeline.py mcsens  [--ks 5 10 20 30]
    python pipeline.py smoke
"""
import os, sys, json, time, hashlib, argparse
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve)

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "input", "BlockTrain_Option1_Wearable_TimeSeries_606641.csv")
CHANNELS = ["spo2", "stress_level", "energy_level", "epoch_to_breaths"]
MODEL_NAMES = ["cnn", "bigru", "cnn_bigru", "cnn_bigru_att"]
PRED_NAMES = MODEL_NAMES + ["blocktrain"]
SEEDS = [11, 22, 33, 44, 55]

# ---------------- protocol constants (protocol items 4-8, 9, 11) ----------------
T = 60                    # window length (steps of 3 min)
MIN_OBS_VALID = 10        # a window is valid if some channel is observed in >= 10 of 60 steps
MIN_OBS_CORRUPT = 5       # corruption channel must have >= 5 observed steps in the window
P_NOISE = 0.6             # probability of value_noise among corrupted windows
SEG_MIN, SEG_MAX = 10, 30  # segment length ~ U{10..30}
NOISE_SD_MULT = 1.0       # noise sd = 1.0 * sigma_j (training-participant SD, raw units)
SPLIT_RNG_SEED = 2026
N_TRAIN_DEFAULT, N_VAL_DEFAULT, N_TEST_DEFAULT = 20000, 2000, 4000
DROPOUT, L2, LR = 0.30, 1e-4, 1e-3
WIDTH = 64
MC_K = 10
EPS = 1e-7


# =====================================================================
# 1. Data loading, 3-min grid, masks
# =====================================================================
_DATA = None


def load_data(csv=CSV, force=False):
    """Read raw csv -> per-participant 3-min grids.

    Steps (protocol 1-3): sort by participant/timestamp, drop duplicate (participant,timestamp),
    stress_level -1/-2 -> NaN, floor timestamp to 3 min, median per bin, grid spans first->last
    record of each participant (empty bins stay NaN).
    Returns dict: pids (sorted full hashes), grid[pid] = float32 array [n_bins, 4] with NaN,
    t0[pid] = Timestamp of bin 0, n_raw_rows[pid], valid_starts[pid] = int array of valid window starts.
    """
    global _DATA
    if _DATA is not None and not force:
        return _DATA
    d = pd.read_csv(csv, sep=";", usecols=range(6))
    d.columns = ["timestamp", "pid"] + CHANNELS
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    n_rows_raw = len(d)
    d.loc[d["stress_level"] < 0, "stress_level"] = np.nan          # sentinels -1, -2
    d = d.sort_values(["pid", "timestamp"], kind="mergesort")
    d = d.drop_duplicates(["pid", "timestamp"], keep="first")
    n_rows_dedup = len(d)
    d["bin"] = d["timestamp"].dt.floor("3min")
    out = dict(grid={}, t0={}, n_raw_rows={}, valid_starts={}, n_bins={}, obs_frac={})
    for pid, g in d.groupby("pid", sort=True):
        agg = g.groupby("bin")[CHANNELS].median()                # NaN-skipping median per bin
        t0, t1 = agg.index.min(), agg.index.max()
        full = pd.date_range(t0, t1, freq="3min")
        agg = agg.reindex(full)
        out["grid"][pid] = agg.to_numpy(dtype=np.float32)
        out["t0"][pid] = t0
        out["n_raw_rows"][pid] = int(len(g))
        out["n_bins"][pid] = int(len(full))
        out["obs_frac"][pid] = np.isfinite(out["grid"][pid]).mean(0).tolist()
        out["valid_starts"][pid] = _valid_starts(out["grid"][pid])
    out["pids"] = sorted(out["grid"].keys())
    out["n_rows_raw"], out["n_rows_dedup"] = n_rows_raw, n_rows_dedup
    _DATA = out
    return out


def _valid_starts(grid):
    """Window start indices s such that grid[s:s+T] lies fully in the grid (always consecutive bins)
    and at least one channel is observed in >= MIN_OBS_VALID of its T steps."""
    n = grid.shape[0]
    if n < T:
        return np.zeros(0, dtype=np.int64)
    obs = np.isfinite(grid).astype(np.int32)
    cs = np.vstack([np.zeros((1, 4), np.int32), np.cumsum(obs, 0)])
    cnt = cs[T:] - cs[:-T]                                        # [n-T+1, 4]
    return np.nonzero((cnt >= MIN_OBS_VALID).any(1))[0].astype(np.int64)


def short(pid):
    return pid[:8]


def make_participant_split(data=None, rng_seed=SPLIT_RNG_SEED):
    """Fixed participant split: 9 train / 2 val / 2 test, RNG seed 2026 (protocol 4).
    The 13 participants are sorted by hash, permuted with default_rng(seed); first 9 train, next 2 val,
    last 2 test. If a val/test participant has < 2000 valid windows, redraw with seed+1, ... (recorded)."""
    data = data or load_data()
    pids = data["pids"]
    tried = []
    s = rng_seed
    while True:
        perm = np.random.default_rng(s).permutation(len(pids))
        tr = [pids[i] for i in perm[:9]]
        va = [pids[i] for i in perm[9:11]]
        te = [pids[i] for i in perm[11:13]]
        nv = {short(p): int(len(data["valid_starts"][p])) for p in va + te}
        ok = all(v >= 2000 for v in nv.values())
        tried.append(dict(rng_seed=s, val=[short(p) for p in va], test=[short(p) for p in te],
                          valid_windows=nv, accepted=bool(ok)))
        if ok:
            break
        s += 1
    return dict(train=tr, val=va, test=te, rng_seed=s, attempts=tried)


# =====================================================================
# 2. Preprocessing (fit on TRAIN participants' raw grid only)
# =====================================================================
class Preprocessor:
    """Per-channel stats from training participants' grid data:
    median of observed values -> imputation value; sd/min/max of observed values (raw units, for the
    corruption); mean/sd of the median-imputed training grid -> z-score."""

    def fit(self, data, train_pids):
        g = np.concatenate([data["grid"][p] for p in train_pids], 0)       # [N,4] with NaN
        self.median = np.nanmedian(g, 0)
        self.raw_sd = np.nanstd(g, 0)
        self.raw_min = np.nanmin(g, 0)
        self.raw_max = np.nanmax(g, 0)
        gi = np.where(np.isfinite(g), g, self.median[None, :])
        self.mean = gi.mean(0)
        self.sd = gi.std(0)
        return self

    def transform(self, Xraw):
        """Xraw [N,T,4] raw with NaN (after corruption) -> X [N,T,8] = 4 z-scored values + 4 masks."""
        m = np.isfinite(Xraw)
        xi = np.where(m, Xraw, self.median[None, None, :])
        z = (xi - self.mean[None, None, :]) / self.sd[None, None, :]
        return np.concatenate([z, m.astype(np.float32)], -1).astype(np.float32)

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ["median", "raw_sd", "raw_min", "raw_max", "mean", "sd"]}


# =====================================================================
# 3. Corruption (label-1 windows only; contract 7)
# =====================================================================
def corrupt_window(xraw, rng, pre):
    """Corrupt ONE raw window xraw [T,4] (NaN = missing). Applied in raw units BEFORE imputation.

    type: value_noise (p=0.6) or missing_block (p=0.4).
    channel: uniform among channels with >= 5 observed steps in the window.
    segment: contiguous, length L ~ U{10..30}, start ~ U{0..T-L}. The start is redrawn (max 50 tries)
             until the segment contains >= 1 observed step of the chosen channel, so that the corruption
             actually changes the window (DEVIATION/clarification of the protocol: otherwise a label-1
             window could be identical to its clean version).
    value_noise: add N(0,(1.0*sigma_j)^2) to the OBSERVED values in the segment, clip to the training
             [min,max] of the channel; masks unchanged.
    missing_block: set values in the segment to NaN (mask -> 0).
    Returns (x_corrupted, info) with info = dict(type, channel, start, length, n_affected, steps[T] bool).
    """
    x = xraw.copy()
    obs = np.isfinite(x)
    cand = np.nonzero(obs.sum(0) >= MIN_OBS_CORRUPT)[0]
    typ = "value_noise" if rng.random() < P_NOISE else "missing_block"
    ch = int(rng.choice(cand))
    L = int(rng.integers(SEG_MIN, SEG_MAX + 1))
    for _ in range(50):
        s = int(rng.integers(0, T - L + 1))
        if obs[s:s + L, ch].any():
            break
    seg = slice(s, s + L)
    o = obs[seg, ch]
    n_aff = int(o.sum())
    if typ == "value_noise":
        noise = rng.normal(0.0, NOISE_SD_MULT * pre.raw_sd[ch], size=L)
        v = x[seg, ch]
        v = np.where(o, np.clip(v + noise, pre.raw_min[ch], pre.raw_max[ch]), np.nan)
        x[seg, ch] = v
    else:
        x[seg, ch] = np.nan
    steps = np.zeros(T, bool)
    steps[seg] = True
    return x, dict(type=typ, channel=ch, start=s, length=L, n_affected=n_aff, steps=steps)


# =====================================================================
# 4. Window sampling + split construction
# =====================================================================
def _sample_split(data, pids, n, rng, pre):
    """Sample n windows uniformly (without replacement) from the pooled valid windows of `pids`;
    half -> label 0 (clean), half -> label 1 (corrupted). No clean/corrupted twin of the same window."""
    pool = np.concatenate([np.stack([np.full(len(data["valid_starts"][p]), i),
                                     data["valid_starts"][p]], 1) for i, p in enumerate(pids)], 0)
    idx = rng.choice(len(pool), size=n, replace=False)
    sel = pool[idx]
    y = np.zeros(n, np.int8)
    y[rng.permutation(n)[: n // 2]] = 1
    Xraw = np.empty((n, T, 4), np.float32)
    for k, (pi, s) in enumerate(sel):
        Xraw[k] = data["grid"][pids[pi]][s:s + T]
    rows, steps = [], np.zeros((n, T), bool)
    Xc = Xraw.copy()
    for k in range(n):
        if y[k] == 1:
            Xc[k], info = corrupt_window(Xraw[k], rng, pre)
            steps[k] = info["steps"]
            rows.append((info["type"], info["channel"], info["start"], info["length"], info["n_affected"]))
        else:
            rows.append(("clean", -1, -1, 0, 0))
    meta = pd.DataFrame(rows, columns=["anomaly_type", "channel", "seg_start", "seg_len", "n_affected"])
    meta.insert(0, "y_true", y.astype(int))
    meta.insert(0, "sample_id", np.arange(n))
    meta["participant"] = [short(pids[pi]) for pi in sel[:, 0]]
    meta["window_start"] = sel[:, 1]
    X = pre.transform(Xc)
    return X, y, meta, steps


def make_split(seed, n_train=N_TRAIN_DEFAULT, n_val=N_VAL_DEFAULT, n_test=N_TEST_DEFAULT, data=None):
    """Build train/val/test arrays for `seed`. Participant split fixed; windows/corruption depend on seed.
    Returns dict: pre, psplit, and for s in train/val/test: X_s [N,60,8], y_s, meta_s, steps_s [N,60]."""
    data = data or load_data()
    ps = make_participant_split(data)
    pre = Preprocessor().fit(data, ps["train"])
    out = dict(pre=pre, psplit=ps, seed=seed)
    for i, (name, n) in enumerate([("train", n_train), ("val", n_val), ("test", n_test)]):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), i]))
        X, y, meta, steps = _sample_split(data, ps[name], n, rng, pre)
        out["X_" + name], out["y_" + name], out["meta_" + name], out["steps_" + name] = X, y, meta, steps
    return out


def split_hash(meta):
    h = hashlib.sha1(pd.util.hash_pandas_object(meta[["participant", "window_start", "y_true",
                                                      "anomaly_type", "channel", "seg_start", "seg_len"]],
                                                index=False).values.tobytes())
    return h.hexdigest()[:12]


# =====================================================================
# 5. Models
# =====================================================================
@tf.keras.utils.register_keras_serializable(package="bt")
class MCDropout(tf.keras.layers.Dropout):
    """Dropout whose stochasticity can be forced on at inference: attribute `mc` (False by default).
    During fit() it behaves as ordinary dropout (training=True); at predict() it is OFF unless mc=True."""

    def __init__(self, rate, mc=False, **kw):
        super().__init__(rate, **kw)
        self.mc = bool(mc)

    def call(self, inputs, training=None):
        return super().call(inputs, training=True if self.mc else training)

    def get_config(self):
        c = super().get_config()
        c["mc"] = self.mc
        return c


@tf.keras.utils.register_keras_serializable(package="bt")
class TemporalAttention(tf.keras.layers.Layer):
    """Additive attention e_t = v^T tanh(W z_t + b); alpha = softmax_t(e); context = sum_t alpha_t z_t.
    call(z) returns (context, alpha). `scores(z)` returns alpha only."""

    def __init__(self, dim=64, **kw):
        super().__init__(**kw)
        self.dim = dim

    def build(self, shape):
        d = int(shape[-1])
        self.W = self.add_weight(name="W", shape=(d, self.dim), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.dim,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.dim,), initializer="glorot_uniform")
        super().build(shape)

    def scores(self, z):
        e = tf.tensordot(tf.tanh(tf.tensordot(z, self.W, axes=1) + self.b), self.v, axes=1)
        return tf.nn.softmax(e, axis=1)

    def call(self, z):
        a = self.scores(z)
        return tf.reduce_sum(z * a[..., None], axis=1), a

    def get_config(self):
        c = super().get_config()
        c["dim"] = self.dim
        return c


def _reg():
    return tf.keras.regularizers.l2(L2)


def _cnn_branch(x):
    x = tf.keras.layers.Conv1D(WIDTH, 3, padding="same", activation="relu", kernel_regularizer=_reg(), name="cnn_conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="cnn_bn1")(x)
    x = tf.keras.layers.Conv1D(WIDTH, 5, padding="same", activation="relu", kernel_regularizer=_reg(), name="cnn_conv5")(x)
    x = tf.keras.layers.BatchNormalization(name="cnn_bn2")(x)
    return MCDropout(DROPOUT, name="cnn_mc_dropout")(x)


def _gru_branch(x):
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(WIDTH, return_sequences=True, kernel_regularizer=_reg(),
                            recurrent_regularizer=_reg()), name="bigru")(x)
    x = tf.keras.layers.BatchNormalization(name="gru_bn")(x)
    return MCDropout(DROPOUT, name="gru_mc_dropout")(x)


def build_model(name, window=T, n_feat=8, lr=LR):
    """name in cnn | bigru | cnn_bigru | cnn_bigru_att. Output: sigmoid probability of label 1.
    Layer names used by mc_uattention_predict: feature_fusion, attention, head_dense, head_dropout, probability."""
    inp = tf.keras.Input(shape=(window, n_feat), name="input")
    if name == "cnn":
        z = _cnn_branch(inp)
    elif name == "bigru":
        z = _gru_branch(inp)
    elif name in ("cnn_bigru", "cnn_bigru_att"):
        z = tf.keras.layers.Concatenate(axis=-1, name="feature_fusion")([_cnn_branch(inp), _gru_branch(inp)])
    else:
        raise ValueError(name)
    if name == "cnn_bigru_att":
        c, _ = TemporalAttention(WIDTH, name="attention")(z)
    else:
        c = tf.keras.layers.GlobalAveragePooling1D(name="pool")(z)
    h = tf.keras.layers.Dense(WIDTH, activation="relu", kernel_regularizer=_reg(), name="head_dense")(c)
    h = MCDropout(DROPOUT, name="head_dropout")(h)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(h)
    m = tf.keras.Model(inp, out, name=name)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.BinaryCrossentropy())
    return m


def load_trained(path):
    m = tf.keras.models.load_model(path, compile=False)
    set_mc(m, False)
    return m


def set_mc(model, on):
    for l in model.layers:
        if isinstance(l, MCDropout):
            l.mc = bool(on)
    return model


def predict_deterministic(model, X, batch=1024):
    """Dropout OFF, BatchNorm in inference mode, single pass."""
    set_mc(model, False)
    return model.predict(X, batch_size=batch, verbose=0).reshape(-1).astype(np.float64)


def train_model(model, Xtr, ytr, Xva, yva, batch=32, epochs=100, patience=10, verbose=0):
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    set_mc(model, False)
    t0 = time.time()
    h = model.fit(Xtr, ytr.astype("float32"), validation_data=(Xva, yva.astype("float32")), batch_size=batch,
                  epochs=epochs, callbacks=cb, verbose=verbose, shuffle=True)
    set_mc(model, False)
    hist = h.history
    return dict(epochs_run=len(hist["loss"]), best_epoch=int(np.argmin(hist["val_loss"])) + 1,
                best_val_loss=float(np.min(hist["val_loss"])), train_seconds=time.time() - t0,
                loss=[float(v) for v in hist["loss"]], val_loss=[float(v) for v in hist["val_loss"]])


# =====================================================================
# 6. BlockTrain: MC dropout inference + U-Attention (protocol 11)
# =====================================================================
def mc_uattention_predict(model, X, K=MC_K, use_u=True, batch=None, seed=None):
    """Stochastic inference on a trained cnn_bigru_att model (same weights as the conventional model).

    For each window:
      z^(k) = fused representation with dropout ON (k=1..K; BatchNorm in inference mode)
      zbar_t = mean_k z^(k)_t ; u_raw_t = sum_k ||z^(k)_t - zbar_t||^2 / (K-1)
      u_t = (u_raw_t - min_t)/(max_t - min_t + 1e-7)   (min-max within the window)
      alpha_t = attention(zbar) ; alpha~_t = alpha_t (1-u_t) ; alpha^ = alpha~/sum alpha~
      c = sum_t alpha^_t zbar_t ; head run K times with dropout ON on c -> p^(k)
    Returns prob = mean_k p^(k), var = sum_k (p^(k)-pbar)^2/(K-1), entropy = H(pbar) (nats),
    and u [n,T], alpha [n,T], alpha_hat [n,T].
    use_u=False: identical MC procedure with u_t = 0 (alpha^ = alpha, context from zbar).
    """
    if batch is None:
        batch = max(100, int(10000 // K))      # keeps the [K,B,T,D] tensor small (memory)
    if seed is not None:
        tf.random.set_seed(int(seed))
    set_mc(model, True)
    fusion = model.get_layer("feature_fusion")
    rep = tf.keras.Model(model.input, fusion.output)
    att = model.get_layer("attention")
    hd, hdrop, hout = model.get_layer("head_dense"), model.get_layer("head_dropout"), model.get_layer("probability")
    n = len(X)
    P, V, E, U, A, AH = [], [], [], [], [], []
    for i in range(0, n, batch):
        xb = tf.constant(X[i:i + batch])
        zs = tf.stack([rep(xb, training=False) for _ in range(K)], 0)          # [K,B,T,D]
        zbar = tf.reduce_mean(zs, 0)
        u_raw = tf.reduce_sum(tf.reduce_sum((zs - zbar[None]) ** 2, -1), 0) / (K - 1)   # [B,T]
        umin = tf.reduce_min(u_raw, 1, keepdims=True)
        umax = tf.reduce_max(u_raw, 1, keepdims=True)
        u = (u_raw - umin) / (umax - umin + EPS)
        alpha = att.scores(zbar)
        if use_u:
            at = alpha * (1.0 - u)
        else:
            at = alpha
            u = tf.zeros_like(u)
        ah = at / (tf.reduce_sum(at, 1, keepdims=True) + EPS)
        c = tf.reduce_sum(zbar * ah[..., None], 1)
        ps = tf.stack([hout(hdrop(hd(c), training=False)) for _ in range(K)], 0)[..., 0]  # [K,B]
        pbar = tf.reduce_mean(ps, 0)
        var = tf.reduce_sum((ps - pbar[None]) ** 2, 0) / (K - 1)
        pc = tf.clip_by_value(pbar, EPS, 1 - EPS)
        ent = -(pc * tf.math.log(pc) + (1 - pc) * tf.math.log(1 - pc))
        P.append(pbar.numpy()); V.append(var.numpy()); E.append(ent.numpy())
        U.append(u.numpy()); A.append(alpha.numpy()); AH.append(ah.numpy())
    set_mc(model, False)
    cat = np.concatenate
    return dict(prob=cat(P).astype(np.float64), var=cat(V).astype(np.float64), entropy=cat(E).astype(np.float64),
                u=cat(U).astype(np.float32), alpha=cat(A).astype(np.float32), alpha_hat=cat(AH).astype(np.float32))


# =====================================================================
# 7. Metrics
# =====================================================================
def ece(y, p, n_bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if i == n_bins - 1 else (p < edges[i + 1]))
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def best_threshold(y, p):
    """Threshold maximising F1 on the given (validation) predictions."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.argmax(f1))])


def metrics(y, p, thr):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=np.float64)
    pc = np.clip(p, EPS, 1 - EPS)
    return dict(f1=float(f1_score(y, (p >= thr).astype(int), zero_division=0)),
                auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(np.mean((p - y) ** 2)),
                nll=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                ece=ece(y, p), threshold=float(thr))


# =====================================================================
# 8. Experiment driver
# =====================================================================
def seed_dir(seed, root=None):
    return os.path.join(root or os.path.join(ROOT, "runs", "main"), f"seed{seed}")


def _save_preds(path, meta, p, var=None, ent=None, thr=np.nan):
    df = pd.DataFrame(dict(sample_id=meta["sample_id"], y_true=meta["y_true"], anomaly_type=meta["anomaly_type"],
                           prob=p, pred_var=np.nan if var is None else var,
                           pred_entropy=np.nan if ent is None else ent, threshold=thr))
    df.to_csv(path, index=False)


def run_seed(seed, cfg, out_root=None):
    sd = seed_dir(seed, out_root)
    os.makedirs(os.path.join(sd, "checkpoints"), exist_ok=True)
    log = dict(seed=seed, timings={}, train={})
    t_all = time.time()
    t0 = time.time()
    S = make_split(seed, cfg["n_train"], cfg["n_val"], cfg["n_test"])
    log["timings"]["make_split_s"] = time.time() - t0
    for sp in ["val", "test"]:
        S["meta_" + sp].to_csv(os.path.join(sd, f"{sp}_meta.csv"), index=False)
        np.save(os.path.join(sd, f"{sp}_X.npy"), S["X_" + sp])
        np.save(os.path.join(sd, f"{sp}_y.npy"), S["y_" + sp])
        np.save(os.path.join(sd, f"{sp}_corrupted_steps.npy"), S["steps_" + sp])
    # protocol names test_meta.csv; also keep the explicit val_meta.csv above
    info = dict(seed=seed, n_train=cfg["n_train"],
                hash={sp: split_hash(S["meta_" + sp]) for sp in ["train", "val", "test"]},
                class_balance={sp: float(S["y_" + sp].mean()) for sp in ["train", "val", "test"]},
                anomaly_counts={sp: S["meta_" + sp]["anomaly_type"].value_counts().to_dict() for sp in ["train", "val", "test"]},
                participants_in_sample={sp: S["meta_" + sp]["participant"].value_counts().to_dict() for sp in ["train", "val", "test"]})
    log["split"] = info
    thr_cache = {}
    for name in MODEL_NAMES:
        tf.keras.utils.set_random_seed(int(seed))
        ck = os.path.join(sd, "checkpoints", f"{name}.keras")
        m = build_model(name)
        tl = train_model(m, S["X_train"], S["y_train"], S["X_val"], S["y_val"], batch=cfg["batch"],
                         epochs=cfg["epochs"], patience=cfg["patience"], verbose=2)
        m.save(ck)
        log["train"][name] = {k: v for k, v in tl.items()}
        t1 = time.time()
        for sp in ["val", "test"]:
            p = predict_deterministic(m, S["X_" + sp])
            if sp == "val":
                thr_cache[name] = best_threshold(S["y_val"], p)
            _save_preds(os.path.join(sd, f"preds_{name}_{sp}.csv"), S["meta_" + sp], p, thr=thr_cache[name])
        log["timings"][f"predict_{name}_s"] = time.time() - t1
        print(f"[seed {seed}] {name}: epochs={tl['epochs_run']} best={tl['best_epoch']} "
              f"val_loss={tl['best_val_loss']:.4f} train_s={tl['train_seconds']:.0f}", flush=True)
        if name == "cnn_bigru_att":
            t1 = time.time()
            arrs = {}
            for sp in ["val", "test"]:
                r = mc_uattention_predict(m, S["X_" + sp], K=cfg["K"], use_u=True, seed=seed)
                if sp == "val":
                    thr_cache["blocktrain"] = best_threshold(S["y_val"], r["prob"])
                _save_preds(os.path.join(sd, f"preds_blocktrain_{sp}.csv"), S["meta_" + sp], r["prob"], r["var"],
                            r["entropy"], thr_cache["blocktrain"])
                np.savez_compressed(os.path.join(sd, f"blocktrain_u_{sp}.npz"), u=r["u"], alpha=r["alpha"],
                                    alpha_hat=r["alpha_hat"])
            log["timings"]["blocktrain_mc_s"] = time.time() - t1
            print(f"[seed {seed}] blocktrain MC K={cfg['K']}: {log['timings']['blocktrain_mc_s']:.0f}s", flush=True)
    # test_meta.csv contract columns subset is contained in saved file
    log["thresholds"] = thr_cache
    log["timings"]["seed_total_s"] = time.time() - t_all
    log["preprocessor"] = S["pre"].to_dict()
    log["psplit"] = {k: ([short(p) for p in v] if k in ("train", "val", "test") else v)
                     for k, v in S["psplit"].items()}
    with open(os.path.join(sd, "seed_log.json"), "w") as f:
        json.dump(log, f, indent=1, default=float)
    open(os.path.join(sd, "SEED_DONE"), "w").close()
    return log


def protocol_header(cfg):
    data = load_data()
    ps = make_participant_split(data)
    return dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        tensorflow=tf.__version__, numpy=np.__version__, pandas=pd.__version__,
        settings=cfg,
        fixed=dict(T=T, stride=1, grid_min=3, min_obs_valid=MIN_OBS_VALID, min_obs_corrupt=MIN_OBS_CORRUPT,
                   p_value_noise=P_NOISE, segment_len=[SEG_MIN, SEG_MAX], noise_sd_multiplier=NOISE_SD_MULT,
                   dropout=DROPOUT, l2=L2, lr=LR, width=WIDTH, split_rng_seed_requested=SPLIT_RNG_SEED),
        data=dict(rows_raw=data["n_rows_raw"], rows_after_dedup=data["n_rows_dedup"],
                  participants={short(p): dict(raw_rows=data["n_raw_rows"][p], grid_bins=data["n_bins"][p],
                                               valid_windows=int(len(data["valid_starts"][p])),
                                               obs_fraction=dict(zip(CHANNELS, data["obs_frac"][p])))
                                for p in data["pids"]}),
        participant_split=dict(rng_seed_used=ps["rng_seed"], attempts=ps["attempts"],
                               train=[short(p) for p in ps["train"]], val=[short(p) for p in ps["val"]],
                               test=[short(p) for p in ps["test"]],
                               valid_windows={sp: int(sum(len(data["valid_starts"][p]) for p in ps[sp]))
                                              for sp in ["train", "val", "test"]}),
    )


def cmd_run(a):
    cfg = dict(seeds=a.seeds, n_train=a.n_train, n_val=a.n_val, n_test=a.n_test, batch=a.batch,
               epochs=a.epochs, patience=a.patience, K=a.K)
    outdir = os.path.join(ROOT, "runs", "main")
    os.makedirs(outdir, exist_ok=True)
    prot_path = os.path.join(ROOT, "runs", "PROTOCOL.json")
    prot = protocol_header(cfg)
    prot["seed_logs"] = {}
    t0 = time.time()
    for s in a.seeds:
        if os.path.exists(os.path.join(seed_dir(s), "SEED_DONE")) and not a.redo:
            print(f"seed {s} already done, skipping", flush=True)
            with open(os.path.join(seed_dir(s), "seed_log.json")) as f:
                prot["seed_logs"][str(s)] = json.load(f)
            continue
        lg = run_seed(s, cfg)
        prot["seed_logs"][str(s)] = lg
        prot["wall_time_s_so_far"] = time.time() - t0
        with open(prot_path, "w") as f:
            json.dump(prot, f, indent=1, default=float)
    prot["wall_time_s_total"] = time.time() - t0
    with open(prot_path, "w") as f:
        json.dump(prot, f, indent=1, default=float)
    open(os.path.join(outdir, "DONE"), "w").close()
    print("ALL DONE", time.time() - t0, flush=True)


def cmd_mcsens(a):
    """MC-K sensitivity on the TEST set for BlockTrain (inference only, reuse checkpoints)."""
    rows = []
    for s in SEEDS:
        sd = seed_dir(s)
        m = load_trained(os.path.join(sd, "checkpoints", "cnn_bigru_att.keras"))
        Xt, yt = np.load(os.path.join(sd, "test_X.npy")), np.load(os.path.join(sd, "test_y.npy"))
        # threshold fixed to the validation-selected one of the main BlockTrain run (K=10), identical for all K
        thr = float(pd.read_csv(os.path.join(sd, "preds_blocktrain_val.csv"))["threshold"].iloc[0])
        for K in a.ks:
            t1 = time.time()
            rt = mc_uattention_predict(m, Xt, K=K, use_u=True, seed=2000 + s)
            sec = time.time() - t1
            r = metrics(yt, rt["prob"], thr)
            r.update(seed=s, K=K, mean_pred_var=float(rt["var"].mean()), median_pred_var=float(np.median(rt["var"])),
                     mean_entropy=float(rt["entropy"].mean()), test_infer_seconds=sec)
            rows.append(r)
            print(s, K, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)[["seed", "K", "f1", "auroc", "auprc", "brier", "nll", "ece", "threshold",
                             "mean_pred_var", "median_pred_var", "mean_entropy", "test_infer_seconds"]]
    df.to_csv(os.path.join(ROOT, "runs", "main", "mc_sensitivity.csv"), index=False)


def cmd_smoke(a):
    t0 = time.time()
    d = load_data()
    print("load", time.time() - t0)
    ps = make_participant_split(d)
    print({k: [short(p) for p in ps[k]] for k in ["train", "val", "test"]}, ps["rng_seed"])
    S = make_split(11, 2000, 400, 400)
    print({k: S[k].shape for k in S if k.startswith("X_")}, S["meta_test"].anomaly_type.value_counts().to_dict())
    for n in MODEL_NAMES:
        m = build_model(n)
        t0 = time.time()
        tl = train_model(m, S["X_train"], S["y_train"], S["X_val"], S["y_val"], batch=a.batch, epochs=2, patience=5)
        print(n, m.count_params(), "2 epochs s", round(time.time() - t0, 1))
    r = mc_uattention_predict(m, S["X_test"], K=10)
    r0 = mc_uattention_predict(m, S["X_test"], K=10, use_u=False)
    print(r["prob"][:5], r["var"][:3], r["u"][0][:5], r0["u"].max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "mcsens", "smoke"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n_train", type=int, default=N_TRAIN_DEFAULT)
    ap.add_argument("--n_val", type=int, default=N_VAL_DEFAULT)
    ap.add_argument("--n_test", type=int, default=N_TEST_DEFAULT)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--K", type=int, default=MC_K)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "mcsens": cmd_mcsens, "smoke": cmd_smoke}[a.cmd](a)
