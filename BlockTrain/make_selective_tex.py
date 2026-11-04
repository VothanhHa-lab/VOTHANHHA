#!/usr/bin/env python3
"""Generate text/C_selective.tex from the saved CSVs in runs/selective/<tag>/ (no recomputation).

  python3 make_selective_tex.py                 # uses main_test if present, else valpreds_old (PROVISIONAL)
  python3 make_selective_tex.py --tag main_test

Every number in the output is read from a CSV.  The qualitative wording is chosen by explicit
checks on those CSVs (printed at the end).  The wording was written for the case that the
variance ordering is NOT better than entropy / max-probability / random; if the checks say
otherwise, the generator switches to the alternative sentence AND writes a "% REVIEW" comment:
read those places before pasting.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = os.environ.get("BLOCKTRAIN_MAIN_TEX", os.path.join(ROOT, "main.tex"))
SEL = f"{ROOT}/runs/selective"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default=None)
a = ap.parse_args()
tag = a.tag or ("main_test" if os.path.exists(f"{SEL}/main_test/summary_mean_sd.csv") else "valpreds_old")
D = f"{SEL}/{tag}"
PROV = tag == "valpreds_old"
SPLIT = "validation" if PROV else "test"
SPLIT_SHORT = "validation" if PROV else "test"

S = pd.read_csv(f"{D}/summary_mean_sd.csv").set_index("score")
C = pd.read_csv(f"{D}/rc_curves_summary.csv")
Q = pd.read_csv(f"{D}/quintiles_summary.csv").set_index("quintile")
P = pd.read_csv(f"{D}/paired_aurc_vs_variance.csv").set_index("other")
G = pd.read_csv(f"{D}/diagnostics_per_seed.csv")
T = pd.read_csv(f"{D}/by_anomaly_type_summary.csv").set_index("anomaly_type") \
    if os.path.exists(f"{D}/by_anomaly_type_summary.csv") else None
DET = None
if os.path.exists(f"{SEL}/main_test_cnn_bigru_att/summary_mean_sd.csv") and not PROV:
    DET = pd.read_csv(f"{SEL}/main_test_cnn_bigru_att/summary_mean_sd.csv").set_index("score")
NS = int(S.loc["var", "n_seeds"])
NSW = {3: "three", 4: "four", 5: "five"}.get(NS, str(NS))


def f3(x): return f"{x:.3f}"
def fm(x): return f"${x:.3f}$" if x < 0 else f"{x:.3f}"
def pm(m, s): return f"${m:.3f}\\pm{s:.3f}$"
def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


V = {}
err = S.loc["var", "error_rate_mean"]
V["ERR"] = f3(err); V["ERR_SD"] = f3(S.loc["var", "error_rate_sd"]); V["NS"] = NSW
for sc in ("var", "entropy", "maxprob", "random"):
    k = sc.upper()
    V[f"AURC_{k}"] = f3(S.loc[sc, "aurc_mean"]); V[f"AURC_{k}_SD"] = f3(S.loc[sc, "aurc_sd"])
    V[f"AUROC_{k}"] = f3(S.loc[sc, "error_detection_auroc_mean"])
    for c in ("0.5", "0.8", "0.9"):
        V[f"R{c.replace('.', '')}_{k}"] = f3(S.loc[sc, f"risk_at_{c}_mean"])
V["AURC_ORACLE"] = f3(S.loc["var", "aurc_oracle_mean"])
V["EXC_VAR"] = f3(S.loc["var", "excess_aurc_mean"]); V["EXC_ENT"] = f3(S.loc["entropy", "excess_aurc_mean"])
V["EXC_RANDOM"] = f3(S.loc["random", "excess_aurc_mean"])
V["D_VAR_ENT"] = f3(P.loc["entropy", "mean_diff"] * -1)  # AURC(var) - AURC(entropy)
V["D_VAR_ENT_SD"] = f3(P.loc["entropy", "sd_diff"])
V["D_RAND_VAR"] = f3(P.loc["random", "mean_diff"])      # AURC(random) - AURC(var)
V["D_RAND_VAR_SD"] = f3(P.loc["random", "sd_diff"])
V["SPEAR_VAR_DIST"] = f3(G["spearman_var_abs_p_minus_thr"].mean())
V["SPEAR_VAR_ERR"] = fm(G["spearman_var_error"].mean())
V["SPEAR_ENT_ERR"] = f3(G["spearman_entropy_error"].mean())
for q in range(1, 6):
    V[f"QE{q}"] = f3(Q.loc[q, "error_rate_mean"]); V[f"QE{q}_SD"] = f3(Q.loc[q, "error_rate_sd"])
    V[f"QV{q}"] = sci(Q.loc[q, "mean_var_mean"]); V[f"QP{q}"] = f3(Q.loc[q, "mean_prob_mean"])
cv = C[C.score == "var"].set_index("coverage")
V["R02_VAR"] = f3(cv.loc[0.2, "risk_mean"])
assert abs(cv.loc[0.2, "risk_mean"] - Q.loc[1, "error_rate_mean"]) < 1e-9   # risk@0.2 == Q1 error
above = [c for c in cv.index if c < 1.0 and cv.loc[c, "risk_mean"] > err]
cmax = float(cv.risk_mean.idxmax())
V["CMAX"] = f"{cmax:.2f}"; V["RMAX"] = f3(cv.risk_mean.max())
V["CAB_LO"] = f"{min(above):.2f}" if above else ""
V["CAB_HI"] = f"{max(above):.2f}" if above else ""
if T is not None:
    for t in T.index:
        V[f"T_{t}_ERR"] = f3(T.loc[t, "error_rate_mean"]); V[f"T_{t}_VAR"] = sci(T.loc[t, "mean_var_mean"])
        V[f"T_{t}_N"] = f"{T.loc[t, 'n_mean']:.0f}"

# ---------- qualitative checks
n_ent_better = int(P.loc["entropy", "n_other_better"]); n_rand_better = int(P.loc["random", "n_other_better"])
ent_rel = "worse_all" if n_ent_better == NS else ("better_all" if n_ent_better == 0 else "mixed")      # variance vs entropy
rand_rel = "var_better_all" if n_rand_better == 0 else ("var_worse_all" if n_rand_better == NS else "mixed")
auroc_var = S.loc["var", "error_detection_auroc_mean"]
qe = Q.error_rate_mean.to_numpy()
mono_up = bool(np.all(np.diff(qe) > 0))
q_argmax, q_argmin = int(np.argmax(qe)) + 1, int(np.argmin(qe)) + 1
useful = (rand_rel == "var_better_all") and auroc_var >= 0.55 and ent_rel != "worse_all"
rel_gain = (S.loc["random", "aurc_mean"] - S.loc["var", "aurc_mean"]) / S.loc["random", "aurc_mean"]
ent_eq_mp = np.allclose(S.loc["entropy", "aurc_mean"], S.loc["maxprob", "aurc_mean"], atol=1e-6)
flags = dict(tag=tag, ent_rel=ent_rel, rand_rel=rand_rel, auroc_var=round(auroc_var, 3), mono_up=mono_up,
             q_argmax=q_argmax, q_argmin=q_argmin, useful=useful, rel_gain_vs_random=round(rel_gain, 3),
             entropy_equals_maxprob=ent_eq_mp)
review = []

# ---------- sentences chosen from the checks
if rand_rel == "var_better_all":
    qual = "only slightly lower" if rel_gain < 0.15 else "lower"
    V["S_RAND"] = (f"The AURC of the variance ordering ({V['AURC_VAR']}) is {qual} than that of random rejection "
                   f"({V['AURC_RANDOM']}) in all {NSW} seeds (mean paired difference {V['D_RAND_VAR']}, SD {V['D_RAND_VAR_SD']})")
elif rand_rel == "var_worse_all":
    V["S_RAND"] = (f"The AURC of the variance ordering ({V['AURC_VAR']}) is higher than that of random rejection "
                   f"({V['AURC_RANDOM']}) in all {NSW} seeds")
else:
    V["S_RAND"] = (f"The AURC of the variance ordering ({V['AURC_VAR']}) is lower than that of random rejection "
                   f"({V['AURC_RANDOM']}) in some seeds and higher in others")
if ent_rel == "worse_all":
    V["S_ENT"] = (f"Rejection by predictive entropy or by max-probability confidence gives a markedly lower AURC "
                  f"({V['AURC_ENTROPY']}) than rejection by variance in all {NSW} seeds "
                  f"(mean paired difference {V['D_VAR_ENT']}, SD {V['D_VAR_ENT_SD']})")
elif ent_rel == "better_all":
    V["S_ENT"] = (f"Rejection by variance gives a lower AURC than rejection by entropy or max-probability confidence "
                  f"({V['AURC_ENTROPY']}) in all {NSW} seeds")
    review.append("variance beats entropy/max-prob in all seeds: check every NEW sentence for consistency")
else:
    V["S_ENT"] = (f"Rejection by variance and by entropy/max-probability confidence ({V['AURC_ENTROPY']}) "
                  f"rank differently across seeds")
    review.append("entropy vs variance AURC is mixed across seeds: check wording")
if auroc_var < 0.5:
    V["S_AUROC"] = (f"the error-detection AUROC of $\\sigma_p^2$ is {V['AUROC_VAR']}, i.e.\\ below the value of 0.5 "
                    f"expected of an uninformative score, whereas it is {V['AUROC_ENTROPY']} for entropy and for max-probability confidence")
elif auroc_var < 0.55:
    V["S_AUROC"] = (f"the error-detection AUROC of $\\sigma_p^2$ is {V['AUROC_VAR']}, close to the value of 0.5 "
                    f"expected of an uninformative score, whereas it is {V['AUROC_ENTROPY']} for entropy and for max-probability confidence")
else:
    V["S_AUROC"] = (f"the error-detection AUROC of $\\sigma_p^2$ is {V['AUROC_VAR']}, compared with {V['AUROC_ENTROPY']} "
                    f"for entropy and for max-probability confidence")
if above:
    V["S_ABOVE"] = (f"Under the variance ordering the selective risk is low only at small coverage (risk {V['R02_VAR']} at coverage 0.2) "
                    f"and, for coverage between {V['CAB_LO']} and {V['CAB_HI']}, it is higher than the error rate of the complete "
                    f"set ({V['ERR']}); its maximum on the coverage grid is {V['RMAX']} at coverage {V['CMAX']}. "
                    f"Over that range, rejecting the windows with the highest variance therefore increases, rather than decreases, "
                    f"the error rate of the retained windows.")
else:
    V["S_ABOVE"] = (f"Under the variance ordering the selective risk is {V['R02_VAR']} at coverage 0.2 and does not exceed "
                    f"the error rate of the complete set ({V['ERR']}) at any grid coverage below 1.")
if mono_up:
    V["S_MONO"] = ("In this analysis the error rate increases monotonically with the variance quintile.")
    review.append("quintile error is monotone increasing: the non-monotonic wording elsewhere must be revised")
else:
    V["S_MONO"] = (f"The error rate is not monotone in $\\sigma_p^2$: it is highest in Q{q_argmax} ({V[f'QE{q_argmax}']}) and lowest in "
                   f"Q{q_argmin} ({V[f'QE{q_argmin}']}), and the highest-variance quintile (Q5) has an error rate of {V['QE5']}"
                   + ("." if q_argmin != 5 else ", the lowest of the five quintiles."))
if T is not None and {"clean", "value_noise", "missing_block"} <= set(T.index):
    hi = T.mean_var_mean.idxmax()
    V["S_TYPE"] = (f"By window type, the error rate and mean $\\sigma_p^2$ are {V['T_clean_ERR']} and {V['T_clean_VAR']} for clean windows "
                   f"({V['T_clean_N']} per seed on average), {V['T_value_noise_ERR']} and {V['T_value_noise_VAR']} for value-noise windows "
                   f"({V['T_value_noise_N']}), and {V['T_missing_block_ERR']} and {V['T_missing_block_VAR']} for missing-block windows "
                   f"({V['T_missing_block_N']}). "
                   + ("Mean variance is thus highest for the clean windows, which have the lowest error rate, "
                      "so variance does not separate corrupted from clean windows in the direction a rejection score would require. "
                      if hi == "clean" else
                      "Mean variance is highest for " + hi.replace("_", " ") + " windows. "))
else:
    V["S_TYPE"] = ""
V["S_SPEAR"] = (f"Across seeds the mean Spearman correlation between $\\sigma_p^2$ and the distance of $\\bar p$ from the decision threshold is "
                f"{V['SPEAR_VAR_DIST']}, and between $\\sigma_p^2$ and the error indicator it is {V['SPEAR_VAR_ERR']} "
                f"(entropy: {V['SPEAR_ENT_ERR']}). In this model, higher variance therefore tends to accompany predicted probabilities "
                f"that lie farther from the threshold, and these are the windows that are classified more reliably"
                + (f"; the quintile with the highest error rate (Q{q_argmax}) has mean $\\bar p$ of {V[f'QP{q_argmax}']}, close to the decision threshold." if abs(Q.loc[q_argmax,'mean_prob_mean']-0.5) < 0.05 and not np.isnan(0) else ".")) \
    if float(V["SPEAR_VAR_DIST"]) > 0.1 else ""
if float(V["SPEAR_VAR_DIST"]) <= 0.1:
    review.append("S_SPEAR sentence suppressed: var vs |p-thr| correlation <= 0.1")
if q_argmin == 5:
    V["S_Q5"] = "The highest-variance windows are thus not the least reliable ones; they have the lowest error rate of the five quintiles."
elif float(V["QE5"]) < err:
    V["S_Q5"] = "The highest-variance windows are thus not the least reliable ones; their error rate is below that of the complete set."
else:
    V["S_Q5"] = ""
    review.append("Q5 error is not below overall error: check the 5.6 narrative")
if ent_rel != "worse_all":
    review.append("variance is not worse than entropy in all seeds: NEW texts [Abstract, 5.10, 6.1, 6.5, 7] state the opposite")
V["CAUTION"] = ("These observations are descriptive; the present analysis does not identify why the variance behaves in this way.")

# selective verdict sentences
if not useful:
    V["VERDICT_ABS"] = (f"Selective prediction does not support predictive variance as a rejection score: in the {SPLIT_SHORT} analysis its AURC "
                        f"({V['AURC_VAR']}) is higher than that of predictive entropy or max-probability confidence ({V['AURC_ENTROPY']}) "
                        + (f"and only slightly lower than that of random rejection ({V['AURC_RANDOM']}), " if rand_rel == "var_better_all" and rel_gain < 0.15
                           else f"and {'lower' if rand_rel=='var_better_all' else 'not lower'} than that of random rejection ({V['AURC_RANDOM']}), ")
                        + "and its relationship with prediction error is non-monotonic.")
    V["VERDICT_SHORT"] = (f"Predictive variance was not an effective rejection score (AURC {V['AURC_VAR']}, versus {V['AURC_ENTROPY']} for predictive "
                          f"entropy or max-probability confidence and {V['AURC_RANDOM']} for random rejection).")
else:
    V["VERDICT_ABS"] = (f"Predictive variance gave an AURC of {V['AURC_VAR']}, compared with {V['AURC_ENTROPY']} for entropy/max-probability "
                        f"confidence and {V['AURC_RANDOM']} for random rejection, although its relationship with prediction error is non-monotonic.")
    V["VERDICT_SHORT"] = V["VERDICT_ABS"]
    review.append("USEFUL branch active: the negative framing used elsewhere in NEW texts must be rewritten by hand")

# ---------- OLD strings extracted from main.tex (exact), NEW strings
main = open(MAIN, encoding="utf-8").read()
def old(start, end):
    i = main.index(start); assert main.count(start) == 1, f"start not unique: {start[:40]}"
    j = main.index(end, i) + len(end)
    return main[i:j]

ENT_LAB = "predictive entropy $H(\\bar p)$"
repl = []
def add(where, s, e, new): repl.append((where, old(s, e), new))

add("Abstract", "Predictive variance\nsupports selective prediction", "prediction error is non-monotonic.",
    "Rejecting windows by predictive variance did not reduce selective risk more effectively than rejecting by predictive entropy "
    "or max-probability confidence (area under the risk--coverage curve, AURC: @AURC_VAR@ versus @AURC_ENTROPY@), was @ABS_RAND@ random rejection (@AURC_RANDOM@), "
    "and the error rate was not monotone in variance." if not useful else "@VERDICT_ABS@")
add("Section 1, contribution 4 (\\item A reliability-oriented evaluation protocol)",
    "The framework is evaluated using discrimination and probabilistic", "sensitivity\n    analysis.",
    "The framework is evaluated using discrimination and probabilistic\n    calibration metrics, stochastic predictive variability,\n"
    "    risk--coverage analysis in which predictive variance is compared with predictive entropy, max-probability\n"
    "    confidence and random rejection, uncertainty-quintile analysis, together with component-wise ablation\n"
    "    experiments and MC-sample sensitivity analysis.")
add("Section 3.8.4, first paragraph", "To evaluate whether predictive uncertainty provides a useful ranking\nof prediction reliability",
    "subset.",
    "To evaluate whether a predictive-uncertainty measure provides a useful ranking of prediction reliability, the test windows are sorted in\n"
    "ascending order of a rejection score and progressively retained from the lowest-score subset. Four rejection scores are compared:\n"
    "the predictive variance $\\sigma_p^2$ of BlockTrain; the predictive entropy $H(\\bar p)$; the max-probability confidence score\n"
    "$1-\\max(\\bar p,1-\\bar p)$; and random order (mean over 200 random permutations). The max-probability score is also computed for the\n"
    "deterministic CNN--BiGRU--Attention model. All scores are evaluated at the same validation-selected decision threshold.")
add("Section 3.8.4, last paragraph", "The resulting risk--coverage curve describes how empirical prediction\nrisk changes as increasingly uncertain predictions are rejected.",
    "rejected.",
    "The resulting risk--coverage curve describes how empirical prediction risk changes as increasingly uncertain predictions are rejected,\n"
    "with $\\hat y_i$ obtained at the fixed threshold. The curve is summarised by the area under the risk--coverage curve,\n"
    "$\\mathrm{AURC}=\\frac{1}{n}\\sum_{k=1}^{n}R(k/n)$, by the excess AURC relative to an oracle that retains all correctly classified\n"
    "windows first, and by the selective risk at coverage 0.5, 0.8 and 0.9. A rejection score is useful only if its AURC is lower than\n"
    "that of random rejection, whose expected AURC equals the overall error rate, and if it is competitive with the output-probability\n"
    "scores. The error-detection AUROC, i.e.\\ the probability that a misclassified window receives a higher score than a correctly\n"
    "classified one, is reported as a ranking summary independent of coverage. Results are reported per seed and as mean $\\pm$ SD over seeds.")
add("Section 3.8.5, first paragraph", "The relationship between predictive stochastic variance and\nclassification error is further examined by sorting samples into\nuncertainty quantiles.",
    "empirical prediction error are calculated.",
    "The relationship between predictive stochastic variance and classification error is further examined by sorting the windows of each seed into\n"
    "five equal-count quintiles of $\\sigma_p^2$ (Q1 lowest, Q5 highest). For each quintile, the mean $\\sigma_p^2$ and the empirical error rate\n"
    "at the fixed threshold are calculated. By construction, the selective risk of the variance ordering at coverage 0.2 equals the error rate of Q1.")
add("Section 5.10 (Summary), final paragraph", "Finally, the uncertainty analyses show that stochastic predictive", "preserving competitive predictive\nperformance.",
    "Finally, the uncertainty analyses do not show that stochastic predictive variance is a useful rejection score. In the risk--coverage\n"
    "analysis, predictive variance was not an effective rejection score @AURC_PAREN@, and the error rate is not monotone in variance. The variance is therefore treated\n"
    "as a descriptive indicator of stochastic output variability, not as a calibrated probability of prediction error and not as an\n"
    "established basis for selective prediction. Overall, the results support BlockTrain as a modular framework that couples temporal\n"
    "relevance with stochastic representation variability while preserving competitive predictive performance.")
add("Section 6.1 (Main Findings), 'Fourth' paragraph", "Fourth, predictive uncertainty provides information that is", "deterministic probability of prediction failure.",
    "Fourth, the predictive stochastic variance of BlockTrain did not prove useful for selective prediction. In the risk--coverage analysis, its\n"
    "AURC (@AURC_VAR@) was higher than that of predictive entropy or max-probability confidence (@AURC_ENTROPY@) and @RANDOM_PHRASE@\n"
    "(@AURC_RANDOM@), and the uncertainty--error analysis showed that the relationship between predictive variance and empirical classification\n"
    "error is non-monotonic. Thus, predictive variance should be interpreted neither as a probability of prediction failure nor, on the\n"
    "present evidence, as a basis for rejecting predictions.")
add("Section 6.2 (Mechanistic Interpretation), paragraph on non-monotonic relationship",
    "The non-monotonic uncertainty--error relationship observed in the\nexperiments provides additional support", "related to,\nbut distinct from, classification error.",
    "The non-monotonic uncertainty--error relationship observed in the experiments shows that the output-level predictive variance does not\n"
    "track classification error. Predictive stochastic variance increased substantially across the uncertainty quintiles, while empirical\n"
    "classification error did not follow the same trend. This observation does not test the proposed mechanism: the justification for\n"
    "U-Attention rests on the representation-level variability $u_t$ and on the ablation results, not on the behaviour of the output-level\n"
    "variance $\\sigma_p^2$.")
add("Positioning table, 'Role of uncertainty' item", "while output-level predictive variance is retained\n    separately for selective prediction.",
    "separately for selective prediction.",
    "while output-level predictive variance is retained\n    separately as a descriptive diagnostic.")
add("Section 6.4 (Methodological Implications), 'Third' paragraph", "Third, the results highlight the importance of evaluating", "larger uncertainty must correspond to larger\nclassification error.",
    "Third, the results highlight the importance of evaluating uncertainty-aware models using multiple criteria. Global calibration\n"
    "metrics, reliability diagrams, risk--coverage analysis, and uncertainty--error analysis capture different properties of predictive\n"
    "reliability. In particular, the present risk--coverage and uncertainty--error results show that a larger predictive variance cannot be\n"
    "assumed to indicate a larger error probability, and that a variance-based rejection score should be compared with entropy,\n"
    "max-probability and random rejection before it is described as supporting selective prediction.")
add("Section 6.5 (Practical Implications), two paragraphs", "The stochastic predictive variance can provide an additional signal", "reliability may be undesirable.",
    "The stochastic predictive variance is available at inference as an additional output. The present results, however, do not support\n"
    "using it to decide which predictions to accept or to defer: in the risk--coverage analysis, rejecting the highest-variance windows\n"
    "did not lower the error rate of the retained windows more than rejecting by predictive entropy or max-probability confidence, and\n"
    "@PRACT_RANDOM@. For applications that permit abstention, an output-probability score such as max-probability confidence\n"
    "would be the more appropriate rejection rule on the present evidence, and its usefulness would need to be verified on independent\n"
    "data. Using predictions flagged by any rejection score for additional processing, quality inspection, or human review remains a\n"
    "design option, but the present study does not demonstrate a benefit of variance-based flagging.")
add("Section 6.6 (Limitations), 'Fifth' paragraph", "Fifth, the uncertainty--error relationship is non-monotonic in the", "uncertainty-aware aggregation and selective prediction.",
    "Fifth, the uncertainty--error relationship is non-monotonic in the present experiments. Consequently, predictive variance cannot be\n"
    "treated as a direct surrogate for classification error, and it was not an effective rejection score in the risk--coverage analysis.\n"
    "The selective-prediction analysis uses a balanced, controlled-corruption target, a fixed decision threshold and five seeds; whether\n"
    "variance behaves differently on naturally occurring data-quality failures or under a different threshold was not examined.")
add("Section 6.7 (Future Research), closing 'In summary' paragraph", "BlockTrain provides an explicit mechanism for\nincorporating stochastic representation variability into temporal\naggregation and for supporting selective prediction.",
    "aggregation and for supporting selective prediction.",
    "BlockTrain provides an explicit mechanism for\nincorporating stochastic representation variability into temporal\naggregation; the present results do not show that its predictive variance supports selective prediction.")
add("Section 7 (Conclusion), paragraph on uncertainty analysis", "The uncertainty analysis also demonstrates that predictive variance", "according to their stochastic stability.",
    "The uncertainty analysis also demonstrates that predictive variance and classification error are related but distinct quantities. The\n"
    "predictive variance increased substantially across the uncertainty quintiles, whereas the corresponding empirical error rate exhibited\n"
    "a non-monotonic pattern; in particular, the highest-uncertainty quintile did not have the highest error rate. The risk--coverage\n"
    "analysis further indicates that predictive variance is not an effective rejection score @AURC_PAREN@. This finding supports a conservative\n"
    "interpretation of predictive variance as a descriptive indicator of stochastic output variability, not as a surrogate for prediction\n"
    "error and not as an established basis for selective prediction.")

if rand_rel == "var_better_all":
    V["RANDOM_PHRASE"] = "only slightly lower than that of random rejection" if rel_gain < 0.15 else "lower than that of random rejection"
    V["PRACT_RANDOM"] = ("its AURC was only slightly below that of random rejection" if rel_gain < 0.15 else "its AURC was below that of random rejection")
elif rand_rel == "var_worse_all":
    V["RANDOM_PHRASE"] = "higher than that of random rejection"; V["PRACT_RANDOM"] = "its AURC was higher than that of random rejection"
else:
    V["RANDOM_PHRASE"] = "not consistently lower than that of random rejection"; V["PRACT_RANDOM"] = "its AURC was not consistently below that of random rejection"
V["ABS_RAND"] = {"var_better_all": "only slightly better than" if rel_gain < 0.15 else "better than", "var_worse_all": "worse than", "mixed": "not consistently better than"}[rand_rel]
V["AURC_PAREN"] = f"(AURC {V['AURC_VAR']}, versus {V['AURC_ENTROPY']} for predictive entropy or max-probability confidence and {V['AURC_RANDOM']} for random rejection)"
V["VERDICT_SHORT_NOPERIOD"] = V["VERDICT_SHORT"].rstrip(".")
V["VERDICT_SHORT_NOPERIOD"] = V["VERDICT_SHORT_NOPERIOD"][0].lower() + V["VERDICT_SHORT_NOPERIOD"][1:]

# ---------- 5.5 / 5.6 LaTeX
def row(label, sc, df=S):
    r = df.loc[sc]
    return (f"{label} & {pm(r.aurc_mean, r.aurc_sd)} & {pm(r.excess_aurc_mean, r.excess_aurc_sd)} & {pm(r['risk_at_0.5_mean'], r['risk_at_0.5_sd'])} & "
            f"{pm(r['risk_at_0.8_mean'], r['risk_at_0.8_sd'])} & {pm(r['risk_at_0.9_mean'], r['risk_at_0.9_sd'])} & "
            f"{pm(r.error_detection_auroc_mean, r.error_detection_auroc_sd)} \\\\")
rows = [row("BlockTrain, $\\sigma_p^2$", "var"), row("BlockTrain, $H(\\bar p)$", "entropy"),
        row("BlockTrain, max-probability", "maxprob"), row("BlockTrain, random order", "random")]
if DET is not None:
    rows.append(row("CNN--BiGRU--Att., max-probability", "maxprob", DET))
orc = S.loc["var"]
rows.append(f"Oracle & {pm(orc.aurc_oracle_mean, orc.aurc_oracle_sd)} & {pm(0, 0)} & -- & -- & -- & -- \\\\")
V["TABLE_ROWS"] = "\n".join(rows)
V["DETSENT"] = ""
if DET is not None:
    d = DET.loc["maxprob"]
    V["DETSENT"] = (f"For the deterministic CNN--BiGRU--Attention model, max-probability confidence gives an AURC of {pm(d.aurc_mean, d.aurc_sd)} "
                    f"at an error rate of {pm(d.error_rate_mean, d.error_rate_sd)}.")
HDR = ("% ===== PROVISIONAL: numbers below come from the authors' EARLIER validation predictions (valpreds/, threshold 0.5 assumed),\n"
       "% ===== NOT from the new test-set run. Regenerate with: python3 selective.py --runs runs/main && python3 make_selective_tex.py\n") if PROV else \
      "% Numbers from runs/selective/main_test/*.csv (test predictions, validation-selected threshold, per-seed mean and SD).\n"

SEC55 = r"""
\subsection{Risk--coverage}
\label{subsec:risk_coverage}

The risk--coverage analysis asks whether a rejection score identifies the predictions that are more likely to be wrong. Predictions of BlockTrain on the @SPLIT@ windows were retained in ascending order of one of four scores: the predictive variance $\sigma_p^2$, the predictive entropy $H(\bar p)$, the max-probability confidence score $1-\max(\bar p,1-\bar p)$, and a random order (mean of 200 permutations). Selective risk is the error rate at the fixed decision threshold among the retained windows, as defined in Section~\ref{sec:methods}. The error rate of the complete set (coverage 1), which is the same for all scores, is @ERR@ (SD @ERR_SD@ across seeds). Results are mean $\pm$ SD over @NS@ seeds.

\begin{table}[ht]
\centering
\caption{Selective-prediction summary for BlockTrain on the @SPLIT@ windows (mean $\pm$ SD over @NS@ seeds; error rate of the complete set @ERR@). AURC is the area under the risk--coverage curve (lower is better), the excess AURC is AURC minus the oracle AURC, $R(c)$ is the selective risk at coverage $c$, and the error-detection AUROC is the probability that a misclassified window receives a higher score than a correctly classified one (0.5 = uninformative).}
\label{tab:selective}
\renewcommand{\arraystretch}{1.1}
\small
\begin{tabular}{lcccccc}
\toprule
Rejection score & AURC & Excess AURC & $R(0.5)$ & $R(0.8)$ & $R(0.9)$ & Error-detection AUROC \\
\midrule
@TABLE_ROWS@
\bottomrule
\end{tabular}
\end{table}

Figure~\ref{fig:risk_coverage} shows the curves and Table~\ref{tab:selective} the summary measures. @S_ENT@. @S_RAND@, and @S_AUROC@. @S_ABOVE@ @DETSENT@

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{figs/fig_risk_coverage.pdf}
    \caption{Risk--coverage curves of BlockTrain on the @SPLIT@ windows for four rejection scores: predictive variance $\sigma_p^2$, predictive entropy $H(\bar p)$, max-probability confidence, and random order. Lines are means over @NS@ seeds and bands are $\pm1$ SD. The entropy and max-probability curves coincide because both are monotone functions of $|\bar p-0.5|$ for a binary output. The dotted grey curve is the oracle.}
    \label{fig:risk_coverage}
\end{figure}

The entropy and max-probability scores are monotone functions of the distance of $\bar p$ from 0.5 and therefore produce identical rankings; they should not be counted as two independent baselines. @VARNOTE@ These results do not support the use of $\sigma_p^2$ as a rejection score for BlockTrain. They also do not show that BlockTrain provides better selective prediction than an output-probability score: the max-probability and entropy scores are available from the output probability of any probabilistic classifier.
"""
SEC56 = r"""
\subsection{Uncertainty--error}
\label{subsec:uncertainty_error}

To examine the relationship between stochastic uncertainty and empirical prediction error directly, the @SPLIT@ windows of BlockTrain were divided, within each seed, into five equal-count quintiles of the predictive variance $\sigma_p^2$ (Q1 lowest). The error rate at the fixed threshold and the mean $\sigma_p^2$ were computed for each quintile and then averaged over the @NS@ seeds. The selective risk of the variance ordering at coverage 0.2 equals the error rate of Q1 (@R02_VAR@) by construction.

Figure~\ref{fig:uncertainty_error} presents the results. The mean variance increases from @QV1@ in Q1 to @QV5@ in Q5, so the quintile construction separates the windows by the magnitude of the stochastic predictive variability. The mean error rate is @QE1@ in Q1, @QE2@ in Q2, @QE3@ in Q3, @QE4@ in Q4 and @QE5@ in Q5 (seed-to-seed SD between @QE_SDMIN@ and @QE_SDMAX@). @S_MONO@ @S_Q5@

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth]{figs/fig_uncertainty_quantile.pdf}
    \caption{Error rate (a) and mean predictive variance $\sigma_p^2$ (b) across five equal-count quintiles of $\sigma_p^2$ for BlockTrain on the @SPLIT@ windows. Markers are means over @NS@ seeds and error bars are $\pm1$ SD across seeds; the printed values are the plotted means. The vertical axis of (b) is logarithmic.}
    \label{fig:uncertainty_error}
\end{figure}

@S_TYPE@@S_SPEAR@ @CAUTION@

Taken together with Section~\ref{subsec:risk_coverage}, these results indicate that $\sigma_p^2$ is not a monotone indicator of prediction error in the present experiments. Predictive variance should therefore be reported as a descriptive measure of stochastic output variability, and not as a calibrated probability of error or a validated rejection score.
"""
V["VARNOTE"] = ("In terms of the oracle, the excess AURC of the variance ordering is " + V["EXC_VAR"] + ", compared with " + V["EXC_ENT"] +
                " for entropy and max-probability and " + V["EXC_RANDOM"] + " for random order.")
V["QE_SDMIN"] = f3(Q.error_rate_sd.min()); V["QE_SDMAX"] = f3(Q.error_rate_sd.max())
V["SPLIT"] = SPLIT
if not useful:
    pass

def fill(t):
    for k in sorted(V, key=len, reverse=True):
        t = t.replace(f"@{k}@", str(V[k]))
    assert "@" not in t.replace("\\@", ""), [w for w in t.split() if "@" in w][:5]
    return t

out = [HDR, "% Generated by make_selective_tex.py from runs/selective/" + tag + "/ -- do not edit numbers by hand.\n"]
for r in review:
    out.append(f"% REVIEW: {r}\n")
out.append("\n% ======================================================================\n% PART (a): replacement LaTeX for Sections 5.5 and 5.6\n"
           "% (replace the whole \\subsection{Risk--coverage} and \\subsection{Uncertainty--error} blocks, including their figures;\n"
           "%  the old figure files 'Fig 8. Risk–coverage curve..png' and 'Fig 9. ...png' are replaced by figs/fig_risk_coverage.pdf and\n"
           "%  figs/fig_uncertainty_quantile.pdf)\n% ======================================================================\n")
out.append(fill(SEC55)); out.append(fill(SEC56))
out.append("\n% ======================================================================\n% PART (b): sentence replacements elsewhere in main.tex\n"
           "% Each OLD text is copied verbatim from main.tex (checked to occur exactly once);\n"
           "% NEW is the replacement, with the same macros/labels.\n% ======================================================================\n")
for i, (where, o, n) in enumerate(repl, 1):
    out.append(f"\n% ---------- [{i}] {where}\n")
    out.append("% OLD: " + o.replace("\n", "\n%      ") + "\n")
    out.append("% NEW: " + fill(n).replace("\n", "\n%      ") + "\n")
txt = "".join(out)
os.makedirs(f"{ROOT}/text", exist_ok=True)
open(f"{ROOT}/text/C_selective.tex", "w", encoding="utf-8").write(txt)
print("wrote text/C_selective.tex from", tag)
print(flags)
for r in review:
    print("REVIEW:", r)
