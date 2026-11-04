# BlockTrain — code for the manuscript

Code for *BlockTrain: A Modular Uncertainty-Aware Framework for Reliable Multivariate Wearable Time-Series Analysis under Controlled Data-Quality Corruption* (Vo Thanh Ha, Nguyen Tam Thanh, Vo Quang Vinh, Nguyen Duy Trung).

The scripts implement the protocol of Sections 3–4 and regenerate every result table and figure of Section 5 from saved per-seed predictions.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Data loading, 3-min grid, participant split, window sampling, controlled corruption, train-only preprocessing, models (CNN, BiGRU, CNN–BiGRU, CNN–BiGRU–Attention), MC-dropout inference with U-Attention, metrics; main run and MC-K sensitivity |
| `ablation.py` | U-Attention ablation on the same trained checkpoints (V0 deterministic; V1 K=1, u=0; V2 K=10, u=0; V3 K=10 + U-Attention), paired bootstrap differences, mechanism check against the corrupted time steps |
| `selective.py` | Risk–coverage (variance, entropy, max-probability, random), AURC, uncertainty quintiles |
| `report.py`, `make_tex.py`, `make_selective_tex.py` | Figures and LaTeX tables from the stored predictions |
| `results/` | Summary metrics of the reported run (CSV) and `PROTOCOL.json` with all settings, participant split and window counts |

## Protocol (summary)

- 13 participants, 606,641 records; duplicates removed; stress codes −1/−2 treated as missing; 3-min grid (median per bin), observation masks.
- Participant split 9 / 2 / 2, fixed for all seeds (see `results/PROTOCOL.json`).
- Windows: T = 60, stride 1, valid if at least one channel is observed in ≥ 10 steps.
- Per seed (11, 22, 33, 44, 55): 20,000 / 2,000 / 4,000 windows, 50 % corrupted.
- Corruption (raw units, before imputation and z-scoring): value noise (p = 0.6, N(0, SD_train²), clipped) or missing block (p = 0.4); one channel, one segment of 10–30 steps.
- Training: Adam 1e-3, batch 32, ≤ 100 epochs, early stopping patience 10, dropout 0.3, L2 1e-4. Threshold = max-F1 on validation.
- BlockTrain = the trained CNN–BiGRU–Attention weights evaluated with K = 10 MC-dropout passes and U-Attention (inference only).

## Reproduce

```bash
pip install -r requirements.txt
# put the raw CSV into input/ (see input/README.md)
python pipeline.py run            # 5 seeds, saves runs/main/seed*/ (≈2 h on 2 CPUs)
python pipeline.py mcsens         # K ∈ {5,10,20,30}
python report.py && python make_tex.py
python ablation.py seed 11 22 33 44 55 && python ablation.py aggregate && python ablation.py report
python selective.py --runs runs/main && python make_selective_tex.py
```

## Data

The raw wearable data contain physiological records of human participants and are not included. See the Data availability statement of the article.
