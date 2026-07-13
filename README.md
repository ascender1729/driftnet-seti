# DriftNet: Learn the Decision, Not the Transform

[![DOI](https://zenodo.org/badge/1299566733.svg)](https://doi.org/10.5281/zenodo.21343024)

Code for the paper **"Learn the Decision, Not the Transform: Localizing Where Machine Learning Helps in Narrowband Technosignature Search"** (Dubasi, submitted to *The Astronomical Journal*).

The classical narrowband-SETI detector `turboSETI` computes an incoherent de-Doppler statistic and applies one hand-set threshold uniformly across observing bands. Rather than propose another detector, this work builds a modular one, **DriftNet**, that shares the same de-Doppler front end and ablates each design choice on **real Green Bank Telescope noise**, to localize where learning actually helps. The headline finding: making the de-Doppler kernels trainable adds nothing (McNemar p = 0.42), while replacing the hand-set threshold with a small learned decision rule over the same statistic is what helps, and that advantage transfers across bands.

## Data

All experiments use **public Breakthrough Listen data** (a Voyager-1 X-band cadence and four Enriquez L-band cadences). The run scripts download it directly from the Berkeley archive; no data is bundled with this repository.

## Install

```bash
python -m pip install -r requirements.txt   # numpy MUST be <2.0 (blimpy/setigen)
```

## Reproduce

Each shell script downloads the required public data and runs one stage end-to-end:

```bash
bash run_comp.sh         # five-variant ablation panel + cross-band OOD (Table 1 / Fig. 3)
bash run_cp4.sh          # cadence-level comparison vs real turboSETI (Table 2 / Fig. 4)
bash run_validation.sh   # hard negatives (Table 3) + reverse OOD + Voyager real-signal control
```

Individual experiments (after data is present under `real_data/`):

```bash
python driftnet_comprehensive.py     # ablation panel, 5 seeds
python cp4_cadence_roc.py --band X   # cadence completeness vs turboSETI
python hardneg.py --band X           # hard-negative false-alarm rates
python reverse_ood.py                # train L, test X
python kedd_realdata.py              # curvature-constrained detector (Outlook)
python voyager_control.py            # curved detector on the real Voyager carrier
```

## Key files

| File | Role |
|---|---|
| `driftnet_v2.py` | the DriftNet model (learnable/frozen de-Doppler kernels, CNN + rho-MLP heads) |
| `baselines.py` | classical thresholded de-Doppler baseline and features |
| `cp3_inject_real.py`, `gen_dataset.py` | setigen injection into real GBT off-noise; dataset generation |
| `driftnet_comprehensive.py` | the five-variant ablation panel with McNemar/BH statistics |
| `cp4_cadence_roc.py` | cadence-level matched-false-alarm comparison against real turboSETI |
| `hardneg.py`, `reverse_ood.py`, `voyager_realsignal.py` | validation experiments |
| `kedd_realdata.py`, `curvedrift_*.py`, `voyager_control.py` | curvature-constrained (Outlook) detector |
| `make_figures.py`, `make_cadence_fig.py`, `make_results_fig.py` | paper figures |
| `vlm_eval.py` | zero-shot vision-language-model sanity floor (set `HF_TOKEN` for the open-source path) |

## Citing

Please cite both the paper and this software (see `CITATION.cff`). Software DOI (v1.0.0): [10.5281/zenodo.21343025](https://doi.org/10.5281/zenodo.21343025); concept DOI (latest): [10.5281/zenodo.21343024](https://doi.org/10.5281/zenodo.21343024).

## License

MIT (see `LICENSE`).
