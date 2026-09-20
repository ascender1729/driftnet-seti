# DriftNet: Learn the Decision, Not the Transform

[![DOI](https://zenodo.org/badge/1299566733.svg)](https://doi.org/10.5281/zenodo.21343024)

Code for the paper **"Learn the Decision, Not the Transform: Localizing Where Machine Learning Helps in Narrowband Technosignature Search"** (Dubasi, submitted to *The Astronomical Journal*).

The classical narrowband-SETI detector `turboSETI` computes an incoherent de-Doppler statistic and applies one hand-set threshold uniformly across observing bands. Rather than propose another detector, this work builds a modular one, **DriftNet**, that shares the same de-Doppler front end and ablates each design choice on **real Green Bank Telescope noise**, to localize where learning actually helps. The headline finding: making the de-Doppler kernels trainable adds nothing (frozen and learnable kernels agree within 0.006 in accuracy and the trained kernels move under 0.03 channels per sample), while a small learned readout over the pipeline's own drift spectrum is what helps (0.965 vs 0.907 in band, 0.911 vs 0.822 out of band against a fixed threshold on the identical drift grid), and that advantage transfers across bands. A learned readout over the scalar statistic alone, or a raw-pixel CNN without the de-Doppler front end, only matches the fixed threshold.

## Changelog

- **v1.1.0 (2026-09-20, revision for AJ referee report).** Fixed a fairness bug: the classical thresholded de-Doppler baseline in `baselines.py` searched 9 trial drifts on [-1, +1] channels per sample while the DriftNet variants searched 17 on [-3, +3]; the injected drifts of +-0.2 / +-0.4 Hz/s are +-1.31 / +-2.61 channels per sample, so the old baseline could not reach them. The baseline now uses the identical 17-point grid. Table 1 was re-run (classical row: in-distribution accuracy 0.720 -> 0.907, OOD 0.663 -> 0.822; learned rows unchanged within 0.005). `driftnet_comprehensive.py` now records the trained drift kernels per seed and the classical grid in the results JSON. `make_results_fig.py` draws majority-class chance (0.5 for the 2:1:1 injection set) instead of 1/3. `make_figures.py::concept` places the matched-filter marker on the injected line (the old marker sat on a noise peak) and adds axis ticks and labels. Re-run results are in `results/`.
- **v1.0.0 (2026-07-13).** Initial public release with the submitted manuscript.

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
