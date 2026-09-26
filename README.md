# High-dimensional low-signal MLP experiment

This repository implements the fixed experiment in `configs/main.json`. The
one-dimensional signal is `y + epsilon`. Each of the four informative
dimensions uses the equal-SNR design `y / 4 + epsilon_i / 2`, with independent
standard-normal `epsilon_i`; both signal conditions therefore have population
Bayes R2 equal to 0.5.

Run tests:

```bash
conda run -n chem_ai python -m unittest discover -s tests -v
```

Run the smoke test, all 140 formal models, and the analysis:

```bash
zsh scripts/run_all.sh
```

The runner writes the revised experiment to `results/equal_snr_smoke` and
`results/equal_snr_main` and refuses to overwrite either directory. The
original two-noise experiment under `results/main` is preserved for comparison.

To run only the smoke test:

```bash
export MPLCONFIGDIR="$PWD/results/.mplconfig"
export XDG_CACHE_HOME="$PWD/results/.cache"
conda run -n chem_ai python src/run_experiment.py \
  --config configs/main.json \
  --output-dir results/equal_snr_smoke \
  --smoke-test \
  --device mps
```
