# High-dimensional low-signal MLP experiment

This repository implements the fixed experiment in `configs/main.json`.

Run tests:

```bash
conda run -n chem_ai python -m unittest discover -s tests -v
```

Run the smoke test, all 140 formal models, and the analysis:

```bash
zsh scripts/run_all.sh
```

The runner refuses to overwrite an existing `results/smoke` or `results/main`
directory. Move old results before starting another complete run.

To run only the smoke test:

```bash
export MPLCONFIGDIR="$PWD/results/.mplconfig"
export XDG_CACHE_HOME="$PWD/results/.cache"
conda run -n chem_ai python src/run_experiment.py \
  --config configs/main.json \
  --output-dir results/smoke \
  --smoke-test \
  --device mps
```
