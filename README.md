# specopt — coupled spectral preprocessing & regression model hyperparameter optimisation

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nikesh23/specopt-demo)

Choosing how to preprocess spectra for ML is usually done manually, separately from 
the model hyperparameter optimisation it feeds into, even though the two decisions 
interact strongly.

**specopt** searches over both jointly, minimising leave-one-out prediction error, 
and hands back a ready-to-apply preprocessing pipeline.

While **specopt** is aimed at chemometrics / spectroscopy (NIR, FTIR, Raman), it 
works on any wide, correlated feature matrix.

```python
from specopt import optimise_preprocessing

# X_train, y_train: training spectra (n_samples, n_features) and targets (n_samples,)
result = optimise_preprocessing(X_train, y_train, model="pls")

print(result.params)   # best preprocessing pipeline, e.g.
                       # {'truncate': True, 'start': 0, 'end': 283,
                       #  'savgol': False, 'snv': True, ...}
print(result.rmse)     # its leave-one-out CV RMSE on the training data

X_train_p = result.transform(X_train)   # apply the pipeline to train
X_test_p  = result.transform(X_test)    # and to held-out data
```

`optimise_preprocessing` sees **training data only** and returns the best preprocessing
conditions. The train/test split and the final model fit stay in your own code (see the
notebook), so there is no risk of test data leakage.

## Install

```bash
# from GitHub
pip install git+https://github.com/sn1k3sh/spectral-preprocess-opt.git

# or from a local clone
pip install .
```

Dependencies (`numpy`, `scipy`, `scikit-learn`, `pyswarm`) install automatically.

## How it works

Every candidate pipeline is a single real vector (truncation window, Savitzky–Golay
window / polyorder / derivative, SNV on/off). Particle swarm searches that space, and
each candidate is scored by the best **leave-one-out CV RMSE** for a latent-variable model
whose own hyperparameters (PLS/PCA components, or Ridge penalty) is retuned for that candidate.
Categorical and integer choices are encoded as floats and rounded on decode (the SG
window is encoded as `2·k+1` so it is always odd). Infeasible candidates, such as a
window wider than the truncated spectrum, are rejected by a feasibility check before
the expensive cross-validation runs.

## What you can tune

Arguments to `optimise_preprocessing`:

- **Model**: `"pls"`, `"pcr"`, or `"ridge"`; component count / penalty is tuned internally.
- **`max_components`**: cap on the latent-variable search.
- **Search bounds**: `min_width`, `max_half_window`, `max_polyorder`, `max_deriv`.
- **Swarm settings**: `swarmsize`, `maxiter`, `omega`, `phip`, `phig`, `random_state`.

## Demonstration

On the public Beer NIR dataset, `optimise_preprocessing` cuts PLS test RMSE by a mean of 
43% (median 42%) across 15 random 90/10 splits — mean RMSE 0.224 → 0.115. The spread is 
wide (roughly +11% to +80%) because each split's test set is only 8 samples.

The [Kaggle notebook](https://www.kaggle.com/code/nikesh23/specopt-demo) shows the full train/test walkthrough.

## Package layout

```
src/specopt/
  preprocessing.py    truncate / savgol / snv + ordered pipeline
  encoding.py         vector <-> pipeline encoding + feasibility check
  objectives.py       leave-one-out CV RMSE objective (PLS / PCR / Ridge)
  swarm_optimiser.py  particle-swarm search, optimise_preprocessing entry point
  __init__.py         public API
```

## Provenance

This package generalises a chemometrics pipeline I built for predicting composition
from IR/NIR spectra. It contains none of that original proprietary data or instrument
I/O — only the modelling method, demonstrated here on the public
[Beer NIR dataset](https://www.kaggle.com/datasets/robertoschimmenti/beer-nir).

The approach is inspired by the coupled preprocessing–model optimisation idea in:

C. D. Kappatou, J. Odgers, S. García-Muñoz and R. Misener, *Ind. Eng. Chem. Res.*,
2023, **62**, 6196–6213. [Open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC10119938/)

## License

MIT
