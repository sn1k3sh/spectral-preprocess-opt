# specopt — coupled spectral preprocessing & regression model hyperparameter optimisation

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nikesh23/specopt-demo)

Choosing how to preprocess IR spectra for ML is usually done manually, separately 
from the model hyperparameter optimisation it feeds into, even though the two decisions 
interact strongly.

**specopt** searches over both jointly, minimising leave-one-out prediction error, 
and returning a optimised preprocessing pipeline.

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

`optimise_preprocessing` only takes in training data negating the chance for test data 
leakage into optimisation.

## Install

```bash
# from GitHub
pip install git+https://github.com/sn1k3sh/spectral-preprocess-opt.git

# or from a local clone
pip install .
```

Dependencies (`numpy`, `scipy`, `scikit-learn`, `pyswarm`) install automatically.

## How it works
The search space is composed of a minimal set of preprocessing steps (truncation, 
Savitzky–Golay filtering, and Standard Normal Variate) expressed as single vectors.

Particle swarm searches the space optimising for lowest **leave-one-out CV RMSE** obtained
with model hyperparameters (PLS/PCA components, or Ridge penalty) optimised for each candidate
set of preprocessing conditions. 

## Demonstration

On the public Beer NIR dataset, optimise_preprocessing cuts PLS test RMSE by a mean of 43% (median 42%) against 
unprocessed data across 15 random 90/10 splits, taking mean RMSE from 0.224 to 0.115. The spread is wide 
(roughly +11% to +80%) because each split's test set is only 8 samples. The swarm converges on truncation to the 
signal-rich region plus SNV, which is a conventional manual choice for NIR.

This approach is particularly useful for data where preprocessing choice is not obvious, such as the noisier 
proprietary spectra it was developed for. 

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
from IR/NIR spectra. The approach is inspired by the coupled preprocessing–model 
optimisation idea in:

C. D. Kappatou, J. Odgers, S. García-Muñoz and R. Misener, *Ind. Eng. Chem. Res.*,
2023, **62**, 6196–6213. [Open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC10119938/)

## License

MIT
