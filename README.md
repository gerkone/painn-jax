# PaiNN in jax
Reimplementation of [polarizable interaction neural network (PaiNN)](http://proceedings.mlr.press/v139/schutt21a.html) in jax. Original work by Kristof Schütt, Oliver Unke and Michael Gastegger.

<!-- ## Installation
```
python -m pip install painn-jax
```

Or clone this repository and build locally
```
python -m pip install -e .
```

### GPU support
Upgrade `jax` to the gpu version
```
pip install --upgrade "jax[cuda]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
``` -->

<!-- ## Validation

|                  |  MSE  | Inference [ms]* |
|------------------|-------|-----------------|
| torch (original) |       |                 |
| jax (ours)       |       |                 |

\* remeasured (Quadro RTX 4000)

NOTE: The validation is not well written and is quite convoluted and is here only to compare the performance of the two implementations.

-->

## Acknowledgements
This implementation is almost a minimal translation of the official torch PaiNN implementation from [schnetpack](https://github.com/atomistic-machine-learning/schnetpack).
