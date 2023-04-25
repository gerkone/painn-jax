# PaiNN in jax
Reimplementation of [polarizable interaction neural network (PaiNN)](http://proceedings.mlr.press/v139/schutt21a.html) in jax. Original work by Kristof Schütt, Oliver Unke and Michael Gastegger.

## Installation
```
pip install git+https://github.com/gerkone/painn-jax.git
```

Or clone this repository and build locally
```
git clone https://github.com/gerkone/painn-jax
cd painn-jax
python -m pip install -e .
```

### GPU support
Upgrade `jax` to the gpu version
```
pip install --upgrade "jax[cuda]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Validation
This implementation is validated on QM9 on the dipole moment target. The results are shown in the table below. The timings are remeasured on a single GPU (Quadro RTX 4000) and the results are compared to the ones reported in the original paper.

|                  |  MSE  | Inference [ms]  |
|------------------|-------|-----------------|
| jax (ours)       | 0.014 |      8.42*      |
| torch (original) | 0.012 |     163.23      |

\* padded (naive)

__NOTE: The validation is not well written and is quite convoluted since it uses the [QM9 dataset from schnetpack](https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/datasets/qm9.py). It is here only to compare the performance of the two implementations.__


## Differences to the original
- In schnetpack PaiNN is used as a representation network, and is wrapped in a `NeuralNetworkPotential` for readout and pooling. Here the model is self-contained in the [`painn_jax.PaiNN`](https://github.com/gerkone/painn-jax/blob/b30137884d467877a14fd5d3b09dcc9757ca25f4/painn_jax/painn.py#L217) class, meaning that readout/pooling is parametric and is done directly inside PaiNN.
- Originally the vectors are initialized with zeros. Here if vector features are passed in input they are lifted and used as initialization for the vectors instead.


## Acknowledgements
This implementation of PaiNN itself is almost a minimal translation of the official torch implementation included in [schnetpack](https://github.com/atomistic-machine-learning/schnetpack).
