from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import haiku as hk


def scaled_silu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.silu(x) * 1 / 0.6


# borrowed from OCP PaiNN
# https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/painn/painn.py
class GatedEquivariantBlock(hk.Module):
    """Gated equivariant block (restricted to L=1, vectorial features).

    References:
        [#painn1] Schütt, Unke, Gastegger:
        Equivariant message passing for the prediction of tensorial properties and
        molecular spectra.
        ICML 2021
    """

    # TODO is it useful to have a different number of scalar and vector channels?
    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        name: str = "gated_equivariant_block",
        activation: Callable = scaled_silu,
    ):
        super().__init__(name)
        self._output_channels = output_channels

        self.vec1_proj = hk.Linear(hidden_channels, with_bias=False, name="vec1_proj")
        self.vec2_proj = hk.Linear(output_channels, with_bias=False, name="vec2_proj")

        self.update_net = hk.Sequential(
            [
                hk.Linear(hidden_channels, with_bias=False),
                activation,
                hk.Linear(output_channels * 2, with_bias=False),
            ],
            name="scalar_update_net",
        )

        self.act = activation

    def __call__(
        self, x: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        vec1 = jnp.linalg.norm(self.vec1_proj(v), axis=-2)
        vec2 = self.vec2_proj(v)

        x = jnp.concatenate([x, vec1], axis=-1)
        x, v = jnp.split(self.update_net(x), self._output_channels, axis=-1)
        v = v[..., jnp.newaxis] * vec2
        x = self.act(x)

        return x, v
