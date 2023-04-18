from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph


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
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        vec1 = jnp.reshape(jnp.linalg.norm(self.vec1_proj(v), axis=-2), s.shape)
        vec2 = self.vec2_proj(v)

        s = jnp.concatenate([s, vec1], axis=-1)
        a = self.update_net(s)
        s, v = jnp.split(a, 2, axis=-1)
        v = v * vec2
        s = self.act(s)

        return s, v


def pooling(
    graph: jraph.GraphsTuple,
    aggregate_fn: Callable = jraph.segment_sum,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pools over graph nodes with the specified aggregation.

    Args:
        graph: Input graph
        aggregate_fn: function used to update pool over the nodes

    Returns:
        The pooled graph nodes.
    """
    n_graphs = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs)
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
    batch = jnp.repeat(graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node)
    s = aggregate_fn(graph.nodes.s, batch, n_graphs)
    v = aggregate_fn(graph.nodes.v, batch, n_graphs)

    return s, v
