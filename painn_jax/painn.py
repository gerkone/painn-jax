from typing import Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

from .blocks import GatedEquivariantBlock, LinearXav, pooling


class NodeFeatures(NamedTuple):
    """Simple container for scalar and vectorial node features."""

    s: jnp.ndarray = None
    v: jnp.ndarray = None


def PaiNNReadout(
    hidden_size: int,
    task: str,
    pool: str,
    out_channels: int = 1,
    activation: Callable = jax.nn.silu,
    blocks: int = 2,
    eps: float = 1e-8,
) -> Callable:
    """
    PaiNN readout block.

    Args:
        hidden_size: Number of hidden channels.
        task: Task to perform. Either "node" or "graph".
        pool: pool method. Either "sum" or "avg".
        scalar_out_channels: Number of scalar/vector output channels.
        activation: Activation function.
        blocks: Number of readout blocks.
    """

    assert task in ["node", "graph"], "task must be node or graph"
    assert pool in ["sum", "avg"], "pool must be sum or avg"
    if pool == "avg":
        pool_fn = jraph.segment_mean
    if pool == "sum":
        pool_fn = jraph.segment_sum

    def _readout(graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        s, v = graph.nodes
        s = jnp.squeeze(s)
        for i in range(blocks - 1):
            ith_hidden_size = hidden_size // 2**i
            s, v = GatedEquivariantBlock(
                hidden_size=ith_hidden_size,
                scalar_out_channels=ith_hidden_size // 2,
                vector_out_channels=ith_hidden_size // 2,
                activation=activation,
                eps=eps,
                name=f"readout_block_{i}",
            )(s, v)

        if task == "graph":
            graph = graph._replace(nodes=NodeFeatures(s, v))
            s, v = pooling(graph, aggregate_fn=pool_fn)

        s, v = GatedEquivariantBlock(
            hidden_size=out_channels * 2,
            scalar_out_channels=out_channels,
            vector_out_channels=out_channels,
            activation=activation,
            eps=eps,
            name="readout_block_out",
        )(s, v)

        return s, v

    return _readout


class PaiNNLayer(hk.Module):
    """PaiNN interaction block."""

    def __init__(
        self,
        hidden_size: int,
        activation: Callable,
        layer_num: int,
        blocks: int = 2,
        aggregate_fn: Optional[Callable] = jraph.segment_sum,
        eps: float = 1e-8,
    ):
        """
        Initialize the PaiNN layer, made up of an interaction block and a mixing block.

        Args:
            hidden_size: Number of node features.
            activation: Activation function.
            layer_num: Numbering of the layer.
            blocks: Number of layers in the context networks.
            aggregate_fn: Function to aggregate the neighbors.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__(f"layer_{layer_num}")
        self._hidden_size = hidden_size
        self._eps = eps
        self._aggregate_fn = aggregate_fn

        # inter-particle context net
        self.interaction_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="interaction_block",
        )

        # intra-particle context net
        self.mixing_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="mixing_block",
        )

        # vector channel mix
        self.vector_mixing_block = LinearXav(
            2 * hidden_size,
            with_bias=False,
            name="vector_mixing_block",
        )

    def _message(
        self,
        s: jnp.ndarray,
        v: jnp.ndarray,
        dir_ij: jnp.ndarray,
        Wij: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Message/interaction.

        Args:
            s (jnp.ndarray): Input scalar features.
            v (jnp.ndarray): Input vector features.
            dir_ij (jnp.ndarray): Direction of the edge.
            Wij (jnp.ndarray): Filter.
            senders (jnp.ndarray): Index of the sender node.
            receivers (jnp.ndarray): Index of the receiver node.

        Returns:
            Node features after interaction.
        """
        # inter-particle
        x = self.interaction_block(s)

        xj = x[receivers]
        vj = v[receivers]

        ds, dv1, dv2 = jnp.split(Wij * xj, 3, axis=-1)
        n_nodes = tree.tree_leaves(s)[0].shape[0]
        ds = self._aggregate_fn(ds, senders, n_nodes)
        dv = dv1[:, jnp.newaxis] * dir_ij[..., jnp.newaxis] + dv2[:, jnp.newaxis] * vj
        dv = self._aggregate_fn(dv, senders, n_nodes)

        s = s + jnp.clip(ds, -1e4, 1e4)
        v = v + jnp.clip(dv, -1e4, 1e4)

        return s, v

    def _update(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update/mixing.

        Args:
            s (jnp.ndarray): Input scalar features.
            v (jnp.ndarray): Input vector features.

        Returns:
            Node features after interaction.
        """
        ## intra-atomic
        v_l, v_r = jnp.split(self.vector_mixing_block(v), 2, axis=-1)
        v_norm = jnp.sqrt(jnp.sum(v_l**2, axis=-2) + self._eps)

        ts = jnp.concatenate([s, v_norm], axis=-1)
        ds, dv, dsv = jnp.split(self.mixing_block(ts), 3, axis=-1)
        dv = dv[:, jnp.newaxis] * v_r
        dsv = dsv * jnp.sum(v_l * v_r, axis=1)

        s = s + jnp.clip(ds + dsv, -1e4, 1e4)
        v = v + jnp.clip(dv, -1e4, 1e4)
        return s, v

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        Wij: jnp.ndarray,
    ):
        """Compute interaction output.

        Args:
            graph (jraph.GraphsTuple): Input graph.
            Wij (jnp.ndarray): Filter.

        Returns:
            atom features after interaction
        """
        # TODO make this work with jraph GNN functions
        s, v = graph.nodes
        s, v = self._message(s, v, graph.edges, Wij, graph.senders, graph.receivers)
        s, v = self._update(s, v)
        return graph._replace(nodes=NodeFeatures(s=s, v=v))


class PaiNN(hk.Module):
    """PaiNN - polarizable interaction neural network.

    References:
        [#painn1] Schütt, Unke, Gastegger:
        Equivariant message passing for the prediction of tensorial properties and
        molecular spectra.
        ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        radial_basis_fn: Callable,
        cutoff_fn: Optional[Callable] = None,
        radius: float = 5.0,
        n_rbf: int = 20,
        activation: Callable = jax.nn.silu,
        node_type: str = "discrete",
        task: str = "node",
        pool: str = "avg",
        out_channels: Optional[int] = None,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        eps: float = 1e-8,
    ):
        """Initialize the model.

        Args:
            hidden_size: Determines the size of each embedding vector.
            n_layers: Number of interaction blocks.
            radial_basis_fn: Expands inter-particle distances in a basis set.
            cutoff_fn: Cutoff method. None means no cutoff.
            radius: Cutoff radius.
            n_rbf: Number of radial basis functions.
            activation: Activation function.
            node_type: Type of node features. Either "discrete" or "continuous".
            task: Regression task to perform. Either "node"-wise or "graph"-wise.
            pool: Node readout pool method. Only used in "graph" tasks.
            out_channels: Number of output scalar/vector channels. Used in readout.
            max_z: Maximum atomic number. Used in discrete node feature embedding.
            shared_interactions: If True, share the weights across interaction blocks.
            shared_filters: If True, share the weights across filter networks.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__("painn")

        assert node_type in [
            "discrete",
            "continuous",
        ], "node_type must be discrete or continuous"
        assert task in ["node", "graph"], "task must be node or graph"
        assert radial_basis_fn is not None, "A radial_basis_fn must be provided"

        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._eps = eps
        self._node_type = node_type
        self._shared_filters = shared_filters
        self._shared_interactions = shared_interactions

        self.cutoff_fn = cutoff_fn(radius) if cutoff_fn else None
        self.radial_basis_fn = radial_basis_fn(n_rbf, radius)

        if node_type == "discrete":
            self.scalar_emb = hk.Embed(
                max_z,
                hidden_size,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                name="scalar_embedding",
            )
        else:
            self.scalar_emb = LinearXav(hidden_size, name="scalar_embedding")
        # mix vector channels (only used if vector features are present in input)
        self.vector_emb = LinearXav(
            hidden_size, with_bias=False, name="vector_embedding"
        )

        if shared_filters:
            self.filter_net = LinearXav(3 * hidden_size, name="filter_net")
        else:
            self.filter_net = LinearXav(n_layers * 3 * hidden_size, name="filter_net")

        if self._shared_interactions:
            self.layers = [
                PaiNNLayer(hidden_size, activation, layer_num=0, eps=eps)
            ] * n_layers
        else:
            self.layers = [
                PaiNNLayer(hidden_size, activation, layer_num=i, eps=eps)
                for i in range(n_layers)
            ]

        self.readout = None
        if out_channels is not None:
            self.readout = PaiNNReadout(
                hidden_size,
                task,
                pool,
                out_channels=out_channels,
                activation=activation,
                eps=eps,
            )

    def _embed(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Embed the input nodes."""
        # embeds scalar features
        s = graph.nodes.s
        if self._node_type == "continuous":
            s = jnp.asarray(s, dtype=jnp.float32)
            if len(s.shape) == 1:
                s = s[:, jnp.newaxis]
        if self._node_type == "discrete":
            s = jnp.asarray(s, dtype=jnp.int32)
        s = self.scalar_emb(s)

        # embeds vector features
        if graph.nodes.v is not None:
            # initialize the vector with the global positions
            v = graph.nodes.v
            v = jnp.reshape(v, (s.shape[0], 3, v.shape[-1] // 3))
            v = self.vector_emb(v)
        else:
            # initialize the vector with zeros (as in the paper)
            v = jnp.zeros((s.shape[0], 3, s.shape[-1]))

        return graph._replace(nodes=NodeFeatures(s=s, v=v))

    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute representations/embeddings.

        Args:
            inputs (dict of jnp.ndarray): SchNetPack dictionary of input tensors.

        Returns:
            jnp.ndarray: atom-wise representation.
            list of jnp.ndarray: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # compute atom and pair features
        norm_ij = jnp.sqrt(jnp.sum(graph.edges**2, axis=1, keepdims=True) + self._eps)
        # TODO this only makes sense for displacement vectors as edges/r_ij
        dir_ij = graph.edges / (norm_ij + self._eps)
        phi_ij = self.radial_basis_fn(jnp.squeeze(norm_ij))
        # replace edges with normalized directions
        graph = graph._replace(edges=dir_ij)

        if self.cutoff_fn is not None:
            norm_ij = self.cutoff_fn(norm_ij)  # pylint: disable=not-callable

        # compute filters
        filters = self.filter_net(phi_ij) * norm_ij
        if self._shared_filters:
            filter_list = [filters] * self._n_layers
        else:
            filter_list = jnp.split(filters, self._n_layers, axis=-1)

        graph = self._embed(graph)

        # message passing
        for n, layer in enumerate(self.layers):
            graph = layer(graph, filter_list[n])

        # readout
        if self.readout is not None:
            s, v = self.readout(graph)
        else:
            s, v = graph.nodes
        return jnp.squeeze(s), jnp.squeeze(v)
