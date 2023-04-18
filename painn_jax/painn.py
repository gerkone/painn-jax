from typing import Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

from .blocks import GatedEquivariantBlock, pooling, scaled_silu


class NodeFeatures(NamedTuple):
    """Simple container for scalar and vectorial node features."""

    s: jnp.ndarray
    v: jnp.ndarray


def PaiNNReadout(
    hidden_size: int,
    task: str,
    pool: str,
    output_channels: int = 1,
    activation: Callable = scaled_silu,
    blocks: int = 2,
) -> Callable:
    """
    PaiNN readout block.

    Args:
        hidden_size: Number of hidden channels.
        task: Task to perform. Either "node" or "graph".
        pool: pool method. Either "sum" or "avg".
        output_channels: Number of output channels.
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
        for i in range(blocks - 1):
            s, v = GatedEquivariantBlock(
                hidden_channels=hidden_size,
                output_channels=hidden_size,
                activation=activation,
                name=f"readout_block_{i}",
            )(s, v)

        if task == "graph":
            graph = graph._replace(nodes=NodeFeatures(s, v))
            s, v = pooling(graph, aggregate_fn=pool_fn)

        s, v = GatedEquivariantBlock(
            hidden_channels=hidden_size,
            output_channels=output_channels,
            activation=activation,
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
            [hk.Linear(hidden_size), activation] * (blocks - 1)
            + [hk.Linear(3 * hidden_size)],
            name="interaction_block",
        )

        # intra-particle context net
        self.mixing_block = hk.Sequential(
            [hk.Linear(hidden_size), activation] * (blocks - 1)
            + [hk.Linear(3 * hidden_size)],
            name="mixing_block",
        )

        # vector channel mix
        self.vector_mixing_block = hk.Linear(
            2 * hidden_size, name="vector_mixing_block"
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
        # ds = tree.tree_map(
        #     lambda s_: self._aggregate_fn(s_, senders, n_nodes), ds
        # )
        n_nodes = tree.tree_leaves(s)[0].shape[0]
        ds = self._aggregate_fn(ds, senders, n_nodes)
        dv = dv1 * dir_ij[..., jnp.newaxis] + dv2 * vj
        dv = self._aggregate_fn(dv, senders, n_nodes)

        s = s + ds
        v = v + dv

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
        v_norm = jnp.sqrt(jnp.sum(v_l**2, axis=-2, keepdims=True) + self._eps)

        ts = jnp.concatenate([s, v_norm], axis=-1)
        ds, dv, dsv = jnp.split(self.mixing_block(ts), 3, axis=-1)
        dv = dv * v_r
        dsv = dsv * jnp.sum(v_l * v_r, axis=1, keepdims=True)

        s = s + ds + dsv
        v = v + dv
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
        dir_ij = graph.edges
        s, v = self._message(s, v, dir_ij, Wij, graph.senders, graph.receivers)
        s, v = self._update(s, v)
        graph = graph._replace(nodes=graph.nodes._replace(s=s, v=v))
        return graph


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
        node_type: str = "discrete",
        cutoff_fn: Optional[Callable] = None,
        radius: float = 5.0,
        n_rbf: int = 20,
        activation: Callable = scaled_silu,
        readout: bool = False,
        task: str = "node",
        pool: str = "sum",
        output_channels: Optional[int] = None,
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
            node_type: Type of node features. Either "discrete" or "continuous".
            cutoff_fn: Cutoff method. None means no cutoff.
            radius: Cutoff radius.
            n_rbf: Number of radial basis functions.
            activation: Activation function.
            readout: If True, use a readout layer. If False, returns representations.
            task: Regression task to perform. Either "node"-wise or "graph"-wise.
            pool: Node readout pool method. Only used in "graph" tasks.
            output_channels: Number of output scalar/vector channels. Used in readout.
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
        assert (
            output_channels is not None or not readout
        ), "output_channels must be specified if readout is True"

        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._eps = eps
        self._shared_filters = shared_filters
        self._shared_interactions = shared_interactions

        self.cutoff_fn = cutoff_fn(radius) if cutoff_fn else None
        self.radial_basis_fn = radial_basis_fn(n_rbf, radius)

        if node_type == "discrete":
            self.scalar_embedding = hk.Embed(
                max_z, hidden_size, name="scalar_embedding"
            )
        else:
            self.scalar_embedding = hk.Linear(hidden_size, name="scalar_embedding")
        # mix vector channels
        self.vector_embedding = hk.Linear(hidden_size, name="vector_embedding")

        if shared_filters:
            self.filter_net = hk.Linear(3 * hidden_size, name="filter_net")
        else:
            self.filter_net = hk.Linear(n_layers * 3 * hidden_size, name="filter_net")

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
        if readout:
            self.readout = PaiNNReadout(
                hidden_size,
                task,
                pool,
                output_channels=output_channels,
                activation=activation,
            )

    def _embed(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Embed the input nodes."""
        # TODO check if this does what it should
        s = graph.nodes.s
        if graph.nodes.v is not None:
            # initialize the vector with the global positions
            v = graph.nodes.v
            v = jnp.reshape(v, (s.shape[0], 3, v.shape[-1]))
        else:
            # initialize the vector with zeros (as in the paper)
            # TODO check if they remain zeros after embedding
            v = jnp.zeros((s.shape[0], 3, 1))

        s = self.scalar_embedding(s)[:, jnp.newaxis, :]
        v = self.vector_embedding(v)

        return graph._replace(nodes=graph.nodes._replace(s=s, v=v))

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
        norm_ij = jnp.linalg.norm(graph.edges, axis=1, keepdims=True)
        # TODO this only makes sense for displacement vectors as edges/r_ij
        dir_ij = graph.edges / norm_ij
        phi_ij = self.radial_basis_fn(norm_ij)
        # replace edges with normalized directions
        graph = graph._replace(edges=dir_ij)

        if self.cutoff_fn is not None:
            norm_ij = self.cutoff_fn(norm_ij)  # pylint: disable=not-callable

        # compute filters
        filters = self.filter_net(phi_ij) * norm_ij[..., jnp.newaxis]
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
        return jnp.squeeze(s), jnp.squeeze(v)
