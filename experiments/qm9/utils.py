import os
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import schnetpack.transform as trn
from schnetpack.datasets import QM9, AtomsLoader

from painn_jax.blocks import LinearXav, pooling
from painn_jax.painn import NodeFeatures, PaiNNReadout, ReadoutBuilderFn


def add_offsets(mean: float, atomrefs: jnp.ndarray) -> Callable:
    @jax.jit
    def _postprocess(graph: jraph.GraphsTuple, target: jnp.ndarray) -> jnp.ndarray:
        target = target + mean
        y0 = pooling(graph._replace(nodes=NodeFeatures(s=atomrefs[graph.nodes])))[0]
        target = (target + jnp.squeeze(y0)) * jraph.get_graph_padding_mask(graph)
        return target

    return _postprocess


def remove_offsets(mean: float, atomrefs: jnp.ndarray) -> Callable:
    @jax.jit
    def _postprocess(graph: jraph.GraphsTuple, target: jnp.ndarray) -> jnp.ndarray:
        target = target - mean * graph.n_node
        y0 = pooling(graph._replace(nodes=NodeFeatures(s=atomrefs[graph.nodes])))[0]
        target = (target - jnp.squeeze(y0)) * jraph.get_graph_padding_mask(graph)
        return target

    return _postprocess


def QM9GraphTransform(
    args,
    max_batch_nodes: int,
    max_batch_edges: int,
    train_trn: Callable,
) -> Callable:
    """
    Build a function that converts torch DataBatch into jax GraphsTuple.

    Mostly a quick fix out of lazyness. Rewriting QM9 in jax is not trivial.
    """

    def _to_jax_graph(
        data: Dict, training: bool = True
    ) -> Tuple[jraph.GraphsTuple, jnp.array]:
        senders = jnp.array(data["_idx_i"], dtype=jnp.int32)
        receivers = jnp.array(data["_idx_j"], dtype=jnp.int32)
        loc = jnp.array(data["_positions"])

        nodes = NodeFeatures(
            s=jnp.array(data["_atomic_numbers"], dtype=jnp.int32), v=None
        )

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=loc[senders] - loc[receivers],
            senders=senders,
            receivers=receivers,
            n_node=jnp.array(data["_n_atoms"]),
            n_edge=jnp.array([len(senders)]),  # TODO
            # small hack to get positions into the graph
            globals=jnp.pad(loc, [(0, max_batch_nodes - loc.shape[0] - 1), (0, 0)]),
        )
        graph = jraph.pad_with_graphs(
            graph,
            n_node=max_batch_nodes,
            n_edge=max_batch_edges,
            n_graph=graph.n_node.shape[0] + 1,
        )

        # pad target
        target = jnp.array(data[args.prop])
        if args.task == "node":
            target = jnp.pad(target, [(0, max_batch_nodes - target.shape[0] - 1)])
        if args.task == "graph":
            target = jnp.append(target, 0)

        # normalize targets
        if training and train_trn is not None:
            target = train_trn(graph, target)

        return graph, target

    return _to_jax_graph


def setup_qm9_data(
    args,
) -> Tuple[AtomsLoader, AtomsLoader, AtomsLoader, Callable, Callable, ReadoutBuilderFn]:
    qm9tut = "./qm9tut"
    if not os.path.exists("qm9tut"):
        os.makedirs(qm9tut)
    try:
        os.remove("split.npz")
    except OSError:
        pass

    transforms = [
        trn.SubtractCenterOfMass(),
        trn.MatScipyNeighborList(args.radius),
        trn.CastTo32(),
    ]
    dataset = QM9(
        "./qm9.db",
        num_train=110000,
        num_val=10000,
        batch_size=args.batch_size,
        transforms=transforms,
        remove_uncharacterized=True,
        num_workers=6,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=False,
        load_properties=[args.prop],
    )
    dataset.prepare_data()
    dataset.setup()

    train_target_transform = None
    eval_target_transform = None
    custom_readout = None
    if args.target == "U0":
        mean = float(dataset.get_stats(args.prop, True, True)[0])
        atomref = jnp.array(
            dataset.train_dataset.atomrefs[args.prop], dtype=jnp.float32
        )
        train_target_transform = remove_offsets(mean, atomref)
        eval_target_transform = add_offsets(mean, atomref)
        custom_readout = PaiNNReadout
    if args.target == "mu":
        custom_readout = DipoleReadout

    # TODO: lazy and naive
    max_batch_nodes = int(
        1.3 * max(sum(d["_n_atoms"]) for d in dataset.val_dataloader())
    )
    max_batch_edges = int(1.3 * max(len(d["_idx_i"]) for d in dataset.val_dataloader()))

    to_graphs_tuple = QM9GraphTransform(
        args,
        max_batch_nodes=max_batch_nodes,
        max_batch_edges=max_batch_edges,
        train_trn=train_target_transform,
    )

    loader_train = dataset.train_dataloader()
    loader_val = dataset.val_dataloader()
    loader_test = dataset.test_dataloader()

    return (
        loader_train,
        loader_val,
        loader_test,
        to_graphs_tuple,
        eval_target_transform,
        custom_readout,
    )


def DipoleReadout(
    hidden_size: int,
    task: str,
    pool: str = "sum",
    out_channels: int = 1,
    activation: Callable = jax.nn.silu,
    blocks: int = 2,
    eps: float = 1e-8,
) -> Callable:
    """
    PaiNN readout block modified for dipole moment prediction.

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
        charges = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(out_channels)],
            name="dipole_charge_net",
        )(graph.nodes.s)

        charges = jnp.squeeze(charges)

        sum_charge, _ = pooling(graph._replace(nodes=NodeFeatures(s=charges)), pool_fn)

        mask = jraph.get_node_padding_mask(graph)
        # same as _idx_m in schnetpack
        n_graphs = graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graphs)
        # equivalent to jnp.sum(n_node), but jittable
        sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
        batch = jnp.repeat(graph_idx, graph.n_node, 0, total_repeat_length=sum_n_node)

        charge_correction = -sum_charge / jnp.sum(mask)  # pylint: disable=E1130
        charges = charges + charge_correction[batch]

        loc = graph.globals
        mu = loc * charges[:, jnp.newaxis]

        # aggregate and normalize
        mu, _ = pooling(graph._replace(nodes=NodeFeatures(s=mu)), pool_fn)
        mu = jnp.sqrt(jnp.sum(mu**2, axis=1) + eps)
        return jnp.squeeze(mu), None

    return _readout
