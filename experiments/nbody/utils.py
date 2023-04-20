from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.nn import knn_graph
import jraph

from .datasets import ChargedDataset, GravityDataset
from painn_jax import NodeFeatures


def NbodyGraphTransform(
    dataset_name: str,
    n_nodes: int,
    batch_size: int,
    neighbours: Optional[int] = 6,
    relative_target: bool = False,
) -> Callable:
    """
    Build a function that converts torch DataBatch into jraph.GraphsTuple.
    """

    if dataset_name == "charged":
        # charged system is a connected graph
        full_edge_indices = jnp.array(
            [
                (i + n_nodes * b, j + n_nodes * b)
                for b in range(batch_size)
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            ]
        ).T

    def _to_steerable_graph(data: List) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        loc, vel, _, q, targets = data

        cur_batch = int(loc.shape[0] / n_nodes)

        if dataset_name == "charged":
            edge_indices = full_edge_indices[:, : n_nodes * (n_nodes - 1) * cur_batch]
            senders, receivers = edge_indices[0], edge_indices[1]
        if dataset_name == "gravity":
            batch = torch.arange(0, cur_batch)
            batch = batch.repeat_interleave(n_nodes).long()
            edge_indices = knn_graph(torch.from_numpy(np.array(loc)), neighbours, batch)
            # switched by default
            senders, receivers = jnp.array(edge_indices[0]), jnp.array(edge_indices[1])

        vel_abs = jnp.sqrt(jnp.power(vel, 2).sum(1, keepdims=True))
        norm_loc = loc - loc.mean(1, keepdims=True)

        nodes = NodeFeatures(
            s=jnp.concatenate([vel_abs, q], axis=-1),
            v=jnp.concatenate([norm_loc[:, jnp.newaxis], vel[:, jnp.newaxis]], axis=-1),
        )

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=loc[senders] - loc[receivers],
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([n_nodes] * cur_batch),
            n_edge=jnp.array([len(senders) // cur_batch] * cur_batch),
            globals=None,
        )
        # relative shift as target
        if relative_target:
            targets = targets - loc

        return graph, targets

    return _to_steerable_graph


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.vstack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


def setup_nbody_data(args) -> Tuple[DataLoader, DataLoader, DataLoader, Callable]:
    if args.dataset == "charged":
        dataset_train = ChargedDataset(
            partition="train",
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            n_bodies=args.n_bodies,
        )
        dataset_val = ChargedDataset(
            partition="val",
            dataset_name=args.dataset_name,
            n_bodies=args.n_bodies,
        )
        dataset_test = ChargedDataset(
            partition="test",
            dataset_name=args.dataset_name,
            n_bodies=args.n_bodies,
        )

    if args.dataset == "gravity":
        dataset_train = GravityDataset(
            partition="train",
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )
        dataset_val = GravityDataset(
            partition="val",
            dataset_name=args.dataset_name,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )
        dataset_test = GravityDataset(
            partition="test",
            dataset_name=args.dataset_name,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )

    graph_transform = NbodyGraphTransform(
        n_nodes=args.n_bodies,
        batch_size=args.batch_size,
        neighbours=args.neighbours,
        relative_target=(args.target == "pos"),
        dataset_name=args.dataset,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )

    return loader_train, loader_val, loader_test, graph_transform
