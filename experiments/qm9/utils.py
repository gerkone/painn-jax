import os
import time
from functools import partial
from typing import Callable, Dict, Tuple
import torch

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
import schnetpack.transform as trn
from schnetpack.datasets import QM9, AtomsDataModule

from painn_jax.painn import NodeFeatures


def QM9GraphTransform(
    args,
    max_batch_nodes: int,
    max_batch_edges: int,
) -> Callable:
    """
    Build a function that converts torch DataBatch into jax GraphsTuple.

    Mostly a quick fix out of lazyness. Rewriting QM9 in jax is not trivial.
    """

    def _to_jax_graph(data: Dict) -> Tuple[jraph.GraphsTuple, jnp.array]:
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
            globals=None,
        )
        graph = jraph.pad_with_graphs(
            graph,
            n_node=max_batch_nodes,
            n_edge=max_batch_edges,
            n_graph=graph.n_node.shape[0] + 1,
        )

        # pad target
        if args.task == "node":
            target = jnp.array(data[args.prop])
            target = jnp.pad(target, [(0, max_batch_nodes - target.shape[0])])
        if args.task == "graph":
            target = jnp.append(jnp.array(data[args.prop]), 0)
        return graph, target

    return _to_jax_graph


def setup_qm9_data(args) -> Tuple[AtomsDataModule, Callable]:
    qm9tut = "./qm9tut"
    if not os.path.exists("qm9tut"):
        os.makedirs(qm9tut)

    try:
        os.remove("split.npz")
    except OSError:
        pass

    transforms = []
    if args.target == "U0":
        transforms = [
            trn.MatScipyNeighborList(cutoff=args.radius),
            trn.RemoveOffsets(args.prop, remove_mean=True, remove_atomrefs=True),
        ]
    if args.target == "mu":
        transforms = [
            trn.SubtractCenterOfMass(),
            trn.MatScipyNeighborList(cutoff=args.radius),
        ]
    transforms.append(trn.CastTo32())
    dataset = QM9(
        "./qm9.db",
        num_train=5000,
        num_val=1000,
        num_test=1000,
        batch_size=args.batch_size,
        transforms=transforms,
        num_workers=6,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=False,
        load_properties=[args.prop],
    )
    dataset.prepare_data()
    dataset.setup()

    # TODO: lazy and naive
    max_batch_nodes = int(
        1.3 * max(sum(d["_n_atoms"]) for d in dataset.val_dataloader())
    )
    max_batch_edges = int(1.3 * max(len(d["_idx_i"]) for d in dataset.val_dataloader()))

    to_graphs_tuple = QM9GraphTransform(
        args,
        max_batch_nodes=max_batch_nodes,
        max_batch_edges=max_batch_edges,
    )

    return dataset, to_graphs_tuple


@partial(jax.jit, static_argnames=["model_fn", "task"])
def predict(
    model_fn: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    graph: jraph.GraphsTuple,
    task: str,
) -> Tuple[jnp.ndarray, hk.State]:
    (pred, _), state = model_fn(params, state, graph)
    if task == "node":
        mask = jraph.get_node_padding_mask(graph)
    if task == "graph":
        mask = jraph.get_graph_padding_mask(graph)
    pred = pred * mask
    return pred, state


@partial(jax.jit, static_argnames=["model_fn", "task"])
def train_mse(
    params: hk.Params,
    state: hk.State,
    graph: jraph.GraphsTuple,
    target: jnp.ndarray,
    model_fn: Callable,
    task: str,
) -> Tuple[float, hk.State]:
    pred, state = predict(model_fn, params, state, graph, task)
    assert target.shape == pred.shape
    return jnp.mean(jnp.power(pred - target, 2)), state


def eval_mae(
    params: hk.Params,
    state: hk.State,
    data: Dict,
    graph: jraph.GraphsTuple,
    target: jnp.ndarray,
    model_fn: Callable,
    prop: str,
    task: str,
) -> Tuple[float, hk.State]:
    pred = jax.lax.stop_gradient(predict(model_fn, params, state, graph, task)[0])[:-1]
    target = target[:-1]
    if prop == QM9.U0:
        # lazy hack to get same eval targets as torch
        data[prop] = torch.tensor(pred.tolist())
        pred = jnp.array(
            trn.AddOffsets(prop, add_mean=True, add_atomrefs=True)(data)[prop]
        )
    assert target.shape == pred.shape
    return jnp.mean(jnp.abs(pred - target))


@partial(jax.jit, static_argnames=["loss_fn", "opt_update"])
def update(
    params: hk.Params,
    state: hk.State,
    graph: jraph.GraphsTuple,
    target: jnp.ndarray,
    opt_state: optax.OptState,
    loss_fn: Callable,
    opt_update: Callable,
) -> Tuple[float, hk.Params, hk.State, optax.OptState]:
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)(
        params, state, graph, target
    )
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), state, opt_state


def evaluate(
    loader,
    params: hk.Params,
    state: hk.State,
    loss_fn: Callable,
    graph_transform: Callable,
) -> Tuple[float, float]:
    eval_loss = 0.0
    eval_times = 0.0
    for data in loader:
        graph, target = graph_transform(data)
        eval_start = time.perf_counter_ns()
        loss = jax.lax.stop_gradient(loss_fn(params, state, data, graph, target))
        eval_loss += jax.block_until_ready(loss)
        eval_times += (time.perf_counter_ns() - eval_start) / 1e6

    return eval_times / len(loader), eval_loss / len(loader)
