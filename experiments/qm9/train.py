import time
from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
from schnetpack.datasets import QM9


@partial(jax.jit, static_argnames=["model_fn"])
def predict(
    model_fn: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    graph: jraph.GraphsTuple,
) -> Tuple[jnp.ndarray, hk.State]:
    (pred, _), state = model_fn(params, state, graph)
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
    pred, state = predict(model_fn, params, state, graph)
    if task == "node":
        mask = jraph.get_node_padding_mask(graph)
    if task == "graph":
        mask = jraph.get_graph_padding_mask(graph)
    target = target * mask
    pred = pred * mask
    assert target.shape == pred.shape
    return jnp.sum(jnp.power(pred - target, 2)) / jnp.count_nonzero(mask), state


@partial(jax.jit, static_argnames=["model_fn", "eval_trn", "prop", "task"])
def eval_mae(
    params: hk.Params,
    state: hk.State,
    graph: jraph.GraphsTuple,
    target: jnp.ndarray,
    model_fn: Callable,
    eval_trn: Callable,
    prop: str,
    task: str,
) -> float:
    pred = jax.lax.stop_gradient(predict(model_fn, params, state, graph)[0])
    if prop == QM9.U0 and eval_trn is not None:
        pred = eval_trn(graph, pred)
    if task == "node":
        mask = jraph.get_node_padding_mask(graph)
    if task == "graph":
        mask = jraph.get_graph_padding_mask(graph)
    target = target * mask
    pred = pred * mask
    assert target.shape == pred.shape
    return jnp.sum(jnp.abs(pred - target)) / jnp.count_nonzero(mask)


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
        graph, target = graph_transform(data, training=False)
        eval_start = time.perf_counter_ns()
        loss = jax.lax.stop_gradient(loss_fn(params, state, graph, target))
        eval_loss += jax.block_until_ready(loss)
        eval_times += (time.perf_counter_ns() - eval_start) / 1e6

    return eval_times / len(loader), eval_loss / len(loader)
