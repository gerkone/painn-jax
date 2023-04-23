import time
from functools import partial
from typing import Callable, Iterable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax


def predict(
    model_fn: hk.TransformedWithState,
    params: hk.Params,
    state: hk.State,
    graph: jraph.GraphsTuple,
) -> Tuple[jnp.ndarray, hk.State]:
    pred, state = model_fn(params, state, graph)
    return pred, state


@partial(jax.jit, static_argnames=["model_fn"])
def mse(
    params: hk.Params,
    state: hk.State,
    graph: jraph.GraphsTuple,
    target: jnp.ndarray,
    model_fn: Callable,
) -> Tuple[float, hk.State]:
    (_, pred), state = predict(model_fn, params, state, graph)
    assert target.shape == pred.shape
    return (jnp.power(pred - target, 2)).mean(), state


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
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, state, graph, target
    )
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), state, opt_state


def evaluate(
    loader: Iterable,
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
        loss, _ = jax.lax.stop_gradient(loss_fn(params, state, graph, target))
        eval_loss += jax.block_until_ready(loss)
        eval_times += (time.perf_counter_ns() - eval_start) / 1e6

    return eval_times / len(loader), eval_loss / len(loader)
