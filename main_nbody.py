import argparse
import time
from functools import partial
from typing import Callable, Iterable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import jraph

from experiments.nbody.utils import setup_nbody_data
from painn_jax import PaiNN, cosine_cutoff, gaussian_rbf


key = jax.random.PRNGKey(1337)


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


def train(
    painn: hk.Transformed, loader_train, loader_val, loader_test, graph_transform, args
):
    init_graph, _ = graph_transform(next(iter(loader_train)))
    params, painn_state = painn.init(key, init_graph)

    print(
        f"Starting {args.epochs} epochs on {args.dataset} "
        f"with {hk.data_structures.tree_size(params)} parameters.\n"
        "Jitting..."
    )

    opt_init, opt_update = optax.adamw(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )

    loss_fn = partial(mse, model_fn=painn.apply)
    eval_loss_fn = partial(mse, model_fn=painn.apply)

    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=eval_loss_fn, graph_transform=graph_transform)

    opt_state = opt_init(params)
    avg_time = []
    best_val = 1e10

    for e in range(args.epochs):
        train_loss = 0.0
        train_start = time.perf_counter_ns()
        for data in loader_train:
            graph, target = graph_transform(data)
            # normalize targets
            loss, params, painn_state, opt_state = update_fn(
                params=params,
                state=painn_state,
                graph=graph,
                target=target,
                opt_state=opt_state,
            )
            train_loss += loss
        train_time = (time.perf_counter_ns() - train_start) / 1e6
        train_loss /= len(loader_train)
        print(
            f"[Epoch {e+1:>4}] train loss {train_loss:.6f}, epoch {train_time:.2f}ms",
            end="",
        )
        if e % args.val_freq == 0:
            eval_time, val_loss = eval_fn(loader_val, params, painn_state)
            avg_time.append(eval_time)
            tag = ""
            if val_loss < best_val:
                tag = " (best)"
                best_val = val_loss
                _, test_loss_ckp = eval_fn(loader_test, params, painn_state)
            print(f" - val loss {val_loss:.6f}{tag}, infer {eval_time:.2f}ms", end="")

        print()

    test_loss = 0
    _, test_loss = eval_fn(loader_test, params, painn_state)
    # ignore compilation time
    avg_time = avg_time[2:]
    avg_time = sum(avg_time) / len(avg_time)
    print(
        "Training done.\n"
        f"Final test loss {test_loss:.6f} - checkpoint test loss {test_loss_ckp:.6f}.\n"
        f"Average (model) eval time {avg_time:.2f}ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (number of graphs).",
    )
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument(
        "--dataset",
        type=str,
        default="charged",
        choices=["charged", "gravity"],
        help="Dataset name",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Maximum number of samples in nbody dataset",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=10,
        help="Evaluation frequency (number of epochs)",
    )

    # nbody parameters
    parser.add_argument(
        "--target",
        type=str,
        default="pos",
        help="Target. e.g. pos, force (gravity), alpha (qm9)",
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=20,
        help="Number of connected nearest neighbours",
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        default=5,
        help="Number of bodies in the dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="small",
        choices=["small", "default", "small_out_dist"],
        help="Name of nbody data partition: default (200 steps), small (1000 steps)",
    )

    # Model parameters
    parser.add_argument(
        "--units", type=int, default=64, help="Number of values in the hidden layers"
    )
    parser.add_argument(
        "--layers", type=int, default=4, help="Number of message passing layers"
    )
    parser.add_argument(
        "--double-precision",
        action="store_true",
        help="Use double precision in model",
    )

    args = parser.parse_args()

    # if specified set jax in double precision
    jax.config.update("jax_enable_x64", args.double_precision)

    # build model
    painn = lambda x: PaiNN(
        hidden_size=args.units,
        n_layers=args.layers,
        node_type="continuous",
        radial_basis_fn=gaussian_rbf,
        cutoff_fn=cosine_cutoff,
        task="node",
        pool="avg",
        radius=1000,
        n_rbf=20,
        out_channels=1,
    )(x)
    painn = hk.without_apply_rng(hk.transform_with_state(painn))

    dataset_train, dataset_val, dataset_test, graph_transform = setup_nbody_data(args)

    train(painn, dataset_train, dataset_val, dataset_test, graph_transform, args)
