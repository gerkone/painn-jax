import argparse
import time
from functools import partial
from typing import Callable

import haiku as hk
import jax
import optax
from schnetpack.datasets import QM9, AtomsDataModule

from experiments.qm9.utils import eval_mae, evaluate, setup_qm9_data, train_mse, update
from painn_jax import cutoff, radial
from painn_jax.painn import PaiNN

key = jax.random.PRNGKey(1337)


def train(
    painn,
    qm9: AtomsDataModule,
    graph_transform: Callable,
    args: argparse.Namespace,
):
    loader_train = qm9.train_dataloader()
    loader_val = qm9.val_dataloader()
    loader_test = qm9.test_dataloader()

    init_graph, _ = graph_transform(next(iter(loader_train)))
    params, painn_state = painn.init(key, init_graph)

    print(
        f"Starting {args.epochs} epochs "
        f"with {hk.data_structures.tree_size(params)} parameters.\n"
        "Jitting..."
    )

    total_steps = args.epochs * len(loader_train)

    # set up learning rate and optimizer
    learning_rate = optax.piecewise_constant_schedule(
        args.lr,
        boundaries_and_scales={
            int(total_steps * 0.8): 0.1,
            int(total_steps * 0.9): 0.1,
        },
    )
    opt_init, opt_update = optax.adamw(
        learning_rate=learning_rate, weight_decay=args.weight_decay
    )

    model_fn = painn.apply
    loss_fn = partial(train_mse, model_fn=model_fn, task=args.task)
    eval_loss_fn = partial(eval_mae, model_fn=model_fn, prop=args.prop, task=args.task)

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
    parser.add_argument(
        "--lr-scheduling",
        action="store_true",
        help="Use learning rate scheduling",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-8,
        help="Weight decay",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=10,
        help="Evaluation frequency (number of epochs)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="U0",
        help="Target variable.",
        choices=["U0", "mu"],
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Cutoff radius (Angstrom) between which atoms to add links.",
    )

    # Model parameters
    parser.add_argument(
        "--units",
        type=int,
        default=128,
        help="Number of values in the hidden layers",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of message passing layers",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="gaussian",
        choices=["gaussian", "bessel"],
        help="Radial basis function.",
    )

    parser.add_argument(
        "--double-precision",
        action="store_true",
        help="Use double precision",
    )

    args = parser.parse_args()

    assert hasattr(QM9, args.target), f"Target {args.target} not available"

    args.prop = getattr(QM9, args.target)
    args.task = "node" if args.target == "mu" else "graph"

    # if specified set jax in double precision
    if args.double_precision:
        print("Using double precision")
    jax.config.update("jax_enable_x64", args.double_precision)

    rbf = radial.EXISTING_RBF.get(args.basis, None)
    cutoff_fn = cutoff.cosine_cutoff

    painn = lambda x: PaiNN(
        hidden_size=args.units,
        n_layers=args.layers,
        max_z=20,
        radial_basis_fn=rbf,
        cutoff_fn=cutoff_fn,
        task=args.task,
        pool="avg",
        radius=args.radius,
        n_rbf=20,
        out_channels=1,
    )(x)

    painn = hk.without_apply_rng(hk.transform_with_state(painn))

    qm9, graph_transform = setup_qm9_data(args)

    train(painn, qm9, graph_transform, args)
