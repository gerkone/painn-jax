import argparse
from functools import partial

import haiku as hk
import jax

from experiments import setup_data, train
from painn_jax import cutoff, radial
from painn_jax.painn import PaiNN

key = jax.random.PRNGKey(1337)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size (number of graphs).",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
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

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="qm9",
        help="Dataset.",
        choices=["qm9", "charged", "gravity"],
    )
    # QM9 specific
    parser.add_argument(
        "--target",
        type=str,
        default="mu",
        help="Target variable.",
        choices=["U0", "mu"],
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Cutoff radius (Angstrom) between which atoms to add links.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=110000,
        help="Number of training atoms.",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=10000,
        help="Number of validation atoms.",
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

    # if specified set jax in double precision
    if args.double_precision:
        print("Using double precision")
    jax.config.update("jax_enable_x64", args.double_precision)

    if args.dataset == "qm9":
        from schnetpack.datasets import QM9

        assert hasattr(QM9, args.target), f"Target {args.target} not available"
        args.prop = getattr(QM9, args.target)
        args.task = "graph"
        args.node_type = "discrete"
    if args.dataset in ["charged", "gravity"]:
        args.target = "pos"
        args.task = "node"
        args.radius = 1000.0
        args.node_type = "continuous"

    (
        loader_train,
        loader_val,
        loader_test,
        graph_transform,
        eval_trn,
        custom_readout,
    ) = setup_data(args)

    rbf = radial.EXISTING_RBF.get(args.basis, None)
    cutoff_fn = cutoff.cosine_cutoff

    def painn(x):
        return PaiNN(
            hidden_size=args.units,
            n_layers=args.layers,
            max_z=20,
            node_type=args.node_type,
            radial_basis_fn=rbf,
            cutoff_fn=cutoff_fn,
            task=args.task,
            pool="sum",
            radius=args.radius,
            n_rbf=20,
            out_channels=1,
            readout_fn=custom_readout,
        )(x)

    painn = hk.without_apply_rng(hk.transform_with_state(painn))

    if args.dataset == "qm9":
        from experiments.qm9 import eval_mae, evaluate, train_mse, update

        train_loss = partial(train_mse, task=args.task)
        eval_loss = partial(eval_mae, eval_trn=eval_trn, prop=args.prop, task=args.task)
    if args.dataset in ["charged", "gravity"]:
        from experiments.nbody import evaluate, mse, update

        train_loss = mse
        eval_loss = mse

    train(
        key,
        painn,
        loader_train,
        loader_val,
        loader_test,
        loss_fn=train_loss,
        eval_loss_fn=eval_loss,
        update=update,
        evaluate=evaluate,
        graph_transform=graph_transform,
        args=args,
    )
