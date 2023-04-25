import time
from functools import partial

import haiku as hk
import optax


def train(
    key,
    painn,
    loader_train,
    loader_val,
    loader_test,
    loss_fn,
    eval_loss_fn,
    update,
    evaluate,
    graph_transform,
    args,
):
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
            int(total_steps * 0.6): 0.1,
            int(total_steps * 0.9): 0.1,
        },
    )
    opt_init, opt_update = optax.adamw(
        learning_rate=learning_rate, weight_decay=args.weight_decay
    )

    model_fn = painn.apply

    loss_fn = partial(loss_fn, model_fn=model_fn)
    eval_loss_fn = partial(eval_loss_fn, model_fn=model_fn)
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
                tag = " (best)"
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
