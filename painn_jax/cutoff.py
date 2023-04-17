from typing import Callable

import haiku as hk
import jax.numpy as jnp


def cosine_cutoff(cutoff: float) -> Callable[[jnp.ndarray], Callable]:
    r"""Behler-style cosine cutoff.

    .. math::
        f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
            & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.
    """
    hk.set_state("cutoff", cutoff)
    cutoff = hk.get_state("cutoff")

    def _cutoff(x: jnp.ndarray) -> jnp.ndarray:
        # Compute values of cutoff function
        cuts = 0.5 * (jnp.cos(x * jnp.pi / cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        mask = jnp.array(x < cutoff, dtype=jnp.float32)
        return cuts * mask

    return _cutoff


def mollifier_cutoff(cutoff: float, eps: float) -> Callable[[jnp.ndarray], Callable]:
    r"""Mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff: Cutoff radius.
        eps: Offset added to distances for numerical stability.
    """
    hk.set_state("cutoff", jnp.array([cutoff]))
    hk.set_state("eps", jnp.array([eps]))
    cutoff = hk.get_state("cutoff")
    eps = hk.get_state("eps")

    def _cutoff(x: jnp.ndarray) -> jnp.ndarray:
        # Compute values of cutoff function
        mask = (x + eps < cutoff).float()
        cuts = jnp.exp(1.0 - 1.0 / (1.0 - jnp.power(x * mask / cutoff, 2)))
        return cuts * mask

    return _cutoff
