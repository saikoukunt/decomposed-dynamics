from types import SimpleNamespace

import click
import jax

from decomposed_dynamics.dynamics_models import (
    DecomposedLinearDeltaDynamics,
    HierarchicalDecomposedDeltaDynamics,
)

from test_ring_attractor import main as ring_attractor_main
from test_ring_attractor import run


@click.command(
    params=ring_attractor_main.params,
    context_settings=ring_attractor_main.context_settings,
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    run(
        args,
        HierarchicalDecomposedDeltaDynamics,
        DecomposedLinearDeltaDynamics,
        dt=args.dt,
    )


if __name__ == "__main__":
    with jax.default_device(jax.devices("cpu")[0]):
        main()
