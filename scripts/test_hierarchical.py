import equinox as eqx
import jax
import jax.random as jr

from decomposed_dynamics.dynamics_models import (
    DecomposedLinearDynamics,
    HierarchicalDecomposedDynamics,
)

key = jr.key(0)
model = HierarchicalDecomposedDynamics(3, 6, 2, DecomposedLinearDynamics, key)

print(jax.tree.leaves_with_path(model))
print(model.primitives)
print(model.G)
