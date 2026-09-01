import jax.numpy as jnp
from jax import Array, jit, vmap


@jit
def _bpdn_least_squares(
    c_t: Array, FX_t: Array, X_tplus1: Array, c_tminus1: Array, smooth_coeff: Array
) -> Array:
    reconstruction_loss = _dynamics_recon_loss_one_step(c_t, FX_t, X_tplus1)
    smooth_loss = 0.5 * smooth_coeff * ((c_t - c_tminus1) ** 2).sum()

    return reconstruction_loss + smooth_loss


@jit
def _dynamics_recon_loss_one_step(
    c_t: Array,
    FX_t: Array,
    X_tplus1: Array,
) -> Array:
    prediction = c_t @ FX_t
    return 0.5 * jnp.sum((prediction - X_tplus1) ** 2)

# @jit
# def _dynamics_recon_loss_one_step(
#     c_t: Array,
#     FX_t: Array,
#     X_tplus1: Array,
#     eps: float = 1e-8,
# ) -> Array:
#     prediction = c_t @ FX_t
#     num = jnp.sum((prediction - X_tplus1) ** 2)
#     denom = jnp.sum(X_tplus1 ** 2) + eps
#     return 0.5 * num / denom


@jit
def _dynamics_recon_loss_trial(C: Array, FX: Array, X: Array) -> Array:
    return vmap(_dynamics_recon_loss_one_step, (-1, -1, -1))(C, FX, X).mean()


@jit
# TODO: replace this with lax.scan if this is too much memory
def _dynamics_recon_loss_all(
    C: Array,  # trial (batch) x num_motif (k) x num_timepoints
    X: Array,  # batch x num_latents x num_timepoints
    F: Array,  # num_motifs x num_latents x num_latents
) -> Array:
    FX = jnp.einsum("kij, bjt -> bkit", F, X[:, :, :-1])
    return vmap(_dynamics_recon_loss_trial, (0, 0, 0))(C, FX, X[:, :, 1:]).mean()


@jit
def _operator_decorr_loss(F: Array) -> Array:
    pairwise_corrs = jnp.einsum("kij, lij -> kl", F, F)
    return jnp.sum(jnp.triu(pairwise_corrs**2, k=1))
