import functools
import numpy as np
from tqdm import tqdm
from scipy.optimize import nnls

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.debug as jdbg
# import jax.config as jcfg
from jax import Array, lax, vmap
from jaxopt import ProximalGradient
from jaxopt.prox import prox_non_negative_lasso, prox_lasso, prox_group_lasso
import jax.debug as jdbg
from optax import l2_loss
from scipy.optimize import nnls

from cr.sparse import lop
from cr.sparse.cvx.adm import yall1

from spgl1 import (
    spg_bpdn,
    spgl1,
    norm_l1nn_project,
    norm_l1nn_primal,
    norm_l1nn_dual
)

from decomposed_dynamics.dynamics_models import DecomposedDynamicsModel
from decomposed_dynamics.inference import (
    InferenceHyperparams,
    NoObsInferenceHyperparams,
)
from decomposed_dynamics.observation_models import ObservationModel
from decomposed_dynamics.utils import _reweight_l1

# jcfg.update("jax_enable_x64", True)

@eqx.filter_jit
def bpdn_inference(
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    observations: Array,
    hyperparams: InferenceHyperparams,
):
    if getattr(hyperparams, "solver", "proximal") != "proximal":
        raise ValueError("Observation-based inference supports only proximal; the other solvers require a linear observation objective.")
    infer_one_trial = functools.partial(
        _bpdn_infer_one_trial,
        observation_model,
        dynamics_model,
        hyperparams=hyperparams,
    )
    state = vmap(infer_one_trial)(observations)

    return (
        state[..., : dynamics_model.num_latents],
        state[..., dynamics_model.num_latents :],
    )


@eqx.filter_jit
def _bpdn_infer_one_trial(
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    observations: Array,
    hyperparams: InferenceHyperparams,
) -> Array:
    solver = ProximalGradient(
        _bpdn_least_squares,
        prox_non_negative_lasso,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )

    infer_one_timestep = functools.partial(
        _bpdn_infer_one_timestep,
        observation_model=observation_model,
        dynamics_model=dynamics_model,
        solver=solver,
        hyperparams=hyperparams,
    )
    _, state = lax.scan(
        infer_one_timestep,
        (
            jnp.zeros(dynamics_model.num_latents + dynamics_model.num_operators),
            jnp.bool_(True),
        ),
        observations,
    )

    return state


@eqx.filter_jit
def _bpdn_infer_one_timestep(
    carry: tuple[Array, Array],
    y_t: Array,
    observation_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    solver: ProximalGradient,
    hyperparams: InferenceHyperparams,
) -> tuple[tuple[Array, Array], Array]:

    state, is_first = carry
    x_tminus1 = state[: dynamics_model.num_latents]
    c_tminus1 = state[dynamics_model.num_latents :]

    flows = dynamics_model.compute_operator_flows(x_tminus1)
    smooth_coeff = jnp.where(is_first, 0.0, hyperparams.smooth_coeff)
    l1_coeff = hyperparams.l1_coeff * jnp.ones(
        dynamics_model.num_latents + dynamics_model.num_operators
    )
    l1_coeff[: dynamics_model.num_latents] = 0

    # solve vanilla L1
    state, _ = solver.run(
        jnp.zeros(dynamics_model.num_latents + dynamics_model.num_operators),
        hyperparams_prox=l1_coeff,
        obs_model=observation_model,
        dynamics_model=dynamics_model,
        flows=flows,
        y_t=y_t,
        c_tminus1=c_tminus1,
        x_tminus1=x_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    state, _ = solver.run(
        state,
        hyperparams_prox=_reweight_l1(state, l1_coeff, hyperparams.l1_reweight_coeff),
        obs_model=observation_model,
        dynamics_model=dynamics_model,
        flows=flows,
        y_t=y_t,
        c_tminus1=c_tminus1,
        x_tminus1=x_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    state = jnp.where(jnp.any(jnp.isnan(state)), jnp.zeros_like(state), state)
    return (
        state,
        jnp.bool_(False),
    ), state


@eqx.filter_jit
def _bpdn_least_squares(
    state: Array,
    obs_model: ObservationModel,
    dynamics_model: DecomposedDynamicsModel,
    flows: Array,
    y_t: Array,
    c_tminus1: Array,
    x_tminus1: Array,
    dynamics_loss_coeff: Array,
    smooth_coeff: Array,
) -> Array:
    x_t = state[: dynamics_model.num_latents]
    c_t = state[dynamics_model.num_latents :]

    rates = obs_model.predict_rates(x_t)
    data_nll = obs_model.neg_log_likelihood(rates, y_t)

    predicted_state = dynamics_model.predict_next_state(x_tminus1, c_t, flows)
    dynamics_recon_loss = dynamics_loss_coeff * l2_loss(predicted_state, x_t).sum()

    smooth_loss = smooth_coeff * l2_loss(c_t, c_tminus1).sum()

    return data_nll + dynamics_recon_loss + smooth_loss


def bpdn_inference_no_obs(
    dynamics_model: DecomposedDynamicsModel,
    X: Array,
    hyperparams: NoObsInferenceHyperparams,
) -> Array:
    # print("Using solver: ", getattr(hyperparams, "solver"))
    if getattr(hyperparams, "solver", "proximal") != "proximal":
        return _no_obs_linear_batch(dynamics_model, X, hyperparams)
    infer_one_trial = functools.partial(
        _bpdn_infer_one_no_obs_trial,
        dynamics_model,
        hyperparams=hyperparams,
    )
    return vmap(infer_one_trial)(X)

@eqx.filter_jit
def _bpdn_infer_one_no_obs_trial(
    dynamics_model: DecomposedDynamicsModel,
    latents: Array,
    hyperparams: NoObsInferenceHyperparams,
) -> Array:
    solver = ProximalGradient(
        _bpdn_no_obs_least_squares,
        prox_non_negative_lasso,
        maxiter=hyperparams.max_iter,
        tol=hyperparams.tol,
    )

    infer_one_timestep = functools.partial(
        _bpdn_infer_one_no_obs_timestep,
        dynamics_model=dynamics_model,
        solver=solver,
        hyperparams=hyperparams,
    )
    _, C = lax.scan(
        infer_one_timestep,
        (jnp.zeros(dynamics_model.num_operators), jnp.bool_(True)),
        (latents[:-1, :], latents[1:, :]),
    )

    return C


@eqx.filter_jit
def _bpdn_infer_one_no_obs_timestep(
    carry: tuple[Array, Array],
    xs: tuple[Array, Array],
    dynamics_model: DecomposedDynamicsModel,
    solver: ProximalGradient,
    hyperparams: NoObsInferenceHyperparams,
) -> tuple[tuple[Array, Array], Array]:

    c_tminus1, is_first = carry
    x_tminus1, x_t = xs
    flows = dynamics_model.compute_operator_flows(x_tminus1)
    # norms = jnp.linalg.norm(flows, axis=0, keepdims=True)
    # normalized_flows = flows / jnp.maximum(norms, 1e-8)
    smooth_coeff = jnp.where(is_first, 0.0, hyperparams.smooth_coeff)

    # solve vanilla L1
    c_t, _ = solver.run(
        jnp.zeros(dynamics_model.num_operators),
        hyperparams_prox=hyperparams.l1_coeff,
        dynamics_model=dynamics_model,
        flows=flows,
        x_t=x_t,
        x_tminus1=x_tminus1,
        c_tminus1=c_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    # solve reweighted L1 using previous as warm start
    c_t, _ = solver.run(
        c_t,
        hyperparams_prox=_reweight_l1(
            c_t, hyperparams.l1_coeff, hyperparams.l1_reweight_coeff
        ),
        dynamics_model=dynamics_model,
        flows=flows,
        x_t=x_t,
        x_tminus1=x_tminus1,
        c_tminus1=c_tminus1,
        dynamics_loss_coeff=hyperparams.dynamics_loss_coeff,
        smooth_coeff=smooth_coeff,
    )

    c_t = jnp.where(jnp.any(jnp.isnan(c_t)), jnp.zeros_like(c_t), c_t)
    return (c_t, jnp.bool_(False)), c_t


@eqx.filter_jit
def _bpdn_no_obs_least_squares(
    c_t: Array,
    dynamics_model: DecomposedDynamicsModel,
    flows: Array,
    x_t: Array,
    x_tminus1: Array,
    c_tminus1: Array,
    dynamics_loss_coeff: Array,
    smooth_coeff: Array,
) -> Array:
    predicted_state = dynamics_model.predict_next_state(x_tminus1, c_t, flows)
    reconstruction_loss = l2_loss(predicted_state, x_t).sum()
    null_predictions = dynamics_model.predict_next_state(
        x_tminus1, jnp.zeros_like(c_t), flows
    )
    variance = jnp.maximum(l2_loss(null_predictions, x_t).sum(axis=-1), 1e-3)

    smooth_loss = smooth_coeff * l2_loss(c_t, c_tminus1).sum()

    # jdbg.print("--------> bpdn loss: {var}", var=dynamics_loss_coeff * reconstruction_loss / variance)
    # jdbg.print("------>{var}", var={
    #     "dynamics_loss_coeff": dynamics_loss_coeff,
    #     "smooth_coeff": smooth_coeff,
    #     "variance": variance,
    #     "recon_loss": reconstruction_loss,
    #     "returned_loss": dynamics_loss_coeff * reconstruction_loss / variance + smooth_loss,
    # })
    # return dynamics_loss_coeff * reconstruction_loss / variance + smooth_loss
    return dynamics_loss_coeff * reconstruction_loss + smooth_loss


# ---------------------------------------------------------------------------
# Linear, nonnegative sparse solvers for no-observation inference.
# ---------------------------------------------------------------------------

# def _no_obs_linear_batch(dynamics_model, X, hyperparams):
#     """Solve all trials sequentially, returning coefficients (batch, T-1, K).

#     At time t, let x_prev, x_next in R^D and c in R^K. Assuming
#     predict_next_state(x_prev, c, flows) is affine in c, construct

#         b = x_next - predict_next_state(x_prev, 0, flows),
#         A[:, j] = predict_next_state(x_prev, e_j, flows)
#                   - predict_next_state(x_prev, 0, flows).

#     The original quadratic objective becomes

#         min_{c >= 0} (d/2)||A c - b||_2^2
#                      + (s/2)||c-c_prev||_2^2 + lambda||c||_1,

#     where d=dynamics_loss_coeff, s=smooth_coeff (zero at t=0),
#     and lambda=l1_coeff. This is equivalent to

#         min_{c >= 0} (1/2)||M c - q||_2^2 + lambda||c||_1,
#         M = [sqrt(d) A; sqrt(s) I],
#         q = [sqrt(d) b; sqrt(s) c_prev].

#     A second solve uses lambda_j = _reweight_l1(c, lambda, epsilon).
#     Solver-specific approximations and limitations are in _solve_linear.
#     """
#     X = np.asarray(X)
#     if X.ndim != 3:
#         raise ValueError("X must have shape (batch, time, num_latents)")
#     results = []
#     pbar = tqdm(X, desc="Solving snippets")

#     total_error = 0.0
#     total_steps = 0

#     for trial in pbar:
#         previous = np.zeros(dynamics_model.num_operators)
#         coefficients = []
#         for t in range(len(trial) - 1):
#             x_prev = jnp.asarray(trial[t])
#             flows = dynamics_model.compute_operator_flows(x_prev)
#             k = dynamics_model.num_operators
#             zero = jnp.zeros(k, dtype=x_prev.dtype)
#             base = np.asarray(dynamics_model.predict_next_state(x_prev, zero, flows))
#             columns = []
#             for j in range(k):
#                 e = np.zeros(k)
#                 e[j] = 1.0
#                 pred = dynamics_model.predict_next_state(x_prev, jnp.asarray(e, dtype=x_prev.dtype), flows)
#                 columns.append(np.asarray(pred) - base)
#             A = np.stack(columns, axis=1)
#             b = np.asarray(trial[t + 1]) - base
#             # Fail explicitly if the assumed affine model is not valid.
#             probe = np.full(k, 0.37)
#             actual = np.asarray(dynamics_model.predict_next_state(x_prev, jnp.asarray(probe, dtype=x_prev.dtype), flows))
#             if not np.allclose(actual, base + A @ probe, rtol=1e-4, atol=1e-5):
#                 raise ValueError("Dynamics are not affine in coefficients; linear solvers are invalid")
#             d = float(hyperparams.dynamics_loss_coeff)
#             smooth = float(hyperparams.smooth_coeff) if t else 0.0
#             if d < 0 or smooth < 0:
#                 raise ValueError("Quadratic loss weights must be nonnegative")
#             M = np.vstack((np.sqrt(d) * A, np.sqrt(smooth) * np.eye(k)))
#             q = np.concatenate((np.sqrt(d) * b, np.sqrt(smooth) * previous))
#             lam = np.broadcast_to(np.asarray(hyperparams.l1_coeff, dtype=float), (k,))
#             method = hyperparams.solver.lower()

#             # First solve: initialize from previous timestep
#             initial = previous.copy() if t > 0 else None
#             c = _solve_linear(M, q, lam, method, hyperparams, initial=initial)
#             weights = np.asarray(_reweight_l1(jnp.asarray(c), jnp.asarray(lam), hyperparams.l1_reweight_coeff))
#             c = _solve_linear(M, q, weights, method, hyperparams, initial=c)
#             previous = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
#             coefficients.append(previous.copy())
#             # Normalized reconstruction error
#             reconstruction = A @ previous
#             error = np.linalg.norm(reconstruction - b)**2 / (
#                 np.linalg.norm(b)**2 + 1e-12
#             )

#             total_error += error
#             total_steps += 1
#             avg_error = total_error / total_steps

#             pbar.set_postfix(
#                 t=t,
#                 error=f"{error:.4f}",
#                 avg_error=f"{avg_error:.4f}",
#             )
#         results.append(np.asarray(coefficients).reshape(-1, dynamics_model.num_operators))
#     return jnp.asarray(np.stack(results))
def make_basis_predictor(dynamics_model):
    k = dynamics_model.num_operators

    @jax.jit
    def predict_basis(x_prev):
        flows = dynamics_model.compute_operator_flows(x_prev)
        eye = jnp.eye(k, dtype=x_prev.dtype)
        zero = jnp.zeros(k, dtype=x_prev.dtype)
        base = dynamics_model.predict_next_state(
            x_prev, zero, flows
        )
        predictions = vmap(
            lambda c: dynamics_model.predict_next_state(
                x_prev, c, flows
            )
        )(eye)
        A = jnp.moveaxis(predictions - base, 0, -1)
        return base, A
    return predict_basis

def _no_obs_linear_batch(dynamics_model, X, hyperparams):
    """
    Solve all trials sequentially, returning coefficients (batch, T-1, K).

    Optimizations:
      - Vectorized operator construction using jax.vmap
      - JAX-native matrix and vector construction
      - Warm starts from previous timestep
      - Two-stage reweighted L1 inference
      - Running reconstruction error in tqdm

    Objective:
        min_{c >= 0}
            (d/2) ||A c - b||^2
          + (s/2) ||c - c_prev||^2
          + lambda ||c||_1
    """

    X = jnp.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            "X must have shape (batch, time, num_latents)"
        )

    n_trials, T, D = X.shape
    k = dynamics_model.num_operators

    results = []

    total_error = 0.0
    total_steps = 0

    d = float(hyperparams.dynamics_loss_coeff)
    smooth_coeff = float(hyperparams.smooth_coeff)

    if d < 0 or smooth_coeff < 0:
        raise ValueError(
            "Quadratic loss weights must be nonnegative"
        )

    sqrt_d = jnp.sqrt(jnp.asarray(d, dtype=X.dtype))
    sqrt_s = jnp.sqrt(jnp.asarray(smooth_coeff, dtype=X.dtype))

    eye = jnp.eye(k, dtype=X.dtype)
    zero = jnp.zeros(k, dtype=X.dtype)

    lam = jnp.broadcast_to(
        jnp.asarray(hyperparams.l1_coeff, dtype=X.dtype),
        (k,)
    )

    method = hyperparams.solver.lower()

    # # Vectorized prediction across the K basis coefficient vectors.
    # def predict_basis(x_prev, flows):
    #     base = dynamics_model.predict_next_state(
    #         x_prev, zero, flows
    #     )

    #     def predict_one(c):
    #         return dynamics_model.predict_next_state(
    #             x_prev, c, flows
    #         )

    #     predictions = vmap(predict_one)(eye)

    #     # (K, D) -> (D, K)
    #     A = jnp.moveaxis(predictions - base, 0, -1)

    #     return base, A
    predict_basis = make_basis_predictor(dynamics_model)

    pbar = tqdm(range(n_trials), desc="Solving snippets")

    for trial_idx in pbar:
        trial = X[trial_idx]

        previous = jnp.zeros(k, dtype=X.dtype)
        coefficients = []

        for t in range(T - 1):
            x_prev = trial[t]
            x_next = trial[t + 1]

            # # Compute flows
            flows = dynamics_model.compute_operator_flows(x_prev)

            # # Construct linear system using vmap
            # base, A = predict_basis(x_prev, flows)
            base, A = predict_basis(x_prev)
            b = x_next - base

            # Optional affine-model verification.
            # Remove after confirming predict_next_state is affine.
            probe = jnp.full((k,), 0.37, dtype=X.dtype)

            actual = dynamics_model.predict_next_state(
                x_prev, probe, flows
            )

            if not bool(jnp.allclose(
                actual,
                base + A @ probe,
                rtol=1e-4,
                atol=1e-5
            )):
                raise ValueError(
                    "Dynamics are not affine in coefficients; "
                    "linear solvers are invalid"
                )

            # Temporal smoothness is disabled at t=0
            current_sqrt_s = (
                sqrt_s if t > 0
                else jnp.asarray(0.0, dtype=X.dtype)
            )

            # Construct augmented linear system
            M = jnp.concatenate(
                [
                    sqrt_d * A,
                    current_sqrt_s * eye
                ],
                axis=0
            )

            q = jnp.concatenate(
                [
                    sqrt_d * b,
                    current_sqrt_s * previous
                ],
                axis=0
            )

            # Warm start from previous timestep
            initial = previous if t > 0 else None

            # First L1 solve
            c = _solve_linear(
                M,
                q,
                lam,
                method,
                hyperparams,
                initial=None #initial
            )

            c = jnp.asarray(c, dtype=X.dtype)

            # # Reweight L1 penalty
            # weights = _reweight_l1(
            #     c,
            #     lam,
            #     hyperparams.l1_reweight_coeff
            # )

            # weights = jnp.asarray(weights, dtype=X.dtype)

            # # Second solve, initialized from first solution
            # c = _solve_linear(
            #     M,
            #     q,
            #     weights,
            #     method,
            #     hyperparams,
            #     initial=c
            # )

            # Sanitize coefficients
            previous = jnp.nan_to_num(
                jnp.asarray(c, dtype=X.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            coefficients.append(previous)

            # Normalized dynamics reconstruction error
            residual = np.asarray(A) @ c - np.asarray(b)

            absolute_error = np.linalg.norm(residual)
            target_norm = np.linalg.norm(b)

            normalized_error = (
                absolute_error**2 / (target_norm**2 + 1e-12)
            )

            pbar.set_postfix(
                t=t,
                abs_err=f"{absolute_error:.2e}",
                rel_err=f"{normalized_error:.2e}",
                c_norm=f"{np.linalg.norm(c):.2e}",
            )

        results.append(jnp.stack(coefficients, axis=0))

    return jnp.stack(results, axis=0)

def _solve_linear(M, q, weights, method, hp, initial=None):
    """Solve a nonnegative sparse linear inverse problem.

    Denote f(c) = 0.5 ||M c - q||_2^2 and W=diag(weights).

    inv:   min_{c>=0} ||M c-q||_2^2 (NNLS; ignores W).
    lasso: min_{c>=0} f(c)+||W c||_1 (sklearn Lasso; scalar W only).
    ista:  c_{k+1} = max(0, c_k - alpha M.T(M c_k-q)-alpha w).
    fista: accelerated ISTA with Nesterov momentum.
    omp:   greedy sparse support selection followed by nonnegative LS;
           sparsity is selected by omp_n_nonzero, not W.
    irls:  iterative reweighted quadratic approximation to L1:
           c <- argmin_{c>=0} f(c) + 0.5 c.T D(c_old) c,
           D_jj = w_j / sqrt(c_old_j^2+eps^2).
    spgl1: JAX-only SPGL1-style Pareto search for weighted nonnegative BPDN:
           min_{c>=0} ||W c||_1 subject to ||M c-q||_2 <= sigma.
           This is NOT equivalent to a fixed lambda without calibration.

    ISTA/FISTA, OMP and IRLS are implemented locally with nonnegative
    constraints. SPGL1 uses JAX projected accelerated gradient and
    a Pareto root search; nonnegativity is enforced at each iteration.
    """
    k = M.shape[1]
    w = np.broadcast_to(np.asarray(weights, dtype=float), (k,))
    if np.any(w < 0):
        raise ValueError("L1 weights must be nonnegative")
    iterations = int(getattr(hp, 'spgl1_max_iter', 1000))
    tol = float(getattr(hp, 'tol', 1e-6))
    if method == 'inv':
        return nnls(M, q)[0]
    if method == 'lasso':
        from sklearn.linear_model import Lasso
        # Feature scaling converts weighted L1 to ordinary Lasso.
        # Zero-weight columns cannot be scaled this way.
        if np.any(w == 0):
            raise ValueError('lasso requires strictly positive L1 weights')
        scaled = M / w[None, :]
        model = Lasso(alpha=1.0 / M.shape[0], fit_intercept=False,
                      positive=True, max_iter=iterations, tol=tol)
        model.fit(scaled, q)
        return model.coef_ / w
    if method in ('ista', 'fista'):
        # Native JAX iteration supports nonnegative weighted shrinkage.
        L = max(float(np.linalg.norm(M, ord=2) ** 2), 1e-12)
        c = np.zeros(k) if initial is None else np.maximum(0, initial.copy())
        z = c.copy()
        momentum = 1.0
        for _ in range(iterations):
            nxt = np.maximum(0, z - (M.T @ (M @ z - q) + w) / L)
            if np.linalg.norm(nxt - c) <= tol * max(1., np.linalg.norm(c)):
                c = nxt
                break
            if method == 'fista':
                new_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
                z = nxt + (momentum - 1) / new_momentum * (nxt - c)
                momentum = new_momentum
            else:
                z = nxt
            c = nxt
        return c
    if method == 'omp':
        n = int(getattr(hp, 'omp_n_nonzero', max(1, min(k, M.shape[0]) // 2)))
        n = max(1, min(n, k, M.shape[0]))
        # Greedy selection using positive correlations, with NNLS refitting.
        support = []
        residual = q.copy()
        for _ in range(n):
            correlation = M.T @ residual
            correlation[support] = -np.inf
            j = int(np.argmax(correlation))
            if correlation[j] <= tol:
                break
            support.append(j)
            fitted = nnls(M[:, support], q)[0]
            residual = q - M[:, support] @ fitted
        out = np.zeros(k)
        if support:
            out[support] = fitted
        return out
    if method == 'irls':
        epsilon = float(getattr(hp, 'irls_epsilon', 1e-5))
        if epsilon <= 0:
            raise ValueError('irls_epsilon must be positive')
        c = nnls(M, q)[0] if initial is None else np.maximum(0, initial.copy())
        for _ in range(iterations):
            diagonal = w / np.sqrt(c ** 2 + epsilon ** 2)
            aug_M = np.vstack((M, np.diag(np.sqrt(diagonal))))
            aug_q = np.concatenate((q, np.zeros(k)))
            nxt = nnls(aug_M, aug_q)[0]
            if np.linalg.norm(nxt - c) <= tol * max(1., np.linalg.norm(c)):
                c = nxt
                break
            c = nxt
        return c
    if method == 'spgl1':
        sigma = getattr(hp, "spgl1_sigma", None)
        if sigma is None:
            raise ValueError("spgl1_sigma is required")
        sigma = float(sigma)
        if sigma < 0:
            raise ValueError("spgl1_sigma must be nonnegative")
        if np.any(w <= 0):
            raise ValueError(
                "SPGL1 requires strictly positive L1 weights"
            )

        x0 = (
            None if initial is None
            else np.maximum(0.0, np.asarray(initial, dtype=float))
        )

        c, residual, gradient, info = spgl1(
            np.asarray(M, dtype=float),
            np.asarray(q, dtype=float),
            tau=hp.l1_coeff,
            sigma=sigma, # None puts it in lasso mode
            x0=x0,
            weights=np.ones(M.shape[1]), # Uniform L1 penalty; originally set to w,
            project=norm_l1nn_project,
            primal_norm=norm_l1nn_primal,
            dual_norm=norm_l1nn_dual,
            iter_lim=iterations,
            opt_tol=tol,
            verbosity=0,
        )

        if not np.all(np.isfinite(c)):
            raise RuntimeError("SPGL1 returned nonfinite coefficients")

        return np.asarray(c)
    
    if method == "yall1":
        dtype = jnp.float64

        rho = float(getattr(hp, "yall1_rho", 1.0))
        if rho <= 0:
            raise ValueError("yall1_rho must be positive")

        M_jax = jnp.asarray(M, dtype=dtype)
        q_jax = jnp.asarray(q, dtype=dtype).reshape(-1)
        w_jax = jnp.asarray(w, dtype=dtype).reshape(-1)

        if initial is None:
            x0 = None
        else:
            candidate = jnp.maximum(
                jnp.asarray(initial, dtype=dtype).reshape(-1),
                0.0
            )

            x0 = (
                candidate
                if float(jnp.linalg.norm(candidate)) > 1e-12
                else None
            )

        A = lop.matrix(M_jax)

        sol = yall1.solve(
            A=A,
            b=q_jax,
            x0=x0,
            weights=w_jax,
            nonneg=True,
            rho=rho,
            delta=0.0,
            tolerance=tol,
            max_iters=iterations,
            jit=True,
        )

        c = np.asarray(sol.x)

        if not np.all(np.isfinite(c)):
            raise RuntimeError("YALL1 returned nonfinite coefficients")
        
        # M_np = np.asarray(M)
        # q_np = np.asarray(q).reshape(-1)

        # recon = M_np @ c
        # residual = recon - q_np

        # print("\n=== YALL1 Diagnostics ===")
        # print("M shape:", M_np.shape)
        # print("q shape:", q_np.shape)
        # print("c shape:", c.shape)

        # print("Coefficient min/max:", c.min(), c.max())
        # print("Nonzero coefficients:", np.count_nonzero(c))
        # print("Coefficient norm:", np.linalg.norm(c))

        # print("Target norm:", np.linalg.norm(q_np))
        # print("Reconstruction norm:", np.linalg.norm(recon))
        # print("Residual norm:", np.linalg.norm(residual))

        # print(
        #     "Normalized reconstruction loss:",
        #     np.linalg.norm(residual)**2 /
        #     (np.linalg.norm(q_np)**2 + 1e-12)
        # )

        # print("rho:", rho)
        # print("Weights min/max:", np.min(w), np.max(w))
        # print("Solver result:", sol)
        # print("Iterations:", sol.iterations)

        # print("M norm:", np.linalg.norm(M_np))
        # print("M singular values:", np.linalg.svd(M_np, compute_uv=False))
        # print("M rank:", np.linalg.matrix_rank(M_np))
        # print("M column norms:", np.linalg.norm(M_np, axis=0))


        # c_nnls, _ = nnls(np.asarray(M), np.asarray(q).reshape(-1))

        # print("NNLS coefficients:", c_nnls)
        # print("NNLS nonzeros:", np.count_nonzero(c_nnls))
        # print(
        #     "NNLS reconstruction loss:",
        #     np.linalg.norm(np.asarray(M) @ c_nnls - np.asarray(q).reshape(-1))**2
        #     / np.linalg.norm(q)**2
        # )
        # M_np = np.asarray(M)
        # q_np = np.asarray(q).reshape(-1)

        # c_ls, *_ = np.linalg.lstsq(M_np, q_np, rcond=None)

        # loss_ls = (
        #     np.linalg.norm(M_np @ c_ls - q_np)**2
        #     / np.linalg.norm(q_np)**2
        # )

        # print("Unconstrained LS loss:", loss_ls)
        # print("LS coefficients:", c_ls)
        # print("Negative coefficients:", np.sum(c_ls < 0))

        # print("========================\n")

        return np.maximum(c, 0.0)

    # if method == 'spgl1':
    #     sigma = getattr(hp, 'spgl1_sigma', None)
    #     if sigma is None:
    #         raise ValueError('spgl1_sigma is required for SPGL1')
    #     if float(sigma) < 0:
    #         raise ValueError('spgl1_sigma must be nonnegative')
    #     if np.any(w <= 0):
    #         raise ValueError('spgl1 requires strictly positive L1 weights')
    #     return np.asarray(_jax_spgl1_nonnegative(
    #         jnp.asarray(M), jnp.asarray(q), jnp.asarray(w),
    #         float(sigma), iterations, tol,
    #         int(getattr(hp, 'spgl1_outer_iter', 40)),
    #     ))
    raise ValueError(f'Unknown solver {method!r}')


def _jax_project_nonnegative_l1_ball(v: Array, tau: Array) -> Array:
    """Euclidean projection onto {z >= 0 : 1.T z <= tau}.

    For v in R^K, the KKT conditions give

        z_i = max(v_i - theta, 0),
        theta >= 0, theta * (1.T z - tau) = 0.

    The threshold theta is determined from sorted entries of max(v, 0).
    This is the simplex projection when the L1 constraint is active.
    """
    u = jnp.maximum(v, 0)
    sorted_u = jnp.sort(u)[::-1]
    cssv = jnp.cumsum(sorted_u)
    idx = jnp.arange(1, v.shape[0] + 1)
    candidate = (cssv - tau) / idx
    active = sorted_u > candidate
    rho = jnp.maximum(jnp.sum(active), 1)
    theta = jnp.maximum((cssv[rho - 1] - tau) / rho, 0)
    return jnp.maximum(v - theta, 0)


def _jax_spgl1_nonnegative(
    M: Array, q: Array, weights: Array, sigma: float,
    max_iter: int = 1000, tol: float = 1e-6, outer_iter: int = 40,
) -> Array:
    """JAX-only SPGL1-style solver for nonnegative weighted BPDN.

    Given M in R^(N x K), q in R^N, and positive weights w in R^K,
    approximately solve

        min_{c >= 0} w.T c   subject to ||M c - q||_2 <= sigma.

    Set z = diag(w)c and B = M diag(w)^(-1). The Pareto family is

        phi(tau) = min_{z >= 0, 1.T z <= tau} ||B z - q||_2.

    An outer bisection searches for the smallest tau with
    phi(tau) <= sigma. Each Pareto subproblem minimizes

        f(z) = 0.5 ||B z - q||_2^2

    with accelerated projected gradient and the exact projection onto
    the nonnegative L1 ball. The result c=diag(w)^(-1)z is nonnegative.

    This is an SPGL1-style Pareto method, NOT a reproduction of the
    original SPGL1 spectral projected-gradient/Newton algorithm.
    Convergence depends on inner/outer iteration budgets. If sigma is
    below the minimum attainable residual, the returned result is the
    least-squares approximation rather than a feasible BPDN solution.
    """
    if max_iter < 1 or outer_iter < 1:
        raise ValueError('max_iter and spgl1_outer_iter must be positive')
    B = M / weights[jnp.newaxis, :]
    gram = B.T @ B
    rhs = B.T @ q
    # Safe upper bound on ||B||_2^2 for projected-gradient steps.
    L = jnp.maximum(jnp.sum(B * B), jnp.asarray(1e-12, dtype=B.dtype))
    k = B.shape[1]
    zero = jnp.zeros((k,), dtype=B.dtype)

    def fit(tau, start):
        def step(carry, _):
            z, y, momentum = carry
            next_z = _jax_project_nonnegative_l1_ball(
                y - (gram @ y - rhs) / L, tau)
            next_momentum = (1 + jnp.sqrt(1 + 4 * momentum**2)) / 2
            next_y = next_z + (momentum - 1) / next_momentum * (next_z - z)
            return (next_z, next_y, next_momentum), None
        (z, _, _), _ = lax.scan(
            step, (start, start, jnp.asarray(1., dtype=B.dtype)),
            None, length=max_iter)
        residual = jnp.linalg.norm(B @ z - q)
        return z, residual

    # Construct a feasible radius from a nonnegative least-squares fit.
    # Projected gradient without a ball gives an approximate NNLS solution.
    def nnls_step(z, _):
        return jnp.maximum(z - (gram @ z - rhs) / L, 0), None
    z_upper, _ = lax.scan(nnls_step, zero, None, length=max_iter * 2)
    tau_upper = jnp.sum(z_upper)
    best, best_res = fit(tau_upper, z_upper)
    sigma_jax = jnp.asarray(sigma, dtype=B.dtype)

    def search(carry, _):
        lower, upper, best_z, best_r = carry
        mid = (lower + upper) / 2
        start = _jax_project_nonnegative_l1_ball(best_z, mid)
        z, r = fit(mid, start)
        feasible = r <= sigma_jax
        lower = jnp.where(feasible, lower, mid)
        upper = jnp.where(feasible, mid, upper)
        best_z = jnp.where(feasible, z, best_z)
        best_r = jnp.where(feasible, r, best_r)
        return (lower, upper, best_z, best_r), None

    (lower, upper, best, best_res), _ = lax.scan(
        search, (jnp.asarray(0., dtype=B.dtype), tau_upper, best, best_res),
        None, length=outer_iter)
    # If the origin already satisfies the residual tolerance, it is optimal.
    best = jnp.where(jnp.linalg.norm(q) <= sigma_jax, zero, best)
    return jnp.maximum(best / weights, 0)
