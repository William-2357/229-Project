"""Two-stage convex transfer for the calibration head (relaxed-harness research arc).

Implements the integrated convex transfer-learning idea (see research/journal.md
"CONVEX-IN-PRETRAINING PIVOT"): in a convex model "pretrain-then-finetune" via
initialization is meaningless (the solution is determined by data+regularizer+dictionary,
not the optimization path). Source knowledge can therefore enter the target solve ONLY
through (1) the gate DICTIONARY and (2) the REGULARIZER. This module adds channel (2): an
anchored ADMM whose v-subproblem is biased toward a source-pretrained convex head v_bar.

Stage 1 (convex pre-train): solve the convex ReLU head on source features with a FIXED gate
  dictionary G -> v_bar (per-class convex weights). This is the transferable object.
Stage 2 (anchored calibrate): on the SAME gates, re-solve on the target calibration set with
  an added quadratic anchor (a/2)||v - v_bar||^2. The anchor both transfers source structure
  and regularizes the otherwise-underdetermined low-K solve (the iter-6 failure mode).

The anchored v-update stays a closed-form group-L2 prox:
    v_i = prox_{beta/(rho+a) ||.||2}( (rho*(u_i+lam_i) + a*v_bar_i) / (rho+a) )
so a=0 recovers stock jaxcld ADMM exactly. Everything else (u-update PCG, cone slack s,
duals) is identical to jaxcld.optimizers.admm.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrn

from jax import vmap

from jaxcld.models.cvx_relu_mlp import CVX_ReLU_MLP
from jaxcld.optimizers.pcg import pcg
from jaxcld.preconditioner.nystrom import Nys_Precond, rand_nys_appx


def _proxl2_cols(z, thresh):
    """Group-L2 prox over columns with a PER-COLUMN threshold (thresh scalar or shape (P,)).
    z: (d, P). Shrinks column i by thresh_i. Generalizes jaxcld.proxl2_tensor (scalar thresh)."""
    norms = jnp.linalg.norm(z, axis=0)                      # (P,)
    factor = 1.0 - thresh / jnp.maximum(thresh, norms)      # (P,)
    return z * factor[None, :]


def _batch_proxl2_cols(z, thresh):
    return vmap(lambda Z: _proxl2_cols(Z, thresh))(z)       # vmap over class axis


def sample_gates(d: int, P: int, key) -> tuple[jnp.ndarray, "jax.Array"]:
    """Sample P random hyperplane gates in R^d, mirroring jaxcld.get_hyperplane_cuts.

    Returns (G, next_key). Gates are data-independent N(0, I_d) columns; the shared G IS the
    transferable dictionary (the same hyperplanes index source and target neurons).
    """
    key, subkey = jrn.split(key)
    G = jrn.normal(subkey, (d, P))
    return G, key


def build_fixed_gate_model(Xnorm, y, n_classes, P_S, beta, rho, key, G):
    """CVX_ReLU_MLP whose activation patterns use the SHARED gates G (not freshly sampled)."""
    cld = CVX_ReLU_MLP(X=jnp.asarray(Xnorm), y=jnp.asarray(y.astype("int32")),
                       n_classes=n_classes, P_S=P_S, beta=beta, rho=rho, seed=key)
    d_diags = (cld.X @ G >= 0).astype(jnp.float32)
    cld.d_diags = d_diags
    cld.e_diags = 2 * d_diags - 1
    return cld


def anchored_admm(model, admm_params, v_anchor=None, anchor_a=0.0):
    """jaxcld ADMM with an optional quadratic anchor (a/2)||v - v_anchor||^2 on the v-block.

    Mirrors jaxcld.optimizers.admm.admm exactly except the v-update blends toward v_anchor.
    anchor_a may be a scalar (isotropic anchor) OR a per-pattern array of shape (P,) — the
    adaptive/Mahalanobis-spirit anchor where conserved neurons (small cross-subject variance)
    are held strongly and variable neurons are free to fit the target. Sets model.theta1/theta2
    (the non-convex weights used by stacked_predict) and returns v.
    """
    rank = admm_params['rank']
    beta = admm_params['beta']
    gamma_ratio = admm_params['gamma_ratio']
    admm_iters = admm_params['admm_iters']
    pcg_iters = admm_params['pcg_iters']

    n, d = model.X.shape
    rho = model.rho
    C, P = model.n_classes, model.P_S
    Y = jax.nn.one_hot(model.y, C)

    u = jnp.zeros((C, 2, d, P))
    v = jnp.zeros((C, 2, d, P))
    s = jnp.zeros((C, 2, n, P))
    lam = jnp.zeros((C, 2, d, P))
    nu = jnp.zeros((C, 2, n, P))

    a = jnp.asarray(anchor_a, dtype=jnp.float32)            # scalar or (P,)
    off = bool(jnp.all(a == 0.0)) or v_anchor is None
    if off:
        v_anchor = jnp.zeros((C, 2, d, P))
        a = jnp.zeros((P,), dtype=jnp.float32)
    thresh = beta / (rho + a)                               # per-column prox threshold, (P,) or scalar

    U, S, model.seed = rand_nys_appx(model, rank, model.seed)
    Mnys = Nys_Precond(U, S, d, rho, P)
    b_1 = model.batch_rmatvec_F(Y.T) / rho

    for _ in range(admm_iters):
        b = b_1 + v - lam + model.batch_rmatvec_G(s - nu)
        u, _, _ = pcg(b, model, Mnys, pcg_iters)

        # anchored v-update: blend ADMM point (u+lam) with the (per-pattern) anchor, then group-prox
        q0 = (rho * (u[:, 0, :] + lam[:, 0, :]) + a * v_anchor[:, 0, :]) / (rho + a)
        q1 = (rho * (u[:, 1, :] + lam[:, 1, :]) + a * v_anchor[:, 1, :]) / (rho + a)
        v = v.at[:, 0, :].set(_batch_proxl2_cols(q0, thresh))
        v = v.at[:, 1, :].set(_batch_proxl2_cols(q1, thresh))

        Gu = model.batch_matvec_G(u)
        s = jax.nn.relu(Gu + nu)
        lam = lam + (u - v) * gamma_ratio
        nu = nu + (Gu - s) * gamma_ratio

    model.u, model.v, model.s = u, v, s
    W1, w2 = model.get_ncvx_weights(v)
    model.theta1, model.theta2 = W1, w2
    return v
