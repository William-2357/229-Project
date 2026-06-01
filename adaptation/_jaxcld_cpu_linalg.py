"""Pin jaxcld's Nyström-preconditioner dense decompositions to CPU.

On Modal A10G GPUs, cuSolver raises `XlaRuntimeError: INTERNAL: cuSolver
internal error` when jaxcld builds its Nyström preconditioner. The culprit is
`jaxcld.preconditioner.nystrom.rand_nys_appx`, which runs four cuSolver
routines on the GPU: `qr`, `cholesky`, `solve_triangular`, and `svd`. Every CLD
method funnels through this:

  - the library `jaxcld.optimizers.admm.admm()`  (cld / ea_cld / foundation_cld
    / foundation_sft_cld), and
  - our inline `_admm_warm()` in `foundation_sft_anchored_cld.py`.

These decompositions act on tiny matrices (features are PCA-reduced to ~63
dims; `Core` is rank x rank and the SVD is economy), so running them on CPU is
negligible in cost and completely avoids the cuSolver bug. The expensive sketch
matvecs (`model.matvec_A`) and all downstream PCG/ADMM matmuls stay on the GPU —
those use cuBLAS, which is unaffected.

Importing this module monkey-patches `rand_nys_appx` in both the preconditioner
module and `optimizers.admm` (which bound the name at its own import time).
`foundation_sft_anchored_cld.py` imports `rand_nys_appx_cpu` directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import lax
from jax.scipy.linalg import solve_triangular

from jaxcld.utils.linops_utils import tensor_to_vec, vec_to_tensor

_CPU = jax.devices("cpu")[0]


def rand_nys_appx_cpu(model, rank: int, key):
    """`rand_nys_appx` with all cuSolver ops (qr/cholesky/solve/svd) on CPU.

    Numerically identical to the upstream implementation; only the device
    placement of the dense decompositions differs. The random sketch via
    `model.matvec_A` stays on the default (GPU) device.
    """
    d = model.X.shape[1]
    N = 2 * model.P_S * d
    key, subkey = jrn.split(key)

    # QR (cuSolver geqrf/orgqr) on CPU, then move the orthonormal test matrix
    # back to the default device for the GPU sketch.
    with jax.default_device(_CPU):
        Omega = jrn.normal(subkey, (N, rank))
        Omega = jnp.linalg.qr(Omega)[0]
    Omega = jax.device_put(Omega)

    def compute_sketch(col):
        col_tensor = vec_to_tensor(col, d, model.P_S)
        col_A = model.matvec_A(col_tensor)
        return tensor_to_vec(col_A)

    Y = jax.vmap(compute_sketch)(Omega.T).T  # GPU sketch

    v = jnp.sqrt(rank) * 10 ** -16 * (jnp.linalg.norm(Y))
    Y = Y + v * Omega  # Add shift
    Core = Omega.T @ Y

    # Cholesky + triangular solve + SVD (all cuSolver) on CPU.
    with jax.default_device(_CPU):
        Core = jax.device_put(Core, _CPU)
        Y = jax.device_put(Y, _CPU)
        v = jax.device_put(v, _CPU)
        C = jnp.linalg.cholesky(Core)
        B = solve_triangular(C, Y.T, lower=True)
        U, S, _ = lax.linalg.svd(B.T, full_matrices=False)
        S = jax.nn.relu(S ** 2 - v)  # Subtract off shift

    return U, S, key


def install() -> None:
    """Replace `rand_nys_appx` everywhere jaxcld bound it. Idempotent."""
    import jaxcld.preconditioner.nystrom as _nys
    import jaxcld.optimizers.admm as _admm

    _nys.rand_nys_appx = rand_nys_appx_cpu
    _admm.rand_nys_appx = rand_nys_appx_cpu


# Patch on import so any module that imports this gets the fix applied.
install()
