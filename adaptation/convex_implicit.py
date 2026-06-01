"""Differentiable (unrolled) solver for the convex two-layer-ReLU reformulation (equation 2 of
the team ADMM paper / Pilanci-Ergen), in PyTorch, so the convex head's OPTIMAL solution can be
backpropped into the backbone — coupling the representation to the convex classifier through the
KKT/optimality conditions (vs iter-19/20 which froze the head -> no dV/dZ).

Equation 2 (per class c), fixed activation patterns D_i = diag(1[Z g_i >= 0]):
    min_{v_i,w_i}  1/2 || sum_i D_i Z (v_i - w_i) - y ||^2 + beta sum_i (||v_i|| + ||w_i||)
    s.t.  (2D_i - I) Z v_i >= 0,  (2D_i - I) Z w_i >= 0.

Because diffcp/cvxpylayers are unavailable, we differentiate by UNROLLING proximal-gradient steps
warm-started at the optimum V0 (found by the fast jax ADMM, detached). Starting AT the fixed point,
unrolling is the Neumann-series approximation of the exact implicit-function-theorem gradient of the
argmin, so dV/dZ (hence dloss/dbackbone) flows correctly. The hard cone is handled as a quadratic
penalty (lam_cone); warm-started at the hard-cone optimum it stays ~feasible. Pattern indicators
1[Zg>=0] are piecewise-constant -> detached (standard for these reformulations).
"""

from __future__ import annotations
import torch


def group_prox(V, thresh):
    """Group-L2 prox per neuron-column. V: (2, d, P); shrink each column V[b,:,i] (norm over d)."""
    norms = V.norm(dim=1, keepdim=True)                       # (2,1,P)
    return V * torch.clamp(1.0 - thresh / norms.clamp_min(1e-12), min=0.0)


def _Fpred(Z, D, Vc):
    """sum_i D_i Z (v_i - w_i)  ->  (n,). Vc: (2,d,P), Z:(n,d), D:(n,P). Matches jaxcld matvec_F."""
    diff = Vc[0] - Vc[1]                                       # (d,P)
    return (D * (Z @ diff)).sum(dim=1)                         # (n,)


def _cone_pen(Z, D, Vc):
    """Quadratic penalty for the cone constraints (2D-I) Z v >= 0 (and for w)."""
    e = 2.0 * D - 1.0                                          # (n,P) in {-1,+1}
    vv = torch.relu(-(e * (Z @ Vc[0])))
    ww = torch.relu(-(e * (Z @ Vc[1])))
    return (vv ** 2).sum() + (ww ** 2).sum()


def _g_obj(Z, D, Vc, y, lam_cone):
    return 0.5 * ((_Fpred(Z, D, Vc) - y) ** 2).sum() + 0.5 * lam_cone * _cone_pen(Z, D, Vc)


def solve_unrolled(Z, D, Yoh, V0, beta, gamma, lam_cone, T):
    """Unrolled prox-gradient on eq.2 per class, warm-started at V0 (detached optimum).
    Z:(n,d) WITH grad, D:(n,P) detached, Yoh:(n,C), V0:(C,2,d,P) detached. Returns V:(C,2,d,P)
    differentiable in Z (Neumann approx of dV*/dZ)."""
    thr = beta * gamma
    Vs = []
    for c in range(Yoh.shape[1]):
        # warm-start leaf (its grad is ignored; Z's gradient flows through the unrolled steps —
        # the Neumann-from-fixed-point approximation of the implicit dV*/dZ).
        V = V0[c].detach().clone().requires_grad_(True)
        y = Yoh[:, c]
        for _ in range(T):
            grad = torch.autograd.grad(_g_obj(Z, D, V, y, lam_cone), V, create_graph=True)[0]
            V = group_prox(V - gamma * grad, thr)
        Vs.append(V)
    return torch.stack(Vs)                                    # (C,2,d,P)


def eq2_logits(Z, D, V):
    """Convex-head prediction F(V) per class -> (n, C). On the head's training features this equals
    the reconstructed ReLU net relu(Z@W1)@W2 (the convex reformulation's defining identity)."""
    return torch.stack([_Fpred(Z, D, V[c]) for c in range(V.shape[0])], dim=1)
