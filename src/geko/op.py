"""Compute the point-like operators."""

import pathlib

import eko.basis_rotation as br
import numpy as np
from eko.beta import beta_qcd
from eko.constants import NC, ed2, eu2
from eko.couplings import Couplings
from eko.evolution_operator.flavors import pids_from_intrinsic_unified_evol
from eko.interpolation import XGrid
from eko.io.items import Evolution
from eko.mellin import Path
from ekore.anomalous_dimensions import exp_matrix_2D
from ekore.anomalous_dimensions.unpolarized.space_like import gamma_ns, gamma_singlet
from ekore.harmonics import cache as c
from scipy.integrate import quad

from . import anomalous_dimensions

_PID_NSP = br.non_singlet_pids_map["ns+"]
_PID_S = br.evol_basis_pids[br.evol_basis.index("S")]

_MELLIN_CUT = 1e-6
_MELLIN_EPS_ABS = 1e-14
_MELLIN_EPS_REL = 1e-6
_MELLIN_LIMIT = 100


def ns_as0_exact(n: complex, a1: float, a0: float, nf: int) -> complex:
    """|LO| non-singlet exact solution."""
    cache = c.reset()
    gamma0: complex = gamma_ns((1, 0), _PID_NSP, n, nf, [0.0] * 7, True)[0]
    beta0 = beta_qcd((2, 0), nf)
    ker = (
        anomalous_dimensions.ns_as0(n, nf, cache)
        / (gamma0 + beta0)
        * (np.exp(gamma0 * np.log(a1 / a0) / beta0) / a0 - 1.0 / a1)
    )
    return complex(ker)


def singlet_as0_exact(n: complex, a1: float, a0: float, nf: int) -> np.ndarray:
    """|LO| singlet exact solution."""
    cache = c.reset()
    gamma0 = gamma_singlet((1, 0), n, nf, [0.0] * 7, True)[0]
    beta0 = beta_qcd((2, 0), nf)
    _exp, lambda_p, lambda_m, e_p, e_m = exp_matrix_2D(gamma0)
    tot = np.zeros((2, 2), np.complex128)
    for lam, e in ((lambda_m, e_m), (lambda_p, e_p)):
        tot += (
            (
                1.0
                / (lam + beta0)
                * (np.exp(lam * np.log(a1 / a0) / beta0) / a0 - 1.0 / a1)
            )
            * e
            @ anomalous_dimensions.singlet_as0(n, nf, cache)
        )
    return tot


def ns_iterate(
    n: complex,
    a1: float,
    a0: float,
    nf: int,
    order_qcd: int,
    ev_op_iterations: int = 30,
) -> complex:
    """Non-singlet iterative solution."""
    cache = c.reset()
    # gamma is a list of gamma coefficients! (similar for the others)
    gamma = gamma_ns((order_qcd, 0), _PID_NSP, n, nf, [0.0] * 7, True)
    beta = [beta_qcd((2, 0), nf)]
    if order_qcd >= 2:
        beta += [beta_qcd((3, 0), nf)]
    beta = np.array(beta)
    k = anomalous_dimensions.ns(order_qcd, n, nf, cache)
    as_iter = np.geomspace(a0, a1, 1 + ev_op_iterations)
    iter_distance = as_iter[1:] - as_iter[:-1]
    as_half = 0.5 * (as_iter[1:] + as_iter[:-1])
    # to turn a list of coefficients into an actual polynomial, we can write it as
    # ordinary scalar product with a list containing variable powers. (here variable = coupling)
    # `order_qcd` is in EKO language, i.e. it is counting powers of the QCD splitting kernels, i.e.
    # 1 = as^1 = |LO|. However, `range` start from 0 up to, but not including the last - so we need to correct twice.
    pows = range(order_qcd - 1 + 1)
    # step 0 is separated
    as_half0_pow = np.power(as_half[0], pows)
    as0_pow = np.power(a0, pows)
    # in this way f(x) = c_0 * x^0 + c_1 * x^1
    #                  = {c_0, c_1} @ {x^0, x^1}
    #                  = c @ x
    # For the EKO: Keep in mind that beta starts at as^2, which can be dealt with as global pre-factor.
    # Same holds for gamma with as^1, so eventually a global division by one as remains
    # For the remaining kernel: k starts at as^0, so no global factor, i.e. we only account for beta.
    ker = (
        np.exp(
            (gamma @ as_half0_pow)
            / (beta @ as_half0_pow)
            / as_half[0]
            * iter_distance[0]
        )
        * (k @ as0_pow)
        / (beta @ as0_pow)
        / a0**2
        * iter_distance[0]
    )
    # now iterate, but the last
    for j in range(1, ev_op_iterations):
        as_half_pow = np.power(as_half[j], pows)
        E = np.exp(
            (gamma @ as_half_pow) / (beta @ as_half_pow) / as_half[j] * iter_distance[j]
        )
        aj_pow = np.power(as_iter[j], pows)
        ker = E * (
            ker
            + (k @ aj_pow)
            / (beta @ aj_pow)
            / as_iter[j] ** 2
            * (iter_distance[j - 1] + iter_distance[j])
        )
    # and finally do the last step
    a1_pow = np.power(a1, pows)
    ker += (k @ a1_pow) / (beta @ a1_pow) / a1**2 * iter_distance[-1]
    return complex(0.5 * ker)


def singlet_iterate(
    n: complex,
    a1: float,
    a0: float,
    nf: int,
    order_qcd: int,
    ev_op_iterations: int = 30,
) -> np.ndarray:
    """Singlet iterative solution."""
    cache = c.reset()
    gamma = gamma_singlet((order_qcd, 0), n, nf, [0.0] * 7, True)
    beta = [beta_qcd((2, 0), nf)]
    if order_qcd >= 2:
        beta += [beta_qcd((3, 0), nf)]
    beta = np.array(beta)
    k = anomalous_dimensions.singlet(order_qcd, n, nf, cache)
    as_iter = np.geomspace(a0, a1, 1 + ev_op_iterations)
    iter_distance = as_iter[1:] - as_iter[:-1]
    as_half = 0.5 * (as_iter[1:] + as_iter[:-1])
    # prepare powers again - see ns
    pows = range(order_qcd - 1 + 1)
    # step 0 is separated
    as_half0_pow = np.power(as_half[0], pows)
    as0_pow = np.power(a0, pows)
    ker = (
        exp_matrix_2D(
            np.dot(gamma.T, as_half0_pow).T
            / (beta @ as_half0_pow)
            / as_half[0]
            * iter_distance[0]
        )[0]
        @ np.dot(k.T, as0_pow).T
        / (beta @ as0_pow)
        * iter_distance[0]
    )
    # now iterate, but the last
    for j in range(1, ev_op_iterations):
        as_half_pow = np.power(as_half[j], pows)
        E = exp_matrix_2D(
            np.dot(gamma.T, as_half_pow).T
            / (beta @ as_half_pow)
            / as_half[j]
            * iter_distance[j]
        )[0]
        aj_pow = np.power(as_iter[j], pows)
        ker = E @ (
            ker
            + np.dot(k.T, aj_pow).T
            / (beta @ aj_pow)
            / as_iter[j] ** 2
            * (iter_distance[j - 1] + iter_distance[j])
        )
    # and finally do the last step
    a1_pow = np.power(a1, pows)
    ker += np.dot(k.T, a1_pow).T / (beta @ a1_pow) / a1**2 * iter_distance[-1]
    return 0.5 * ker


def quad_ker(
    t: float,
    mode: int,
    logx: float,
    a1: float,
    a0: float,
    nf: int,
    order_qcd: int,
    ev_op_iterations: int,
) -> float:
    """Integration kernel."""
    p = Path(t, logx, False if mode == _PID_NSP else True)
    ker = 0j
    if mode == _PID_NSP:
        # ker = ns_as0_exact(p.n, a1, a0, nf)
        ker = ns_iterate(p.n, a1, a0, nf, order_qcd, ev_op_iterations)

    else:
        # vker = singlet_as0_exact(p.n, a1, a0, nf)
        vker = singlet_iterate(p.n, a1, a0, nf, order_qcd, ev_op_iterations)
        if mode == 21:
            ker = vker[1]
        else:
            ker = vker[0]
    return float(np.real(p.prefactor * p.jac * ker * np.exp(-logx * p.n)))


def blowup(op: dict[int, np.ndarray], nf: int) -> np.ndarray:
    """Convert anomalous dimension basis to flavor basis."""
    nfu = nf // 2
    nfd = nf - nfu
    e2tot = nfu * eu2 + nfd * ed2
    # prepare rotation
    flavour_to_intrinsic_unified_evol = np.zeros(
        (len(br.flavor_basis_pids), len(br.flavor_basis_pids))
    )
    basis = ["S", "g", "Sdelta", "ph", "V", "Vdelta", "Td3", "Vd3", "t+", "t-"]
    if nf == 3:
        basis += ["c+", "c-", "b+", "b-"]
    elif nf == 4:
        basis += ["Tu3", "Vu3", "b+", "b-"]
    elif nf == 5:
        basis += ["Tu3", "Vu3", "Td8", "Vd8"]
    for idx, lab in enumerate(basis):
        flavour_to_intrinsic_unified_evol[idx] = pids_from_intrinsic_unified_evol(
            lab, nf, False
        )
    # map raw elements to basis vectors
    ev = np.zeros((len(br.flavor_basis_pids), len(op[21])))
    # The singlet coupling is Eq. 2.29 of :cite:`Moch:2001im`,
    ev[basis.index("S")] = NC * e2tot * op[_PID_S]
    ev[basis.index("g")] = NC * e2tot * op[21]
    # but the non-singlet coupling is given by the quark contributions inside Sdelta
    ev[basis.index("Sdelta")] = NC * nfd * (eu2 - ed2) * op[_PID_NSP]
    res = np.linalg.inv(flavour_to_intrinsic_unified_evol) @ ev
    return res


def compute_one(
    xgrid: XGrid,
    sc: Couplings,
    seg: Evolution,
    path: pathlib.Path,
    order_qcd: int,
    ev_op_iterations: int,
) -> None:
    """Compute operator for the given evolution patch."""
    # iterate sectors
    op = {}
    a1 = sc.a_s(seg.target, seg.nf)
    a0 = sc.a_s(seg.origin, seg.nf)
    for lab in (_PID_NSP, 21, _PID_S):
        # iterate all points
        vec = []
        for x in xgrid.raw:
            # do the inversion
            args = (lab, np.log(x), a1, a0, seg.nf, order_qcd, ev_op_iterations)
            res = quad(
                quad_ker,
                0.5,
                1.0 - _MELLIN_CUT,
                args=args,
                epsabs=_MELLIN_EPS_ABS,
                epsrel=_MELLIN_EPS_REL,
                limit=_MELLIN_LIMIT,
            )
            vec.append(res[0])
        op[lab] = np.array(vec)
    # blow up
    out = blowup(op, seg.nf)
    # save
    np.save(path, out)
