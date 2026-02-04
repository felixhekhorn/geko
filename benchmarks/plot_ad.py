"""Plot anomalous dimensions."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from eko.mellin import Path
from ekore.harmonics import cache as c
from scipy.integrate import quad
from scipy.special import spence

from geko.anomalous_dimensions import ns
from geko.op import _MELLIN_CUT, _MELLIN_EPS_ABS, _MELLIN_EPS_REL, _MELLIN_LIMIT

_XS = np.linspace(5e-2, 0.99, 30)


def integrand(u: float, x: float, order: int, nf: int, is_disg: bool) -> float:
    """Integration kernel."""
    is_singlet = False  # ns_mode is None
    path = Path(u, np.log(x), is_singlet)
    integrand = path.prefactor * x ** (-path.n) * path.jac
    cache = c.reset()
    if integrand == 0.0:
        return 0.0

    # if is_singlet:
    #     gamma = gamma_singlet((order + 1, 0), path.n, nf, n3lo_variation)
    #     if L != 0:
    #         gamma = gamma_variation(gamma, (order + 1, 0), nf, L)
    #     idx1, idx2 = map_singlet_entries[entry]
    #     gamma = gamma[order, idx1, idx2]
    # else:
    gamma = ns(order, path.n, nf, cache, is_disg)
    gamma = gamma[order - 1]

    # recombine everything
    return np.real(gamma * integrand)


def compute(pto: int, is_disg: bool) -> npt.NDArray[np.float64]:
    """Compute ad on grid."""
    res = []
    for x in _XS:
        args = (x, pto, 3, is_disg)
        # recover the minus in the definition of γ = - M[P]
        res.append(
            -quad(
                integrand,
                0.5,
                1.0 - _MELLIN_CUT * 0.001,
                args=args,
                epsabs=_MELLIN_EPS_ABS,
                epsrel=_MELLIN_EPS_REL,
                limit=_MELLIN_LIMIT,
            )[0]
        )
    return np.array(res)


def nlo_disg_summand(zz):
    """NLO DISg correction."""
    return (
        (
            7.0
            - 10.0 * zz
            - np.pi**2 / 6.0 * (6.0 - 12.0 * zz + 16.0 * zz**2)
            + (1.0 - 16.0 * zz + 32.0 * zz**2) * np.log(zz)
            + (1.0 - 2.0 * zz + 4.0 * zz**2) * (np.log(zz)) ** 2
            - (5.0 - 36.0 * zz + 32.0 * zz**2) * np.log(1.0 - zz)
            + (4.0 - 8.0 * zz + 8.0 * zz**2)
            * np.log(1.0 - zz)
            * (np.log(1.0 - zz) - np.log(zz))
            + (2.0 - 4.0 * zz + 8.0 * zz**2) * spence(1.0 - zz)
        )
        * 2.0
        * 4.0
        / 3.0
        * 2.0
    )


def nlo_msbar(zz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """NLO MSbar x-space.

    TODO: find reference.
    Taken from VG implementation."""
    return (
        (
            4.0
            - 9.0 * zz
            - (1.0 - 4.0 * zz) * np.log(zz)
            - (1.0 - 2.0 * zz) * (np.log(zz)) ** 2
            + 4.0 * np.log(1.0 - zz)
            + (
                4.0 * np.log(zz)
                - 4.0 * np.log(zz) * np.log(1.0 - zz)
                + 2.0 * (np.log(zz)) ** 2
                - 4.0 * np.log(1.0 - zz)
                + 2.0 * (np.log(1.0 - zz)) ** 2
                - 2.0 * np.pi**2 / 3.0
                + 10.0
            )
            * (zz**2 + (1.0 - zz) ** 2)
        )
        * 2.0
        * 4.0
        / 3.0
        * 2
    )


def lo(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """LO x-space.

    TODO: find reference"""
    return 4.0 * (z**2 + (1.0 - z) ** 2)


def ref(order: int, is_disg: bool) -> npt.NDArray[np.float64]:
    """Collect reference values."""
    if order == 1:
        ext = lo(_XS)
    elif order == 2:
        ext = nlo_msbar(_XS)
        if is_disg:
            ext -= nlo_disg_summand(_XS)

    else:
        raise ValueError(f"order unknown {order}")
    return ext


def plot(mf: npt.NDArray[np.float64], ext: npt.NDArray[np.float64], name: str) -> None:
    """Plot gEKO vs. reference."""
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(_XS, mf, label="gEKO")
    ax[0].plot(_XS, ext, label="X")
    ax[0].set_ylabel("k(z)")
    ax[1].plot(_XS, np.ones_like(_XS), color="black", linestyle="--")
    ax[1].plot(_XS, mf / ext)
    ax[1].set_xlabel("z")
    fig.legend()
    fig.savefig(name)
    # print(name)
    # print(_XS)
    # print(mf)
    # print(ext)


plot(compute(1, False), ref(1, False), "ns-LO.pdf")
plot(compute(2, False), ref(2, False), "ns-NLO.pdf")
plot(compute(2, True), ref(2, True), "ns-NLO-DISg.pdf")
# TODO compare NS + S + g
