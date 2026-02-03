"""Plot anomalous dimensions."""

import matplotlib.pyplot as plt
import numpy as np
from eko.mellin import Path
from ekore.harmonics import cache as c
from scipy.integrate import quad

from geko.anomalous_dimensions import ns
from geko.op import _MELLIN_CUT, _MELLIN_EPS_ABS, _MELLIN_EPS_REL, _MELLIN_LIMIT

_XS = np.linspace(1e-2, 0.99, 30)


def integrand(u, x, order, nf):
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
    gamma = ns(order, path.n, nf, cache, False)
    gamma = gamma[order - 1]

    # recombine everything
    return np.real(gamma * integrand)


def compute():
    res = []
    for x in _XS:
        args = (x, 2, 3)
        # recover the minus in the definition of γ = - M[P]
        res.append(
            -quad(
                integrand,
                0.5,
                1.0 - _MELLIN_CUT,
                args=args,
                epsabs=_MELLIN_EPS_ABS,
                epsrel=_MELLIN_EPS_REL,
                limit=_MELLIN_LIMIT,
            )[0]
        )
    return res


# def vadim_nlo(zz):
#     return (7.0-10.0*zz-np.pi**2/6.*(6.0-12.0*zz+16.0*zz**2)
#      +(1.0-16.0*zz+32.0*zz**2)*np.log(zz)
#      +(1.0-2.0*zz+4.0*zz**2)*(np.log(zz))**2
#      -(5.0-36.0*zz+32.0*zz**2)*np.log(1.0-zz)
#      +(4.0-8.0*zz+8.0*zz**2)*np.log(1.0-zz)*(np.log(1.0-zz)-np.log(zz))
#      +(2.0-4.0*zz+8.0*zz**2)*spence(1.-zz))*zz


def vadim_nlo_msbar(zz):
    return (
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


# TODO make this a proper thing
# TODO compare NS + S + g
# TODO compare LO + NLO@MSbar + NLO@DISg
plt.plot(_XS, compute(), label="EKO")
# plt.plot(_XS, 4*(_XS**2 + (1.-_XS)**2))
plt.plot(_XS, 2.0 * np.pi * vadim_nlo_msbar(_XS), label="X")
plt.legend()
plt.savefig("ns.pdf")
