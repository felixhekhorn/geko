r"""Point-like anomalous dimensions.

Note that we adopt the normalization of [Moch:2001im], in particular we implement

.. math::
    \delta_p^{-1} k_p

This yields additional factors with respect to [Gluck:1991ee]:

- one (-1) for the Mellin definition of the anomalous dimension, see [Gluck:1991ee](Eq. 2.3)
- one 2 for either the a_em and/or a_s definition

The NC is factored both in [Moch:2001im] and [Gluck:1991ee].

References
----------
- [Gluck:1991ee] https://inspirehep.net/literature/321270
- [Moch:2001im] https://inspirehep.net/literature/564916
"""

import numpy as np
from eko.constants import CF
from ekore.anomalous_dimensions.unpolarized.space_like.as1 import gamma_gq, gamma_ns
from ekore.harmonics import cache as c


def ns_as0(n: complex, _nf: int, _cache: np.ndarray, _is_disg: bool = True) -> complex:
    """LO non-singlet contribution.

    Implements Eq. (2.9) of [Gluck:1991ee].
    """
    return -2.0 * 2.0 * (n**2 + n + 2.0) / (n * (n + 1.0) * (n + 2.0))


def ns_as1_msbar(n: complex, nf: int, cache: np.ndarray) -> complex:
    """NLO non-singlet contribution.

    Implements Eq. (2.9) of [Gluck:1991ee].
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    k0 = ns_as0(n, nf, cache)
    res = (
        (S1**2 - S2 + (5.0 / 2.0)) * k0 * (-0.5)
        - 4.0 / (n**2) * S1
        + (11.0 * n**4 + 26.0 * n**3 + 15.0 * n**2 + 8.0 * n + 4.0)
        / (n**3 * (n + 1.0) ** 3 * (n + 2.0))
    )
    return res * CF * (-4.0)


def ns_as1(n: complex, nf: int, cache: np.ndarray, is_disg: bool = True) -> complex:
    """NLO non-singlet contribution.

    Shift to DISγ using Eq. (4.20) of [Moch:2001im].
    """
    res = ns_as1_msbar(n, nf, cache)
    if is_disg:
        res -= gamma_ns(n, cache) * dis_gamma_coeff_as0(n, nf, cache)
    return res


def ns(
    order_qcd: int, n: complex, nf: int, cache: np.ndarray, is_disg: bool = True
) -> np.ndarray:
    """Tower of non-singlet contributions."""
    res = [ns_as0(n, nf, cache, is_disg)]
    if order_qcd >= 2:
        res += [ns_as1(n, nf, cache, is_disg)]
    return np.array(res)


def singlet_as0(
    n: complex, nf: int, cache: np.ndarray, is_disg: bool = True
) -> np.ndarray:
    """LO non-singlet contribution.

    Implements Eq. (2.6) and (2.9) of [Gluck:1991ee].
    """
    return np.array([ns_as0(n, nf, cache, is_disg), 0.0], dtype=np.complex_)


def gluon_as1_msbar(n: complex, _nf: int, _cache: np.ndarray) -> complex:
    """NLO gluon contribution.

    Implements Eq. (2.10) of [Gluck:1991ee]."""
    num = 2 * n**6 + 4 * n**5 + n**4 - 10 * n**3 - 5 * n**2 - 4 * n - 4
    den = (n - 1) * n**3 * (n + 1) ** 3 * (n + 2)
    return CF * -2 * -4 * num / den


def singlet_as1(
    n: complex, nf: int, cache: np.ndarray, is_disg: bool = True
) -> np.ndarray:
    """NLO singlet contribution.

    Implements Eq. (2.6) and (2.9) of [Gluck:1991ee].
    Shift to DISγ using Eq. (4.20) of [Moch:2001im].
    """
    # at NLO the quark sector is still the same
    sing = ns_as1(n, nf, cache, is_disg)
    glue = gluon_as1_msbar(n, nf, cache)
    if is_disg:
        glue -= gamma_gq(n) * dis_gamma_coeff_as0(n, nf, cache)
    return np.array([sing, glue], dtype=np.complex_)


def singlet(
    order_qcd: int, n: complex, nf: int, cache: np.ndarray, is_disg: bool = True
) -> np.ndarray:
    """Tower of singlet contributions."""
    res = [singlet_as0(n, nf, cache, is_disg)]
    if order_qcd >= 2:
        res += [singlet_as1(n, nf, cache, is_disg)]
    return np.array(res)


def dis_gamma_coeff_as0(n: complex, _nf: int, cache: np.ndarray) -> complex:
    """LO DISγ F2 contribution.

    Implements Eq. (5.8) of [Moch:2001im], in Mellin space.
    """
    S1 = c.get(c.S1, cache, n)
    c2g = (
        4.0
        * (2.0 + n + 4.0 * n**2 - n**3 - n * (2.0 + n + n**2) * S1)
        / (n**2 * (n + 1) * (n + 2))
    )
    return c2g
