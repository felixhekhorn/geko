import numpy as np
from eko.constants import CF
from ekore.harmonics import cache as c

from geko.anomalous_dimensions import dis_gamma_coeff_as0, ns, singlet


def test_compare_to_mvv():  # compares to MVV, Appendix A
    ref_q = np.array(
        [
            [-4.0 / 3.0, CF * (-148.0 / 27.0)],
            [-11.0 / 15.0, CF * (-56317.0 / 9000.0)],
            [-11.0 / 21.0, CF * (-296083.0 / 46305.0)],
        ]
    )
    ref_gluon = np.array(
        [
            [0.0, CF * (40.0 / 27.0)],
            [0.0, CF * (2951.0 / 4500.0)],
            [0.0, CF * (15418.0 / 46305.0)],
        ]
    )
    ref_coeff = np.array([-1.0, -133 / 90.0, -1777.0 / 1260.0])
    for j, (q, gluon, cgamma1) in enumerate(zip(ref_q, ref_gluon, ref_coeff)):
        n = 2.0 * (j + 1)
        cache = c.reset()
        np.testing.assert_allclose(
            cgamma1, dis_gamma_coeff_as0(n, 3, cache), err_msg=f"cgamma1: {n=}"
        )
        np.testing.assert_allclose(q, ns(2, n, 3, cache, False), err_msg=f"ns: {n=}")
        np.testing.assert_allclose(
            np.array([q, gluon]).T,
            singlet(2, n, 3, cache, False),
            err_msg=f"singlet: {n=}",
        )


def test_shape():
    for order_qcd in [1, 2]:
        for n in [1.2, 1.0 + 1j, 2.0]:
            cache = c.reset()
            for nf in [4, 5]:
                assert ns(order_qcd, n, nf, cache).shape == (order_qcd,)
                assert singlet(order_qcd, n, nf, cache).shape == (order_qcd, 2)
