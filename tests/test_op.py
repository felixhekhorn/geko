import numpy as np
from eko.beta import beta_qcd
from eko.interpolation import XGrid
from eko.io.items import Evolution
from eko.kernels.non_singlet import lo_exact
from ekore.anomalous_dimensions.unpolarized.space_like import gamma_ns

from geko.op import _PID_NSP, compute_one, ns_as0_exact, ns_iterate, singlet_iterate


def test_eko_ns():
    a0 = 0.1
    a1 = 0.01
    nf = 4
    ev_op_iterations = 60
    beta0 = beta_qcd((2, 0), nf)
    as_iter = np.geomspace(a0, a1, 1 + ev_op_iterations)
    iter_distance = as_iter[1:] - as_iter[:-1]
    as_half = 0.5 * (as_iter[1:] + as_iter[:-1])
    for n in [1.0, 1.0 + 1j, 2.0]:
        gamma0 = gamma_ns((1, 0), _PID_NSP, n, nf, [0.0] * 7, True)[0]
        E = 1.0
        for aj, dist in zip(as_half, iter_distance):
            betaj = beta0 * aj**2
            E *= np.exp(gamma0 * aj / betaj * dist)
        np.testing.assert_allclose(
            E, lo_exact([gamma0], a1, a0, [beta0]), rtol=2e-4, err_msg=f"{n=}"
        )


def test_ns_iterate(monkeypatch):
    for n in [1.0, 1.0 + 1j, 2.0]:
        for nf in [4, 5]:
            # no evolution is a zero
            a0 = 0.1
            a1 = 0.1
            iter_sol = ns_iterate(n, a1, a0, nf, 1)
            np.testing.assert_allclose(iter_sol, 0.0, atol=1e-8)
            # Integral is linear in k
            a0 = 0.1
            a1 = 0.2
            monkeypatch.setattr("geko.anomalous_dimensions.ns", lambda *_args: [0.0])
            iter_sol = ns_iterate(n, a1, a0, nf, 1, 10)
            np.testing.assert_allclose(iter_sol, 0.0, atol=1e-8)
    # compare with analytic solution
    n = 1.0
    for nf in [4, 5]:
        a0 = 0.1
        a1 = 0.001
        monkeypatch.setattr("geko.anomalous_dimensions.ns", lambda *_args: [1.0])
        monkeypatch.setattr("eko.beta.beta_qcd", lambda *_args: 1.0)
        # monkeypatch.setattr(
        #     "ekore.anomalous_dimensions.unpolarized.space_like", lambda *_args: 0.0
        # )
        beta0 = beta_qcd((2, 0), nf)
        iter_sol = ns_iterate(n, a1, a0, nf, 1, 1)
        np.testing.assert_allclose(
            iter_sol,
            1.0 / beta0 * (1.0 / a0**2 + 1.0 / a1**2) * (a1 - a0) / 2.0,
            rtol=1e-6,
        )
        iter_sol = ns_iterate(n, a1, a0, nf, 1, 2)
        amid = 0.01
        np.testing.assert_allclose(
            iter_sol,
            0.5
            / beta0
            * (
                (1.0 / a0**2 + 1.0 / amid**2) * (amid - a0)
                + (1.0 / amid**2 + 1.0 / a1**2) * (a1 - amid)
            ),
            rtol=1e-3,
        )


def test_ns_iterate_compare():
    # compare with exact solution
    for n in [1.0, 1.0 + 1j, 2.0]:
        for nf in [4]:
            a0 = 0.1
            a1 = 0.01
            exact = ns_as0_exact(n, a1, a0, nf)
            iter_sol = ns_iterate(n, a1, a0, nf, 1, 70)
            np.testing.assert_allclose(iter_sol, exact, rtol=1e-3, err_msg=f"{n=}")


def test_shape():
    a0 = 0.1
    a1 = 0.01
    for order_qcd in [1, 2]:
        for n in [1.2, 1.0 + 1j, 2.0]:
            for nf in [4, 5]:
                ns = ns_iterate(n, a0, a1, nf, order_qcd, 1)
                assert isinstance(ns, complex)
                assert np.isfinite(ns)
                assert np.abs(ns) > 1e-5
                sng = singlet_iterate(n, a0, a1, nf, order_qcd, 1)
                assert sng.shape == (2,)
                assert (np.isfinite(sng)).all()
                assert (np.abs(sng) > 1e-5).all()


def test_compute_one(tmp_path, monkeypatch):
    x = XGrid([0.1, 0.5, 1.0], True)

    # check zero evolution
    class FakeConstCoupling:
        def a_s(self, *_args):
            return 0.1

    target = tmp_path / "zero.npy"
    compute_one(x, FakeConstCoupling(), Evolution(10, 20, 3), target, 1, 10)
    zero = np.load(target)
    np.testing.assert_allclose(zero, np.zeros_like(zero))
    # check non-zero evolution
    monkeypatch.setattr("geko.op.quad_ker", lambda *_args, **_kwargs: 1.0)

    class FakeNonConstCoupling:
        def a_s(self, q2, *_args):
            return 1.0 / q2

    target = tmp_path / "non-zero.npy"
    compute_one(x, FakeNonConstCoupling(), Evolution(10, 100, 6), target, 1, 10)
    ones = np.load(target)
    expected = [0.0] + [1.0 / 3.0, 1.0 / 12.0] * 3 + [2.5] + [1.0 / 12.0, 1.0 / 3.0] * 3
    np.testing.assert_allclose(
        ones, np.array(expected * 3).reshape(3, 14).T, rtol=2e-6, atol=5e-6
    )
