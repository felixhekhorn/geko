"""Execute benchmarks.

References
----------
- [Gluck:1991jc] https://inspirehep.net/literature/322191
- [VG] photon_pdfs_v7.pdf

"""

import argparse
import pathlib
from collections.abc import Callable

import eko.basis_rotation as br
import grvphoton
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eko.beta import beta_qcd
from eko.interpolation import InterpolatorDispatcher, XGrid
from eko.io.types import EvolutionPoint
from run import EKODIR, GEKODIR, operator_base

import geko

# Setup paths
_HERE = pathlib.Path(__file__).parent
_COMPAREDIR = _HERE / "compare"
_PLOTSDIR = _HERE / "plots"
_QCDNUMDIR = _HERE / "qcdnum_ref"

_LABEL_GRV = "GRV"
_LABEL_QCDNUM = "QCDNUM"


# check alpha_s
def a_s_grv(q2: float, nf: int, lambda2: float, pto: int) -> float:
    """Evaluate strong coupling using GRV prescription.

    Implements Eq. (8) of [Gluck:1991jc]."""
    lg = np.log(q2 / lambda2)
    # LO
    beta0 = beta_qcd((2, 0), nf)
    a_s = 1.0 / (beta0 * lg)
    if pto > 0:  # NLO
        a_s -= beta_qcd((3, 0), nf) / beta0**3 * np.log(lg) / lg**2
    return a_s


class GRV:
    """LHAPDF-like wrapper for GRV."""

    pto: int
    """Perturbative order."""

    def __init__(self, pto: int):
        self.pto = pto

    def hasFlavor(self, pid: int) -> bool:
        return abs(pid) < 6 or pid == 21

    def xfxQ2(self, pid, x, Q2):
        # order is: UL, DL, SL, CL, BL, GL
        idx = 5  # = g = 21
        apid = abs(pid)
        if apid == 2:  # u
            idx = 0
        elif apid == 1:  # d
            idx = 1
        elif apid in [3, 4, 5]:  # s-b
            idx = apid - 1
        elif apid == 6:  # t
            return 0.0
        if self.pto == 1:
            return grvphoton.grvgho(x, Q2)[idx]
        return grvphoton.grvglo(x, Q2)[idx]


def pid_weights(pid: str) -> np.ndarray:
    """Cast pid to flavor projection."""
    w = np.zeros_like(br.flavor_basis_pids)
    try:
        ipid = int(pid)
        if ipid in br.flavor_basis_pids:
            w[br.flavor_basis_pids.index(ipid)] = 1
        elif ipid in br.evol_basis_pids:
            w = br.rotate_flavor_to_evolution[br.evol_basis_pids.index(ipid)]
        elif ipid in br.unified_evol_basis_pids:
            w = br.rotate_flavor_to_unified_evolution[
                br.unified_evol_basis_pids.index(ipid)
            ]
    except ValueError:
        if pid in br.flavor_basis_names:
            w[br.flavor_basis_names.index(pid)] = 1
        elif pid in br.evol_basis:
            w = br.rotate_flavor_to_evolution[br.evol_basis.index(pid)]
        elif pid in br.unified_evol_basis:
            w = br.rotate_flavor_to_unified_evolution[br.unified_evol_basis.index(pid)]
    return w


_DF_VS_X_TYPE = Callable[[int, dict, EvolutionPoint, str], pd.DataFrame]


def df_vs_grv(pto: int, evolved: dict, ep: EvolutionPoint, pid: str) -> pd.DataFrame:
    """Compare EKO vs. GRV."""
    weights = pid_weights(pid)
    res = weights @ np.array([evolved[ep]["pdfs"][pid] for pid in br.flavor_basis_pids])
    df = pd.DataFrame()
    xgrid = operator_base["xgrid"]
    df["x"] = xgrid
    df["eko"] = res * xgrid
    ref = GRV(pto)
    df[_LABEL_GRV] = [
        weights @ np.array([ref.xfxQ2(pid, x, ep[0]) for pid in br.flavor_basis_pids])
        for x in xgrid
    ]
    df["absErr"] = df["eko"] - df[_LABEL_GRV]
    df["relErr"] = (df["eko"] - df[_LABEL_GRV]) / df[_LABEL_GRV]
    return df


def df_vs_qcdnum(pto: int, evolved: dict, ep: EvolutionPoint, pid: str) -> pd.DataFrame:
    """Compare EKO vs. QCDNUM."""
    labels = ["u", "d", "s", "c", "S", "g"]
    ep_map = {
        (16.0, 4): "4",
        (100.0, 5): "10",
        (400.0, 5): "20",
    }
    if pid not in labels:
        raise ValueError(f"QCDNUM has no PID '{pid}'")
    # load QCDNUM
    if pto != 1:
        raise ValueError("QCDNUM comparison only available at NLO")
    qcdnum_file = _QCDNUMDIR / f"GRV_evol_Q{ep_map[ep]}_nlo_high1.dat"
    qcdnum = pd.read_csv(qcdnum_file, sep=r"\s+", header=None)
    qcdnum.columns = ["x", "uv", "dv", "u", "d", "s", "c", "g", "S"]
    # load gEKO
    res = pid_weights(pid) @ np.array(
        [evolved[ep]["pdfs"][pid] for pid in br.flavor_basis_pids]
    )
    df = pd.DataFrame()
    # rotate gEKO to QCDNUM
    xgrid = operator_base["xgrid"]
    interp = InterpolatorDispatcher(
        XGrid(xgrid, operator_base["configs"]["interpolation_is_log"]),
        operator_base["configs"]["interpolation_polynomial_degree"],
        False,
    )
    rot = interp.get_interpolation(qcdnum["x"].to_list())
    df["x"] = qcdnum["x"]
    df["eko"] = rot @ (res * xgrid)
    df[_LABEL_QCDNUM] = qcdnum[pid]
    df["absErr"] = df["eko"] - df[_LABEL_QCDNUM]
    df["relErr"] = (df["eko"] - df[_LABEL_QCDNUM]) / df[_LABEL_QCDNUM]
    return df


def plot(
    pto: int,
    eko_path: pathlib.Path,
    pl_path: pathlib.Path,
    cmp: _DF_VS_X_TYPE,
    label: str,
    is_abs: bool = False,
) -> None:
    """Generate comparison plot EKO vs. GRV."""
    pids = np.array([["u", "d"], ["s", "c"], ["S", "g"]])
    evolved = geko.apply_pdf_paths(GRV(pto), eko_path, pl_path)
    fig, axs = plt.subplots(*pids.shape, sharex=True, figsize=(7, 7))
    for axs_, pids_ in zip(axs, pids):
        for ax, pid in zip(axs_, pids_):
            keys = [*evolved.keys()]
            keys.sort()
            for ep in keys:
                cmp_df = cmp(pto, evolved, ep, pid)
                lab = (
                    rf"$Q^2 = {ep[0]} \ \mathrm{{GeV}}^2,\ {ep[1]} \ n_f$"
                    if pid == "u"
                    else None
                )
                if is_abs:
                    ax.plot(cmp_df["x"], cmp_df["eko"], label=lab)
                    ax.plot(cmp_df["x"], cmp_df[label])
                else:
                    ax.plot(cmp_df["x"], cmp_df["eko"] / cmp_df[label], label=lab)
                if lab is not None:
                    ax.legend(prop={"size": 9})
            ax.set_title(f"{pid}")
            ax.set_xscale("log")
            ax.set_xlim(1e-4, 1.0)
            ax.tick_params(
                which="both",
                direction="in",
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
    for ax in axs[-1]:
        ax.set_xlabel("x")
    if is_abs:
        for ax in axs.flatten()[:4]:
            ax.set_ylim(0.0, 1.0)
        axs[-1][0].set_ylim(0.0, 10.0)
        axs[-1][1].set_ylim(0.0, 50.0)
        fig.suptitle(f"EKO,{label} PTO={pto}")
    else:
        for ax in axs.flatten():
            ax.set_ylim(0.95, 1.05)
        fig.suptitle(f"γEKO/{label} PTO={pto}")
    fig.tight_layout()
    abs_tag = "abs-" if is_abs else ""
    fig.savefig(_PLOTSDIR / f"{abs_tag}{label}-{pto}.pdf")
    plt.close(fig)


def compare_df(pto: int, nep: int, pids: str, cmp: _DF_VS_X_TYPE, label: str) -> None:
    """Show DataFrame comparison"""
    evolved = geko.apply_pdf_paths(GRV(pto), EKODIR[pto], GEKODIR[pto])
    ep = list(evolved.keys())[nep]
    for pid in pids.split(","):
        pid = pid.strip()
        if pid == "":
            continue
        cmp_df = cmp(pto, evolved, ep, pid)
        print(f"PID = {pid} at {ep} vs. {label}")
        print(cmp_df)
        cmp_df.to_csv(_COMPAREDIR / f"{label}-{pto}-{ep}-{pid}.csv")


def cli() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pto", help="Perturbative order: 0 = LO, 1 = NLO")
    parser.add_argument("-nep", default=0, help="Number evolution point")
    parser.add_argument("-pid", default="g", help="PID")
    parser.add_argument(
        "--alphas-VG", help="Compare alpha_s to Vadim", action="store_true"
    )
    # GRV
    parser.add_argument(
        "--df-grv", help="Compare (g)EKO vs. GRV via DataFrame", action="store_true"
    )
    parser.add_argument(
        "--plot-grv",
        help="Compare (g)EKO vs. GRV in a plot relatively",
        action="store_true",
    )
    parser.add_argument(
        "--plot-abs-grv",
        help="Compare (g)EKO vs. GRV in a plot absolutely",
        action="store_true",
    )
    # QCDNUM
    parser.add_argument(
        "--df-qcdnum",
        help="Compare (g)EKO vs. QCDNUM via DataFrame",
        action="store_true",
    )
    parser.add_argument(
        "--plot-qcdnum",
        help="Compare (g)EKO vs. QCDNUM in a plot relatively",
        action="store_true",
    )
    parser.add_argument(
        "--plot-abs-qcdnum",
        help="Compare (g)EKO vs. QCDNUM in a plot absolutely",
        action="store_true",
    )

    # prepare args
    args = parser.parse_args()
    pto_: int = int(args.pto)
    nep_: int = int(args.nep)
    # do something
    if args.alphas_VG:
        if pto_ == 0:
            print(
                "alpha_s^LO(MZ = 91.2, nf = 5) = ",
                a_s_grv(91.2**2, 5, 0.153**2, 0) * 4.0 * np.pi,
                " =?= 0.128 [VG, Eq. (4.3)]",
            )
        elif pto_ == 1:
            print(
                "alpha_s^NLO(MZ = 91.2, nf = 5) = ",
                a_s_grv(91.2**2, 5, 0.131**2, 1) * 4.0 * np.pi,
                " =?= 0.109 [VG, Eq. (4.5)]",
            )
    # GRV
    if args.df_grv:
        compare_df(pto_, nep_, args.pid, df_vs_grv, _LABEL_GRV)
    if args.plot_grv:
        plot(pto_, EKODIR[pto_], GEKODIR[pto_], df_vs_grv, _LABEL_GRV, False)
    if args.plot_abs_grv:
        plot(pto_, EKODIR[pto_], GEKODIR[pto_], df_vs_grv, _LABEL_GRV, True)
    # QCDNUM
    if args.df_qcdnum:
        compare_df(pto_, nep_, args.pid, df_vs_qcdnum, _LABEL_QCDNUM)
    if args.plot_qcdnum:
        plot(pto_, EKODIR[pto_], GEKODIR[pto_], df_vs_qcdnum, _LABEL_QCDNUM, False)
    if args.plot_qcdnum:
        plot(pto_, EKODIR[pto_], GEKODIR[pto_], df_vs_qcdnum, _LABEL_QCDNUM, True)


if __name__ == "__main__":
    cli()
