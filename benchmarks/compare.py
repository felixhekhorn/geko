"""Execute benchmarks.

References
----------
- [Gluck:1991jc] https://inspirehep.net/literature/322191
- [VG] photon_pdfs_v7.pdf

"""

import argparse
import pathlib

import eko.basis_rotation as br
import grvphoton
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eko.beta import beta_qcd
from eko.io.types import EvolutionPoint
from run import EKODIR, GEKODIR, operator_base

import geko

# Setup paths
_HERE = pathlib.Path(__file__).parent
_COMPAREDIR = _HERE / "compare"
_PLOTSDIR = _HERE / "plots"


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


def df_vs_grv(
    pto: int, evolved: dict, ep: EvolutionPoint, weights: np.ndarray
) -> pd.DataFrame:
    """Compare EKO vs. GRV."""
    res = weights @ np.array([evolved[ep]["pdfs"][pid] for pid in br.flavor_basis_pids])
    df = pd.DataFrame()
    xgrid = operator_base["xgrid"]
    df["x"] = xgrid
    df["eko"] = res * xgrid
    ref = GRV(pto)
    df["grv"] = [
        weights @ np.array([ref.xfxQ2(pid, x, ep[0]) for pid in br.flavor_basis_pids])
        for x in xgrid
    ]
    df["absErr"] = df["eko"] - df["grv"]
    df["relErr"] = (df["eko"] - df["grv"]) / df["grv"]
    return df


def plot_vs_grv(
    pto: int, eko_path: pathlib.Path, pl_path: pathlib.Path, is_abs: bool = False
) -> None:
    """Generate comparison plot EKO vs. GRV."""
    pids = np.array([["u", "d"], ["s", "c"], ["S", "g"]])
    evolved = geko.apply_pdf_paths(GRV(pto), eko_path, pl_path)
    fig, axs = plt.subplots(*pids.shape, sharex=True, figsize=(7, 7))
    for axs_, pids_ in zip(axs, pids):
        for ax, pid in zip(axs_, pids_):
            for ep in evolved.keys():
                cmp = df_vs_grv(pto, evolved, ep, pid_weights(pid))
                label = f"{ep}" if pid == "u" else None
                if is_abs:
                    ax.plot(cmp["x"], cmp["eko"], label=label)
                    ax.plot(cmp["x"], cmp["grv"])
                else:
                    ax.plot(cmp["x"], cmp["eko"] / cmp["grv"], label=label)
                if label is not None:
                    ax.legend()
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
        fig.suptitle(f"EKO,GRV PTO={pto}")
    else:
        for ax in axs.flatten():
            ax.set_ylim(0.95, 1.05)
        fig.suptitle(f"EKO/GRV PTO={pto}")
    fig.tight_layout()
    abs_tag = "-abs" if is_abs else ""
    fig.savefig(_PLOTSDIR / f"compare{abs_tag}-{pto}.pdf")
    plt.close(fig)


# def write_evolved_pdfs(
#     pto: int, evolved: dict, weights: np.ndarray, out_dir: pathlib.Path
# ) -> None:
#     """Store rotated evolved PDFs as CSV: x, u, d, s, c, S, g."""
#     out_dir.mkdir(parents=True, exist_ok=True)
#     xgrid = operator_base["xgrid"]
#     labels = ["u", "d", "s", "c", "S", "g"]

#     for ep, data in evolved.items():
#         # Collect PDFs from flavor basis
#         fb_pdfs = np.array([data["pdfs"][pid] for pid in br.flavor_basis_pids])

#         # Rotate using weights
#         rotated = weights @ fb_pdfs  # shape: (7, len(xgrid))

#         # Create DataFrame
#         df = pd.DataFrame({"x": xgrid})
#         for i, label in enumerate(labels):
#             df[label] = rotated[i] * xgrid  # match EKO convention

#         # Save CSV
#         ep_str = "-".join(map(str, ep)) if isinstance(ep, tuple) else str(ep)
#         csv_path = out_dir / f"evolved-pto_{pto}-{ep_str}.csv"
#         df.to_csv(csv_path, index=False)


def cli() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pto", help="Perturbative order: 0 = LO, 1 = NLO")
    parser.add_argument("-nep", default=0, help="Number evolution point")
    parser.add_argument("-pid", default="g", help="PID")
    parser.add_argument(
        "--alphas-VG", help="Compare alpha_s to Vadim", action="store_true"
    )
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
    # parser.add_argument(
    #     "--write-pdfs", help="Write evolved PDFs to dataset files", action="store_true"
    # )

    # prepare args
    args = parser.parse_args()
    pto_: int = int(args.pto)
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
    if args.df_grv:
        evolved = geko.apply_pdf_paths(GRV(pto_), EKODIR[pto_], GEKODIR[pto_])
        ep = list(evolved.keys())[int(args.nep)]
        for pid_ in args.pid.split(","):
            pid_ = pid_.strip()
            if pid_ == "":
                continue
            cmp = df_vs_grv(pto_, evolved, ep, pid_weights(pid_))
            print(f"PID = {pid_} at {ep}")
            print(cmp)
            cmp.to_csv(_COMPAREDIR / f"{pto_}-{ep}-{pid_}.csv")
    if args.plot_grv:
        plot_vs_grv(pto_, EKODIR[pto_], GEKODIR[pto_], False)
    if args.plot_abs_grv:
        plot_vs_grv(pto_, EKODIR[pto_], GEKODIR[pto_], True)
    # if args.write_pdfs:
    #     evolved = geko.apply_pdf_paths(GRV(pto_), EKODIR[pto_], GEKODIR[pto_])
    #     weights = np.stack([pid_weights(pid) for pid in ["u", "d", "s", "c", "S", "g"]])
    #     write_evolved_pdfs(pto_, evolved, weights, pathlib.Path("evolved_pdfs"))


if __name__ == "__main__":
    cli()
