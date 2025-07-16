"""Compute benchmark operators."""

import argparse
import pathlib
import shutil
from math import nan

from eko.interpolation import lambertgrid
from eko.io.runcards import OperatorCard, TheoryCard
from eko.runner.managed import solve as eko_solve

import geko

# Setup paths
_HERE = pathlib.Path(__file__).parent
_DATADIR = _HERE / "data"
EKODIR = {0: _DATADIR / "lo-eko.tar", 1: _DATADIR / "nlo-eko.tar"}
"""Path to EKO objects."""
GEKODIR = {0: _DATADIR / "lo-geko", 1: _DATADIR / "nlo-geko"}
"""Path to gEKO objects."""

# define basic settings
theory_base = {
    "order": [1, 0],  # This is overwritten by the CLI
    "couplings": {
        "alphas": 0.128,  # This may be overwritten by the CLI
        "alphaem": 1.0 / 137.0,
        "scale": 91.2,
        "num_flavs_ref": 5,
        "max_num_flavs": 6,
        "em_running": False,
    },
    "heavy": {
        "masses": ((1.5, nan), (4.5, nan), (100.0, nan)),
        "masses_scheme": "pole",
        "matching_ratios": (1.0, 1.0, 1.0),
        "num_flavs_init": 4,
        "num_flavs_max_pdf": 6,
        "intrinsic_flavors": [],
    },
    "xif": 1.0,
    "n3lo_ad_variation": (0, 0, 0, 0, 0, 0, 0),
}
operator_base = {
    "mu0": 1.51,
    "mugrid": [(4.0, 4), (10.0, 5), (20.0, 5)],
    "xgrid": lambertgrid(30, 1e-5),
    "configs": {
        "evolution_method": "iterate-exact",
        "ev_op_max_order": [10, 0],
        "ev_op_iterations": 30,
        "scvar_method": None,
        "inversion_method": None,
        "interpolation_polynomial_degree": 4,
        "interpolation_is_log": True,
        "polarized": False,
        "time_like": False,
        "n_integration_cores": 6,
    },
    "debug": {"skip_singlet": False, "skip_non_singlet": False},
}


def update_theory_nlo(tc: TheoryCard):
    """Update theory settings for NLO run."""
    tc.couplings.alphas = 0.1109
    # tc.couplings.alphas = 0.1090  # comparing to VG


def compute_eko(pto: int, path: pathlib.Path, overwrite: bool = False) -> None:
    """Compute an EKO."""
    if path.exists():
        if overwrite:
            print(f"EKO {path} already exists! overwriting it ...")
            path.unlink()
        else:
            print(f"EKO {path} already exists! doing nothing ...")
            return
    tc = TheoryCard.from_dict(theory_base)
    tc.order = (1 + pto, 0)
    if pto == 1:
        update_theory_nlo(tc)
    # do it!
    eko_solve(tc, OperatorCard.from_dict(operator_base), path)


def compute_geko(
    pto: int, path: pathlib.Path, eko_path: pathlib.Path, overwrite: bool = False
) -> None:
    """Compute an gEKO."""
    if path.exists():
        if overwrite:
            print(f"gEKO {path} already exists! overwriting it ...")
            shutil.rmtree(path)
        else:
            print(f"gEKO {path} already exists! doing nothing ...")
            return
    tc = TheoryCard.from_dict(theory_base)
    tc.order = (1 + pto, 0)
    if pto == 1:
        update_theory_nlo(tc)
    # do it!
    geko.compute(
        tc,
        OperatorCard.from_dict(operator_base),
        path,
        eko_path,
    )


def cli() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pto", help="Perturbative order: 0 = LO, 1 = NLO")
    parser.add_argument("-eko", help="Compute EKO", action="store_true")
    parser.add_argument(
        "-geko",
        help="Compute gEKO",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite", help="Overwrite files if already present", action="store_true"
    )
    # prepare args
    args = parser.parse_args()
    pto_: int = int(args.pto)
    overwrite_: bool = bool(args.overwrite)
    if args.eko:
        compute_eko(pto_, EKODIR[pto_], overwrite_)
    if args.geko:
        compute_geko(pto_, GEKODIR[pto_], EKODIR[pto_], overwrite_)


if __name__ == "__main__":
    cli()
