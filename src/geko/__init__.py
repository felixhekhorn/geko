"""Compute the point-like contribution to the photon evolution."""

import json
import pathlib
from dataclasses import asdict
from typing import Any

import eko
import eko.basis_rotation as br
import eko.runner.commons as rcom
import numpy as np
from eko.io.inventory import encode
from eko.io.items import Evolution, Target
from eko.io.runcards import OperatorCard, TheoryCard
from eko.io.types import EvolutionPoint
from eko.matchings import Atlas, Segment
from ekobox.apply import apply_pdf

from .op import compute_one

__version__ = "0.1.1"

THEORYFILE = "theory.json"
OPERATORFILE = "operator.json"

PARTSDIR = "parts"
OPERATORSDIR = "operators"
RECIPESDIR = "recipes"

OPERATOR_EXT = ".npy"
HEADER_EXT = ".json"


def build_recipes(ep: EvolutionPoint, atlas: Atlas) -> list[Evolution]:
    """Construct recipe for given |EP|."""
    recipes = []
    blocks = atlas.matched_path(ep)
    for block in blocks:
        # TODO check matching
        if not isinstance(block, Segment):
            continue
        cliff = block.target in atlas.walls
        recipe = Evolution.from_atlas(block, cliff=cliff)
        recipes.append(recipe)
    return recipes


def build_all_recipes(evolgrid: list[EvolutionPoint], atlas: Atlas) -> list[Evolution]:
    """Construct all recipes for all |EP|."""
    recipes = []
    for ep in evolgrid:
        recipes.extend(build_recipes(ep, atlas))
    return list(set(recipes))


def combine_operator(
    ep: EvolutionPoint, atlas: Atlas, path: pathlib.Path, ekop: eko.EKO
) -> None:
    """Combine all parts to a true operator."""
    recipes = build_recipes(ep, atlas)
    op = 0.0
    # replace the path after j with geko
    for j in range(len(recipes)):
        # eko is (potentially) a list, but
        eko_recipes = recipes[:j]
        # geko is just one integral - the summation is happening right here
        geko_recipe = recipes[j]
        part_path = path / PARTSDIR / (encode(geko_recipe) + OPERATOR_EXT)
        geko_op = np.load(part_path)
        # ekos are multiplied from the right!
        for eko_recipe in reversed(eko_recipes):
            geko_op = np.einsum("akbj,bj->ak", ekop.parts[eko_recipe].operator, geko_op)
        op += geko_op
    head = Target.from_ep(ep)
    head_hash = encode(head)
    head_path = path / OPERATORSDIR / (head_hash + HEADER_EXT)
    head_path.write_text(json.dumps(asdict(head)))
    op_path = path / OPERATORSDIR / (head_hash + OPERATOR_EXT)
    np.save(op_path, op)
    # drop last part if possible
    last_part = recipes[-1]
    if not last_part.cliff:
        part_path = path / PARTSDIR / (encode(last_part) + OPERATOR_EXT)
        part_path.unlink()


def compute(
    theory_card: TheoryCard,
    operator_card: OperatorCard,
    path: pathlib.Path,
    eko_path: pathlib.Path,
) -> None:
    """Compute point-like contribution."""
    # prepare path
    if path.exists():
        raise FileExistsError(f"Output directory '{path}' already exists!")
    path.mkdir()
    (path / PARTSDIR).mkdir()
    (path / OPERATORSDIR).mkdir()
    (path / RECIPESDIR).mkdir()
    # dump cards
    with open(path / THEORYFILE, "w", encoding="utf8") as f:
        json.dump(theory_card.public_raw, f)
    with open(path / OPERATORFILE, "w", encoding="utf8") as f:
        json.dump(operator_card.public_raw, f)
    # compute recipes
    sc = rcom.couplings(theory_card, operator_card)
    atlas = rcom.atlas(theory_card, operator_card)
    recipes = build_all_recipes(operator_card.evolgrid, atlas)
    order_qcd = theory_card.order[0]
    ev_op_iterations = operator_card.configs.ev_op_iterations
    # execute recipes
    for recipe in recipes:
        # write recipe
        recipe_path = path / RECIPESDIR / (encode(recipe) + HEADER_EXT)
        recipe_path.write_text(json.dumps(asdict(recipe)))
        # execute it
        part_path = path / PARTSDIR / (encode(recipe) + OPERATOR_EXT)
        compute_one(
            operator_card.xgrid, sc, recipe, part_path, order_qcd, ev_op_iterations
        )
        # remove recipe again
        recipe_path.unlink()
    # combine operators
    with eko.EKO.read(eko_path) as ekop:
        for ep in operator_card.evolgrid:
            combine_operator(ep, atlas, path, ekop)


def load(path: pathlib.Path) -> dict[EvolutionPoint, np.ndarray]:
    """Load from disk."""
    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist!")
    # load op card
    operator_card = None
    with open(path / OPERATORFILE, encoding="utf8") as f:
        operator_card = json.load(f)
    operator_card = OperatorCard.from_dict(operator_card)
    # load operators
    out = {}
    for ep in operator_card.evolgrid:
        head = Target.from_ep(ep)
        op_path = path / OPERATORSDIR / (encode(head) + OPERATOR_EXT)
        out[ep] = np.load(op_path)
    return out


def apply_pdf_paths(
    lhapdf_like: Any, eko_path: pathlib.Path, pl_path: pathlib.Path
) -> dict[EvolutionPoint, dict[str, np.ndarray]]:
    r"""Evolve |PDF| with eko + geko.

    The |PDF| `lhapdf_like` must be normalized as :math:`f / \alpha_{em}`,
    and the result returned has the same normalization.
    """
    # hadronic contributions
    evolved: dict[EvolutionPoint, dict[str, np.ndarray]] = {}
    with eko.EKO.read(eko_path) as evolution_operator:
        evolved = apply_pdf(evolution_operator, lhapdf_like)
    # point-like contributions
    pl = load(pl_path)
    if len(evolved.keys()) != len(pl.keys()):
        raise ValueError(
            "Point-like and hadronic contributions don't contain the same number of evolution points!"
        )
    for ep in evolved:
        for idx, pid in enumerate(br.flavor_basis_pids):
            evolved[ep]["pdfs"][pid] += pl[ep][idx] / (4.0 * np.pi)
    return evolved
