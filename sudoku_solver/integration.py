"""integration.py -- End-to-end pipeline that ties NN + CSP + Search together.

Project 7.

Pipeline (per spec):
    1. Read the 9x9 image mosaic.
    2. NN classifies each *given* cell, producing (digit, prob) candidates.
    3. AC-3 (in the CSP) detects conflicts in the read puzzle.
       - If the puzzle is already solvable, the CSP solves it. Done.
       - Otherwise: search for the cheapest correction set (max MAX_CORRECTIONS).
    4. For each candidate correction set, the CSP is run again to test
       solvability. The first solvable one wins.
    5. If no correction works, report "unreadable" and list the least
       confident cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .csp import SudokuCSP
from .data import Mosaic, build_mosaic, generate_puzzle, generate_solution
from .nn import DigitClassifier
from .search import (
    CellReading,
    Correction,
    MAX_CORRECTIONS,
    SearchResult,
    SearchStats,
    search_corrections,
)


# ---------------------------------------------------------------------------
# Reading: from mosaic + classifier -> CellReading dict + read puzzle
# ---------------------------------------------------------------------------

def read_mosaic(classifier: DigitClassifier, mosaic: Mosaic,
                top_k: int = 3) -> tuple[np.ndarray, dict[tuple[int, int], CellReading]]:
    """Read the mosaic using `classifier`. Returns (read_puzzle, readings)
    where `read_puzzle` is the 9x9 array of NN argmax digits (0 for blank
    cells) and `readings` is the dict of CellReading objects for given cells
    only (blank cells are not classified)."""
    read_puzzle = np.zeros((9, 9), dtype=np.int8)
    readings: dict[tuple[int, int], CellReading] = {}

    for r in range(9):
        for c in range(9):
            img = mosaic.image_for_cell(r, c)
            if img is None:
                continue  # blank cell -- leave as 0
            P = classifier.predict_proba(img[None, ...])[0]
            # Order: classes [1..9], so candidate list = sorted (digit, prob)
            order = np.argsort(-P)
            cands = tuple((int(d + 1), float(P[d])) for d in order[:top_k])
            readings[(r, c)] = CellReading(r=r, c=c, candidates=cands)
            read_puzzle[r, c] = cands[0][0]

    return read_puzzle, readings


# ---------------------------------------------------------------------------
# End-to-end solve
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    found: bool
    read_puzzle: np.ndarray
    solution: Optional[np.ndarray]
    corrections: list[Correction]
    correction_cost: float
    least_confident_cells: list[tuple[tuple[int, int], float]]
    search_mode: str
    csp_mode: str
    search_stats: SearchStats
    csp_stats: dict
    search_trace: list[str] = field(default_factory=list)
    mosaic: Optional[Mosaic] = None

    def correction_log(self) -> list[dict]:
        """Correction log per spec: list of {cell, read_digit, new_digit, cost}."""
        out = []
        for corr in self.corrections:
            # Look up the read digit from the mosaic if available.
            rd = self.read_puzzle[corr.r, corr.c]
            out.append({
                "cell": (corr.r, corr.c),
                "read_digit": int(rd),
                "new_digit": int(corr.new_digit),
                "cost": float(corr.cost),
            })
        return out


def solve_sudoku(
    classifier: DigitClassifier,
    mosaic: Mosaic,
    search_mode: str = "ucs",
    csp_mode: str = "ac3",
    max_corrections: int = MAX_CORRECTIONS,
    require_unique: bool = False,
    top_k: int = 3,
    verbose: bool = False,
) -> PipelineResult:
    """Run the full pipeline on a mosaic and return a PipelineResult."""

    # Step 1: read mosaic.
    read_puzzle, readings = read_mosaic(classifier, mosaic, top_k=top_k)
    if verbose:
        print("[integration] Read puzzle:")
        _print_grid(read_puzzle)

    # Step 2: try to solve directly. If solvable, no corrections needed.
    csp = SudokuCSP(puzzle=read_puzzle)
    sol = csp.solve(mode=csp_mode)
    csp_stats = csp.stats.as_dict()
    if sol is not None:
        return PipelineResult(
            found=True,
            read_puzzle=read_puzzle,
            solution=sol,
            corrections=[],
            correction_cost=0.0,
            least_confident_cells=[],
            search_mode=search_mode,
            csp_mode=csp_mode,
            search_stats=SearchStats(),
            csp_stats=csp_stats,
            mosaic=mosaic,
        )

    # Step 3: search for cheapest correction set.
    sr: SearchResult = search_corrections(
        base_puzzle=read_puzzle,
        readings=readings,
        mode=search_mode,
        csp_mode=csp_mode,
        max_corrections=max_corrections,
        require_unique=require_unique,
        verbose=verbose,
    )

    if sr.found:
        return PipelineResult(
            found=True,
            read_puzzle=read_puzzle,
            solution=sr.solution,
            corrections=sr.corrections,
            correction_cost=sr.cost,
            least_confident_cells=[],
            search_mode=search_mode,
            csp_mode=csp_mode,
            search_stats=sr.stats,
            csp_stats=csp_stats,
            search_trace=sr.trace,
            mosaic=mosaic,
        )

    # Step 4: not found -- list least-confident cells (per spec).
    least_conf = sorted(
        ((cell, rd.current_prob) for cell, rd in readings.items()),
        key=lambda kv: kv[1],
    )[:5]
    return PipelineResult(
        found=False,
        read_puzzle=read_puzzle,
        solution=None,
        corrections=[],
        correction_cost=float("inf"),
        least_confident_cells=least_conf,
        search_mode=search_mode,
        csp_mode=csp_mode,
        search_stats=sr.stats,
        csp_stats=csp_stats,
        search_trace=sr.trace,
        mosaic=mosaic,
    )


# ---------------------------------------------------------------------------
# Pretty-printer used by run_demo.py
# ---------------------------------------------------------------------------

def _print_grid(grid: np.ndarray) -> None:
    g = np.where(grid == 0, ".", grid.astype(str))
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("------+-------+------")
        row = " ".join(g[r, c] for c in range(9))
        # Insert separators at every 3 cols.
        parts = [row.split(" ")[i:i + 3] for i in range(0, 9, 3)]
        print(" | ".join(" ".join(p) for p in parts))


# ---------------------------------------------------------------------------
# Convenience: build a single mosaic and solve it
# ---------------------------------------------------------------------------

def solve_random_puzzle(
    classifier: DigitClassifier,
    rng: np.random.Generator,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_givens: int = 30,
    n_injected: int = 0,
    search_mode: str = "ucs",
    csp_mode: str = "ac3",
    max_corrections: int = MAX_CORRECTIONS,
    top_k: int = 3,
    verbose: bool = False,
) -> tuple[PipelineResult, Mosaic]:
    sol = generate_solution(rng)
    puzzle = generate_puzzle(rng, n_givens=n_givens)
    # We need the puzzle and the solution to match (puzzle is a partial view
    # of the solution; generate_puzzle already guarantees this).
    givens_mask = puzzle > 0
    mosaic = build_mosaic(
        puzzle=puzzle, solution=sol,
        X_pool=X_pool, y_pool=y_pool,
        rng=rng, n_injected_misreads=n_injected,
        givens_mask=givens_mask,
    )
    result = solve_sudoku(
        classifier=classifier, mosaic=mosaic,
        search_mode=search_mode, csp_mode=csp_mode,
        max_corrections=max_corrections, top_k=top_k,
        verbose=verbose,
    )
    return result, mosaic
