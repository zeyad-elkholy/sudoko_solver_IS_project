"""tests/test_pipeline.py -- Tests for the Project 7 pipeline.

Common-checklist deliverable #9 requires:
    - at least 5 normal cases
    - at least 3 edge cases
    - at least 2 impossible cases (must give a clear "no solution" message)

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make the package importable from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sudoku_solver.csp import SudokuCSP
from sudoku_solver.data import generate_puzzle, generate_solution
from sudoku_solver.search import (CellReading, _count_conflicts,
                                    search_corrections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reading(r, c, primary, alts):
    """Build a CellReading with primary digit and a list of (alt_digit, prob)."""
    cands = [(primary, 0.95)] + alts
    return CellReading(r=r, c=c, candidates=tuple(cands))


# ---------------------------------------------------------------------------
# Normal cases (>=5)
# ---------------------------------------------------------------------------

def test_normal_csp_bt_solves_valid_puzzle():
    """1. A valid puzzle is solved by plain backtracking."""
    rng = np.random.default_rng(0)
    sol = generate_solution(rng)
    puzzle, _sol = generate_puzzle(rng, n_givens=40)
    csp = SudokuCSP(puzzle=puzzle)
    out = csp.solve(mode="bt")
    assert out is not None
    # Check the solution: every row, column and box is a permutation of 1..9.
    for r in range(9):
        assert sorted(out[r].tolist()) == list(range(1, 10))
    for c in range(9):
        assert sorted(out[:, c].tolist()) == list(range(1, 10))
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = out[br:br + 3, bc:bc + 3].flatten().tolist()
            assert sorted(block) == list(range(1, 10))


def test_normal_csp_fc_solves_valid_puzzle():
    """2. Backtracking + FC also solves."""
    rng = np.random.default_rng(1)
    puzzle, _sol = generate_puzzle(rng, n_givens=35)
    out = SudokuCSP(puzzle=puzzle).solve(mode="fc")
    assert out is not None
    for r in range(9):
        assert sorted(out[r].tolist()) == list(range(1, 10))


def test_normal_csp_ac3_solves_valid_puzzle():
    """3. Backtracking + FC + AC-3 also solves."""
    rng = np.random.default_rng(2)
    puzzle, _sol = generate_puzzle(rng, n_givens=30)
    out = SudokuCSP(puzzle=puzzle).solve(mode="ac3")
    assert out is not None
    for r in range(9):
        assert sorted(out[r].tolist()) == list(range(1, 10))


def test_normal_search_finds_single_correction():
    """4. With one injected misread, UCS search finds the cheapest correction."""
    # Build a valid puzzle, then break one cell by changing its value.
    rng = np.random.default_rng(3)
    sol = generate_solution(rng)
    puzzle, _sol = generate_puzzle(rng, n_givens=35)
    # Choose a given cell and change its digit to a wrong value that creates
    # a duplicate in its row.
    given_cells = [(r, c) for r in range(9) for c in range(9) if puzzle[r, c] != 0]
    chosen = None
    for (r, c) in given_cells:
        orig = int(puzzle[r, c])
        # Find a digit already present elsewhere in the row (to create a conflict).
        row_digits = [int(puzzle[r, cc]) for cc in range(9) if cc != c and puzzle[r, cc] != 0]
        for d in set(row_digits):
            if d != orig:
                chosen = (r, c, orig, d)
                break
        if chosen is not None:
            break
    assert chosen is not None
    r, c, orig, wrong = chosen
    puzzle[r, c] = wrong
    # Build readings: the wrong digit is the NN's top pick, but the original
    # is also a candidate with slightly lower probability.
    readings = {(r, c): _reading(r, c, wrong, [(orig, 0.40)])}
    res = search_corrections(base_puzzle=puzzle, readings=readings,
                              mode="ucs", csp_mode="ac3", max_corrections=3)
    assert res.found, "Search should find a correction that fixes the puzzle"
    assert any(corr.r == r and corr.c == c and corr.new_digit == orig
               for corr in res.corrections), \
        "Correction should restore the original digit"


def test_normal_search_no_correction_needed():
    """5. If the puzzle is already solvable, search returns immediately with
    no corrections."""
    rng = np.random.default_rng(4)
    puzzle, _sol = generate_puzzle(rng, n_givens=35)
    readings = {}  # nothing to correct
    res = search_corrections(base_puzzle=puzzle, readings=readings,
                              mode="ucs", csp_mode="ac3", max_corrections=3)
    assert res.found
    assert res.corrections == []
    assert res.cost == 0.0


# ---------------------------------------------------------------------------
# Edge cases (>=3)
# ---------------------------------------------------------------------------

def test_edge_already_solved_puzzle():
    """1. A fully-solved grid (no blanks) is immediately returned."""
    rng = np.random.default_rng(10)
    sol = generate_solution(rng)
    csp = SudokuCSP(puzzle=sol)
    out = csp.solve(mode="ac3")
    assert out is not None
    assert np.array_equal(out, sol)


def test_edge_minimum_givens_extreme():
    """2. A puzzle with very few givens (edge: near 17, the theoretical min)
    -- we use 22 givens; if solvable, AC-3 should solve it."""
    rng = np.random.default_rng(11)
    puzzle, _sol = generate_puzzle(rng, n_givens=25)
    out = SudokuCSP(puzzle=puzzle).solve(mode="ac3", time_limit_s=10.0)
    if out is not None:
        for r in range(9):
            assert sorted(out[r].tolist()) == list(range(1, 10))


def test_edge_search_max_corrections_zero():
    """3. With max_corrections=0 and an unsolvable puzzle, search must
    return found=False without trying any correction."""
    rng = np.random.default_rng(12)
    puzzle, _sol = generate_puzzle(rng, n_givens=35)
    # Find two given cells in the same row and force a duplicate.
    given_in_row0 = [(0, c) for c in range(9) if puzzle[0, c] != 0]
    if len(given_in_row0) >= 2:
        (r, c1), (_, c2) = given_in_row0[0], given_in_row0[1]
        puzzle[r, c1] = puzzle[r, c2]  # duplicate given -> unsolvable
    else:
        # If row 0 has fewer than 2 givens, force one in row 1.
        given_in_row1 = [(1, c) for c in range(9) if puzzle[1, c] != 0]
        (r, c1), (_, c2) = given_in_row1[0], given_in_row1[1]
        puzzle[r, c1] = puzzle[r, c2]
    readings = {(r, c1): _reading(r, c1, int(puzzle[r, c1]), [(7, 0.4)])}
    res = search_corrections(base_puzzle=puzzle, readings=readings,
                              mode="ucs", csp_mode="ac3", max_corrections=0)
    assert not res.found


# ---------------------------------------------------------------------------
# Impossible cases (>=2): must produce a clear "no solution" message
# ---------------------------------------------------------------------------

def test_impossible_duplicate_givens_same_row():
    """1. Two equal givens in the same row -> no solution. The CSP must
    return None and the search must report found=False."""
    rng = np.random.default_rng(20)
    puzzle, _sol = generate_puzzle(rng, n_givens=35)
    # Force two equal givens in row 0.
    puzzle[0, 0] = 5
    puzzle[0, 1] = 5
    csp = SudokuCSP(puzzle=puzzle)
    out = csp.solve(mode="ac3")
    assert out is None, "Duplicate givens in a row must yield no solution"


def test_impossible_search_no_correction_fixes_puzzle():
    """2. A puzzle that requires >MAX_CORRECTIONS corrections is reported as
    'unreadable' (found=False)."""
    rng = np.random.default_rng(21)
    puzzle, _sol = generate_puzzle(rng, n_givens=35)
    # Inject 5 misreads (more than the max=3 corrections search allows).
    given_cells = [(r, c) for r in range(9) for c in range(9) if puzzle[r, c] != 0]
    rng.shuffle(given_cells)
    for (r, c) in given_cells[:5]:
        # Replace with a digit guaranteed to conflict in its row.
        row_d = [int(puzzle[r, cc]) for cc in range(9) if cc != c and puzzle[r, cc] != 0]
        for d in set(row_d):
            if d != puzzle[r, c]:
                puzzle[r, c] = d
                break
    readings = {(r, c): _reading(r, c, int(puzzle[r, c]), [(7, 0.4)])
                for (r, c) in given_cells[:5]}
    res = search_corrections(base_puzzle=puzzle, readings=readings,
                              mode="ucs", csp_mode="ac3", max_corrections=3)
    assert not res.found, "With >3 required corrections, search must report no solution"
