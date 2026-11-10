"""search.py -- Correction search over uncertain cells.

Project 7.

The pipeline is: NN reads the mosaic and reports, for each *given* cell,
the top-k digit candidates with their probabilities. Those probabilities
define a *cost* for changing a cell from its current reading to a different
digit (the cost is the drop in log-probability, never negative).

Search tries to find the cheapest set of at most MAX_CORRECTIONS corrections
such that the corrected puzzle is solvable by the CSP.

State   : frozenset of corrections already made (cell -> new digit).
Actions : change one uncertain cell to one of its top-k alternatives that is
          not already the current reading.
Cost    : -log(p_new / p_current)   (>= 0 since p_new <= p_current)
Goal    : the corrected puzzle is solvable by the CSP (one solution).
Limit   : at most MAX_CORRECTIONS corrections.

Three algorithms (per common-checklist deliverable #5 and project spec):
  - Uniform Cost Search (UCS) -- optimal w.r.t. the cost defined above.
  - Greedy Best-First     -- expands the node that looks most promising
                              (fewest corrections / highest total prob).
  - A*                    -- UCS + heuristic: at least one cell of any
                              duplicate-digit conflict must change. The
                              heuristic is admissible (lower bound on the
                              number of corrections; each correction costs
                              >= 0, so the heuristic never overestimates
                              actual remaining cost).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from .csp import SudokuCSP


# Maximum number of corrections Search is allowed to make (per spec).
MAX_CORRECTIONS = 3


# ---------------------------------------------------------------------------
# Reading data structure produced by the NN
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellReading:
    """NN reading for a single given cell.

    Attributes:
        r, c           -- cell coordinates.
        candidates     -- list of (digit, probability) sorted by descending
                          probability. The first element is the *current*
                          reading (what the NN reports as the digit).
    """
    r: int
    c: int
    candidates: tuple  # tuple of (digit, prob) sorted desc by prob

    @property
    def current_digit(self) -> int:
        return self.candidates[0][0]

    @property
    def current_prob(self) -> float:
        return self.candidates[0][1]

    def cost_of(self, new_digit: int) -> float:
        """Cost of changing the current reading to `new_digit`.

        Defined as -log(p_new / p_current) which is >= 0 because p_new <=
        p_current (the current reading is the most probable)."""
        p_new = 0.0
        for d, p in self.candidates:
            if d == new_digit:
                p_new = p
                break
        if p_new <= 0:
            return float("inf")
        p_cur = max(self.current_prob, 1e-12)
        return max(0.0, -math.log(p_new / p_cur))


@dataclass(frozen=True)
class Correction:
    r: int
    c: int
    new_digit: int
    cost: float

    def __repr__(self) -> str:
        return f"({self.r},{self.c})->{self.new_digit}({self.cost:.3f})"


@dataclass
class SearchStats:
    nodes_expanded: int = 0
    nodes_generated: int = 0
    goal_tests: int = 0
    max_frontier: int = 0
    csp_calls: int = 0

    def as_dict(self) -> dict:
        return {"nodes_expanded": self.nodes_expanded,
                "nodes_generated": self.nodes_generated,
                "goal_tests": self.goal_tests,
                "max_frontier": self.max_frontier,
                "csp_calls": self.csp_calls}


# ---------------------------------------------------------------------------
# Helper: duplicate-digit conflicts heuristic (admissible for A*)
# ---------------------------------------------------------------------------

def _count_conflicts(puzzle: np.ndarray) -> int:
    """Return the number of duplicate-digit conflicts in rows, columns and
    boxes. Each duplicate pair adds 1; conflicts indicate that at least one
    cell must change for the puzzle to be solvable."""
    conflicts = 0
    for r in range(9):
        seen = {}
        for c in range(9):
            v = int(puzzle[r, c])
            if v == 0:
                continue
            seen[v] = seen.get(v, 0) + 1
        conflicts += sum(max(0, cnt - 1) for cnt in seen.values())

    for c in range(9):
        seen = {}
        for r in range(9):
            v = int(puzzle[r, c])
            if v == 0:
                continue
            seen[v] = seen.get(v, 0) + 1
        conflicts += sum(max(0, cnt - 1) for cnt in seen.values())

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = {}
            for dr in range(3):
                for dc in range(3):
                    v = int(puzzle[br + dr, bc + dc])
                    if v == 0:
                        continue
                    seen[v] = seen.get(v, 0) + 1
            conflicts += sum(max(0, cnt - 1) for cnt in seen.values())
    return conflicts


def _conflict_cells(puzzle: np.ndarray) -> set[tuple[int, int]]:
    """Return the set of cells that are involved in a duplicate-digit
    conflict (same digit appears more than once in their row/col/box)."""
    out: set[tuple[int, int]] = set()
    # Rows
    for r in range(9):
        seen = {}
        for c in range(9):
            v = int(puzzle[r, c])
            if v == 0:
                continue
            seen.setdefault(v, []).append((r, c))
        for v, cells in seen.items():
            if len(cells) > 1:
                out.update(cells)
    # Columns
    for c in range(9):
        seen = {}
        for r in range(9):
            v = int(puzzle[r, c])
            if v == 0:
                continue
            seen.setdefault(v, []).append((r, c))
        for v, cells in seen.items():
            if len(cells) > 1:
                out.update(cells)
    # Boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = {}
            for dr in range(3):
                for dc in range(3):
                    r, c = br + dr, bc + dc
                    v = int(puzzle[r, c])
                    if v == 0:
                        continue
                    seen.setdefault(v, []).append((r, c))
            for v, cells in seen.items():
                if len(cells) > 1:
                    out.update(cells)
    return out


def _heuristic_min_cost(readings: dict[tuple[int, int], CellReading],
                          puzzle: np.ndarray) -> float:
    """Admissible heuristic: at least one cell of each conflict must change.
    Each conflict contributes at least the cheapest correction available
    across all conflicting cells (zero if a conflict's cell is already a
    single-candidate cell -- but in our setting every given cell has at
    least 2 candidates)."""
    h = 0.0
    # For each conflict group (row/col/box), one correction is needed.
    # The cheapest correction available is the minimum over all cells in
    # that unit of the cost of changing to any non-conflicting alternative.
    # We use a per-unit lower bound: the smallest "second-best" cost in
    # that unit. This is admissible because at least one cell must change.
    units = []
    for r in range(9):
        units.append([(r, c) for c in range(9)])
    for c in range(9):
        units.append([(r, c) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            units.append([(br + dr, bc + dc) for dr in range(3) for dc in range(3)])

    for unit in units:
        # Find duplicate digits in this unit.
        seen = {}
        for (r, c) in unit:
            v = int(puzzle[r, c])
            if v == 0:
                continue
            seen.setdefault(v, []).append((r, c))
        for v, cells in seen.items():
            if len(cells) <= 1:
                continue
            # At least one cell must change. Cheapest correction across
            # these conflicting cells:
            best = float("inf")
            for (r, c) in cells:
                if (r, c) not in readings:
                    continue
                rd = readings[(r, c)]
                # Try every alternative digit for this cell.
                for d, _p in rd.candidates[1:]:
                    cost = rd.cost_of(d)
                    if cost < best:
                        best = cost
            if best == float("inf"):
                best = 0.0  # no readings to change -> can't fix
            h += best
    return h


# ---------------------------------------------------------------------------
# Goal test: the corrected puzzle is solvable.
# ---------------------------------------------------------------------------

def _build_corrected_puzzle(base_puzzle: np.ndarray,
                             corrections: frozenset[Correction]) -> np.ndarray:
    out = base_puzzle.copy()
    for corr in corrections:
        out[corr.r, corr.c] = corr.new_digit
    return out


def _is_goal(puzzle: np.ndarray, csp_mode: str,
              stats: SearchStats, require_unique: bool = False) -> tuple[bool, Optional[np.ndarray]]:
    """Test whether the corrected puzzle is solvable by the CSP. If
    `require_unique` is True, the corrected puzzle must have *exactly* one
    solution (optional extension in the project spec)."""
    stats.goal_tests += 1
    stats.csp_calls += 1
    csp = SudokuCSP(puzzle=puzzle)
    if require_unique:
        n = csp.count_solutions(mode=csp_mode, cap=2)
        if n == 1:
            sol = np.zeros((9, 9), dtype=np.int8)
            for (r, c), d in csp.domains.items():
                sol[r, c] = next(iter(d)) if len(d) == 1 else 0
            return True, sol
        return False, None
    sol = csp.solve(mode=csp_mode)
    return (sol is not None), sol


# ---------------------------------------------------------------------------
# Search algorithms
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    found: bool
    corrections: list[Correction]
    cost: float
    solution: Optional[np.ndarray]
    stats: SearchStats
    trace: list  # list of strings describing each expansion (for demo)

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "n_corrections": len(self.corrections),
            "cost": self.cost,
            "stats": self.stats.as_dict(),
            "corrections": [
                {"r": c.r, "c": c.c, "old": None, "new": c.new_digit,
                 "cost": c.cost}
                for c in self.corrections
            ],
        }


def _candidate_actions(readings: dict[tuple[int, int], CellReading],
                        applied: frozenset[Correction],
                        conflict_cells: set[tuple[int, int]]
                        ) -> Iterable[Correction]:
    """Yield candidate corrections: change one cell that hasn't been changed
    yet to one of its top-k alternatives. Restrict to cells in
    `conflict_cells` for efficiency (cells involved in duplicate-digit
    conflicts are the only ones that can fix the puzzle)."""
    changed_cells = {(c.r, c.c) for c in applied}
    for (r, c) in conflict_cells:
        if (r, c) in changed_cells:
            continue
        if (r, c) not in readings:
            continue
        rd = readings[(r, c)]
        # Top alternatives = all candidates except the current reading.
        for d, _p in rd.candidates[1:]:
            cost = rd.cost_of(d)
            if cost == float("inf"):
                continue
            yield Correction(r=r, c=c, new_digit=d, cost=cost)


def _state_key(corrections: frozenset[Correction]) -> tuple:
    """Hashable state key. Since corrections don't repeat cells, we can use
    a sorted tuple of (r, c, new_digit)."""
    return tuple(sorted((c.r, c.c, c.new_digit) for c in corrections))


def search_corrections(
    base_puzzle: np.ndarray,
    readings: dict[tuple[int, int], CellReading],
    mode: str = "ucs",
    csp_mode: str = "ac3",
    max_corrections: int = MAX_CORRECTIONS,
    require_unique: bool = False,
    verbose: bool = False,
) -> SearchResult:
    """Search for the cheapest set of corrections that makes `base_puzzle`
    solvable.

    `mode` is one of "ucs", "greedy", "a*".

    Returns a SearchResult.
    """
    stats = SearchStats()
    trace: list[str] = []

    # Quick goal test on the base puzzle (no corrections).
    ok, sol = _is_goal(base_puzzle, csp_mode, stats, require_unique)
    if ok:
        trace.append("Base puzzle already solvable -- no corrections needed.")
        return SearchResult(found=True, corrections=[], cost=0.0,
                            solution=sol, stats=stats, trace=trace)
    trace.append("Base puzzle unsolvable. Starting search for corrections...")

    # Restrict candidate corrections to cells involved in duplicate-digit
    # conflicts (massive speed-up).
    conflict_cells = _conflict_cells(base_puzzle)
    # Keep only cells that have readings (givens classified by the NN).
    conflict_cells = {c for c in conflict_cells if c in readings}
    if verbose:
        trace.append(f"  conflict cells: {sorted(conflict_cells)}")

    # Priority queue of (priority, tiebreak, state_key, corrections, cost).
    # For UCS:      priority = g (cost so far)
    # For Greedy:   priority = h (heuristic only)
    # For A*:        priority = g + h
    frontier: list[tuple[float, int, tuple, frozenset, float]] = []
    counter = 0
    start = frozenset()
    g0 = 0.0
    h0 = _heuristic_min_cost(readings, base_puzzle) if mode != "ucs" else 0.0
    if mode == "greedy":
        p0 = h0
    elif mode == "a*":
        p0 = g0 + h0
    else:  # ucs
        p0 = g0
    heapq.heappush(frontier, (p0, counter, _state_key(start), start, g0))
    counter += 1
    visited: set[tuple] = set()

    best: Optional[SearchResult] = None

    while frontier:
        stats.max_frontier = max(stats.max_frontier, len(frontier))
        priority, _, key, corrections, g = heapq.heappop(frontier)
        if key in visited:
            continue
        visited.add(key)
        stats.nodes_expanded += 1

        # Goal test at EXPANSION time (not generation). This is correct for
        # UCS/A* (best-first), and is a reasonable approximation for Greedy.
        if corrections:
            puzzle = _build_corrected_puzzle(base_puzzle, corrections)
            ok, sol = _is_goal(puzzle, csp_mode, stats, require_unique)
            if ok:
                if best is None or g < best.cost:
                    best = SearchResult(
                        found=True,
                        corrections=sorted(corrections, key=lambda x: (x.r, x.c)),
                        cost=g,
                        solution=sol,
                        stats=stats,
                        trace=trace + [f"  GOAL: corrections="
                                       + ", ".join(str(c) for c in sorted(corrections,
                                            key=lambda x: (x.r, x.c)))]
                        )
                if mode != "greedy":
                    return best  # UCS/A*: first goal is optimal
                else:
                    return best  # Greedy: first goal is good enough

        if verbose:
            trace.append(f"  expand: cost={g:.4f} corrections="
                         + ", ".join(str(c) for c in sorted(corrections,
                                                              key=lambda x: (x.r, x.c))))

        if len(corrections) >= max_corrections:
            continue

        for corr in _candidate_actions(readings, corrections, conflict_cells):
            new_corr = corrections | {corr}
            new_key = _state_key(new_corr)
            if new_key in visited:
                continue
            new_g = g + corr.cost
            stats.nodes_generated += 1

            # Build the corrected puzzle to compute the heuristic.
            if mode != "ucs":
                puzzle = _build_corrected_puzzle(base_puzzle, new_corr)
                h = _heuristic_min_cost(readings, puzzle)
            else:
                h = 0.0
            if mode == "greedy":
                new_p = h
            elif mode == "a*":
                new_p = new_g + h
            else:
                new_p = new_g
            heapq.heappush(frontier, (new_p, counter, new_key, new_corr, new_g))
            counter += 1

    if best is not None:
        return best
    return SearchResult(found=False, corrections=[], cost=float("inf"),
                       solution=None, stats=stats, trace=trace)


# ---------------------------------------------------------------------------
# Convenience: run all three search modes for the same puzzle and return
# a small comparison table (used by experiments).
# ---------------------------------------------------------------------------

def compare_search_modes(
    base_puzzle: np.ndarray,
    readings: dict[tuple[int, int], CellReading],
    csp_mode: str = "ac3",
    max_corrections: int = MAX_CORRECTIONS,
) -> dict[str, SearchResult]:
    out = {}
    for mode in ("ucs", "greedy", "a*"):
        out[mode] = search_corrections(
            base_puzzle=base_puzzle,
            readings=readings,
            mode=mode,
            csp_mode=csp_mode,
            max_corrections=max_corrections,
        )
    return out
