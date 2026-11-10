"""csp.py -- Sudoku as a Constraint Satisfaction Problem.

Project 7.

The CSP has 81 variables (one per cell), values in {1..9} for blank cells
and a single fixed value for given clues. Three algorithms are implemented
as **switchable** parts (per common-checklist deliverable #6):

- `Backtracking` (BT)
- `Backtracking + Forward Checking` (BT+FC)
- `Backtracking + Forward Checking + AC-3` (BT+FC+AC-3)

Each run reports counters:
- `assignments` -- the number of variable assignments tried
- `backtracks`   -- the number of times an assignment had to be undone
- `nodes`        -- the number of search nodes expanded (== assignments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sudoku constants
# ---------------------------------------------------------------------------

DIGITS = list(range(1, 10))  # 1..9
N = 9


def _units_for(r: int, c: int) -> list[list[tuple[int, int]]]:
    """Return the row, column and 3x3 box that contain (r, c)."""
    row = [(r, cc) for cc in range(N)]
    col = [(rr, c) for rr in range(N)]
    br, bc = (r // 3) * 3, (c // 3) * 3
    box = [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]
    return [row, col, box]


def _all_units() -> list[list[tuple[int, int]]]:
    units = []
    for r in range(N):
        units.append([(r, c) for c in range(N)])
    for c in range(N):
        units.append([(r, c) for r in range(N)])
    for br in range(0, N, 3):
        for bc in range(0, N, 3):
            units.append([(br + dr, bc + dc)
                          for dr in range(3) for dc in range(3)])
    return units


PEERS: dict[tuple[int, int], set[tuple[int, int]]] = {}
for _r in range(N):
    for _c in range(N):
        peers = set()
        for unit in _units_for(_r, _c):
            peers.update(unit)
        peers.discard((_r, _c))
        PEERS[(_r, _c)] = peers
del _r, _c


# ---------------------------------------------------------------------------
# CSP state
# ---------------------------------------------------------------------------

@dataclass
class CSPStats:
    assignments: int = 0
    backtracks: int = 0
    nodes: int = 0
    ac3_calls: int = 0
    fc_calls: int = 0
    rule_violations: int = 0

    def as_dict(self) -> dict:
        return {"assignments": self.assignments, "backtracks": self.backtracks,
                "nodes": self.nodes, "ac3_calls": self.ac3_calls,
                "fc_calls": self.fc_calls,
                "rule_violations": self.rule_violations}


@dataclass
class SudokuCSP:
    """A Sudoku CSP. `puzzle` is the 9x9 array of clue values (0 = blank)."""

    puzzle: np.ndarray
    stats: CSPStats = field(default_factory=CSPStats)

    # Derived state (rebuilt on reset).
    domains: dict[tuple[int, int], set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reset_domains()

    # ------------------------------------------------------------------
    # Domain construction
    # ------------------------------------------------------------------

    def reset_domains(self) -> None:
        self.domains = {}
        for r in range(N):
            for c in range(N):
                v = int(self.puzzle[r, c])
                if v != 0:
                    self.domains[(r, c)] = {v}
                else:
                    self.domains[(r, c)] = set(DIGITS)

    def clone(self) -> "SudokuCSP":
        csp = SudokuCSP(puzzle=self.puzzle.copy())
        csp.domains = {k: set(v) for k, v in self.domains.items()}
        csp.stats = CSPStats()
        return csp

    # ------------------------------------------------------------------
    # Consistency helpers
    # ------------------------------------------------------------------

    def is_consistent_assignment(self, r: int, c: int, v: int) -> bool:
        """Check whether assigning v to (r, c) is locally consistent given
        the *current* domains of its peers. Used by plain BT (no propagation)."""
        for (pr, pc) in PEERS[(r, c)]:
            if self.domains[(pr, pc)] == {v} and (pr, pc) != (r, c):
                self.stats.rule_violations += 1
                return False
        return True

    def has_empty_domain(self) -> bool:
        return any(len(s) == 0 for s in self.domains.values())

    # ------------------------------------------------------------------
    # Forward Checking
    # ------------------------------------------------------------------

    def forward_check(self, r: int, c: int, v: int) -> list[tuple[tuple[int, int], int]]:
        """After assigning v to (r, c), remove v from the domain of each
        blank peer. Returns the list of (peer, removed_value) so the change
        can be undone on backtrack."""
        removed: list[tuple[tuple[int, int], int]] = []
        for (pr, pc) in PEERS[(r, c)]:
            if v in self.domains[(pr, pc)] and len(self.domains[(pr, pc)]) > 1:
                self.domains[(pr, pc)].discard(v)
                removed.append(((pr, pc), v))
                if len(self.domains[(pr, pc)]) == 0:
                    # We will detect this on the next pick; but record it.
                    pass
        self.stats.fc_calls += 1
        return removed

    def restore(self, removed: list[tuple[tuple[int, int], int]]) -> None:
        for (cell, val) in removed:
            self.domains[cell].add(val)

    # ------------------------------------------------------------------
    # AC-3 (arc consistency)
    # ------------------------------------------------------------------

    def ac3(self) -> bool:
        """Run AC-3 over the Sudoku constraint graph.

        Each binary constraint "x != y" between two peers (cells) is enforced
        by removing from x's domain any value that has no support in y's
        domain. Returns False if some domain becomes empty (inconsistent).
        """
        self.stats.ac3_calls += 1
        # Queue of (x, y) arcs.
        queue: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for x in self.domains:
            for y in PEERS[x]:
                queue.append((x, y))

        while queue:
            x, y = queue.pop()
            if self._revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in PEERS[x]:
                    if z != y:
                        queue.append((z, x))
        return True

    def _revise(self, x: tuple[int, int], y: tuple[int, int]) -> bool:
        """Return True if x's domain was revised."""
        revised = False
        for v in list(self.domains[x]):
            # x != y constraint: v needs at least one value in y's domain
            # that is different from v.
            support = any(u != v for u in self.domains[y])
            if not support:
                self.domains[x].discard(v)
                revised = True
        return revised

    # ------------------------------------------------------------------
    # Variable selection heuristics (MRV)
    # ------------------------------------------------------------------

    def pick_unassigned(self) -> Optional[tuple[int, int]]:
        """Return the unassigned cell with the smallest domain (MRV).
        Returns None if all cells are assigned."""
        best, best_size = None, 99
        for cell, dom in self.domains.items():
            sz = len(dom)
            if sz == 1 and self.puzzle[cell] != 0:
                continue  # clue, not "unassigned"
            if 1 < sz < best_size:
                best, best_size = cell, sz
                if sz == 2:
                    break  # can't be smaller
        return best

    def unassigned_count(self) -> int:
        return sum(1 for cell, dom in self.domains.items()
                   if len(dom) > 1 and self.puzzle[cell] == 0)

    # ------------------------------------------------------------------
    # Public solve API
    # ------------------------------------------------------------------

    def solve(self, mode: str = "bt", time_limit_s: float = 30.0) -> Optional[np.ndarray]:
        """Solve the puzzle.

        `mode` is one of:
          - "bt"     -- plain backtracking
          - "fc"     -- backtracking + forward checking
          - "ac3"    -- backtracking + forward checking + AC-3 preprocessing
                        (and maintained during search via re-running ac3)
          - "bt+fc"      -- alias for "fc"
          - "bt+fc+ac3"  -- alias for "ac3"

        Returns the solved grid (9x9 int array) or None if no solution
        exists (e.g. two equal givens in the same row).
        """
        import time
        self.stats = CSPStats()
        self.reset_domains()

        # Sanity check: any contradictory givens?
        for r in range(N):
            for c in range(N):
                v = int(self.puzzle[r, c])
                if v != 0 and not self.is_consistent_assignment(r, c, v):
                    return None  # contradictory givens -> impossible

        if mode in ("ac3", "bt+fc+ac3"):
            if not self.ac3():
                return None  # AC-3 found an inconsistency

        deadline = time.time() + time_limit_s
        ok = self._backtrack(mode, deadline)
        if not ok:
            return None

        # Reconstruct the grid.
        out = np.zeros((N, N), dtype=np.int8)
        for (r, c), dom in self.domains.items():
            out[r, c] = next(iter(dom)) if len(dom) == 1 else 0
        return out

    def _backtrack(self, mode: str, deadline: float) -> bool:
        import time
        cell = self.pick_unassigned()
        if cell is None:
            return True  # all assigned
        if time.time() > deadline:
            return False

        self.stats.nodes += 1
        r, c = cell
        for v in sorted(self.domains[cell]):
            self.stats.assignments += 1
            if mode == "bt":
                # Plain backtracking: only local consistency check.
                if not self.is_consistent_assignment(r, c, v):
                    continue
                old_dom = self.domains[cell]
                self.domains[cell] = {v}
                if self._backtrack(mode, deadline):
                    return True
                self.domains[cell] = old_dom
                self.stats.backtracks += 1
            else:
                # FC and AC-3 modes: snapshot BEFORE making any changes so we
                # can fully restore on backtrack (forward_check + AC-3 reduce
                # peer domains, not just the assigned cell).
                snapshot = {k: set(d) for k, d in self.domains.items()}
                self.domains[cell] = {v}
                # Local consistency: any peer domain already == {v}?
                conflict = any(self.domains[p] == {v} for p in PEERS[(r, c)])
                ok = False
                if not conflict:
                    self.forward_check(r, c, v)
                    if mode in ("ac3", "bt+fc+ac3"):
                        ok = self.ac3()
                    else:
                        ok = not self.has_empty_domain()
                if ok:
                    if self._backtrack(mode, deadline):
                        return True
                # Restore from snapshot (covers cell + all peers).
                self.domains = {k: set(d) for k, d in snapshot.items()}
                self.stats.backtracks += 1
        return False

    # ------------------------------------------------------------------
    # Counting solutions (for uniqueness testing)
    # ------------------------------------------------------------------

    def count_solutions(self, mode: str = "ac3", cap: int = 2) -> int:
        """Count solutions up to `cap`. Used to test puzzle uniqueness and to
        implement the optional extension "require exactly one solution"."""
        self.stats = CSPStats()
        self.reset_domains()
        if mode in ("ac3", "bt+fc+ac3"):
            if not self.ac3():
                return 0
        return self._count(mode, cap, 0)

    def _count(self, mode: str, cap: int, found: int) -> int:
        if found >= cap:
            return found
        cell = self.pick_unassigned()
        if cell is None:
            return found + 1
        r, c = cell
        for v in sorted(self.domains[cell]):
            self.stats.assignments += 1
            if mode == "bt":
                if not self.is_consistent_assignment(r, c, v):
                    continue
                old_dom = self.domains[cell]
                self.domains[cell] = {v}
                found = self._count(mode, cap, found)
                self.domains[cell] = old_dom
                if found >= cap:
                    return found
                self.stats.backtracks += 1
            else:
                snapshot = {k: set(d) for k, d in self.domains.items()}
                self.domains[cell] = {v}
                conflict = any(self.domains[p] == {v} for p in PEERS[(r, c)])
                ok = False
                if not conflict:
                    self.forward_check(r, c, v)
                    if mode in ("ac3", "bt+fc+ac3"):
                        ok = self.ac3()
                    else:
                        ok = not self.has_empty_domain()
                if ok:
                    found = self._count(mode, cap, found)
                self.domains = {k: set(d) for k, d in snapshot.items()}
                if found >= cap:
                    return found
                self.stats.backtracks += 1
        return found
