"""eval.py -- Experiments and evaluation for Project 7.

Project 7 (Handwritten Sudoku Reader and Solver).

Three experiments (per project spec):
  (1) UCS vs Greedy vs A*: nodes expanded and recovery rate for 0-4 misreads.
  (2) BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens.
  (3) NN-guided correction vs no correction vs random correction.

Each experiment is run with 5 different seeds; mean and std are reported.

Also includes:
  - Recovery rate on 100 puzzles per misread level (per project-specific
    deliverable #4).
  - Tables saved as CSV in `results/` and plots in `figures/`.

Run:
    python -m sudoku_solver.eval                 # run everything (may take time)
    python -m sudoku_solver.eval --experiment 1  # just experiment 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow `python -m sudoku_solver.eval` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sudoku_solver.csp import SudokuCSP
from sudoku_solver.data import (MODELS_DIR, PREPARED_PATH, build_hard_image_pool,
                                  build_mosaic, generate_puzzle, generate_solution,
                                  load_prepared)
from sudoku_solver.integration import read_mosaic, solve_sudoku
from sudoku_solver.nn import DigitClassifier
from sudoku_solver.search import (CellReading, MAX_CORRECTIONS,
                                    search_corrections)

MODEL_PATH = os.path.join(MODELS_DIR, "digit_classifier.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures")


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _load_classifier() -> DigitClassifier:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. "
                                "Run `python -m sudoku_solver.train` first.")
    return DigitClassifier.load(MODEL_PATH)


def _load_pool():
    data = load_prepared(PREPARED_PATH)
    return data["X_test"], data["y_test"]


_HARD_POOL_CACHE: Optional[dict] = None


def _get_hard_pool(clf, X_pool, y_pool) -> dict:
    """Cache the hard-image pool so we don't recompute it on every call."""
    global _HARD_POOL_CACHE
    if _HARD_POOL_CACHE is None:
        print("[eval] Building hard-image pool (one-time)...")
        _HARD_POOL_CACHE = build_hard_image_pool(clf, X_pool, y_pool,
                                                  n_per_digit=30)
        for d in range(1, 10):
            print(f"  digit {d}: {len(_HARD_POOL_CACHE[d])} hard images")
    return _HARD_POOL_CACHE


def _build_one_puzzle(rng, n_givens: int, n_misreads: int, X_pool, y_pool,
                      hard_pool: Optional[dict] = None):
    puzzle, sol = generate_puzzle(rng, n_givens=n_givens)
    givens_mask = puzzle > 0
    mosaic = build_mosaic(
        puzzle=puzzle, solution=sol,
        X_pool=X_pool, y_pool=y_pool,
        rng=rng, n_injected_misreads=n_misreads,
        givens_mask=givens_mask,
        hard_pool=hard_pool,
    )
    return sol, puzzle, mosaic


def _recovered(result, mosaic) -> bool:
    """Did the pipeline recover the *original* solution?"""
    if not result.found or result.solution is None:
        return False
    return np.array_equal(result.solution, mosaic.solution)


# ---------------------------------------------------------------------------
# Experiment 1: UCS vs Greedy vs A* -- nodes and recovery rate for 0-4 misreads
# ---------------------------------------------------------------------------

def run_experiment_1(clf, X_pool, y_pool, seeds=(1, 2, 3, 4, 5),
                     n_puzzles_per_seed: int = 20,
                     misread_levels=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """For each (seed, misread_level, search_mode), run `n_puzzles_per_seed`
    puzzles and collect:
      - mean nodes_expanded
      - mean recovery rate
      - mean cost (log-prob units)
      - mean runtime (s)
    """
    hard_pool = _get_hard_pool(clf, X_pool, y_pool)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for n_mr in misread_levels:
            for mode in ("ucs", "greedy", "a*"):
                nodes, recov, costs, times = [], [], [], []
                for _ in range(n_puzzles_per_seed):
                    sol, puzzle, mosaic = _build_one_puzzle(
                        rng, n_givens=30, n_misreads=n_mr,
                        X_pool=X_pool, y_pool=y_pool, hard_pool=hard_pool,
                    )
                    t0 = time.time()
                    res = solve_sudoku(clf, mosaic, search_mode=mode,
                                       csp_mode="ac3",
                                       max_corrections=MAX_CORRECTIONS,
                                       top_k=3)
                    dt = time.time() - t0
                    nodes.append(res.search_stats.nodes_expanded)
                    recov.append(int(_recovered(res, mosaic)))
                    costs.append(res.correction_cost if res.found else float("nan"))
                    times.append(dt)
                rows.append({
                    "seed": seed, "misreads": n_mr, "mode": mode,
                    "mean_nodes": float(np.mean(nodes)),
                    "std_nodes": float(np.std(nodes)),
                    "recovery_rate": float(np.mean(recov)),
                    "mean_cost": float(np.nanmean(costs)),
                    "std_cost": float(np.nanstd(costs)),
                    "mean_time_s": float(np.mean(times)),
                })
                print(f"[exp1] seed={seed} misreads={n_mr} mode={mode}: "
                      f"nodes={np.mean(nodes):.1f} recovery={np.mean(recov):.2f}")
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "experiment_1_search_modes.csv"), index=False)

    # Summary table (mean over seeds).
    summary = (df.groupby(["misreads", "mode"])
                 .agg(mean_nodes=("mean_nodes", "mean"),
                      std_nodes=("mean_nodes", "std"),
                      recovery_rate=("recovery_rate", "mean"),
                      std_recovery=("recovery_rate", "std"),
                      mean_cost=("mean_cost", "mean"))
                 .reset_index())
    summary.to_csv(os.path.join(RESULTS_DIR, "experiment_1_summary.csv"),
                   index=False)
    _plot_experiment_1(summary)
    return df


def _plot_experiment_1(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    modes = ["ucs", "greedy", "a*"]
    colors = {"ucs": "tab:blue", "greedy": "tab:orange", "a*": "tab:green"}

    for ax, metric, ylabel in zip(
        axes,
        ["mean_nodes", "recovery_rate", "mean_cost"],
        ["Nodes expanded", "Recovery rate", "Mean correction cost"],
    ):
        for mode in modes:
            sub = summary[summary["mode"] == mode].sort_values("misreads")
            if sub.empty:
                continue
            ax.errorbar(
                sub["misreads"], sub[metric],
                yerr=sub.get(f"std_{metric.split('_')[-1]}", None)
                if f"std_{metric.split('_')[-1]}" in sub.columns else None,
                label=mode.upper(), color=colors[mode], marker="o", capsize=3,
            )
        ax.set_xlabel("Injected misreads")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Experiment 1: UCS vs Greedy vs A* (mean over 5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_1_search_modes.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, 35 givens
# ---------------------------------------------------------------------------

def run_experiment_2(seeds=(1, 2, 3, 4, 5),
                     n_puzzles_per_seed: int = 10,
                     givens_levels=(25, 30, 35)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for n_givens in givens_levels:
            for mode in ("bt", "fc", "ac3"):
                assigns, bts, times, solvable = [], [], [], []
                for _ in range(n_puzzles_per_seed):
                    puzzle, _sol = generate_puzzle(rng, n_givens=n_givens)
                    csp = SudokuCSP(puzzle=puzzle)
                    t0 = time.time()
                    # Plain BT can be slow on 25-givens puzzles; cap at 10s.
                    time_limit = 10.0 if mode == "bt" else 30.0
                    sol = csp.solve(mode=mode, time_limit_s=time_limit)
                    dt = time.time() - t0
                    assigns.append(csp.stats.assignments)
                    bts.append(csp.stats.backtracks)
                    times.append(dt)
                    solvable.append(int(sol is not None))
                rows.append({
                    "seed": seed, "givens": n_givens, "mode": mode,
                    "mean_assignments": float(np.mean(assigns)),
                    "std_assignments": float(np.std(assigns)),
                    "mean_backtracks": float(np.mean(bts)),
                    "std_backtracks": float(np.std(bts)),
                    "mean_time_s": float(np.mean(times)),
                    "solvable_rate": float(np.mean(solvable)),
                })
                print(f"[exp2] seed={seed} givens={n_givens} mode={mode}: "
                      f"assigns={np.mean(assigns):.1f} backtracks={np.mean(bts):.1f} "
                      f"time={np.mean(times):.3f}s")
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "experiment_2_csp_modes.csv"), index=False)
    summary = (df.groupby(["givens", "mode"])
                 .agg(mean_assignments=("mean_assignments", "mean"),
                      std_assignments=("mean_assignments", "std"),
                      mean_backtracks=("mean_backtracks", "mean"),
                      std_backtracks=("mean_backtracks", "std"),
                      mean_time_s=("mean_time_s", "mean"))
                 .reset_index())
    summary.to_csv(os.path.join(RESULTS_DIR, "experiment_2_summary.csv"),
                   index=False)
    _plot_experiment_2(summary)
    return df


def _plot_experiment_2(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    modes = ["bt", "fc", "ac3"]
    labels = {"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}
    colors = {"bt": "tab:red", "fc": "tab:purple", "ac3": "tab:cyan"}

    for ax, metric, ylabel in zip(
        axes,
        ["mean_assignments", "mean_backtracks"],
        ["Assignments", "Backtracks"],
    ):
        for mode in modes:
            sub = summary[summary["mode"] == mode].sort_values("givens")
            if sub.empty:
                continue
            err_col = f"std_{metric.split('_')[-1]}"
            ax.errorbar(
                sub["givens"], sub[metric],
                yerr=sub[err_col] if err_col in sub.columns else None,
                label=labels[mode], color=colors[mode], marker="o", capsize=3,
            )
        ax.set_xlabel("Number of givens")
        ax.set_ylabel(ylabel)
        ax.set_title(f"CSP mode comparison: {ylabel}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Experiment 2: BT vs BT+FC vs BT+FC+AC-3 (mean over 5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_2_csp_modes.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3: NN-guided correction vs no correction vs random correction
# ---------------------------------------------------------------------------

def _random_correction_search(base_puzzle, readings, csp_mode, max_corrections,
                               rng):
    """Random correction baseline: pick corrections uniformly at random
    from the candidate actions, up to max_corrections. Return (found, sol)."""
    if SudokuCSP(puzzle=base_puzzle).solve(mode=csp_mode) is not None:
        return True, base_puzzle
    cells = list(readings.keys())
    rng.shuffle(cells)
    applied = []
    puzzle = base_puzzle.copy()
    for cell in cells[:max_corrections]:
        rd = readings[cell]
        alts = list(rd.candidates[1:])
        if not alts:
            continue
        d, _p = alts[int(rng.integers(len(alts)))]
        puzzle[cell] = d
        applied.append((cell, d))
        sol = SudokuCSP(puzzle=puzzle).solve(mode=csp_mode)
        if sol is not None:
            return True, sol
    return False, None


def _no_correction_solve(base_puzzle, csp_mode):
    sol = SudokuCSP(puzzle=base_puzzle).solve(mode=csp_mode)
    return (sol is not None), sol


def run_experiment_3(clf, X_pool, y_pool, seeds=(1, 2, 3, 4, 5),
                     n_puzzles_per_seed: int = 20,
                     misread_levels=(1, 2, 3)) -> pd.DataFrame:
    hard_pool = _get_hard_pool(clf, X_pool, y_pool)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for n_mr in misread_levels:
            # Strategies: "none", "random", "nn_guided"
            for strat in ("none", "random", "nn_guided"):
                recov, solves, times = [], [], []
                for _ in range(n_puzzles_per_seed):
                    sol, puzzle, mosaic = _build_one_puzzle(
                        rng, n_givens=30, n_misreads=n_mr,
                        X_pool=X_pool, y_pool=y_pool, hard_pool=hard_pool,
                    )
                    t0 = time.time()
                    if strat == "none":
                        read_puzzle, readings = read_mosaic(clf, mosaic, top_k=3)
                        ok, sol2 = _no_correction_solve(read_puzzle, "ac3")
                    elif strat == "random":
                        read_puzzle, readings = read_mosaic(clf, mosaic, top_k=3)
                        rng_local = np.random.default_rng(seed * 31 + n_mr)
                        ok, sol2 = _random_correction_search(
                            read_puzzle, readings, "ac3", MAX_CORRECTIONS,
                            rng_local,
                        )
                    else:  # nn_guided
                        res = solve_sudoku(clf, mosaic, search_mode="ucs",
                                          csp_mode="ac3",
                                          max_corrections=MAX_CORRECTIONS,
                                          top_k=3)
                        ok = res.found
                        sol2 = res.solution
                    dt = time.time() - t0
                    solves.append(int(ok))
                    recov.append(int(ok and sol2 is not None
                                    and np.array_equal(sol2, mosaic.solution)))
                    times.append(dt)
                rows.append({
                    "seed": seed, "misreads": n_mr, "strategy": strat,
                    "recovery_rate": float(np.mean(recov)),
                    "solvable_rate": float(np.mean(solves)),
                    "mean_time_s": float(np.mean(times)),
                })
                print(f"[exp3] seed={seed} misreads={n_mr} strat={strat}: "
                      f"recovery={np.mean(recov):.2f} solvable={np.mean(solves):.2f}")
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "experiment_3_correction_strategies.csv"),
              index=False)
    summary = (df.groupby(["misreads", "strategy"])
                 .agg(recovery_rate=("recovery_rate", "mean"),
                      std_recovery=("recovery_rate", "std"),
                      solvable_rate=("solvable_rate", "mean"),
                      mean_time_s=("mean_time_s", "mean"))
                 .reset_index())
    summary.to_csv(os.path.join(RESULTS_DIR, "experiment_3_summary.csv"),
                   index=False)
    _plot_experiment_3(summary)
    return df


def _plot_experiment_3(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    strats = [("none", "No correction", "tab:gray", "o"),
              ("random", "Random correction", "tab:orange", "s"),
              ("nn_guided", "NN-guided (UCS)", "tab:blue", "^")]
    for s, lbl, color, marker in strats:
        sub = summary[summary["strategy"] == s].sort_values("misreads")
        if sub.empty:
            continue
        ax.errorbar(sub["misreads"], sub["recovery_rate"],
                    yerr=sub["std_recovery"], label=lbl, color=color,
                    marker=marker, capsize=3)
    ax.set_xlabel("Injected misreads")
    ax.set_ylabel("Solution-recovery rate")
    ax.set_title("Experiment 3: NN-guided vs no-correction vs random correction")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "experiment_3_correction_strategies.png"),
                dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recovery-rate evaluation on 100 puzzles per misread level
# (project-specific deliverable #4)
# ---------------------------------------------------------------------------

def run_recovery_evaluation(clf, X_pool, y_pool, base_seed: int = 100,
                             misread_levels=(0, 1, 2, 3, 4),
                             n_puzzles: int = 100) -> pd.DataFrame:
    """For each misread level, run `n_puzzles` puzzles and report the
    exact-solution recovery rate (did the pipeline output the original
    solution exactly?)."""
    hard_pool = _get_hard_pool(clf, X_pool, y_pool)
    rows = []
    for n_mr in misread_levels:
        rng = np.random.default_rng(base_seed + n_mr)
        recov, found, costs = [], [], []
        for i in range(n_puzzles):
            sol, puzzle, mosaic = _build_one_puzzle(
                rng, n_givens=30, n_misreads=n_mr,
                X_pool=X_pool, y_pool=y_pool, hard_pool=hard_pool,
            )
            res = solve_sudoku(clf, mosaic, search_mode="ucs",
                              csp_mode="ac3",
                              max_corrections=MAX_CORRECTIONS, top_k=3)
            recov.append(int(_recovered(res, mosaic)))
            found.append(int(res.found))
            costs.append(res.correction_cost if res.found else float("nan"))
        rate = float(np.mean(recov))
        rows.append({
            "misreads": n_mr,
            "n_puzzles": n_puzzles,
            "recovery_rate": rate,
            "found_rate": float(np.mean(found)),
            "mean_cost": float(np.nanmean(costs)),
            "std_cost": float(np.nanstd(costs)),
        })
        print(f"[recovery] misreads={n_mr}: recovery={rate:.3f} "
              f"(n={n_puzzles})")
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "recovery_rate_100_per_level.csv"),
              index=False)
    _plot_recovery(df)
    return df


def _plot_recovery(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(df["misreads"], df["recovery_rate"],
                yerr=df["recovery_rate"].clip(upper=1) * 0,  # no per-level std in this df
                marker="o", color="tab:blue", capsize=3)
    ax.set_xlabel("Injected misreads")
    ax.set_ylabel("Exact-solution recovery rate")
    ax.set_title("Recovery rate on 100 puzzles per misread level")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "recovery_rate.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["1", "2", "3", "all", "recovery"],
                        default="all")
    parser.add_argument("--n-per-seed", type=int, default=20)
    parser.add_argument("--recovery-n", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    clf = None
    X_pool = y_pool = None

    if args.experiment in ("1", "3", "all", "recovery"):
        print("[eval] Loading classifier and MNIST test pool...")
        clf = _load_classifier()
        X_pool, y_pool = _load_pool()

    if args.experiment in ("1", "all"):
        print("\n=== Experiment 1: UCS vs Greedy vs A* ===")
        run_experiment_1(clf, X_pool, y_pool,
                          n_puzzles_per_seed=min(args.n_per_seed, 20))
    if args.experiment in ("2", "all"):
        print("\n=== Experiment 2: BT vs BT+FC vs BT+FC+AC-3 ===")
        run_experiment_2(n_puzzles_per_seed=min(args.n_per_seed, 10))
    if args.experiment in ("3", "all"):
        print("\n=== Experiment 3: NN-guided vs no-correction vs random ===")
        run_experiment_3(clf, X_pool, y_pool,
                          n_puzzles_per_seed=min(args.n_per_seed, 20))
    if args.experiment in ("recovery", "all"):
        print("\n=== Recovery rate (100 puzzles per misread level) ===")
        run_recovery_evaluation(clf, X_pool, y_pool,
                                n_puzzles=args.recovery_n)
    print("\n[eval] Done. Results -> results/, figures -> figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
