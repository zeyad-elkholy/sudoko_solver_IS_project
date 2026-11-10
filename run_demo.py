"""run_demo.py -- Integrated demo of the full pipeline.

Project 7 (Handwritten Sudoku Reader and Solver).

This script:
  1. Loads (or trains) the digit classifier.
  2. Builds a Sudoku puzzle + image mosaic, optionally injecting 1-2 misreads.
  3. Reads the mosaic with the NN, runs the CSP solver, and (if needed) runs
     the correction search.
  4. Prints the input mosaic, the read puzzle, the corrections applied, and
     the final solution.
  5. Saves a figure showing how the three components cooperate:
        mosaic | read puzzle | correction search trace | solved grid

Run:
    python run_demo.py
    python run_demo.py --seed 7 --misreads 2 --search a*
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Allow `python run_demo.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sudoku_solver.data import (DATA_DIR, MODELS_DIR, PREPARED_PATH,
                                  build_hard_image_pool, build_mosaic,
                                  generate_puzzle,
                                  generate_solution, load_prepared)
from sudoku_solver.integration import solve_sudoku
from sudoku_solver.nn import DigitClassifier


MODEL_PATH = os.path.join(MODELS_DIR, "digit_classifier.pkl")
FIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "figures", "demo_pipeline.png")


def _plot_demo(mosaic, read_puzzle, solution, corrections, trace, fig_path):
    """Render the input mosaic, read puzzle, search trace and solution."""
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.25)

    # 1) Mosaic of MNIST digit images (top-left, large).
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    big = np.zeros((9 * 28, 9 * 28), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            img = mosaic.image_for_cell(r, c)
            if img is not None:
                big[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img
    ax1.imshow(big, cmap="gray_r", vmin=0, vmax=1)
    ax1.set_title("Input: handwritten Sudoku mosaic", fontsize=11)
    # Mark injected misreads with red squares.
    for (r, c, _o, _n) in mosaic.injected:
        ax1.add_patch(plt.Rectangle((c * 28 - 1, r * 28 - 1), 28, 28,
                                    fill=False, edgecolor="red", linewidth=1.5))
    ax1.set_xticks([]); ax1.set_yticks([])

    # 2) Read puzzle (top-right).
    ax2 = fig.add_subplot(gs[0, 2])
    _draw_grid(ax2, read_puzzle, title="NN read puzzle",
               highlight=mosaic.injected)
    # 3) Corrections trace (top-right side).
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis("off")
    if corrections:
        msg = "Corrections applied:\n"
        for c in corrections:
            rd = int(read_puzzle[c.r, c.c])
            msg += f"  ({c.r},{c.c}): {rd} -> {c.new_digit}  cost={c.cost:.3f}\n"
    else:
        msg = "No corrections needed\n(puzzle was already solvable)"
    ax3.text(0.0, 0.5, msg, fontsize=10, family="monospace", va="center")
    ax3.set_title("Search result", fontsize=11)

    # 4) Solution (bottom-right).
    ax4 = fig.add_subplot(gs[1, 2:4])
    if solution is not None:
        _draw_grid(ax4, solution, title="Solved Sudoku",
                   corrections=corrections)
    else:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "UNSOLVABLE\nReported 'unreadable'.",
                 ha="center", va="center", fontsize=14, color="red",
                 transform=ax4.transAxes)

    fig.suptitle("Project 7 -- Handwritten Sudoku Reader and Solver\n"
                 "(NN reads -> AC-3 detects conflict -> Search corrects -> CSP solves)",
                 fontsize=12)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] Figure saved to {fig_path}")


def _draw_grid(ax, grid, title="", highlight=None, corrections=None):
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(-0.5, 9, 1))
    ax.set_yticks(np.arange(-0.5, 9, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(False)
    ax.invert_yaxis()
    # Light grid lines, heavy every 3.
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color="black", linewidth=lw)
        ax.axvline(i - 0.5, color="black", linewidth=lw)
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c]) if grid is not None else 0
            if v != 0:
                color = "black"
                weight = "bold"
                # Highlight corrected cells in blue.
                if corrections and any(cc.r == r and cc.c == c for cc in corrections):
                    color = "blue"
                    weight = "bold"
                # Highlight injected (true misread) cells in red.
                if highlight and any(h[0] == r and h[1] == c for h in highlight):
                    color = "red"
                ax.text(c, r, str(v), ha="center", va="center",
                        fontsize=12, color=color, weight=weight)


def main() -> int:
    parser = argparse.ArgumentParser(description="Project 7 demo.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--misreads", type=int, default=2,
                        help="Number of injected misreads (default 2).")
    parser.add_argument("--n-givens", type=int, default=30)
    parser.add_argument("--search", choices=["ucs", "greedy", "a*"],
                        default="ucs")
    parser.add_argument("--csp", choices=["bt", "fc", "ac3"], default="ac3")
    parser.add_argument("--max-corrections", type=int, default=3)
    args = parser.parse_args()

    # 1. Load MNIST.
    if not os.path.exists(PREPARED_PATH):
        print("[demo] MNIST not prepared. Run `python make_data.py` first.")
        return 1
    data = load_prepared(PREPARED_PATH)

    # 2. Load (or quickly train) the classifier.
    if not os.path.exists(MODEL_PATH):
        print("[demo] No trained model found. Run `python -m sudoku_solver.nn.train` first.")
        return 1
    clf = DigitClassifier.load(MODEL_PATH)

    # 3. Build a puzzle + mosaic with injected misreads.
    rng = np.random.default_rng(args.seed)
    puzzle, sol = generate_puzzle(rng, n_givens=args.n_givens)
    givens_mask = puzzle > 0

    # Use a subset of MNIST test images as the digit pool (so we don't pull
    # training images for the mosaic).
    X_pool, y_pool = data["X_test"], data["y_test"]
    hard_pool = build_hard_image_pool(clf, X_pool, y_pool, n_per_digit=30) \
        if args.misreads > 0 else None
    mosaic = build_mosaic(
        puzzle=puzzle, solution=sol,
        X_pool=X_pool, y_pool=y_pool,
        rng=rng, n_injected_misreads=args.misreads,
        givens_mask=givens_mask,
        hard_pool=hard_pool,
    )

    print("\n=== INPUT HANDWRITTEN MOSAIC ===")
    print(f"  {len(mosaic.injected)} injected misread(s).")
    print(f"  Puzzle has {int(mosaic.givens_mask.sum())} givens.")

    # 4. Run the full pipeline.
    result = solve_sudoku(
        classifier=clf, mosaic=mosaic,
        search_mode=args.search, csp_mode=args.csp,
        max_corrections=args.max_corrections, top_k=3,
        verbose=True,
    )

    print("\n=== READ PUZZLE ===")
    _print_grid(result.read_puzzle)

    if result.found:
        print("\n=== SOLUTION ===")
        _print_grid(result.solution)
        print(f"\nCorrections applied: {len(result.corrections)}  "
              f"cost={result.correction_cost:.4f}")
        for corr in result.corrections:
            rd = int(result.read_puzzle[corr.r, corr.c])
            print(f"  cell ({corr.r},{corr.c}): {rd} -> {corr.new_digit}  cost={corr.cost:.4f}")
        print(f"\nSearch stats: {result.search_stats.as_dict()}")
        print(f"CSP stats: {result.csp_stats}")
    else:
        print("\n=== UNSOLVABLE ===")
        print("Reported 'unreadable'. Least-confident cells:")
        for (cell, p) in result.least_confident_cells:
            print(f"  cell {cell}: confidence={p:.4f}")

    # 5. Save the figure.
    _plot_demo(mosaic, result.read_puzzle, result.solution,
               result.corrections, result.search_trace, FIG_PATH)
    print(f"\n[demo] Done. Figure -> {FIG_PATH}")
    return 0


def _print_grid(grid: np.ndarray) -> None:
    g = np.where(grid == 0, ".", grid.astype(str))
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("------+-------+------")
        parts = [list(g[r, i:i + 3]) for i in range(0, 9, 3)]
        print(" | ".join(" ".join(p) for p in parts))


if __name__ == "__main__":
    raise SystemExit(main())
