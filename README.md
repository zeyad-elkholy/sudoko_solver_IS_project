# Project 7 -- Handwritten Sudoku Reader and Solver

> A system that reads a Sudoku puzzle written in handwritten digits,
> detects digits the neural network may have misread, corrects the
> cheapest set of misreads so the puzzle becomes solvable, and then solves
> it with a constraint-satisfaction solver. Three components cooperate:
> a **Neural Network** (digit classifier), a **Search** over correction
> sets (UCS / Greedy / A\*), and a **CSP solver** (Backtracking / Forward
> Checking / AC-3).

## Deliverables map (common checklist + project-specific)

| Common deliverable                                | Where to find it |
|---------------------------------------------------|------------------|
| 1.  Source modules (data, nn, search, csp, ...)   | `sudoku_solver/` |
| 2.  Data script `make_data.py`                    | `make_data.py` |
| 3.  Data dictionary                                | `docs/DATA_DICTIONARY.md` |
| 4.  Trained NN + baseline test results             | `models/digit_classifier.pkl`, `models/training_summary.json` |
| 5.  Search w/ counters, tested on small example    | `sudoku_solver/search.py`, `tests/test_pipeline.py` |
| 6.  CSP (BT / FC / AC-3 switchable) w/ counters    | `sudoku_solver/csp.py` |
| 7.  Integrated demo                                | `run_demo.py`, `figures/demo_pipeline.png` |
| 8.  README (this file)                              | `README.md` |
| 9.  Tests (5 normal / 3 edge / 2 impossible)        | `tests/test_pipeline.py` |
| 10. Experiments tables + plots (5 seeds, mean+std)  | `results/`, `figures/experiment_*.png` |
| 11. Report (max 12 pages)                           | `docs/report.pdf` |
| 12. Presentation (10-min + 5-min demo notes)        | `docs/presentation.pptx` |
| 13. Contribution statement + git log                | `docs/contribution_statement.md`, `docs/git_log.txt` |

| Project-specific deliverable (P7)                  | Where to find it |
|---------------------------------------------------|------------------|
| Calibrated classifier vs logistic regression       | `models/training_summary.json`, `figures/nn_confusion_*.png`, `figures/nn_reliability_*.png` |
| Puzzle and mosaic generator with seed              | `sudoku_solver/data.py:generate_puzzle, build_mosaic, build_hard_image_pool` |
| Correction log for each solved puzzle               | `PipelineResult.correction_log()` in `sudoku_solver/integration.py`; printed by `run_demo.py` |
| Recovery rate on 100 puzzles per misread level      | `results/recovery_rate_100_per_level.csv`, `figures/recovery_rate.png` |

## Environment setup (pinned requirements)

Python 3.12 (any 3.10+ works). Pinned versions in `requirements.txt`:

```
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
pytest==9.0.2
torch==2.15.0+cpu     # optional -- falls back to sklearn MLP if absent
```

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (optional) pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Exact run commands

```bash
# 1. Download + prepare MNIST (one-time; ~30s)
python make_data.py

# 2. Train the digit classifier and the LR baseline (~15s on CPU)
python -m sudoku_solver.train --epochs 12

# 3. Run the tests
python -m pytest tests/ -v

# 4. Run the integrated demo
python run_demo.py --seed 1 --misreads 2 --search ucs

# 5. Run all experiments (3 experiments x 5 seeds + recovery on 100 puzzles)
python -m sudoku_solver.eval --experiment all --n-per-seed 10 --recovery-n 100
# Or run individually:
python -m sudoku_solver.eval --experiment 1 --n-per-seed 10
python -m sudoku_solver.eval --experiment 2 --n-per-seed 5
python -m sudoku_solver.eval --experiment 3 --n-per-seed 10
python -m sudoku_solver.eval --experiment recovery --recovery-n 100
```

## Project layout

```
sudoku_solver/
├── README.md
├── requirements.txt
├── make_data.py               # MNIST download + prepare
├── run_demo.py                # integrated demo
├── sudoku_solver/
│   ├── __init__.py
│   ├── data.py                # MNIST loader, puzzle/mosaic generator, hard-image pool
│   ├── nn.py                  # MLP classifier + LR baseline + calibration
│   ├── train.py               # training script (run as: python -m sudoku_solver.train)
│   ├── csp.py                 # Sudoku CSP (BT, FC, AC-3 switchable)
│   ├── search.py              # correction search (UCS, Greedy, A*)
│   ├── integration.py         # end-to-end pipeline
│   └── eval.py                # experiments + recovery evaluation
├── tests/
│   └── test_pipeline.py       # 5 normal + 3 edge + 2 impossible
├── data/                      # MNIST cache (auto-generated)
├── models/                    # trained classifiers + summary
├── results/                   # experiment CSVs
├── figures/                   # experiment plots + demo figure
└── docs/                      # report, presentation, contribution statement, git log
```

## Design summary

**Pipeline.** A handwritten Sudoku is given as a 9x9 mosaic of MNIST digit
images (blank cells are simply absent). The classifier reads each *given*
cell, returning a probability distribution over digits 1-9. The CSP solver
runs AC-3 on the read puzzle; if AC-3 finds a contradiction (an empty domain)
or backtracking fails, the puzzle is unsolvable as read. The correction
search then enumerates correction sets (cell -> alternative digit), using the
negative log-probability ratio as the cost, and the first corrected puzzle
that the CSP can solve wins. If no correction set of size <= 3 makes the
puzzle solvable, the system reports "unreadable" and lists the least
confident cells.

**Maximum 3 corrections** per puzzle (per spec). The **A\* heuristic** is
admissible: it counts the cheapest correction that must be made in each
conflicting row/column/box.

## Reproducibility

* Every random step (MNIST split, puzzle generation, mosaic building, hard-image
  pool) takes a `seed` argument and uses `numpy.random.default_rng(seed)`.
* `make_data.py --seed 42` produces a deterministic 50k/10k/10k split.
* `python -m sudoku_solver.train --seed 42` reproduces the trained model.
* Experiments use seeds 1..5 and report mean ± std across seeds.

## Licence

* **MNIST:** Yann LeCun & Corinna Cortes; public dataset. We redistribute
  nothing -- `make_data.py` downloads it on demand.
* **Project code:** MIT (see `LICENSE` if present).
