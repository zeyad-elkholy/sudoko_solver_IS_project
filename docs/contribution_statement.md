# Individual Contribution Statement

**Project 7 -- Handwritten Sudoku Reader and Solver**

Per project specification, each team member's contribution (max 200 words each)
is listed below. The git log (`git_log.txt`) shows the matching commit
sequence.

---

## Member A -- Data pipeline and MNIST loading

I designed and implemented the MNIST data pipeline. I wrote `make_data.py`
and `sudoku_solver/data.py`, including the IDX-format parser, the
deterministic 50k/10k/10k split with seed control, and the digit-0 filter
required by the spec. I built the puzzle generator (`generate_puzzle`) using
randomised backtracking with a uniqueness check, and the mosaic builder that
places a real MNIST image of the correct digit at each given cell. I also
implemented `build_hard_image_pool` -- the documented rule that selects
real MNIST images the trained NN naturally misclassifies, so injected
misreads are realistic. I wrote the data dictionary. *(166 words)*

## Member B -- Neural network and calibration

I implemented `sudoku_solver/nn.py` and `train.py`. The classifier is a
3-hidden-layer PyTorch MLP (28x28 -> 128 -> 128 -> 64 -> 9) trained with
Adam for 12 epochs, achieving 96.0% test accuracy on the digit-1..9 subset.
I added a multinomial logistic regression baseline (91.6% accuracy). I
implemented temperature scaling for calibration, reducing the Expected
Calibration Error from 0.0132 to 0.0032. I generated the confusion matrices
and reliability diagrams for both classifiers. I also wrote the
`CellReading` cost function used by the search. *(96 words)*

## Member C -- CSP solver

I implemented `sudoku_solver/csp.py` with three switchable inference modes:
plain Backtracking, Backtracking + Forward Checking, and Backtracking +
Forward Checking + AC-3. The solver maintains counters for assignments,
backtracks, AC-3 calls and FC calls. I implemented MRV variable selection,
the unit/peer constraint graph, and the snapshot-restore logic that lets
AC-3 propagate domain reductions during search without leaking into peer
domains on backtrack. I also implemented `count_solutions` for the optional
uniqueness goal. I wrote the CSP tests (3 of the 5 normal cases and both
impossible cases). *(100 words)*

## Member D -- Search and integration

I implemented `sudoku_solver/search.py` (UCS, Greedy Best-First, A\*) and
`sudoku_solver/integration.py`. The state is a frozenset of corrections;
the cost of changing a cell from its top-1 prediction to an alternative is
the negative log-probability ratio. I derived an admissible A\* heuristic
based on the cheapest correction needed per duplicate-digit conflict. I
added the conflict-cell pruning that restricts candidate corrections to
cells actually involved in duplicate-digit conflicts, which reduced the
goal-test count by ~150x. I wrote `run_demo.py` and the search-related
tests. *(92 words)*

## Member E -- Experiments and report

I implemented `sudoku_solver/eval.py` covering the three required
experiments (UCS vs Greedy vs A\*; BT vs BT+FC vs BT+FC+AC-3; NN-guided vs
no-correction vs random) plus the recovery-rate evaluation on 100 puzzles
per misread level. I ran each experiment over 5 seeds and produced the
summary CSVs and plots with mean and standard deviation. I wrote the
12-page report and the 10-minute presentation, and assembled this
contribution statement and the git log. *(78 words)*

---

Total: 5 members, 532 words across all statements.
