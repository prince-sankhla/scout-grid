# IR-01: Energy-Aware Robot Grid Exploration

This repository contains a pure software simulation for the IR-01 problem.

## What the program does

`scout_grid.py` creates a deterministic 30x30 grid using NumPy's PCG64 random number generator with seed `20260911`. Every non-base cell becomes an obstacle independently with probability `0.20`. If fewer than 70% of all free cells are reachable from the base `(0, 0)`, the program generates another grid using the same RNG stream until the condition is satisfied.

The controller does not receive the hidden grid. It only sees:

- its current position,
- cells that have already been revealed,
- its previous actions, and
- its remaining energy.

At every position, the simulator reveals the true occupancy of every cell whose Manhattan distance from the current position is at most 2.

The controller uses a simple online frontier strategy. A frontier is a known-free cell that has at least one still-unknown neighbor. The controller walks through known-free cells toward the nearest frontier, then tries an unknown neighboring cell to reveal more of the grid. A failed move into an obstacle costs one energy unit and leaves the position unchanged, exactly as specified.

## Energy safety rule

Before making an exploratory move, the controller finds the shortest route from its current position back to the base using only cells it already knows are free.

Suppose that route needs `d` steps.

A move costs 1 energy. The controller requires:

`remaining_energy >= d + 2`

before trying an unknown cell.

Why the extra 2? There are two cases:

1. The unknown cell is free. The controller spends 1 energy and moves there. It can still get home through the current known route plus the new step.
2. The unknown cell is an obstacle. The controller spends 1 energy but stays where it is. It still needs the original `d` steps to get home.

So requiring `d + 2` means that after spending the next energy unit, there is still enough energy for the known return path even in the bad case. When this condition is no longer true, the controller stops exploring and turns back toward the base instead.

## Important online-information rule

The true grid exists only inside the simulator/environment. The controller never receives the true grid, the total number of reachable cells, or the status of an unrevealed cell. It makes decisions from the revealed map only.

The simulator is allowed to use the hidden grid to execute actions and to calculate the final evaluation metrics.

## Output

Running:

```bash
python scout_grid.py
```

prints:

- final coverage percentage,
- whether the run returned successfully to the base,
- `CkEk`, where `CkEk = coverage * (remaining_energy / 400)` for a successful return and is `0` otherwise,
- number of actions,
- final remaining energy, and
- number of grid-generation attempts.

It also displays a matplotlib plot of the generated grid, obstacles, the path, and the base.

## Dependencies

```bash
pip install numpy matplotlib
```

Python 3.10+ is recommended.
