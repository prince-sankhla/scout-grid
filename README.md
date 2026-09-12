# IR-01: Energy-Aware Grid Exploration

This repository contains a pure software simulation for the IR-01 problem.

## Run it

```bash
pip install numpy matplotlib
python scout_grid.py
```

## What the program does

`scout_grid.py` creates a deterministic 30x30 grid with NumPy PCG64 and seed `20260911`. Every non-base cell is independently made an obstacle with probability `0.20`. The grid is regenerated until at least 70% of all free cells are reachable from base `(0, 0)` using 4-directional movement.

The controller is genuinely online. It receives only the revealed map, its current position, previous action history, and remaining energy. It never receives the hidden grid, the total reachable-cell count, or the status of an unrevealed cell.

At every position, the simulator reveals the true occupancy of every cell within Manhattan distance 2. The controller then chooses what to do using only that revealed information.

## Exploration strategy

The controller uses a simple frontier strategy. A frontier is a cell that is already known to be free and has at least one unknown neighbor. The controller moves through known-free cells to the nearest reachable frontier, then tries an unknown neighboring cell. This makes the behavior easy to explain and fully online.

## Energy safety: the important rule

Before exploring from the current position, the controller finds the shortest route back to base using only cells it already knows are free. Call that distance `d`.

An exploratory move costs 1 energy. The destination might be an obstacle, in which case the position does not change, or it might be free, in which case the robot moves one cell farther away.

To guarantee that exploration never strands the robot, the controller requires:

```text
remaining_energy >= d + 3
```

The extra 3 has a clear meaning:

- 1 energy unit pays for the exploratory move.
- Up to `d + 1` energy units are enough to return through the known route even when the unknown destination is free.
- 1 additional energy unit is reserved so that, after reaching base, the run can still issue `DONE` before energy reaches zero.

When this condition is no longer true, the controller stops exploring and follows its known-free return path to base instead.

For ordinary movement through already-known-free cells, the controller performs the same check after the proposed move: after spending 1 energy, enough must remain to return to base and still keep 1 unit available for `DONE`.

## Coverage and CkEk

The simulator computes true reachable free cells for evaluation, but the controller never sees that information.

Coverage is:

```text
observed reachable free cells / total reachable free cells
```

The final score value is:

```text
CkEk = coverage * (remaining_energy / 400)
```

when the controller successfully returns to base and outputs `DONE`. Otherwise the remaining-energy fraction is defined as 0, so `CkEk = 0`.

## Logged data

The local runner stores:

- `trajectory`: every position visited,
- `coverage_over_time`: coverage after every action,
- `energy_over_time`: remaining energy after every action,
- `actions`: MOVE / ATTEMPT_OBSTACLE / DONE,
- final coverage, return status, and CkEk.

The matplotlib figure shows the generated evaluation grid, the path, and the base.

## Files

- `scout_grid.py` — complete ready-to-run simulator, online controller, metrics, and visualization.
- `README.md` — simple explanation of the approach and safety rule.
