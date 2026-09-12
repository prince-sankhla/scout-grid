# RoverMind — IR-01 Energy-Aware Robot Grid Exploration

RoverMind is a pure Python simulation for IR-01. It does not use any external system; the moving agent is only a position variable inside the virtual grid.

## Run

```bash
pip install -r requirements.txt
python rovermind.py
```

The program creates the required 30x30 map, reveals only the allowed local view, runs an online greedy explorer, prints metrics, writes step-by-step logs, and saves a Matplotlib visualization.

## Approach

The explorer looks for the nearest **frontier**: a cell already known to be free that touches at least one unknown cell. It walks to a nearby frontier through known-free cells and then tests one unknown neighbor. When several frontiers are equally close, it prefers the one with more unknown neighbors because that can reveal more new information.

The controller never receives the hidden full map, the true reachable-cell set, or the occupancy of an unrevealed cell. Its decision function receives only the revealed map, current position, previous action history, and remaining energy.

## Energy safety

Let `d` be the shortest distance from the current position to the base using only cells that are already known to be free.

For an unknown-cell attempt, the controller requires:

```text
remaining_energy >= d + 3
```

The meaning is simple:

1. `1` energy pays for the exploratory attempt.
2. If the unknown cell is free, the position can become one step farther from base, so up to `d + 1` additional energy can be needed to return through the old known-safe route.
3. `1` more energy must still remain after reaching base; otherwise the run would end at zero before the controller gets another decision turn to output `DONE`.

For a move through a known-free cell, the controller calculates the return distance from the destination before moving. After paying the 1-unit move cost, at least 1 energy must still remain beyond that return distance.

When exploration no longer satisfies the safety condition, the controller immediately turns back using the known-free route to base.

## Outputs

Running the script creates:

- `logs/development_trajectory.csv` — every simulated step, position, energy, coverage, action, and decision reason.
- `logs/development_metrics.txt` — final development metrics.
- `logs/exploration.png` — generated grid, exploration path, and base.

The committed `logs/exploration.svg` is a repository-friendly vector copy of the development visualization.

`CkEk = coverage * (remaining_energy / 400)` when the controller returns successfully and outputs `DONE`; otherwise the remaining-energy fraction is 0.

## Development run

With the required seed `20260911`, the current controller produced:

```text
reachable_free_cells=700
steps=398
energy_used=398
remaining_energy=2
final_coverage_pct=99.428571
returned_successfully=True
ckek=0.004971
```
