# RoverMind — IR-01 Energy-Aware Robot Grid Exploration

RoverMind is a pure Python simulation for IR-01.

## Run

```bash
pip install -r requirements.txt
python rovermind.py
```

The program creates the required 30x30 map, reveals only the allowed local view, runs an online greedy explorer, prints metrics, writes step-by-step logs, and saves a plot.

## Approach

The explorer looks for the nearest **frontier**: a cell already known to be free that touches at least one unknown cell. It walks to a nearby frontier through known-free cells and then tests one unknown neighbor. When several frontiers are equally close, it prefers the one with more unknown neighbors because that usually gives more new information.

The important rule is that the explorer never receives the hidden full map. It only gets the current revealed map, current position, previous actions, and remaining energy.

## Energy safety

Let `d` be the shortest distance from the current position to the base using only cells that are already known to be free.

For an unknown-cell attempt, the explorer requires:

`remaining_energy >= d + 2`

Why? The move itself costs 1. If the unknown cell is free, the new position can be one step farther from the base, so up to `d + 1` more energy may be needed to return. The extra reserve also ensures the run can reach the base before energy becomes zero, allowing `DONE` on the next decision.

For a move through a known-free cell, the explorer first calculates the return distance from that destination. It moves only when, after paying the move cost, enough energy remains to return to base with one unit left.

When exploration is no longer safe, the explorer immediately follows the known-free path back to base.

## Output

The script creates:

- `logs/development_trajectory.csv` — every position, energy level, coverage value, action, and decision reason.
- `logs/development_metrics.txt` — final development metrics.
- `logs/exploration.png` — map, path, and base.

`CkEk = coverage * (remaining_energy / 400)` when the explorer returns successfully; otherwise the remaining-energy fraction is 0.
