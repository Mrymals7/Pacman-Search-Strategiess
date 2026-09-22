# Pacman Search Strategies

Implementations of classic AI search and adversarial game-playing algorithms in the UC Berkeley Pacman AI framework, applied to maze navigation and multi-agent ghost-avoidance problems.

## Overview

This project is built on the widely used **Berkeley CS188 Pacman AI projects**, which provide a game environment (`pacman.py`) and testing/autograding infrastructure. On top of that framework, this project implements:

1. **Uninformed and informed search algorithms** for single-agent maze navigation
2. **Adversarial search algorithms** for multi-agent games (Pacman vs. ghosts)

## Part 1: Search (`search/`)

Implements classic search algorithms to help Pacman navigate mazes, reach corners, and eat all the food efficiently.

**Algorithms implemented (`search.py`, `searchAgents.py`):**
- **Depth-First Search (DFS)** — explores as far as possible along a branch before backtracking
- **Breadth-First Search (BFS)** — explores all neighbors level by level, guaranteeing shortest path in unweighted graphs
- **Uniform Cost Search (UCS)** — expands the lowest-cost node first, optimal for weighted graphs
- **A\* Search** — combines path cost with a heuristic to find optimal solutions more efficiently than UCS

**Problems solved:**
- Finding a path to a fixed point in a maze
- Visiting all four corners of a maze
- Eating all the food dots with an admissible, consistent heuristic

**Run it:**
```bash
cd search
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

## Part 2: Multi-Agent Search (`multiagent/`)

Implements adversarial and probabilistic search algorithms so Pacman can make optimal decisions while being chased by ghosts.

**Algorithms implemented (`multiAgents.py`):**
- **Reflex Agent** — a hand-designed evaluation function reacting to the immediate game state
- **Minimax** — assumes ghosts play optimally against Pacman
- **Alpha-Beta Pruning** — a minimax optimization that prunes branches that can't affect the final decision, enabling deeper search
- **Expectimax** — models ghosts as acting randomly rather than optimally, better reflecting suboptimal real-world opponents

**Run it:**
```bash
cd multiagent
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3
```

## Repository Structure

```
├── search/            # Uninformed & informed search (DFS, BFS, UCS, A*)
│   ├── search.py          # Core search algorithm implementations
│   ├── searchAgents.py    # Agents and problems using the search algorithms
│   ├── pacman.py          # Game engine
│   └── layouts/           # Maze layout files
├── multiagent/         # Adversarial & probabilistic search (Minimax, Alpha-Beta, Expectimax)
│   ├── multiAgents.py     # Core agent implementations
│   ├── pacman.py          # Game engine
│   └── layouts/           # Maze layout files
└── README.md
```

## Tech Stack

- Python 3

## Credits

Built on the [Berkeley CS188 Pacman AI Projects](http://ai.berkeley.edu/project_overview.html) framework, developed at UC Berkeley for educational use in AI coursework.
