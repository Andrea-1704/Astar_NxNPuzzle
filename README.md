# LAB 3 Andrea Mirenda

In this lab I tried to implement a solution for the N^2-1 Puzzle.

## Problem description 

The **N²-1 puzzle problem** is a classic combinatorial optimization challenge involving a sliding puzzle. It consists of:

- A square grid with \( N^2 - 1 \) numbered tiles and 1 empty space, where \( N \) is the grid's dimension.
- Example configurations:
  - \( 3 \times 3 \) grid: 8 tiles (numbered 1 through 8) and 1 blank space.
  - \( 4 \times 4 \) grid: 15 tiles and 1 blank space.

#### Objective
Rearrange the tiles from a given starting configuration to a target configuration (often sequentially ordered, with the blank in the bottom-right corner) by sliding adjacent tiles into the empty space.

#### Key Features
- **Moves**: Only adjacent tiles can be moved into the blank space.
- **Complexity**: The puzzle's difficulty comes from constraints on tile movement and the fact that not all configurations are solvable.

#### Solvability
The solvability of a configuration depends on:
1. The number of **inversions** (pairs of tiles out of order).
2. The grid's dimensions.

The N²-1 puzzle is a benchmark problem for evaluating optimization and search strategies due to its well-defined state space and computational complexity.


## Methodology

At first, I tried to solve the problem by using a depth search and a breath search aproach. The problems of these algorithm were the fact that they didn't scale up with the complexity of the problem space. These techniques, infact, were perfectly able to solve a 8 puzzle in a resonable time, but did not solve the 15 puzzle.

I tried to use the A* algorithm, which is a graph traversal and pathfinding algorithm that finds the shortest path between nodes by combining the actual cost from the start node (g) and a heuristic estimate to the goal (h). It balances exploration and efficiency, ensuring an optimal path if the heuristic is admissible (and we are using a tree-like representation).

```python
def enhanced_a_star(initial_state: np.ndarray, final_state: np.ndarray) -> Tuple[Union[list, None], float]:
    heuristic_service = PuzzleHeuristicService(final_state)

    def calculate_heuristic(state: np.ndarray) -> int:
        return heuristic_service.combined_heuristic(state)

    # Priority queue: (f_score, g_score, current_state, path)
    open_set = []
    heappush(open_set, (calculate_heuristic(initial_state), 0, initial_state.tobytes(), []))
    visited = set()
    optimum = state_to_bytes(final_state)

    cost = 0

    while open_set:
        # Extract node with the lowest f score (f score= cost)
        f_score, g_score, current_bytes, path = heappop(open_set)
        current_state = np.frombuffer(current_bytes, dtype=initial_state.dtype).reshape(initial_state.shape)
        current_score = state_to_bytes(current_state)

        # Check if we finished already:
        if current_score == optimum:
            return path, float(cost) 

        # Add current node to visited
        visited.add(current_score)

        # Generate possible moves:
        for act in available_actions(current_state):
            next_state = do_action(current_state, act)
            next_score = state_to_bytes(next_state)
            if next_score in visited:
                continue

            cost += 1

            # update scores:
            new_g_score = g_score + 1
            new_f_score = new_g_score + calculate_heuristic(next_state)

            # Add new state to openset
            heappush(open_set, (new_f_score, new_g_score, next_state.tobytes(), path + [act]))

    return None, float('inf')  # No solution found
```

The first euristic function used was the Manhattan distance. Manhattan distance is a measure of distance between two points in a grid-based system, calculated as the sum of the absolute differences of their Cartesian coordinates. It reflects the total horizontal and vertical steps needed to travel between the points, resembling movement along a city street grid.

```python
def heuristic_manhattan_distance(self, position):
        distance = 0
        size = len(position)
        for i in range(size):
            for j in range(size):
                tile = position[i][j]
                if tile != 0:
                    target_row = (tile - 1) // size
                    target_col = (tile - 1) % size
                    distance += abs(i - target_row) + abs(j - target_col)
        return distance
```


Using this aproach I was now able to solve the 8 puzzle problem in an optimal way within some minutes.

Since the execution time depends on the hardware component that executes the program, from now on we will rely on different measurement of the efficiency of the program, which is the total number of evaluation computed by the algorithm, we will name this measurement "cost". 

Since the aim of the algorithm should be to find the minimum number of operations, we will name "quality" of a solution the length of the actions vector.

### Starting point
As a starting point a random initial state of the matrix was provided in the following way:
```python
RANDOMIZE_STEPS = 100_000
state = np.array([i for i in range(1, PUZZLE_DIM**2)] + [0]).reshape((PUZZLE_DIM, PUZZLE_DIM))
for r in tqdm(range(RANDOMIZE_STEPS), desc='Randomizing'):
    state = do_action(state, choice(available_actions(state)))
```

### Results first aproach
In the following are reported some results, but to be more precise the starting point shoulb be provided too, in the following tables we are considering a mean of the values obtained.

|    Puzzle dimension       | 3 | 8 |
|-----------|-----------|-----------|
| time (s) | 0  | 9.6  |
| cost | 12  | 3185  |
| Quality| 6  | 24  | 


## Second aproach
It became immediatly clear that the main problem to solve was not the efficiency of the solution, but the ability to solve it. I tried to focus more on faster aproach to get a solution, even if they were not the best one. 

To do so, I tried to over-estimate the solution in a way that the A* algorithm would act similar to a depth search, using a FIFO priority queue. To do so, for the 15 and 24 puzzle problem I first decided to use more heurist functions:
```python
    def heuristic_linear_conflict(self, position):
        conflict = 0
        size = len(position)

        # Row conflicts
        for row in range(size):
            max_val = -1
            for col in range(size):
                value = position[row][col]
                if value != 0 and (value - 1) // size == row:
                    if value > max_val:
                        max_val = value
                    else:
                        conflict += 2

        # Column conflicts
        for col in range(size):
            max_val = -1
            for row in range(size):
                value = position[row][col]
                if value != 0 and (value - 1) % size == col:
                    if value > max_val:
                        max_val = value
                    else:
                        conflict += 2

        return conflict

    def heuristic_walking_distance(self, position):
        # Create a grid to store the walking distances
        size = len(position)
        distance_grid = [[0] * size for _ in range(size)]

        for row in range(size):
            for col in range(size):
                value = position[row][col]
                if value != 0:
                    target_row = (value - 1) // size
                    target_col = (value - 1) % size
                    distance_grid[row][col] = abs(row - target_row) + abs(col - target_col)

        # Calculate the walking distance
        walking_distance = 0
        for row in range(size):
            for col in range(size):
                walking_distance += distance_grid[row][col]

        return walking_distance
```

The value provided for a given state was computed by considering a sum over all these metrics (included the Manhattan distance show previously).

This solution worked increadibly well for problem instances till the 24 puzzle problem. 

### Results second aproach

|    Puzzle dimension       | 3 | 8 | 15 |24|
|-----------|-----------|-----------|-----------|-----------|
| time (s) | 0  | 0  | 0.4  | 29.3  |
| cost | 7  | 87  | 7968  | 340656 |
| Quality| 6  | 24  | 52  | 142  |



## Third aproach
I kept on over-estimating the heuristic to speed up the execution of the code and obtain a first suitable solution for the problem. The idea was simply to multiply the resutlt of the heuristicb by a scaling factor. The scaling factor was fine tuned to balance goodness of the solution and the cost.

The management of the parameter and of the heuristic in general is the following:
```python
    def combined_heuristic(self, state: np.ndarray) -> int:
    
        if PUZZLE_DIM<=3:
            return self.heuristic_manhattan_distance(state)

        if PUZZLE_DIM<=5:
            return 1*(self.heuristic_manhattan_distance(state) + self.heuristic_linear_conflict(state) + self.heuristic_walking_distance(state))

        if PUZZLE_DIM==6:
            return 100*(self.heuristic_manhattan_distance(state) + self.heuristic_linear_conflict(state) + self.heuristic_walking_distance(state))

        return 10_000*(self.heuristic_manhattan_distance(state) + self.heuristic_linear_conflict(state) + self.heuristic_walking_distance(state))

```

### Results third aproach

|    Puzzle dimension       | 3 | 8 | 15 |24|35|48|
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| time (s) | 0  | 0  | 0.4  | 96.7  |29.3  |10.3 |
| cost | 7  | 87  | 7968  | 340656 |759556  |67946  |
| Quality| 6  | 24  | 52  | 142  |464  |952  |