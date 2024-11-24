# CI2024_lab3

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

Initially, I attempted to solve the problem using both depth-first and breadth-first search approaches. However, these algorithms struggled to scale with the increasing complexity of the problem space. While they were capable of solving the 8-puzzle within a reasonable time, they failed to solve the 15-puzzle efficiently.

```python
def depth_limited_search(initial_state: np.ndarray, final_state: np.ndarray, max_depth: int) -> list[action] or None:
    stack = deque([(initial_state, [], 0)])  # Stack of (current state, path to reach it, current depth)
    visited = set()  # Set of visited states for current path only
    optimum = matrix_score(final_state)

    while stack:
        current_state, path, depth = stack.pop()
        current_score = matrix_score(current_state)

        # Check if we reached the goal
        if current_score == optimum:
            return path
        
        # # Backtrack if depth limit reached
        if depth >= max_depth:
            continue
        
        # Add current state to visited set (track only in current path)
        visited.add(current_score)
        
        # Generate and iterate over all possible moves
        for act in available_actions(current_state):
            next_state = do_action(current_state, act)
            
            # Check if the next state has already been visited in the current path
            if matrix_score(next_state) not in visited:
                # Add the new state and path to stack, increase depth
                stack.append((next_state, path + [act], depth + 1))
        
        # Remove the current state from visited set after backtracking
        #visited.remove(matrix_score(current_state))
    
    return None  # Return None if no solution is found within depth limit

# Iterative Deepening Depth-First Search (IDDFS) wrapper function
def iterative_deepening_dfs(initial_state: np.ndarray, final_state: np.ndarray, max_depth: int = 50) -> list[action] or None:
    for depth in range(1, max_depth + 1):
        result = depth_limited_search(initial_state, final_state, depth)
        if result is not None:
            return result
    return None
```

I also tried to speed up the process by implementing a Dijkstra aproach but the results were not so interesting.

To address this, I turned to the A* algorithm, a well-known graph traversal and pathfinding technique. A* finds the shortest path between nodes by combining the actual cost from the start node (g) with a heuristic estimate of the cost to reach the goal (h). This approach effectively balances exploration and efficiency, ensuring an optimal solution when the heuristic is admissible and a tree-like structure is used.

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

To prevent loops during the search, a set of previously visited nodes is maintained. The most efficient way to store the nodes appeared to be using the following function:

```python
def state_to_bytes(state: np.ndarray) -> bytes:
    return state.tobytes()
```
This function converts a NumPy array (state) into a compact byte representation using the tobytes() method, making it efficient for storage and comparison.
I also tried different aproaches, like implementing a function by my self in the following way:
```python
def matrix_score(matrix):
    matrice = matrix.flatten()
    score = 0
    for i in range(len(matrice)):
        score += (10**i)*matrice[i]
    return int(score)
```
But among all the possible options considered, the .__tobytes()__ function showed the best results.

The first heuristic function used was the Manhattan distance. This metric quantifies the distance between two points in a grid-based system by summing the absolute differences of their Cartesian coordinates. It represents the total number of horizontal and vertical steps required to travel between the points, similar to movement along a city street grid.

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


With this approach, I was able to solve the 8-puzzle problem optimally in just a few minutes.

Since execution time can vary depending on the hardware running the program, we will use a different metric to evaluate the algorithm's efficiency: the total number of evaluations performed, which we will refer to as the '__cost__.'

As the goal of the algorithm is to minimize the number of operations, we will define the '__quality__' of a solution as the length of the action vector.

### Starting point
As a starting point a random initial state of the matrix was provided in the following way:
```python
RANDOMIZE_STEPS = 100_000
state = np.array([i for i in range(1, PUZZLE_DIM**2)] + [0]).reshape((PUZZLE_DIM, PUZZLE_DIM))
for r in tqdm(range(RANDOMIZE_STEPS), desc='Randomizing'):
    state = do_action(state, choice(available_actions(state)))
```

### Results first aproach
The following results are presented, but for clarity, the starting point should also be specified. In the tables below, I report the average values obtained.

|    Puzzle dimension       | 3 | 8 |
|-----------|-----------|-----------|
| time (s) | 0  | 9.6  |
| cost | 12  | 3185  |
| Quality| 6  | 24  | 


## Second aproach
It quickly became clear that the primary challenge was not the efficiency of the solution, but the ability to find a solution at all. As a result, I shifted my focus toward faster approaches, even if they weren’t optimal.

To achieve this, I attempted to overestimate the solution in such a way that the A* algorithm would behave more like a depth-first search, using a FIFO priority queue. For the 15-puzzle and 24-puzzle problems, I decided to experiment with additional heuristic functions:
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

The value assigned to a given state was calculated by summing all of these metrics, including the Manhattan distance mentioned earlier.

This approach worked incredibly well for problem instances up to the 24-puzzle.

### Results second aproach

|    Puzzle dimension       | 3 | 8 | 15 |24|
|-----------|-----------|-----------|-----------|-----------|
| time (s) | 0  | 0  | 0.4  | 29.3  |
| cost | 7  | 87  | 7968  | 340656 |
| Quality| 6  | 24  | 52  | 142  |



## Third aproach
Based on suggestions from my colleague Stefano Fumero, I continued to overestimate the heuristic to speed up the execution and obtain an initial, suitable solution for the problem. The approach involved multiplying the heuristic result by a scaling factor. This factor was fine-tuned to balance the quality of the solution with the associated cost.

The management of the parameter and the heuristic is as follows:
```python
    def combined_heuristic(self, state: np.ndarray) -> int:
        if PUZZLE_DIM<=3:
            return self.heuristic_manhattan_distance(state)
        if PUZZLE_DIM<=5:
            return 1*(self.heuristic_manhattan_distance(state) + self.heuristic_linear_conflict(state) + self.heuristic_walking_distance(state))
        return 5*(self.heuristic_manhattan_distance(state) + self.heuristic_linear_conflict(state) + self.heuristic_walking_distance(state))

```

### Results third aproach

|    Puzzle dimension       | 3 | 8 | 15 |24|35|48|
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| time (s) | 0  | 0  | 0.4  | 96.7  |21.7  |20.9 |
| cost | 7  | 87  | 7968  | 340656 |78992  |55516  |
| Quality| 6  | 24  | 52  | 142  |388  |592  |

The 48 puzzle isn't always solvable, which I believe may also be due to the fact that some initial configurations are inherently unsolvable.

# Conclusions

In this lab, I explored different strategies for solving the N²-1 puzzle problem, a classic sliding puzzle that involves rearranging tiles to reach a target configuration. Through several iterative approaches, I was able to evaluate different algorithms and heuristics, focusing on balancing the trade-off between solution quality and computational efficiency.

### Key Findings:
1. **Depth-First and Breadth-First Search**: Initially, I experimented with depth-first and breadth-first search algorithms. While these methods performed well on smaller puzzle sizes (e.g., the 3x3 puzzle), they struggled to solve larger puzzles like the 4x4 or 5x5 efficiently. These approaches, although conceptually simple, proved inefficient for larger state spaces due to their exponential growth in time and space complexity.

2. **A* Algorithm with Heuristics**: Implementing the A* algorithm significantly improved performance, especially for larger puzzles. The Manhattan distance heuristic provided a reliable way to estimate the cost to the goal, leading to optimal solutions for smaller puzzles within reasonable time. However, as the puzzle size increased, the algorithm's performance was still constrained by the number of states explored.

3. **Heuristic Refinement**: As the puzzle size increased (e.g., from 8-puzzle to 15-puzzle and beyond), it became evident that the key challenge was not just efficiency but also the ability to find a solution in a reasonable time. By introducing more complex heuristics, such as linear conflict and walking distance, I was able to achieve faster solutions for larger puzzle sizes. These additional heuristics helped reduce the search space, although at the cost of longer computation times.

4. **Overestimating Heuristics for Faster Solutions**: To further optimize for faster solutions, I experimented with scaling the heuristics. This involved adjusting the weights of the heuristic functions based on the puzzle size. By overestimating the heuristic values, the algorithm behaved more like a breadth-first search, reducing the time needed to find a solution, although this sometimes resulted in suboptimal solutions.

### Performance Summary:
- The third approach, using a combined heuristic with scaling factors, provided a good balance between solution quality and execution time, especially for larger puzzles. This approach allowed for faster solutions with an acceptable increase in the number of evaluated states (cost).
- For smaller puzzles (3x3 and 4x4), simpler heuristics like Manhattan distance performed well, providing both optimal solutions and fast execution times.
- For larger puzzles (15, 24, 35, 48), more advanced heuristics combined with scaling factors allowed the algorithm to quickly find solutions, though the solution quality (length of the action sequence) was not always optimal.