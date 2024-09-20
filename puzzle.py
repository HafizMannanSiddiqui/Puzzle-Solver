import heapq
import random
import os

# A* algorithm implementation
class PuzzleSolver:
    def __init__(self, size, max_moves, initial_state):
        self.size = size  # Grid size, e.g., 2x2, 3x3, etc.
        self.max_moves = max_moves  # Maximum allowed moves
        self.initial_state = tuple(initial_state)  # Tuple of initial state
        self.empty_tile = 0  # Representation of the empty tile
        self.goal_state = self.get_goal_state()  # Hardcoded goal state
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Possible moves: Up, Down, Left, Right

    def get_goal_state(self):
        """Return the hardcoded goal state based on the size of the puzzle."""
        if self.size == 2:
            return (1, 2, 3, 0)
        elif self.size == 3:
            return (1, 2, 3, 4, 5, 6, 7, 8, 0)
        elif self.size == 4:
            return tuple(range(1, 16)) + (0,)
        elif self.size == 5:
            return tuple(range(1, 25)) + (0,)
        else:
            raise ValueError("Invalid grid size!")

    def manhattan_distance(self, current_state):
        """Calculate the Manhattan distance heuristic."""
        dist = 0
        for i in range(self.size * self.size):
            if current_state[i] == 0:
                continue  # Skip the empty tile
            correct_position = self.goal_state.index(current_state[i])
            current_row, current_col = divmod(i, self.size)
            goal_row, goal_col = divmod(correct_position, self.size)
            dist += abs(current_row - goal_row) + abs(current_col - goal_col)
        return dist

    def is_solvable(self, state):
        """Check if the puzzle is solvable."""
        inversion_count = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] != 0 and state[j] != 0 and state[i] > state[j]:
                    inversion_count += 1
        return inversion_count % 2 == 0

    def get_neighbors(self, state):
        """Return all possible neighboring states."""
        empty_index = state.index(self.empty_tile)
        empty_row, empty_col = divmod(empty_index, self.size)
        neighbors = []

        for move_row, move_col in self.moves:
            new_row, new_col = empty_row + move_row, empty_col + move_col
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                new_index = new_row * self.size + new_col
                new_state = list(state)
                new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
                neighbors.append(tuple(new_state))

        return neighbors

    def solve(self):
        """Perform A* search to solve the puzzle."""
        if not self.is_solvable(self.initial_state):
            return "The puzzle is unsolvable."

        frontier = []
        heapq.heappush(frontier, (0 + self.manhattan_distance(self.initial_state), 0, self.initial_state, []))
        explored = set()

        while frontier:
            estimated_cost, moves_made, current_state, path = heapq.heappop(frontier)

            if current_state == self.goal_state:
                return path

            if moves_made < self.max_moves:
                explored.add(current_state)

                for neighbor in self.get_neighbors(current_state):
                    if neighbor not in explored:
                        new_path = path + [neighbor]
                        new_cost = moves_made + 1
                        priority = new_cost + self.manhattan_distance(neighbor)
                        heapq.heappush(frontier, (priority, new_cost, neighbor, new_path))

        return "No solution found within the move limit."

def generate_random_solvable_state(size):
    """Generate a random solvable initial state for the puzzle."""
    while True:
        state = list(range(size * size))
        random.shuffle(state)
        if PuzzleSolver(size, 0, state).is_solvable(state):
            return state

def generate_from_known_state(size):
    """Generate a random solvable state by making moves from a known state."""
    known_state = list(range(1, size * size)) + [0]  # A solved state
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    empty_index = known_state.index(0)

    for _ in range(100):  # Number of random moves to make
        random_move = random.choice(moves)
        new_row, new_col = divmod(empty_index, size)
        new_row += random_move[0]
        new_col += random_move[1]

        if 0 <= new_row < size and 0 <= new_col < size:
            # Swap empty tile with the adjacent tile
            new_index = new_row * size + new_col
            known_state[empty_index], known_state[new_index] = known_state[new_index], known_state[empty_index]
            empty_index = new_index

    return known_state

def read_user_input():
    """Get user input for puzzle size, then use hardcoded max moves and generate random states."""
    size = int(input("Enter the grid size (2 for 2x2, 3 for 3x3, 4 for 4x4, 5 for 5x5): "))
    if size not in [2, 3, 4, 5]:
        raise ValueError("Invalid grid size! Please choose 2, 3, 4, or 5.")
    
    # Hardcoded max moves for each grid size
    max_moves_dict = {
        2: 5,    # For 2x2 puzzle
        3: 30,   # For 3x3 puzzle
        4: 80,   # For 4x4 puzzle
        5: 200   # For 5x5 puzzle
    }
    
    max_moves = max_moves_dict[size]

    # Generate random solvable initial state
    initial_state = generate_random_solvable_state(size)

    print(f"Initial state: {initial_state}")
    print(f"Goal state: {PuzzleSolver(size, max_moves, initial_state).goal_state}")  # Display goal state
    print(f"Maximum number of moves allowed: {max_moves}")

    return size, max_moves, initial_state

def write_solution_to_file(filename, size, max_moves, initial_state, goal_state, solution):
    """Write the solution and details to the output file."""
    with open(filename, 'w') as file:
        # Writing the puzzle details to the output file
        file.write(f"Enter the grid size (2 for 2x2, 3 for 3x3, 4 for 4x4, 5 for 5x5): {size}\n")
        file.write(f"Initial state: {initial_state}\n")
        file.write(f"Goal state: {PuzzleSolver(size, max_moves, initial_state).goal_state}\n")  # Write hardcoded goal state
        file.write(f"Maximum number of moves allowed: {max_moves}\n")
        
        if isinstance(solution, str):
            file.write(solution)
        else:
            file.write("Solution found:\n")
            for step in solution:
                file.write(f"{step}\n")

if __name__ == "__main__":
    # Read user input for puzzle configuration
    size, max_moves, initial_state = read_user_input()

    # Initialize puzzle solver
    solver = PuzzleSolver(size, max_moves, initial_state)

    # Solve the puzzle
    solution = solver.solve()

    # Write output to a file
    output_filename = "output.txt"
    write_solution_to_file(output_filename, size, max_moves, initial_state, solver.goal_state, solution)

    print(f"Solution has been written to {output_filename}.")
