import numpy as np

# Constants
GRID_SIZE = 5
DISCOUNT_FACTOR = 0.95
THRESHOLD = 0.01
PROB_INTENT = 0.85
PROB_OTHER = 0.05

# Define the grid
grid = np.array([
    ['.', '.', '.', '.', 'T'],
    ['.', 'M', 'L', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', 'M', 'M', '.'],
    ['R', '.', '.', '.', '.']
])

# Reward structure
reward_grid = np.zeros((GRID_SIZE, GRID_SIZE))
reward_grid[0, 4] = 1  # Treasure
reward_grid[1, 2] = -1  # Lightning

# Policy (encoded as actions)
policy_grid = np.array([
    ['R', 'R', 'R', 'R', 'U'],
    ['L', 'U', 'L', 'L', 'U'],
    ['U', 'U', 'R', 'R', 'R'],
    ['U', 'D', 'D', 'D', 'U'],
    ['U', 'R', 'R', 'U', 'U']
])

# Actions mapping directions to grid changes
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

# Check if the next state is valid
def is_valid(state):
    x, y = state
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] != 'M'

# Compute next state for a given action
def get_next_state(state, action):
    x, y = state
    dx, dy = actions[action]
    next_state = (x + dx, y + dy)
    return next_state if is_valid(next_state) else state

# Get possible transitions from a state based on the action
def get_transition_probabilities(state, action):
    transitions = [(get_next_state(state, action), PROB_INTENT)]
    transitions += [(get_next_state(state, other_action), PROB_OTHER) for other_action in actions if other_action != action]
    return transitions

# Convert (x, y) to 1D index
def to_index(state):
    return state[0] * GRID_SIZE + state[1]

# Convert 1D index back to (x, y)
def to_state(index):
    return divmod(index, GRID_SIZE)

def print_grid(grid):
    for row in grid:
        print(' '.join(row))
    print()

def print_policy(policy):
    for row in policy:
        print(' '.join(row))
    print()