from part_7 import *
import numpy as np

# Constants
GRID_SIZE = 5
DISCOUNT_FACTOR = 0.95
THRESHOLD = 0.01
DEFAULT_REWARD = -0.04  # Small negative reward to encourage movement

# Define the grid and reward structures
grid = np.array([
    ['.', '.', '.', '.', 'T'],
    ['.', 'M', 'L', '.', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', '.', 'M', 'M', '.'],
    ['R', '.', '.', '.', '.']
])

# Update the reward structure to have a default negative reward for non-terminal states
reward_grid = np.full((GRID_SIZE, GRID_SIZE), DEFAULT_REWARD)
reward_grid[0, 4] = 1    # Treasure
reward_grid[1, 2] = -1   # Lightning

actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
action_list = ['U', 'D', 'L', 'R']

# Get next state based on action, ensuring the next state is valid
def get_next_state(state, action):
    x, y = state
    dx, dy = actions[action]
    next_state = (x + dx, y + dy)
    if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE and grid[next_state] != 'M':
        return next_state
    return state

# Get possible transitions and their probabilities
def get_transition_probabilities(state, action):
    transitions = [(get_next_state(state, action), 0.85)]  # Intentional move
    for other_action in actions:
        if other_action != action:
            transitions.append((get_next_state(state, other_action), 0.05))  # Unintentional move
    return transitions

# Value iteration algorithm
def value_iteration(reward_grid, discount_factor, threshold):
    num_states = GRID_SIZE * GRID_SIZE
    V = np.zeros(num_states)  # Initialize value function to zero
    policy = np.empty((GRID_SIZE, GRID_SIZE), dtype=str)  # Placeholder for the learned policy
    
    def to_index(state):
        x, y = state
        return x * GRID_SIZE + y
    
    def to_state(index):
        return divmod(index, GRID_SIZE)

    # Initialize robot start position value to -2
    robot_start = (4, 0)
    V[to_index(robot_start)] = -2
    
    iteration = 0
    while True:
        delta = 0
        new_V = V.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                current_state = (x, y)
                current_index = to_index(current_state)
                
                # Skip if the state is a terminal state (treasure or lightning)
                if grid[x, y] in ['T', 'L']:
                    new_V[current_index] = reward_grid[x, y]  # Ensure terminal states keep their reward
                    policy[x, y] = ''  # No action needed in terminal states
                    continue
                
                max_value = float('-inf')
                best_action = None
                
                # Iterate over all actions and compute expected returns
                for action in actions:
                    v = 0
                    transitions = get_transition_probabilities(current_state, action)
                    for next_state, prob in transitions:
                        next_index = to_index(next_state)
                        v += prob * (reward_grid[x, y] + discount_factor * V[next_index])
                    
                    if v > max_value:
                        max_value = v
                        best_action = action
                
                # Update the value function and the policy
                new_V[current_index] = max_value
                policy[x, y] = best_action
                delta = max(delta, abs(new_V[current_index] - V[current_index]))
        
        V = new_V
        iteration += 1
        
        # Print the value function after each iteration
        print(f"\nValue Function after iteration {iteration}:")
        print(V.reshape((GRID_SIZE, GRID_SIZE)))
        
        # If the change is smaller than the threshold, stop the iteration
        if delta < threshold:
            break
    
    return V.reshape((GRID_SIZE, GRID_SIZE)), policy, iteration

# Perform value iteration
optimal_value_function, optimal_policy, num_iterations = value_iteration(reward_grid, DISCOUNT_FACTOR, THRESHOLD)

# Report the results
print("\nFinal Optimal Value Function:")
print(optimal_value_function)

print("\nFinal Optimal Policy:")
print(optimal_policy)

print(f"\nNumber of iterations for convergence: {num_iterations}")
