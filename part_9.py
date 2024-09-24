from part_8 import *
import numpy as np

# Define constants
rows, cols = 5, 5  # Grid size
actions = ['up', 'down', 'left', 'right']
gamma = 0.95  # Discount factor
theta = 0.01  # Convergence threshold
action_prob = 0.85  # Probability of going in the intended direction
other_prob = 0.05  # Probability of going in any other direction

# Rewards and transition matrix
reward_grid = np.zeros((rows, cols))
reward_grid[0, 3] = 1  # Treasure
reward_grid[1, 2] = -1  # Lightning

# Obstacles (mountains)
mountains = [(1, 1), (3, 2), (3, 3)]

# Possible actions
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Initialize random policy (uniform distribution over actions)
policy = np.random.choice(actions, size=(rows, cols))

# Policy iteration algorithm
def policy_evaluation(policy, V, reward_grid):
    while True:
        delta = 0
        for r in range(rows):
            for c in range(cols):
                if (r, c) in mountains or (r == 0 and c == 3):  # Skip mountains and treasure cell
                    continue
                v = V[r, c]
                new_v = 0
                for action in actions:
                    action_prob = 0.85 if action == policy[r, c] else 0.05
                    new_r, new_c = r + action_dict[action][0], c + action_dict[action][1]
                    if 0 <= new_r < rows and 0 <= new_c < cols and (new_r, new_c) not in mountains:
                        new_v += action_prob * (reward_grid[new_r, new_c] + gamma * V[new_r, new_c])
                    else:
                        new_v += action_prob * (reward_grid[r, c] + gamma * V[r, c])
                V[r, c] = new_v
                delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
    return V

def policy_improvement(V):
    policy_stable = True
    new_policy = policy.copy()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in mountains or (r == 0 and c == 3):  # Skip mountains and treasure cell
                continue
            action_values = []
            for action in actions:
                new_r, new_c = r + action_dict[action][0], c + action_dict[action][1]
                if 0 <= new_r < rows and 0 <= new_c < cols and (new_r, new_c) not in mountains:
                    action_values.append(reward_grid[new_r, new_c] + gamma * V[new_r, new_c])
                else:
                    action_values.append(reward_grid[r, c] + gamma * V[r, c])
            best_action = actions[np.argmax(action_values)]
            if best_action != policy[r, c]:
                policy_stable = False
            new_policy[r, c] = best_action
    return new_policy, policy_stable

# Initialize value function to zeros
V = np.zeros((rows, cols))

# Policy iteration with iteration counter
iteration = 0

while True:
    iteration += 1
    print(f"Iteration {iteration}:")
    
    # Display the current policy
    for row in policy:
        print(row)
    
    # Policy Evaluation
    V = policy_evaluation(policy, V, reward_grid)
    
    # Policy Improvement
    policy, stable = policy_improvement(V)
    
    print()  # Add a blank line for readability

    # Check if policy is stable
    if stable:
        print(f"Optimal Policy found after {iteration} iterations:")
        for row in policy:
            print(row)
        break

# Display the final value function
print("\nFinal Value Function:")
print(V)
