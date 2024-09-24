from part_2 import *

import pandas as pd #for naming the rows and columns of the transitions matrix

def build_transition_and_reward_matrices(policy_grid, reward_grid):
    num_states = GRID_SIZE * GRID_SIZE
    P_pi = np.zeros((num_states, num_states))
    R_pi = np.zeros(num_states)

    # To store (x, y) labels for the states
    state_labels = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            current_state = (x, y)
            current_index = x * GRID_SIZE + y

            action = policy_grid[x, y]
            reward = reward_grid[x, y]
            transitions = [(get_next_state(current_state, action), 0.85)]  # Intentional move
            for other_action in actions:
                if other_action != action:
                    transitions.append((get_next_state(current_state, other_action), 0.05))  # Unintended move

            for next_state, prob in transitions:
                next_index = next_state[0] * GRID_SIZE + next_state[1]
                P_pi[current_index, next_index] += prob

            R_pi[current_index] = reward

    # Convert the P_pi matrix into a Pandas DataFrame
    P_pi_df = pd.DataFrame(P_pi, index=state_labels, columns=state_labels)
    return P_pi_df, R_pi


#Store the Transition and reward matrices
transition_matrix, reward_vector = build_transition_and_reward_matrices(policy_grid, reward_grid)


# Print the descriptive transition matrix
print("Transition Matrix P_pi (with state labels):")
print(transition_matrix)

# Optional: if you want to print the reward vector as well, you can use:
#reward_df = pd.Series(R_pi, index=[(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)])
#print("\nReward Vector R_pi:")
#print(reward_df)
