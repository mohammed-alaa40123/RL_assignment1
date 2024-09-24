from part_3 import *
def analytical_solution(policy_grid, reward_grid, discount_factor):
    P_pi, R_pi = build_transition_and_reward_matrices(policy_grid, reward_grid)
    identity = np.eye(P_pi.shape[0])

    # Solve (I - gamma * P_pi) * V = R_pi for V
    V_pi = np.linalg.inv(identity - discount_factor * P_pi) @ R_pi
    return V_pi.reshape((GRID_SIZE, GRID_SIZE))

ValueFunctionForState = analytical_solution(policy_grid, reward_grid, DISCOUNT_FACTOR)

print("Analytical solution, value function for each state: ")
print(ValueFunctionForState)

