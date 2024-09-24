from part_5 import *
# Iterative policy evaluation with fixed number of iterations
errors = []
def iterative_solution_with_fixed_iterations(policy_grid, reward_grid, discount_factor, num_iterations):
    num_states = GRID_SIZE * GRID_SIZE
    V = np.zeros(num_states)
    def compute_value_for_state(state_index, P_pi, R_pi, V):
        return R_pi[state_index] + discount_factor * np.dot(P_pi[state_index], V)

    P_pi, R_pi = build_transition_and_reward_matrices(policy_grid, reward_grid)
    P_pi = P_pi.values  # Convert DataFrame to numpy array
    
    for iteration in range(num_iterations):
        new_V = V.copy()
        
        for i in range(num_states):
            new_V[i] = compute_value_for_state(i, P_pi, R_pi, V)
        errors.append(np.linalg.norm((ValueFunctionForState.flatten()-new_V), ord=np.inf))
        V = new_V
    
    return V.reshape((GRID_SIZE, GRID_SIZE))

# Perform iterative policy evaluation with the calculated number of iterations
value_function_iterative_fixed = iterative_solution_with_fixed_iterations(
    policy_grid, reward_grid, DISCOUNT_FACTOR, num_iterations
)
print("\nValue Function (Iterative Solution with fixed iterations):")
print(value_function_iterative_fixed)
