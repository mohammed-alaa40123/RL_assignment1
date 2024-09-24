
#JUSTIFICATION: IN THE MARKDOWN ABOVE
from part_4 import *

# New Constants
DESIRED_ACCURACY = 0.01

# Formula to calculate the number of iterations
def calculate_iterations(discount_factor, desired_accuracy):
    return int(np.ceil(np.log(desired_accuracy * (1 - discount_factor)) / np.log(discount_factor)))

# Calculate the number of iterations required for convergence
num_iterations = calculate_iterations(DISCOUNT_FACTOR, DESIRED_ACCURACY) + 1 # +1 to account for the initial iteration (guarenteed to be less than the desired accuracy)
print(f"Number of iterations needed for convergence to an accuracy of {DESIRED_ACCURACY}: {num_iterations}")
