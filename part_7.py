from part_6 import *
import matplotlib.pyplot as plt

# Plot the sequence of errors
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), errors, label=r'$\|V_t - V_\pi\|_\infty$')
plt.xlabel('Iterations')
plt.ylabel('Error (Infinity Norm)')
plt.title(r'Max Error $\|V_t - V_\pi\|_\infty$ vs. Iterations')
plt.grid(True)
plt.legend()
plt.show()
