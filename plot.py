import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Data
sizes = np.array([
    4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 
    12288, 13312, 14336, 15360, 16384, 20480, 24576, 32768
])

strassen2_speedups = np.array([
    1.148, 1.049, 1.058, 1.111, 1.026, 1.019, 1.054, 1.051,
    1.049, 1.046, 1.048, 1.045, 1.046, 1.041, 1.133, 1.146
])

strassen_speedups = np.array([
    1.491, 1.575, 1.592, 1.647, 1.415, 1.527, 1.616, 1.552,
    1.575, 1.576, 1.578, 1.616, 1.466, 1.611, 1.750, 1.633
])

triton_speedups = np.array([
    1.921, 1.872, 1.934, 1.996, 1.826, 1.870, 1.909, 1.906,
    1.891, 1.927, 1.921, 1.915, 1.932, 1.891, 2.101, 2.074
])

# Create interpolation functions
f_strassen2 = interp1d(sizes, strassen2_speedups, kind='linear', fill_value='extrapolate')
f_strassen = interp1d(sizes, strassen_speedups, kind='linear', fill_value='extrapolate')
f_triton = interp1d(sizes, triton_speedups, kind='linear', fill_value='extrapolate')

# Predict values for size 65536
size_65536 = 65536
predicted_strassen2 = f_strassen2(size_65536)
predicted_strassen = f_strassen(size_65536)
predicted_triton = f_triton(size_65536)

print(f"Predicted speedups for size 65536:")
print(f"Strassen(depth=2): {predicted_strassen2:.3f}x")
print(f"Strassen(depth=1): {predicted_strassen:.3f}x")
print(f"Triton: {predicted_triton:.3f}x")

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(sizes, strassen2_speedups, 'b-', label='Strassen(depth=2)', marker='o')
plt.plot(sizes, strassen_speedups, 'r-', label='Strassen(depth=1)', marker='s')
plt.plot(sizes, triton_speedups, 'g-', label='Triton', marker='^')

plt.title('Comparing the speeds of Strassen(depth=2), Strassen(depth=1), and Triton for Matmuls')
plt.xlabel('Matrix Size')
plt.ylabel('Speedup (x)')
plt.grid(True)
plt.legend()

# Use log scale for x-axis since matrix sizes grow exponentially
plt.xscale('log', base=2)

# Add annotations for the predicted values
plt.annotate(f'Predicted at 65536:\nStrassen(d=2): {predicted_strassen2:.3f}x\n'
            f'Strassen(d=1): {predicted_strassen:.3f}x\n'
            f'Triton: {predicted_triton:.3f}x',
            xy=(sizes[-1], triton_speedups[-1]),
            xytext=(sizes[-1]*1.2, triton_speedups[-1]),
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

plt.tight_layout()
plt.savefig('matmul_speedups.png')
plt.show()