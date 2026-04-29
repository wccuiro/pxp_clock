import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters matching your Rust code
dt = 0.001
t_final = 10.0
num_steps = int(round(t_final / dt))

# Create the time array (num_steps + 1 points to include t=0 and t=10)
t = np.linspace(0, t_final, num_steps + 1)

plt.figure(figsize=(12, 7))

# Read and parse the CSV file
with open("../rust/occupation_time_alpha.csv", "r") as f:
    for line in f:
        # Split the comma-separated line
        parts = line.strip().split(',')
        
        # Extract the sweep parameters (first 4 columns)
        alpha = float(parts[0])
        gp = float(parts[1])
        gm = float(parts[2])
        omega = float(parts[3])
        
        # Extract the fidelity values
        # The observables start at index 4 as triplets: n, nn, fidelity
        # Fidelity is at index 6, 9, 12, etc. (parts[6::3])
        fid = [float(x) for x in parts[6::3]]
        
        # Plot the data line for this specific parameter configuration
        plt.plot(t, fid, label=fr"$\alpha={alpha}, \gamma_+={gp}, \gamma_-={gm}$")

# Format and label the plot
plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.title("Fidelity Dynamics vs Time")

# Place the legend outside the plot area so it doesn't cover the data
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and/or display the figure
# plt.savefig("fidelity_plot.png", dpi=150)
plt.show()