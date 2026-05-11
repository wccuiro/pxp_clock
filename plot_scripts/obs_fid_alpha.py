import matplotlib.pyplot as plt
import numpy as np

dt = 0.001
t_final = 10.0
num_steps = int(round(t_final / dt))
t = np.linspace(0, t_final, num_steps + 1)

data = []
with open("../rust/occupation_time_alpha_8.csv", "r") as f:
    for line in f:
        parts = line.strip().split(',')
        alpha = float(parts[0])
        gp = float(parts[1])
        gm = float(parts[2])
        omega = float(parts[3])
        # Extract fidelity (starts at index 6, every 3rd element)
        fid = [float(x) for x in parts[6::3]]
        data.append({
            'alpha': alpha, 'gp': gp, 'gm': gm, 'omega': omega, 'fid': fid
        })

# Get unique parameter combinations of (gp, gm, omega)
combinations = list(set([(d['gp'], d['gm'], d['omega']) for d in data]))
combinations.sort()

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, combo in enumerate(combinations):
    gp, gm, omega = combo
    ax = axes[i]
    
    # Filter data for this combination
    subset = [d for d in data if d['gp'] == gp and d['gm'] == gm and d['omega'] == omega]
    subset.sort(key=lambda x: x['alpha'])
    
    # Plot each alpha value
    for d in subset:
        ax.plot(t, d['fid'], label=fr"$\alpha={d['alpha']:0.02f}$")
        
    ax.set_title(fr"$\gamma_+={gp}, \gamma_-={gm}, \Omega={omega}$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("fidelity_comparison_alpha.png", dpi=150)
plt.show()