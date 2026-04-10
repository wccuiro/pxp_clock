import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
# Assuming header=None. If you have a header row, remove header=None
df = pd.read_csv('../rust/occupation_time_10.csv', header=None)

dg = pd.read_csv('../rust/occupation.csv')


# Initialize the figure with 3 subplots sharing the x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Iterate through each row in the dataframe
for i in range(1,len(df)):
    gamma_plus = df.iloc[i, 0]
    gamma_minus = df.iloc[i, 1]
    omega = df.iloc[i, 2]
    
    gamma = gamma_plus / gamma_minus
    
    # Extract the raw interleaved data
    raw_data = df.iloc[i, 3:].dropna().values
    
    # Slice the data
    # <n>  = Occupation (even indices: 0, 2, 4...)
    # <nn> = Correlation (odd indices: 1, 3, 5...)
    n_avg = raw_data[::2]  # <n>
    nn_avg = raw_data[1::2] # <nn>
    
    # Ensure both arrays are the same length before calculation
    # (In case the row has an odd number of data points, truncate the extra <n>)
    min_len = min(len(n_avg), len(nn_avg))
    n_avg = n_avg[:min_len]
    nn_avg = nn_avg[:min_len]
    
    
    n_ss = dg.iloc[i, 3] * np.ones(min_len) 
    nn_ss = dg.iloc[i, 4] * np.ones(min_len)
    
    n_renorm = (n_avg - n_ss) / (n_avg[0] - n_ss[0])
    nn_renorm = (nn_avg - nn_ss) / (nn_avg[0] - nn_ss[0])

    # Calculate Delta based on the formula: 
    # Delta = gamma * <nn> - (3*gamma + 1) * <n> + gamma
    delta = gamma * nn_avg - ((3 * gamma + 1) * n_avg) + gamma
    
    delta_ss = gamma * nn_ss - ((3 * gamma + 1) * n_ss) + gamma
    delta_renorm = (delta - delta_ss) / (delta[0] - delta_ss)
    
    time_steps = range(min_len)
    
    # Plot 1: Occupation <n>
    ax1.plot(time_steps, n_renorm, label=r'$\gamma_-={gamma_minus}, \gamma_+={gamma_plus}, \Omega={omega}$'.format(gamma_minus=gamma_minus, gamma_plus=gamma_plus, omega=omega))
    
    # Plot 2: Correlation <nn>
    ax2.plot(time_steps, nn_renorm, label=r'$\gamma_-={gamma_minus}, \gamma_+={gamma_plus}, \Omega={omega}$'.format(gamma_minus=gamma_minus, gamma_plus=gamma_plus, omega=omega))

    # Plot 3: Delta
    ax3.plot(time_steps, delta_renorm, label=r'$\gamma_-={gamma_minus}, \gamma_+={gamma_plus}, \Omega={omega}$'.format(gamma_minus=gamma_minus, gamma_plus=gamma_plus, omega=omega))

# --- Configure Subplot 1: Occupation ---
ax1.set_ylabel(r'Occupation $\langle n \rangle$')
ax1.set_title(r'Occupation $\langle n \rangle$ in Time')
# ax1.set_ylim(0, 1.0) # Adjusted range, change if needed
ax1.grid(True)
ax1.legend(loc='upper right', fontsize='small')

# --- Configure Subplot 2: Correlation ---
ax2.set_ylabel(r'Correlation $\langle nn \rangle$')
ax2.set_title(r'Correlation $\langle nn \rangle$ in Time')
ax2.grid(True)

# --- Configure Subplot 3: Delta ---
ax3.set_ylabel(r'$\Delta$')
ax3.set_title(r'$\Delta = \gamma \langle nn \rangle - (3\gamma+1)\langle n \rangle + \gamma$')
ax3.set_xlabel('Time Step')
ax3.grid(True)

# Global x-axis limit
# plt.xlim(0, 40000)

plt.tight_layout()
plt.show()