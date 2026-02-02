import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (assuming no header based on your file structure)
df = pd.read_csv('../rust/occupation_time.csv', header=None)

# Initialize the figure
plt.figure(figsize=(12, 7))

# Iterate through each row in the dataframe
for i in range(len(df)):
    # The first column is Gamma
    gamma = df.iloc[i, 0]
    # The second column is Omega
    omega = df.iloc[i, 1]
    
    # The rest of the columns contain the time series data
    # We drop NaN values to handle potential differing lengths or trailing empty cells
    occupation = df.iloc[i, 2:].dropna().values
    
    # Create a range for the time steps based on the data length
    time_steps = range(len(occupation))
    
    # Plot the data with the corresponding label
    plt.plot(time_steps, occupation, label=f'$\gamma={gamma}, \Omega={omega}$')

# Add labels, title, legend, and grid
plt.xlabel('Time')
plt.ylabel('Occupation')
plt.title('Occupation in Time')
plt.ylim(0, 0.51)
plt.xlim(0, 40000)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()