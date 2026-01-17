# 1. Output Settings
set terminal pngcairo size 1000,600 enhanced font 'Arial,12'
set output 'pxp_trajectories.png'

# 2. Graph Formatting
set title "PXP Model: Quantum Trajectories (Monte Carlo)"
set xlabel "Time (1/Omega)"
set ylabel "Magnetization <Sz>"
set grid
set yrange [-1.1:1.1]

# --- NEW: Define Expected Value ---
# If it is a constant (Steady State), set it here:
ExpectedSz = -0.6425108401711591
# Or define a function if it varies with time:
# ExpectedSz(x) = exp(-0.1*x) * cos(x)

# 3. CSV Settings
set datafile separator ","
set key off  # Turn off the legend (too messy for 50 lines)

# 4. Define Parameters
N = 49       # Corresponds to (0..num_trajectories-1) in Rust
TrajDir = "./trajectories/"

# 5. Plot Loop
# We use a loop to generate the filename strings dynamically.
# 'lc rgb "#220000FF"' sets the line color to Blue with high transparency 
# (if your gnuplot version supports ARGB).
# If transparency fails, use: lc rgb "blue" lw 0.5

plot for [i=0:N] sprintf("%straj_%d.csv", TrajDir, i) \
     using 1:3 with lines lc rgb "#440000FF" lw 0.05, \
     ExpectedSz with lines lc rgb "black" lw 3 dt 2 title "Expected Value"