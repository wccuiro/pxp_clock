import numpy as np

def simulate_trajectory(N, gamma_plus, gamma_minus, t_final, initial_state):
    """
    Simulate a facilitated spin chain using the Gillespie algorithm.

    N: number of spins
    gamma_plus: rate for 0->1 flip
    gamma_minus: rate for 1->0 flip
    t_final: final simulation time
    initial_state: list of 0/1 of length N
    """
    n = np.array(initial_state, dtype=int)
    t = 0.0

    while t < t_final:
        rates = []

        # Compute rates for each central spin i+1 (1..N-2)
        for i in range(N-2):
            chi = (n[i] == 0) and (n[i+2] == 0)
            # up-flip
            r_plus = gamma_plus * chi if n[i+1] == 0 else 0.0
            # down-flip
            r_minus = gamma_minus * chi if n[i+1] == 1 else 0.0
            rates.append(r_plus)
            rates.append(r_minus)

        R_tot = sum(rates)
        if R_tot == 0:
            break  # no possible flips
        # next event time
        dt = -np.log(np.random.rand()) / R_tot
        t += dt

        # choose event
        r = np.random.rand() * R_tot
        cumulative = 0.0
        for idx, rate in enumerate(rates):
            cumulative += rate
            if r < cumulative:
                spin_idx = idx // 2 + 1   # central spin index
                if idx % 2 == 0:
                    n[spin_idx] = 1  # up flip
                else:
                    n[spin_idx] = 0  # down flip
                break

    return n

# Example usage
N = 8
t_final = 10.0
gamma_plus = 1.0
gamma_minus = 1.0

initial_states = {
    "all_zero": [0]*N,
    "neel": [0,1]* (N//2),
    "all_one": [1]*N
}

for name, state in initial_states.items():
    final_state = simulate_trajectory(N, gamma_plus, gamma_minus, t_final, state)
    print(f"{name} initial -> final: {final_state}")
