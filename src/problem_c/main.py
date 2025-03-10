import logging
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pickle

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(processName)s %(levelname)s: %(message)s'
)

# Global constants and parameters
N = 100       # Lattice size
J = 1         # Interaction strength
KB = 1        # Boltzmann constant
steps = 200_000   # Total Monte Carlo steps
burnin = 50_000   # Burn-in period

def MCMC(lattice_spins, temp, steps, B):
    m_values = []  # List to store magnetization values
    t = 0
    while t < steps:
        # Randomly select a spin to attempt a flip
        i, j = np.random.randint(N), np.random.randint(N)
        delta_energy = 0
        for k, l in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            i_neigh = (i + k) % N
            j_neigh = (j + l) % N
            delta_energy += 2 * J * lattice_spins[i, j] * lattice_spins[i_neigh, j_neigh]
        delta_energy += 2 * B * lattice_spins[i, j]
        
        if delta_energy <= 0:
            lattice_spins[i, j] *= -1
        else:
            prob = np.exp(-delta_energy / (KB * temp))
            if np.random.random() < prob:
                lattice_spins[i, j] *= -1
        m_values.append(np.mean(lattice_spins))
        t += 1
    return m_values

def compute_m(args):
    T, B = args
    logging.info(f"Starting computation for T={T:.3f}, B={B:.3f}")
    if T < 2.269 and B != 0:
        lattice = np.ones((N, N)) if B > 0 else -np.ones((N, N))
    else:
        lattice = 2 * (np.random.randint(2, size=(N, N)) - 0.5)
    
    m_values = MCMC(lattice, temp=T, steps=steps, B=B)
    m_avg = np.mean(m_values[burnin:])
    logging.info(f"Finished computation for T={T:.3f}, B={B:.3f}")
    return m_avg

if __name__ == '__main__':
    T_range = np.linspace(0.1, 5, 100)    # Temperatures from 0.1 to 5
    B_range = np.linspace(-1.0, 1.0, 100)   # Magnetic fields from -1.0 to 1.0

    grid_args = [(T, B) for B in B_range for T in T_range]
    
    logging.info("Starting multiprocessing pool")
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_m, grid_args)
    logging.info("Multiprocessing pool complete")

    M_data = np.array(results).reshape((len(B_range), len(T_range)))

    with open('m_data.pkl', 'wb') as f:
        pickle.dump(M_data, f)
        pickle.dump(T_range, f)
        pickle.dump(B_range, f)
    

