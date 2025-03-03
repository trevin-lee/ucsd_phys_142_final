import numpy as np 
import numpy.typing as npt

class MCMC():

    def __init__(
            self, 
            spins: npt.NDArray,
            N = 100,
            J = 1,
            KB = 1,
        ):
        self.spins = spins
        self.N = N
        self.J = J
        self.KB = KB
        
    def acceptance_criterion(self, delta_energy, current_spin, KB, T):
        if delta_energy <= 0:
            return current_spin * -1
        else:
            if np.random.rand() < np.exp(-delta_energy / (KB * T)):
                return current_spin * -1
            else:
                return current_spin

    def mcmc(self, spins, T, steps):
        m_values = []
        N = spins.shape[0]
        J = KB = 1

        for _ in range(steps):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            delta_energy = 0
            
            for k, l in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                i_neighbor = (i + k) % N
                j_neighbor = (j + l) % N
                delta_energy += 2 * J * spins[i, j] * spins[i_neighbor, j_neighbor]

            spins[i, j] = self.acceptance_criterion(delta_energy, spins[i, j], KB, T)
            
            m_values.append(np.mean(spins))
    
        return np.array(m_values)

    def run_simulation(self, spins, T, steps):
        return self.mcmc(spins, T, steps)
        