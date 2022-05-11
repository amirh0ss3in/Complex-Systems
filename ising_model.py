import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

""" The two-dimensional ISING model. """

tqdm.monitor_interval = 0
cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

np.random.seed(42)

def show_lattice(lattice, save_path):
    plt.imshow(lattice, cmap='Blues')
    plt.save(save_path)
    plt.show()

def calcEnergy(lattice):
    '''
    Energy of a given lattice
    '''
    energy = 0 
    L = len(lattice)
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            S = lattice[i,j]
            nb = lattice[(i+1)%L, j] + lattice[i,(j+1)%L] + lattice[(i-1)%L, j] + lattice[i,(j-1)%L]
            energy += -nb*S
    return energy

def ISING(L, T, MCmoves):
    Magnetization = 0

    spin_lattice = np.random.choice([-1, 1], size=(L, L))
    initEnergy = calcEnergy(spin_lattice)
    Energy = 0
    
    # E_set = {0, 4, 8, -8, -4}
    exp_E_T = {0:np.exp(0), 2:np.exp(-4/T), 4:np.exp(-8/T), -4:np.exp(8/T) , -2:np.exp(4/T)}

    for MCmove in tqdm(range(MCmoves)):
        i, j = np.random.randint(0, L, 2)
        E = spin_lattice[i, j] * (spin_lattice[(i + 1) % L, j] + spin_lattice[i, (j + 1) % L] + spin_lattice[i, (j - 1) % L] + spin_lattice[(i - 1) % L, j])

        if np.random.rand() < min(1, exp_E_T[E]):
            spin_lattice[i, j] = - spin_lattice[i, j]

        if spin_lattice[(i + 1) % L, j] == spin_lattice[i, (j + 1) % L] == spin_lattice[i, (j - 1) % L] == spin_lattice[(i - 1) % L, j]:
            initEnergy -= E
            Energy += E
            
        else:
            initEnergy -= E/2
            Energy += E
        
    return spin_lattice , - Energy / (MCmoves)

def main():
    L_list = [32, 64, 128, 256]
    MCmoves_list = [1_000_000, 1_000_000, 2_000_000, 5_000_000]
    save_path = cwd + 'ISING Model Results/'
    for i in tqdm(range(len(L_list))):
        Energy_list = []
        temp = np.linspace(1.53, 3.28, 64)
        
        for t in tqdm(temp):
            spin_lattice, Energy = ISING(L = L_list[i], T = t, MCmoves = MCmoves_list[i])
            Energy_list.append(Energy)
        
        plt.plot(temp, Energy_list, 'o')
        # plt.show()
        plt.savefig(save_path + 'E_T_' + str(L_list[i]) + 'x'+ str(L_list[i]) + '.svg')
        plt.close()

        break # only run for one lattice size

if __name__ == '__main__':
    main()
