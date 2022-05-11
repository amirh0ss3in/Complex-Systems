import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

""" The two-dimensional ISING model. """

np.random.seed(42)

def show_lattice(lattice):
    plt.imshow(lattice, cmap='Blues')
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
    Energy = calcEnergy(spin_lattice)
    Energy_c = 0
    ee = 0
    for MCmove in tqdm(range(1, MCmoves)):
        i, j = np.random.randint(0, L, 2)
        E = spin_lattice[i, j] * (spin_lattice[(i + 1) % L, j] + spin_lattice[i, (j + 1) % L] + spin_lattice[i, (j - 1) % L] + spin_lattice[(i - 1) % L, j])
        if np.random.rand() < min(1, np.exp(- 2 * E / T)):
            spin_lattice[i, j] = - spin_lattice[i, j]

        if spin_lattice[(i + 1) % L, j] == spin_lattice[i, (j + 1) % L] == spin_lattice[i, (j - 1) % L] == spin_lattice[(i - 1) % L, j]:
            Energy -= E
            Energy_c += E
            
        else:
            Energy -= E/2
            Energy_c += E
        
    return spin_lattice , - Energy_c / (MCmoves)

def main():
    L_list = [32, 64, 128, 256]
    MCmoves_list = [100_000, 500_000, 1_000_000, 5_000_000]
    # spin_lattice, Energy = ISING(L = L_list[0], T = 1.5, MCmoves = 20_000)
    # show_lattice(spin_lattice)

    # for i in range(len(L_list)):
    #     spin_lattice = ISING(L = L_list[i], T = 1.5, MCmoves = MCmoves_list[i])
    #     show_lattice(spin_lattice)
    
    Energy_list = []
    temp = np.linspace(1.53, 3.28, 32)
    for t in tqdm(temp):
        spin_lattice, Energy = ISING(L = 64, T = t, MCmoves = 1_000_000)
        Energy_list.append(Energy)
    plt.plot(temp, Energy_list, 'o')
    plt.show()

if __name__ == '__main__':
    main()
