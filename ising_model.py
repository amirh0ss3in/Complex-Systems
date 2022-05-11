import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

""" The two-dimensional ISING model. """


def show_lattice(lattice):
    plt.imshow(lattice, cmap='Blues')
    plt.show()


def ISING(L, T, MCmoves):
    spin_lattice = np.random.choice([-1, 1], size=(L, L))
    for _ in tqdm(range(MCmoves)):
        i, j = np.random.randint(0, L, 2)
        E = 2 * spin_lattice[i, j] * (spin_lattice[(i + 1) % L, j] + spin_lattice[i, (j + 1) % L] + spin_lattice[i, (j - 1) % L] + spin_lattice[(i - 1) % L, j])
        
        if np.random.rand() < min(1, np.exp(- E / T)):
            spin_lattice[i, j] = - spin_lattice[i, j]

    return spin_lattice 

def main():
    L_list = [32, 64, 128, 256, 512]
    MCmoves_list = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

    for i in range(len(L_list)):
        spin_lattice = ISING(L = L_list[i], T = 1.5, MCmoves = MCmoves_list[i])
        show_lattice(spin_lattice)

if __name__ == '__main__':
    main()
