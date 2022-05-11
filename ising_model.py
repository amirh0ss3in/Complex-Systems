import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

""" The two-dimensional ISING model. """

tqdm.monitor_interval = 0
cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

np.random.seed(42)


def show_lattice(lattice, save_path = None):
    plt.imshow(lattice, cmap='Blues')
    if save_path is not None:
        plt.savefig(save_path)
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
    spin_lattice = np.random.choice([1], size=(L, L))
    initEnergy = calcEnergy(spin_lattice)
    Energy = 0
    
    initMagnetization = np.sum(spin_lattice)
    # E_set = {0, 4, 8, -8, -4}
    exp_E_T = {0:np.exp(0), 2:np.exp(-4/T), 4:np.exp(-8/T), -4:np.exp(8/T) , -2:np.exp(4/T)}

    for MCmove in tqdm(range(MCmoves)):
        i, j = np.random.randint(0, L, 2)
        E = spin_lattice[i, j] * (spin_lattice[(i + 1) % L, j] + spin_lattice[i, (j + 1) % L] + spin_lattice[i, (j - 1) % L] + spin_lattice[(i - 1) % L, j])
        # Magnetization +=  initMagnetization
        Magnetization += np.sum(spin_lattice)
        if np.random.rand() < min(1, exp_E_T[E]):
            spin_lattice[i, j] = - spin_lattice[i, j]
            # Magnetization += 2 * spin_lattice[i, j]

        if spin_lattice[(i + 1) % L, j] == spin_lattice[i, (j + 1) % L] == spin_lattice[i, (j - 1) % L] == spin_lattice[(i - 1) % L, j]:
            initEnergy -= E
            Energy += E
            
        else:
            initEnergy -= E/2
            Energy += E
        
    return spin_lattice , - Energy / MCmoves , Magnetization / (MCmoves*L*L)


def main():
    L_list = [32, 64, 128, 256]
    MCmoves_list = [500_000, 1_000_000, 2_000_000, 5_000_000]
    save_path = cwd + 'ISING Model Results/'
    for i in tqdm(range(len(L_list))):
        Energy_list = []
        Magnetization_list = []
        temp = np.linspace(1.6, 3.2, 64)
        
        for t in tqdm(temp):
            spin_lattice, Energy, Magnetization = ISING(L = L_list[i], T = t, MCmoves = MCmoves_list[i])
            Energy_list.append(Energy)
            Magnetization_list.append(Magnetization)
        specific_heat = np.gradient(Energy_list)

        plt.plot(temp, Energy_list, 'o')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Energy", fontsize=20)
        plt.tight_layout()
        plt.savefig(save_path + 'E_T_' + str(L_list[i]) + 'x'+ str(L_list[i]) + '.svg')
        plt.close()
        # plt.show()
        
        plt.plot(temp, Magnetization_list, 'o')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Magnetization", fontsize=20)
        plt.tight_layout()
        plt.savefig(save_path + 'M_T_' + str(L_list[i]) + 'x'+ str(L_list[i]) + '.svg')
        plt.close()
        # plt.show()

        plt.plot(temp, specific_heat, 'o')
        plt.title(r'Critical Temperature $T_{c}$: '+ str(np.round(temp[np.argmax(specific_heat)], 4)))
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Specific Heat", fontsize=20)
        plt.tight_layout()
        plt.savefig(save_path + 'C_T_' + str(L_list[i]) + 'x'+ str(L_list[i]) + '.svg')
        plt.close()
        # plt.show()


if __name__ == '__main__':
    main()
