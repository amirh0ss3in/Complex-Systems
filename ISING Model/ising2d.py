"""  
    Simulation of 2D ISING model using fast Numba JIT Compiler.
    By 
        Amirhossein Rezaei
"""

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

@nb.njit(nogil=True)
def initialstate(N):   
    ''' 
    Generates a random spin configuration for initial condition in compliance with the Numba JIT compiler.
    '''
    state = np.empty((N,N),dtype=np.int8)
    for i in range(N):
        for j in range(N):
            state[i,j] = 2*np.random.randint(2)-1
    return state


@nb.njit(nogil=True, fastmath=True)
def mcmove(lattice, beta, N):
    '''
    Monte Carlo move using Metropolis algorithm 
    '''
    
    # calculate the exponentials of the energy differences only once. (A tricky optimization).
    exp_betas = np.exp(-beta*np.arange(0,9))

    for _ in range(N * N):
        a = np.random.randint(0, N)
        b = np.random.randint(0, N)
        s =  lattice[a, b]
        dE = lattice[(a+1)%N,b] + lattice[a,(b+1)%N] + lattice[(a-1)%N,b] + lattice[a,(b-1)%N]
        cost = 2*s*dE

        if cost < 0:
            s = -s
        
        elif np.random.rand() < exp_betas[cost]:
            s = -s

        lattice[a, b] = s

    return lattice



@nb.njit(nogil=True)
def calcEnergy(lattice, N):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    
    for i in range(len(lattice)):
        for j in range(len(lattice)):
            S = lattice[i,j]
            dE = lattice[(i+1)%N, j] + lattice[i,(j+1)%N] + lattice[(i-1)%N, j] + lattice[i,(j-1)%N]
            energy += -dE*S
    return energy//2


@nb.njit(nogil=True)
def calcMag(lattice):
    '''
    Magnetization of a given configuration
    '''
    mag = np.sum(lattice, dtype=np.int32)
    return mag

@nb.njit(nogil=True, parallel=True)
def ISING_model(nT, N, burnin, mcSteps):

    """ 
    nT      :         Number of temperature points.
    N       :         Size of the lattice, N x N.
    burnin  :         Number of MC sweeps for equilibration (Burn-in).
    mcSteps :         Number of MC sweeps for calculation.

    """


    T       = np.linspace(1.2, 3.8, nT)
    E,M,C,X = np.empty(nT, dtype= np.float32), np.empty(nT, dtype= np.float32), np.empty(nT, dtype= np.float32), np.empty(nT, dtype= np.float32)
    n1, n2  = 1/(mcSteps*N*N), 1/(mcSteps*mcSteps*N*N) 


    for temperature in nb.prange(nT):
        lattice = initialstate(N)         # initialise

        E1 = M1 = E2 = M2 = 0
        iT = 1/T[temperature]
        iT2= iT*iT
        
        for _ in range(burnin):           # equilibrate
            mcmove(lattice, iT, N)        # Monte Carlo moves

        for _ in range(mcSteps):
            mcmove(lattice, iT, N)           
            Ene = calcEnergy(lattice, N)  # calculate the Energy
            Mag = calcMag(lattice)        # calculate the Magnetisation
            E1 += Ene
            M1 += Mag
            M2 += Mag*Mag 
            E2 += Ene*Ene

        E[temperature] = n1*E1
        M[temperature] = n1*M1
        C[temperature] = (n1*E2 - n2*E1*E1)*iT2
        X[temperature] = (n1*M2 - n2*M1*M1)*iT

    return T,E,M,C,X


def main():
    
    N = 128
    start_time = timer()
    T,E,M,C,X = ISING_model(nT = 64, N = N, burnin = 2 * 10**5, mcSteps = 2 * 10**5)
    end_time = timer()

    print("Elapsed time: %g seconds" % (end_time - start_time))

    f = plt.figure(figsize=(18, 10)); #  

    # figure title
    f.suptitle(f"Ising Model: 2D Lattice\nSize: {N}x{N}", fontsize=20)

    _ =  f.add_subplot(2, 2, 1 )
    plt.plot(T, E, '-o', color='Blue') 
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Energy ", fontsize=20)
    plt.axis('tight')


    _ =  f.add_subplot(2, 2, 2 )
    plt.plot(T, abs(M), '-o', color='Red')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization ", fontsize=20)
    plt.axis('tight')


    _ =  f.add_subplot(2, 2, 3 )
    plt.plot(T, C, '-o', color='Green')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Specific Heat ", fontsize=20)
    plt.axis('tight')


    _ =  f.add_subplot(2, 2, 4 )
    plt.plot(T, X, '-o', color='Black')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    plt.axis('tight')


    plt.show()

if __name__ == '__main__':
    main()
