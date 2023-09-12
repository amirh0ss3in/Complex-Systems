"""  
    Simulation of 2D ISING model using 
    fast Numba JIT Compiler,
    parallel processing using Numba's prange,
    branchless computation,
    look up table for exponential function,
    and further optimization using custom random number generator.
    
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

@nb.njit(inline="always")
def rol64(x, k):
    return (x << k) | (x >> (64 - k))

@nb.njit(inline="always")
def xoshiro256ss_init():
    state = np.empty(4, dtype=np.uint64)
    maxi = (np.uint64(1) << np.uint64(63)) - np.uint64(1)
    for i in range(4):
        state[i] = np.random.randint(0, maxi)
    return state

@nb.njit(inline="always")
def xoshiro256ss(state):
    result = rol64(state[1] * np.uint64(5), np.uint64(7)) * np.uint64(9)
    t = state[1] << np.uint64(17)
    state[2] ^= state[0]
    state[3] ^= state[1]
    state[1] ^= state[2]
    state[0] ^= state[3]
    state[2] ^= t
    state[3] = rol64(state[3], np.uint64(45))
    return result

@nb.njit(inline="always")
def xoshiro_gen_values(N, state):
    '''
    Produce 2 integers between 0 and N and a simple-precision floating-point number.
    N must be a power of two less than 65536. Otherwise results will be biased (ie. not random).
    N should be known at compile time so for this to be fast
    '''
    rand_bits = xoshiro256ss(state)
    a = (rand_bits >> np.uint64(32)) % N
    b = (rand_bits >> np.uint64(48)) % N
    c = np.uint32(rand_bits) * np.float32(2.3283064370807974e-10)
    return (a, b, c)

@nb.njit(nogil=True)
def mcmove_generic(lattice, beta, N):
    '''
    Monte Carlo move using Metropolis algorithm.
    N must be a small power of two and known at compile time
    '''

    state = xoshiro256ss_init()

    lut = np.full(16, np.nan)
    for cost in (0, 4, 8, 12, 16):
        lut[cost] = np.exp(-cost*beta)

    for _ in range(N):
        for __ in range(N):
            a, b, c = xoshiro_gen_values(N, state)
            s =  lattice[a, b]
            dE = lattice[(a+1)%N,b] + lattice[a,(b+1)%N] + lattice[(a-1)%N,b] + lattice[a,(b-1)%N]
            cost = 2*s*dE

            # Branchless computation of s
            tmp = (cost < 0) | (c < lut[cost])
            s *= 1 - tmp * 2

            lattice[a, b] = s

    return lattice

@nb.njit(nogil=True)
def mcmove(lattice, beta, N):
    assert N in [16, 32, 64, 128, 256]
    if N == 16: return mcmove_generic(lattice, beta, 16)
    elif N == 32: return mcmove_generic(lattice, beta, 32)
    elif N == 64: return mcmove_generic(lattice, beta, 64)
    elif N == 128: return mcmove_generic(lattice, beta, 128)
    elif N == 256: return mcmove_generic(lattice, beta, 256)
    else: raise Exception('Not implemented')

@nb.njit(nogil=True)
def calcEnergy(lattice, N):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    # Center
    for i in range(1, len(lattice)-1):
        for j in range(1, len(lattice)-1):
            S = lattice[i,j]
            nb = lattice[i+1, j] + lattice[i,j+1] + lattice[i-1, j] + lattice[i,j-1]
            energy -= nb*S
    # Border
    for i in (0, len(lattice)-1):
        for j in range(1, len(lattice)-1):
            S = lattice[i,j]
            nb = lattice[(i+1)%N, j] + lattice[i,(j+1)%N] + lattice[(i-1)%N, j] + lattice[i,(j-1)%N]
            energy -= nb*S
    for i in range(1, len(lattice)-1):
        for j in (0, len(lattice)-1):
            S = lattice[i,j]
            nb = lattice[(i+1)%N, j] + lattice[i,(j+1)%N] + lattice[(i-1)%N, j] + lattice[i,(j-1)%N]
            energy -= nb*S
    return energy/2

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


    # make more data points near critcial temperature Tc = 2.269:
    T = np.linspace(0.5, 4.0, nT)
    Tc = 2.269
    T = np.concatenate((T[T<Tc], Tc + (T[T>=Tc]-Tc)**2))

    nT = len(T)

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
    
    N = 64
    start_time = timer()
    T,E,M,C,X = ISING_model(nT = 64, N = N, burnin = 8 * 10**4, mcSteps = 16 * 10**4)
    end_time = timer()

    print("Elapsed time: %g seconds" % (end_time - start_time))

    f = plt.figure(figsize=(18, 10))
    
    # figure title
    f.suptitle(f"Ising Model: 2D Lattice\nSize: {N}x{N}", fontsize=20)

    _ =  f.add_subplot(2, 2, 1 )
    plot_style = '-o'
    plt.plot(T, E, plot_style, color='Blue') 
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Energy ", fontsize=20)
    plt.axis('tight')


    _ =  f.add_subplot(2, 2, 2 )
    plt.plot(T, abs(M), plot_style, color='Red')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Magnetization ", fontsize=20)
    plt.axis('tight')


    _ =  f.add_subplot(2, 2, 3 )
    plt.plot(T, C, plot_style, color='Green')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Specific Heat ", fontsize=20)
    plt.axis('tight')


    _ =  f.add_subplot(2, 2, 4 )
    plt.plot(T, X, plot_style, color='Black')
    plt.xlabel("Temperature (T)", fontsize=20)
    plt.ylabel("Susceptibility", fontsize=20)
    plt.axis('tight')


    plt.show()

if __name__ == '__main__':
    main()
