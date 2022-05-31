import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Program for simulation of Relaxation Random Deposition.

""" Authours:
        Paria Norouzi Nik
        Amirhossein Rezaei
        Amirhossein Sarkaboodi
"""
np.random.seed(42)

def find_height(arr):
    """ finds the height of the surface """
    h = len(arr)
    for y in range(h - 1, -1, -1):
        if arr[y] == 1:
            return y + 1
    else:
        return 0


def RRD(surface, particles):
    """ performs the Relaxation Random Deposition algorithm """
    l, h = surface.shape
    maxs = []
    for _ in tqdm(range(particles)):
        pos = int(np.random.uniform(1, l - 1))
        arr_h1 = find_height(surface[pos - 1,:])
        arr_h2 = find_height(surface[pos, :])
        arr_h3 = find_height(surface[pos + 1, :])
        min_ = min(arr_h1, arr_h2, arr_h3)

        if min_ < h:
          if arr_h1 < arr_h2 and arr_h1 < arr_h3:
            if arr_h1 == -1:
              surface[pos-1, 0] = 1
              continue
            else:
              surface[pos-1, arr_h1 + 1] = 1

          elif arr_h2 <= arr_h1 and arr_h2 <= arr_h3:
            if arr_h2 == -1:
              surface[pos, 0] = 1
              continue
            else:
              surface[pos, arr_h2 + 1] = 1

          elif arr_h3 < arr_h1 and arr_h3 < arr_h2:
            if arr_h3 == -1:
              surface[pos + 1, 0] = 1
              continue
            else:
              surface[pos + 1, arr_h3 + 1] = 1
        av_heihgt = []
        max_i = []
        for i in range(1,l-1):
            max_i.append([i, find_height(surface[i, :])-1])
        maxs.append(max_i)
        av_heihgt = np.mean(max_i, axis=0)

    maxs = np.array(maxs)

    return surface , maxs , av_heihgt

  
def find_ones(surface):
    """ finds all the position of particles in the surface """
    XY = []
    for ix, x in enumerate(surface):
      for iy, y in enumerate(x):
        if y == 1:
          XY.append([ix, iy])
    return np.array(XY)

def plot_RRD(l, h, particles):
    
    surface = np.zeros([l, h])
    surface , maxs, av_heihgt = RRD(surface, particles)
    wt = []
    for i in maxs:
        wt.append(np.std(i,axis=0)[1])
    
    plt.plot(wt)
    plt.xlabel(xlabel='Number of Particles', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    XY = find_ones(surface)
    plt.plot(XY[:, 0], XY[:, 1], 'o', markersize = 2)
    plt.plot(maxs[-1][:,0], maxs[-1][:,1])
    plt.xlim([0, l]) 
    plt.ylim([0, h])
    plt.gca().set_aspect('equal')
    plt.show()


def main():
    plot_RRD(l = 100, h = 500, particles = 20000)

if __name__ == '__main__':
    main()
