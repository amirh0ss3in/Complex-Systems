import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


# Program for simulation of Ballistic Deposition.

""" Authours:
        Amirhossein Rezaei
        Paria Norouzi Nik
        Ali Farhadi
        Amirhossein Sar Kaboudi

"""
np.random.seed(42)

def find_height(arr):
    """ finds the height of the surface """
    h = len(arr)
    for y in range(h - 1, -1, -1):
        if arr[y] == 1:
            return y
    else:
        return 0


def BD(surface, particles):
    """ performs the Ballistic Deposition algorithm """
    l, h = surface.shape
    maxs = []
    for _ in tqdm(range(particles)):
        pos = int(np.random.uniform(1, l - 1))
        arr_h1 = find_height(surface[pos - 1,:])
        arr_h2 = find_height(surface[pos, :])
        arr_h3 = find_height(surface[pos + 1, :])
        max_ = max(arr_h1, arr_h2, arr_h3)
        
        if max_ < h:
            if surface[pos,max_] == 1:
                surface[pos, max_+1] = 1
                continue
            elif max_ == -1:
                surface[pos, 0] = 1
                continue
            else:
                surface[pos, max_] = 1

        max_i = []
        for i in range(1,l-1):
            max_i.append([i, find_height(surface[i, :])-1])
        maxs.append(max_i)
    maxs = np.array(maxs)

    return surface , maxs 

  
def find_ones(surface):
    """ finds all the position of particles in the surface """
    XY = []
    for ix, x in enumerate(surface):
      for iy, y in enumerate(x):
        if y == 1:
          XY.append([ix, iy])
    return np.array(XY)

def plot_BD(l = 100, h = 250, particles = 10000, mid_bar = False, bar_height = 50):
    """ Plot surface using Ballistic Deposition """
    
    surface = np.zeros([l, h])

    if mid_bar:
        surface[l//2, :bar_height] = 1

    surface , maxs = BD(surface, particles)
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
    plot_BD()


if __name__ == '__main__':
    main()
