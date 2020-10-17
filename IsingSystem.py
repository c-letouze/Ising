#! /usr/env/bin python3
# IsingSystem.py
# Coraline Letouze, 17 oct 2020
# part of the Ising project
# provides class IsingSystem

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm, binary

class IsingSystem:
    """A class that describes the spin system."""

    def __init__(self, n=40, J=1):
        """
        This is a docString.

        arg n: the number of sites on each side of the lattice
        arg J: the coupling constant ; J> 0 for a ferromagnetic material; J<0 for an antiferromagnetic one
        """
        self.nb_sites = n
        self.coupling = J
        rng = np.random.default_rng()
        self.lattice = rng.choice(np.array([-1, 1]), size=(n,n))

    def plot(self):
        """
        Plot the system as a n x n array.
        Black spots correspond to a spin up; white ones to a spin down.
        """
        plt.imshow(self.lattice, interpolation="none", cmap=coolwarm)
        plt.colorbar()
        plt.show()

    def hamiltonian(self):
        """ 
        Return the hamiltonian of the system.
        
        It takes into account the interactions between first neighbors only
        and assumes periodic boundary conditions.
        """

        array_up = np.vstack((self.lattice, np.zeros((self.nb_sites))))
        neighbors_up = np.vstack((self.lattice[-1, :], self.lattice))
        coupling_up = array_up * neighbors_up

        array_left = np.hstack((self.lattice, np.zeros((self.nb_sites, 1))))
        neighbors_left = np.hstack((self.lattice[:, -1].reshape(self.nb_sites, 1), self.lattice))
        coupling_left = array_left * neighbors_left

        coupling_total = coupling_up[:-1, :] + coupling_left[:, :-1]

        return - self.coupling * np.sum(coupling_total)
        
        
# Instanciate the IsingSystem object
ising = IsingSystem()
#ising.plot()
print(ising.hamiltonian())

test = False

if test:
    # test lattice
    print("Test lattice")
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    print(a)

    # coupling with upper neighbors
    print("Coupling with upper neighbors")
    a_up = np.vstack((a, np.zeros((3))))
    print(a_up)
    b_up = np.vstack((a[-1, :], a))
    print(b_up)
    print("result:")
    c = a_up * b_up
    print(c[:-1, :])

    # coupling with left neightbors
    print("Coupling with left neighbors")
    a_left = np.hstack((a, np.zeros((3, 1))))
    print(a_left)
    b_left = np.hstack((a[:, -1].reshape(3, 1), a))
    print(b_left)
    print("result:")
    d = a_left * b_left
    print(d[:, :-1])

    # result
    print("Sum over a numpy array")
    print(np.sum(a))
