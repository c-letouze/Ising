#! /usr/env/bin python3
# IsingSystem.py
# Coraline Letouze
# last revision 25 oct 2020
# part of the Ising project
# provides class IsingSystem

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm

class IsingSystem:
    """A class that describes the spin system."""

    def __init__(self, n=40, J=1):
        """
        The class constructor builds a lattice of shape (n, n) 
        with a random initialization of the spin sites.

        arg n: the number of sites on each side of the lattice
        arg J: the coupling constant ; J>0 for a ferromagnetic material;
         J<0 for an antiferromagnetic one
        """
        self.nb_sites = n
        self.coupling = J
        rng = np.random.default_rng()
        self.lattice = rng.choice(np.array([-1, 1]), size=(n,n))
        self.build_periodic_lattice()
        self.compute_hamiltonian()

    def __str__(self):
        """Print a minimal description of the system."""
        return '{0}x{0} Ising system'.format(self.nb_sites)

    def __del__(self):
        print("The {0} has been deleted.".format(self))

    def build_periodic_lattice(self):
        """ 
        Build an extended array that integrates the periodic 
        boundary condition. 
        """
        vperiodic = np.vstack((self.lattice[-1, :],
                              self.lattice,
                               self.lattice[0, :]))
        self.periodic = np.hstack((vperiodic[:, -1].reshape(self.nb_sites+2, 1),
                                   vperiodic,
                                   vperiodic[:, 0].reshape(self.nb_sites+2, 1)))

    def compute_hamiltonian(self):
        """ 
        Return the hamiltonian of the system.
        
        It only takes into account the interactions between first neighbors
        and assumes periodic boundary conditions.
        """
        # coupling with the upper neighor
        array_up = np.vstack((self.lattice, np.zeros((self.nb_sites))))
        neighbors_up = np.vstack((self.lattice[-1, :], self.lattice))
        coupling_up = array_up * neighbors_up
        # coupling with the left-side neighbor
        array_left = np.hstack((self.lattice, np.zeros((self.nb_sites, 1))))
        neighbors_left = np.hstack((self.lattice[:, -1].reshape(self.nb_sites, 1), self.lattice))
        coupling_left = array_left * neighbors_left
        # resulting coupling
        coupling_total = coupling_up[:-1, :] + coupling_left[:, :-1]
        self.hamiltonian = - self.coupling * np.sum(coupling_total)
        return self.hamiltonian

    def delta_hamiltonian(self, spin_location):
        """
        Compute the energy difference of the system when one spin is flipped.
        
        arg spin_location: the flipped spin location (i, j)
        """
        # using the extended array with periodic boundaries
        # location (i, j) becomes (i+1, j+1) in self.periodic
        i, j = spin_location[0]+1, spin_location[1]+1
        spin_orientation = self.periodic[i, j]
        neighbors_sum = self.periodic[i, j-1] + self.periodic[i, j+1] \
            + self.periodic[i-1, j] + self.periodic[i+1, j]
        return 2 * self.coupling * spin_orientation * neighbors_sum

    def plot(self, title_string):
        """ Plot the system as a n x n array. """
        plt.clf()
        plt.imshow(self.lattice, interpolation="none", cmap=coolwarm)
        plt.colorbar()
        plt.title(title_string)
        plt.show()

    #def save(self, index=0):
        #file_name = "n{}_T{}_{}.csv".format(self.nb_sites, self.temperature,
        #                                    index)
        #np.savetxt(file_name, self.lattice)
