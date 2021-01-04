#! /usr/env/bin python3
# Metropolis.py
# Coraline Letouze
# last revision 25 oct 2020
# part of the Ising project
# provides class Metropolis

import ising
import numpy as np

class Metropolis():
    """ Makes the IsingSystem evolve toward equilibrium at a specified temperature. """

    k_B = 1    
    
    def __init__(self, T):
        """
        The class constructor.
        
        arg isingSys: an IsingSystem object
        arg T: the thermal bath's temperature
        """
        self.isingSys = ising.IsingSystem()
        self.temperature = T
        # time evolution
        self.iteration = self.isingSys.nb_sites**2
        self.time = 0

    def __str__(self):
        return 'Simulating a {0} at temperature {1}.'.format(self.isingSys, self.temperature)

    def __del__(self):
        print("Simulation of a {0} at temperature {1} has been deleted.".format(self.isingSys, self.temperature))

    # offer

    def print_offer(self, spin_location, delta_H):
        """ Print the offer. """
        print("\nOffer n° {}".format(self.iteration_cnt))
        print("The selected spin is located at {}".format(spin_location))
        i, j = spin_location[0]+1, spin_location[1]+1
        spin_center = self.isingSys.periodic[i, j]
        spin_up = self.isingSys.periodic[i-1, j]
        spin_down = self.isingSys.periodic[i+1, j]
        spin_left = self.isingSys.periodic[i, j-1]
        spin_right = self.isingSys.periodic[i, j+1]
        print("    {}   ".format(spin_up))
        print("{}  {}  {}".format(spin_left, spin_center, spin_right))
        print("    {}   ".format(spin_down))
        print("Delta H = {}".format(delta_H))
        
    def update(self, verbose=False):
        """
        Update the system configuration by accepting or rejecting the flipping proposition.

        arg verbose: if True, print information about the offer
        
        If delta_H < 0, exp(- beta * delta_H) > 1 > np.random.random()  and the offer is always accepted.
        """
        # pick a random spin and present the offer
        spin_row, spin_col = np.random.randint(self.isingSys.nb_sites, size=2)
        delta_H = self.isingSys.delta_hamiltonian((spin_row, spin_col))
        if verbose: 
            self.print_offer((spin_row, spin_col), delta_H)
        # decision regarding the offer   
        if np.exp(- delta_H / self.temperature / self.k_B ) > np.random.random() :
            self.isingSys.lattice[spin_row, spin_col] *= -1
            self.isingSys.build_periodic_lattice()
            self.isingSys.hamiltonian += delta_H
            if verbose: 
                print("the offer is accepted")
        else:
            if verbose: 
                print("the offer is rejected")

    # dynamics

    def run_iteration(self, verbose=False):
        """ Make as many offers as there are spins in the lattice. """
        cnt = 0
        while cnt < self.iteration:
            self.update( verbose)
            cnt += 1

    def evolve(self, n_iter, verbose=False):
        """
        Make the system evolve during a specified number 
        of iterations and keep track of its physical properties.

        arg n_iter: number of iterations;
        arg verbose: if True, display information
        """
        cnt_iter = 0
        record = np.empty((n_iter+1, 2))
        energy_per_site = self.isingSys.hamiltonian / self.isingSys.nb_sites ** 2
        record[cnt_iter] = np.array([self.time, energy_per_site])
        while cnt_iter < n_iter:
            self.run_iteration(verbose)
            self.time += 1
            cnt_iter += 1
            energy_per_site = self.isingSys.hamiltonian / self.isingSys.nb_sites ** 2
            record[cnt_iter] = np.array([self.time, energy_per_site])
        return record

    # save configurations
    
    def plot(self):
        """ Plot the system as a n x n array. """
        self.isingSys.plot("Temperature: {}".format(self.temperature))
    
    def generate_configurations(self, nb_config, verbose=False):
        pass

    # Physical properties

    def compute_magnetization(self):
        """ Return the magnetization of the Ising system."""
        return np.sum(self.isingSys.lattice)

    def compute_susceptibility(self):
        pass

    def compute_heat_capacity(self):
        pass