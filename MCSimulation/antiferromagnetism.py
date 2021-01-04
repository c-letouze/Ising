#! /usr/bon/env python3
# Coraline Letouze
# last revision 7 nov 2020
# generate_samples.py
# Part of a Machine Learning project on the Ising model
# this script runs a specified algorithm
# to generate (and save) a given number of spin configurations
# thoses configurations must be taken at equilibrium
# and be uncorrelated

import sys
import time
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm

    
# initialization
def init_simulation(shape, init='random'):
    """
    This is a DocString.
    
    arg shape: a tuple (height, width)
    arg init: choose the initial configuration before relaxation;
        either 'random' (default) or 'uniform'
    
    return: (arr, param)
    """
    # initialize the array
    if init=='uniform':
        arr = np.ones(shape)
    else:
        rng = np.random.default_rng()
        arr = rng.choice(np.array([-1, 1]), shape) 
    return arr
    
# Physical quantities
def measure_magnetization(arr, param):
    """ 
    Return the magnetization per site of the system.
    
    arg arr:
    arg param: (beta, coupling, mag_moment
    
    return: a float
    """
    mag_moment = param[2]
    return mag_moment * np.mean(arr) 
    
def measure_energy(arr, param):
    """ 
    Return the energy per site of the system.
    
    It only takes into account the interactions between 
    first neighbors and assumes periodic boundary conditions.
    
    arg arr:
    arg param: (beta, coupling, mag_moment)
    
    return: a float
    """
    width = arr.shape[0]
    height = arr.shape[1]
    coupling_cst = param[1]
    
    # coupling with the upper neighor
    array_up = np.vstack((arr, np.zeros((width))))
    neighb_up = np.vstack((arr[-1, :], arr))
    coupling_up = array_up * neighb_up
    # coupling with the left-side neighbor
    array_left = np.hstack((arr, np.zeros((height, 1))))
    neighb_left = np.hstack((arr[:, -1].reshape(height, 1), arr))
    coupling_left = array_left * neighb_left
    # resulting coupling
    coupling_total = coupling_up[:-1, :] + coupling_left[:, :-1]
    return - coupling_cst * np.mean(coupling_total)
    
# local dynamics 
def delta_hamiltonian(arr, loc, param):
    """
    Compute the energy difference of the system when one spin 
    is flipped.
    
    arg arr: the array representing the Ising system 
    arg loc: (tuple) the flipped spin location (i, j)
    arg param
        
    return: a float
    """
    width = arr.shape[0]
    coupling = param[1]
    # build an extended array with boundary conditions
    vperiodic = np.vstack((arr[-1, :], arr, arr[0, :]))
    periodic = np.hstack((vperiodic[:, -1].reshape(width+2, 1),
                          vperiodic,
                          vperiodic[:, 0].reshape(width+2, 1)))
    # location (i, j) becomes (i+1, j+1) in 'periodic'
    i, j = loc[0]+1, loc[1]+1
    spin_orientation = periodic[i, j]
    neighbors_sum = periodic[i, j-1] + periodic[i, j+1] \
        + periodic[i-1, j] + periodic[i+1, j]
    return 2 * coupling * spin_orientation * neighbors_sum
 
def make_offer(arr, param):
    """
    Update the system configuration by accepting or rejecting 
    the flipping proposition.
        
    If delta_H < 0, exp(- beta * delta_H) > 1 > np.random.random()  
    and the offer is always accepted.
            
    arg arr: the array representing the Ising system
    arg param: (beta, coupling, mag_moment)
        
    return: (int)
        1 if the offer is accepted
        0 if the offer is rejected
    """
    beta = param[0]
    coupling = param[1]
    # pick a random spin and present the offer
    spin_row = np.random.randint(arr.shape[0])
    spin_col = np.random.randint(arr.shape[1])
        
    delta_H = delta_hamiltonian(arr, (spin_row, spin_col), param)
    # decision regarding the offer   
    if np.exp(- delta_H * beta) > np.random.random() :
        arr[spin_row, spin_col] *= -1
        return 1
    else:
        return 0
      
# time evolution     
def evolve(arr, n_iter, param):
    """
    Make the system evolve during a specified number of iterations

    arg arr:
    arg n_iter: (int) number of iterations
    arg param: (beta, coupling, mag_moment)
    """
    for cnt in range(int(n_iter * arr.size)):
        make_offer(arr, param)  
    
def monitor(arr, n_iter, param):
    """ 
    Make the system evolve during a specified number of iterations
    and keep track of its physical properties
    
    arg arr: 
    arg n_iter: (int) number of iterations
    arg param: (beta, coupling, mag_moment)
    
    return: a ndarray of shape (n_iter, 2) containing 
        for each iteration the energy per site and 
        magnetization per site of the system.
    """
    record = np.empty((n_iter, 2))
    
    for cnt_iter in range(n_iter):
        for cnt_loc in range(arr.size):
            make_offer(arr, param)
        record[cnt_iter, 0] = measure_energy(arr, param)
        record[cnt_iter, 1] = measure_magnetization(arr, param)
        
    return record   
    
# relaxation
    
def relax(arr, param, average_time=10, e=1e-2):
    """
    Make the freshly-initialized system evolve toward equilibrium
    and estimate the correlation time
    
    arg arr:
    arg average_time: (int) the number of iterations used for time average 
    arg prec: (float, >0) a precision criterion for equilibrium
    arg param: (beta, coupling, mag_moment)
    
    return: (float) the estimated correlation time
    """
    delta = 2*e
    mean = measure_energy(arr, param)
    t_cnt = 0
    while delta > e:
        rec = monitor(arr, average_time, param)
        mean_next = np.mean(rec[:, 0])
        delta = abs(mean_next - mean)
        mean = mean_next
        t_cnt += average_time
    return t_cnt
    
def randomize(arr, percent=0.1):
    assert percent > 0 and percent < 1
    width = arr.shape[0]
    height = arr.shape[1] 
    nb_spins = int(percent * width*height) 
    rng = np.random.default_rng() 
    for spin in range(nb_spins):
        spin_row = np.random.randint(arr.shape[0])
        spin_col = np.random.randint(arr.shape[1])
        arr[spin_row, spin_col] = rng.choice(np.array([-1, 1]), ) 

# loop
def generate_samples(shape, t_corr, nb_samples, params, path_to_pickle):
    """
    Generate nb_samples of the Ising system and save them with 'pickle'.
    
    arg shape: (nb rows (int), nb col (int)) the shape of the array to extract
    arg t_corr:(int) the estimated correlation time
    arg nb_samples: (int) the number of samples to generate
    arg param: (beta, coupling, mag_moment)
    arg path_to_pickle: (string) relative path to the pickle file for output    
    
    return: a list
    """
    print("Begin simulation")
    arr = init_simulation(shape)
    beta = param[0]
    for cnt_sample in range(nb_samples):
        print("    generate sample n°{}".format(cnt_sample))
        evolve(arr, t_corr, param)
        with open(path_to_pickle, 'ab') as file_out:
            pickle.dump(arr, file_out)
    print("End simulation")

# input/output
def plot_system(arr, temp):
    """ Plot the array."""
    plt.clf()       
    title_str = "A {} Ising system at T={}".format(np.shape(arr), temp)
    plt.imshow(arr, interpolation="none", cmap=coolwarm, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(title_str)
    plt.show()
    
def plot_history(hist):
    """ Plot the monitored physical quantities."""
    plt.clf()
    plt.suptitle("History")
    plt.subplot(2, 2, 1)
    plt.plot(hist['time'], hist['energy'], 'o--')
    plt.title("energy")
    plt.subplot(2, 2, 2)
    plt.plot(hist['time'], hist['heat_capa'], 'o--')
    plt.title("heat capacity")
    plt.subplot(2, 2, 3)
    plt.plot(hist['time'], hist['magnet'], 'o--')
    plt.title("magnetization")
    plt.subplot(2, 2, 4)
    plt.plot(hist['time'], hist['heat_capa'], 'o--')
    plt.title("magnetic susceptibility")
    plt.show()
    
def load_state(path_to_pickle, shape):
    """
    Load the last state saved in a pickle file.
    
    arg path_to_pickle: (string) relative path to the pickle file to load
    arg shape: (nb rows (int), nb col(int)) the shape of the array to extract
    
    return : an array of the specified shape
    """
    with open(path_to_pickle, 'rb') as file_in:
        raw = pickle.load(file_in)
    arr = raw[-np.prod(shape)].reshape(shape)
    return arr
    

    
#################################################################"

### global parameters

# arg temp: (float>0) temperature, between 0 and 4 here
temp_list = np.arange(4.0, 0.0, step=-0.25)
k_B = 1.0
beta_list = 1 / (k_B * temp_list)
# arg coupling: (float) the coupling constant J between 
#       first-neighbor spins
#       J>0 yields a ferromagnetism behavior;
#       J<0 leads to an anti-ferromagnetism transition
coupling = -1.0
mag_moment = 1.0

# geometry
width = 40
shape = (width, width)

nb_samples_per_temp = 20

for temp in temp_list:
    
    print("temp n°: ", temp)
    param = (1/ (k_B * temp), coupling, mag_moment)
    arr_t = np.empty((nb_samples_per_temp, width*width))
    
    for sample in range(nb_samples_per_temp):
        print('sample n°: ', sample)
        #initialization
        obj = init_simulation(shape, 'random')
        evolve(obj, 250, param)
        arr_t[sample, :] = obj.reshape((-1, ))
        
    path_to_pickle = '../Data_antiferro2/T={0:.2f}.pkl'.format(temp)   
    with open(path_to_pickle, 'wb') as file_out:
            pickle.dump(arr_t, file_out)

