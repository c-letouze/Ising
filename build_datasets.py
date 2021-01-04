#! /usr/bin/env python3
# Coraline Letouzé 
# last revision 1 jan 2021
# build_datasets.py

import numpy as np
import pickle as pickle

from sklearn.utils import shuffle

################ FUNCTIONS ################
def read_t(t,root="./"):
    """ Load the data from Mehta's dataset at temperature t."""
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    return np.unpackbits(data).astype(int).reshape(-1,1600)
    
def read_t_anti(t,root="./"):
    """ Load the antiferromagnetic at temperature t."""
    data = pickle.load(open(root+'T=%.2f.pkl'%t,'rb'))
    return data.astype(int).reshape(-1,1600)

def data_augmentation(a, shape=(40, 40)):
    """Return several transformations of the array a."""
    a = a.reshape(shape)
    b = np.rot90(a, 1)
    c = np.rot90(a, 2)
    d = np.rot90(a, 3)
    a_ud = np.flipud(a)
    a_lr = np.fliplr(a)
    b_ud = np.flipud(b)
    b_lr = np.fliplr(b)
    arr = np.row_stack((a.flatten(), b.flatten(), c.flatten(), d.flatten(), 
        a_ud.flatten(), a_lr.flatten(), b_ud.flatten(), b_lr.flatten()))
    return np.row_stack((arr, - arr))
    
################### MAIN ##################

# Temperature ranges
ordered_temp = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
critical_temp = [2.0, 2.25, 2.5]
disordered_temp = [2.75, 3.0, 3.25, 3.75, 4.0]

t_critical = 2.27

# build the sets of ferromagnetic samples
ferromagnetic = True
if ferromagnetic:
    
    n_samples_per_t = 10000
    size_sample = 1600
    
    # ordered set
    x = np.empty((n_samples_per_t * len(ordered_temp), size_sample))
    y = np.empty((n_samples_per_t * len(ordered_temp), 2))
    t_cnt = 0
    for t in ordered_temp:
        x_t = read_t(t, root='./Data_Mehta/')
        label = 0 if t < t_critical else 1
        y_t = np.full((n_samples_per_t, 2), [t, label])
        x[n_samples_per_t * t_cnt : n_samples_per_t * (t_cnt+1), :] = x_t
        y[n_samples_per_t * t_cnt : n_samples_per_t * (t_cnt+1), :] = y_t
        t_cnt +=1
    x, y = shuffle(x, y)
    #save with pickle
    path = './DataSets/ferro_ordered_set.pkl'   
    dump_data = (x, y)
    with open(path, 'wb') as file_in:
        pickle.dump(dump_data, file_in)
    
    # disordered set
    x = np.empty((n_samples_per_t * len(disordered_temp), size_sample))
    y = np.empty((n_samples_per_t * len(disordered_temp), 2))
    t_cnt = 0
    for t in disordered_temp:
        x_t = read_t(t, root='./Data_Mehta/')
        label = 0 if t < t_critical else 1
        y_t = np.full((n_samples_per_t, 2), [t, label])
        x[n_samples_per_t * t_cnt : n_samples_per_t * (t_cnt+1), :] = x_t
        y[n_samples_per_t * t_cnt : n_samples_per_t * (t_cnt+1), :] = y_t
        t_cnt +=1
    x, y = shuffle(x, y)
    #save with pickle
    path = './DataSets/ferro_disordered_set.pkl'    
    dump_data = (x, y)
    with open(path, 'wb') as file_in:
        pickle.dump(dump_data, file_in)
        
    # near critical
    x = np.empty((n_samples_per_t * len(critical_temp), size_sample))
    y = np.empty((n_samples_per_t * len(critical_temp), 2))
    t_cnt = 0
    for t in critical_temp:
        x_t = read_t(t, root='./Data_Mehta/')
        label = 0 if t < t_critical else 1
        y_t = np.full((n_samples_per_t, 2), [t, label])
        x[n_samples_per_t * t_cnt : n_samples_per_t * (t_cnt+1), :] = x_t
        y[n_samples_per_t * t_cnt : n_samples_per_t * (t_cnt+1), :] = y_t
        t_cnt +=1
    x, y = shuffle(x, y)
    #save with pickle
    path = './DataSets/ferro_critical_set.pkl'  
    dump_data = (x, y)
    with open(path, 'wb') as file_in:
        pickle.dump(dump_data, file_in)

# build the sets of antiferromagnetic samples
else:
    
    n_samples_per_t = 20
    augmentation = 16
    size_t = n_samples_per_t * augmentation
    size_sample = 1600
    
    
    # ordered set
    x = np.empty((size_t * len(ordered_temp), size_sample))
    y = np.empty((size_t * len(ordered_temp), 2))
    t_cnt = 0
    for t in ordered_temp:
        x_t = read_t_anti(t, root='./Data_antiferro/')
        label = 0 if t < t_critical else 1
        y_t = np.full((size_t, 2), [t, label])
        for i in range(n_samples_per_t):
            index_start = size_t*t_cnt + augmentation*i
            index_stop = size_t*t_cnt + augmentation*(i+1)
            x[index_start : index_stop, : ] = data_augmentation(x_t[i])
        y[size_t * t_cnt : size_t * (t_cnt+1), :] = y_t
        t_cnt +=1
    x, y = shuffle(x, y)
    #save with pickle
    path = './DataSets/anti_ordered_set.pkl'   
    dump_data = (x, y)
    with open(path, 'wb') as file_in:
        pickle.dump(dump_data, file_in)
    
    # disordered set
    x = np.empty((size_t * len(disordered_temp), size_sample))
    y = np.empty((size_t * len(disordered_temp), 2))
    t_cnt = 0
    for t in disordered_temp:
        x_t = read_t_anti(t, root='./Data_antiferro2/')
        label = 0 if t < t_critical else 1
        y_t = np.full((size_t, 2), [t, label])
        for i in range(n_samples_per_t):
            index_start = size_t*t_cnt + augmentation*i
            index_stop = size_t*t_cnt + augmentation*(i+1)
            x[index_start : index_stop, : ] = data_augmentation(x_t[i])
        y[size_t * t_cnt : size_t * (t_cnt+1), :] = y_t
        t_cnt +=1
    x, y = shuffle(x, y)
    #save with pickle
    path = './DataSets/anti_disordered_set.pkl'   
    dump_data = (x, y)
    with open(path, 'wb') as file_in:
        pickle.dump(dump_data, file_in)
        
    # critical
    # ordered set
    x = np.empty((size_t * len(critical_temp), size_sample))
    y = np.empty((size_t * len(critical_temp), 2))
    t_cnt = 0
    for t in critical_temp:
        x_t = read_t_anti(t, root='./Data_antiferro2/')
        label = 0 if t < t_critical else 1
        y_t = np.full((size_t, 2), [t, label])
        for i in range(n_samples_per_t):
            index_start = size_t*t_cnt + augmentation*i
            index_stop = size_t*t_cnt + augmentation*(i+1)
            x[index_start : index_stop, : ] = data_augmentation(x_t[i])
        y[size_t * t_cnt : size_t * (t_cnt+1), :] = y_t
        t_cnt +=1
    x, y = shuffle(x, y)
    #save with pickle
    path = './DataSets/anti_critical_set.pkl'   
    dump_data = (x, y)
    with open(path, 'wb') as file_in:
        pickle.dump(dump_data, file_in)
