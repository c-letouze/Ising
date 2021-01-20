#! /usr/bin/env python3
# Coraline Letouzé 
# last revision 4 jan 2021
# preprocess_CNN.py

import numpy as np
import pickle as pickle
from keras.backend import image_data_format

################ FUNCTIONS ################
def scaling(a):
    """ Scale the array a so that its values spread from 0 to 1."""
    min_a, max_a = np.min(a), np.max(a)
    return (a + min_a) / (max_a - min_a)
    
def preprocess_CNN(path_in, path_out, img_rows=40, img_cols=40):
	
	with open(path_in, 'rb') as file_in:
	    x_load, y = pickle.load(file_in)
	# reshape data, depending on Keras backend
	if image_data_format() == 'channels_first':
	    x = x_load.reshape(x_load.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x = x_load.reshape(x_load.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	# re-scale   
	x_dump = scaling(x)
	#save
	dump_data = (x_dump, y)
	with open(path_out, 'wb') as file_out:
		pickle.dump(dump_data, file_out)
	return
    
################### MAIN ##################
print("Building the ferromagnetic CNN dataset.")
sys_type = 'ferro_'

print(" - Ordered set")
phase = 'ordered_'
path_in = './DataSets/' + sys_type + phase + 'set.pkl'
path_out = './CNN/' + sys_type + phase + 'CNN.pkl'
preprocess_CNN(path_in, path_out)

print(" - Disordered set")
phase = 'disordered_'
path_in = './DataSets/' + sys_type + phase + 'set.pkl'
path_out = './CNN/' + sys_type + phase + 'CNN.pkl'
preprocess_CNN(path_in, path_out)

print(" - Critical set")
phase = 'critical_'
path_in = './DataSets/' + sys_type + phase + 'set.pkl'
path_out = './CNN/' + sys_type + phase + 'CNN.pkl'
preprocess_CNN(path_in, path_out)

print("Building the anti-ferromagnetic CNN dataset.")
sys_type = 'anti_'

print(" - Ordered set")
phase = 'ordered_'
path_in = './DataSets/' + sys_type + phase + 'set.pkl'
path_out = './CNN/' + sys_type + phase + 'CNN.pkl'
preprocess_CNN(path_in, path_out)

print(" - Disordered set")
phase = 'disordered_'
path_in = './DataSets/' + sys_type + phase + 'set.pkl'
path_out = './CNN/' + sys_type + phase + 'CNN.pkl'
preprocess_CNN(path_in, path_out)

print(" - Critical set")
phase = 'critical_'
path_in = './DataSets/' + sys_type + phase + 'set.pkl'
path_out = './CNN/' + sys_type + phase + 'CNN.pkl'
preprocess_CNN(path_in, path_out)




