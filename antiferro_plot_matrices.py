#! /usr/bin/env python3
# antiferro_plot_matrices.py
# Coraline Letouzé 31 dec 2020
# save a couple of antiferro samples in a gnuplot-friendly way

import numpy
import pickle

with open('antiferro0_T=0.5.pkl', 'rb') as file_in:
	x = pickle.load(file_in)
print(x.shape)
numpy.savetxt('../report/anti_samples_T=0.50.dat', x, delimiter=' ')

with open('antiferro0_T=2.25.pkl', 'rb') as file_in:
	x = pickle.load(file_in)
numpy.savetxt('../report/anti_samples_T=2.25.dat', x, delimiter=' ')

with open('antiferro0_T=3.75.pkl', 'rb') as file_in:
	x = pickle.load(file_in)
numpy.savetxt('../report/anti_samples_T=3.75.dat', x, delimiter=' ')

