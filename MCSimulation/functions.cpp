/* functions.cpp
 * author: Coraline Letouzé
 * last revision: 21 dec 2020
 * goal : Monte-Carlo simulation of the Ising Model
 * This file provides functions used in `generator.cpp`
 * (Don't compile)
 */

// some 'include' may be unnecessary :
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xnpy.hpp"

xt::xarray<int> init_simulation(int nrow, int ncol, char i)
/* The configuration 'u' for 'up' makes a "cold" lattice,
 * with every spin = +1. Default is a random configuration 
 * following the uniform distribution of {0, 1}.
 */
{
    if (i == 'u')
    {
        xt::xarray<int> arr = xt::ones<int>({nrow, ncol});
        return arr;
    }
    else
    {
        xt::xarray<int> arr_tmp = xt::random::randint({nrow, ncol}, 0, 2);
        return 2*arr_tmp - 1;
    }   
}

void print_parameters(double* ptr)
/* Prints the physical parameters of the simulation.
 */
{
    std::cout << "Physical parameters:" << std::endl;
    std::cout << " - Beta factor: " << *(ptr) << std::endl;
    std::cout << " - Coupling constant: " << *(ptr+1) << std::endl;
    std::cout << " - Magnetic moment: " << *(ptr+2) << std::endl;
    return;
}

// measurements

double measure_magnetization(xt::xarray<int>* ptr_arr, double* ptr_param)
/* Returns the mean magnetization per spin of the model.
 */
{
    double mag_mmt = *(ptr_param+2);
    double mean = xt::mean(*ptr_arr)();
    return mag_mmt * mean;
}

double measure_energy(xt::xarray<int>* ptr_arr, double* ptr_param)
/* Returns the mean energy per site of the system.
 * Interactions here are between the first neighbors only. They are 
 * computed by creating one lattice moved one row up, and one lattice 
 * moved one column left with perdiodic boundary conditions.
 */
 {
    xt::xarray<double> arr = *ptr_arr;
    int nrow = arr.shape(0);
    int ncol = arr.shape(1);
    double coupling_cst = *(ptr_param+1);
    
    xt::xarray<int> periodic = xt::pad(arr, 1, xt::pad_mode::periodic);

    xt::xarray<int> zeros_row = xt::zeros<int>({1, ncol});
    xt::xarray<int> array_up = xt::vstack(xtuple(arr, zeros_row));
    xt::xarray<int> neighb_up = xt::view(periodic, xt::range(_, -1), 
                                         xt::range(1, -1));
    xt::xarray<int> inter_up = xt::view(array_up*neighb_up, 
                                        xt::range(_, -1), xt::all());
    
    xt::xarray<int> zeros_col = xt::zeros<int>({nrow, 1});
    xt::xarray<int> array_left = xt::hstack(xtuple(arr, zeros_col));
    xt::xarray<int> neighb_left = xt::view(periodic, xt::range(1, -1), 
                                           xt::range(_, -1));
    xt::xarray<int> inter_left = xt::view(array_left*neighb_left, 
                                          xt::all(), xt::range(_, -1));
    
    xt::xarray<int> inter_total = inter_up + inter_left;
    double mean = xt::mean<double>(inter_total)();
    
    return - coupling_cst * mean;
}


// dynamics

double delta_hamiltonian(xt::xarray<int> arr, int i_row, int j_col, 
                            double* ptr_param)
/* Computes the difference in energy between the current spin configuration 
 * and the one where the (i_row, j_col) spin is flipped.
 */
{
    double coupling_cst = *(ptr_param+1);
    
    xt::xarray<int> periodic = xt::pad(arr, 1, xt::pad_mode::periodic);
    
    int spin = periodic(i_row+1, j_col+1);
    int sum_neighbors = periodic(i_row, j_col+1) 
                        + periodic(i_row+1,j_col+2) 
                        + periodic(i_row+2, j_col+1) 
                        + periodic(i_row+1, j_col);
    
    return 2 * coupling_cst * spin * sum_neighbors;
} 

bool make_offer(xt::xarray<int>* ptr_arr, double* ptr_param)
/* Local updating of the spin configration following the Metropolis 
 * algorithm. An offer is made to flip a given spin; it is accepted if 
 * the new configuration is energetically beneficial, or rejected with a 
 * Boltzmann probability.
 */
{
    double beta = *(ptr_param);
    double coupling_cst = *(ptr_param+1);
    
    int i_row = xt::random::randint<int>({1}, 0, ptr_arr->shape(0))();
    int j_col = xt::random::randint<int>({1}, 0, ptr_arr->shape(1))();
    
    double delta_H = delta_hamiltonian(*ptr_arr, i_row, j_col, ptr_param);
    double random_nb = xt::random::rand<double>({1}, 0, 1)();
    if (std::exp(- delta_H * beta) > random_nb)
    {
        (*ptr_arr)(i_row, j_col) *= -1;
        return true;
    }
    else
    {
        return false;
    }
}
    
void evolve(xt::xarray<int>* ptr_arr, int niter, double* ptr_param)
/* Iterate the local updating of the system for a certain time.
 */
{
    int arr_size = ptr_arr->shape(0) * ptr_arr->shape(1);
    for (int iter=0; iter<niter*arr_size; iter++)
    {
        make_offer(ptr_arr, ptr_param);
    }
    return;
}

// monitoring

struct phys_quantities 
{
    double energy;
    double magnetization;
    double heat_capacity;
    double mag_susceptibility;
};

phys_quantities compute_physical_quantities(xt::xarray<int>* ptr_arr, 
                                            int niter, double* ptr_param)
/* Computes averaged-over-time values per site of the main physical 
 * quantities of the Ising model, namely energy and magnetization for the 
 * first-moment based quantities, the heat capacity and magnetic 
 * susceptibility for the second-moment derived quantities. 
 */
{
    xt::xarray<double> rec = xt::empty<double>({niter, 2});
    for (int iter=0; iter<niter; iter++)
    {
        evolve(ptr_arr, 1, ptr_param);
        rec(iter, 0) = measure_energy(ptr_arr, ptr_param);
        rec(iter, 1) = measure_magnetization(ptr_arr, ptr_param);
    }
    phys_quantities res;
    res.energy = xt::mean(xt::col(rec, 0))();
    res.magnetization = xt::mean(xt::col(rec, 1))();
    res.heat_capacity = (*ptr_param) * (*ptr_param) * xt::variance(xt::col(rec, 0))();
    res.mag_susceptibility = (*ptr_param) * xt::variance(xt::col(rec, 1))();
    return res;
}

void monitor_system(xt::xarray<int>* ptr_arr, xt::xarray<double>* ptr_hist,
        int nb_measurements, int duration_measurement, double* ptr_param)
/* Repeats the previous measurements.
 */
{
    for (int measure=0; measure<nb_measurements; measure++)
    {
        phys_quantities record = compute_physical_quantities(ptr_arr, 
                                        duration_measurement, ptr_param);
        (*ptr_hist)(measure, 0) = record.energy;
        (*ptr_hist)(measure, 1) = record.magnetization;
        (*ptr_hist)(measure, 2) = record.heat_capacity;
        (*ptr_hist)(measure, 3) = record.mag_susceptibility;
    }
    return;
}

int set_correlation_time(xt::xarray<int>* ptr_arr, double* ptr_param,
                        int average_time, int t_max, double prec=0.01)
/* Estimates the correlation time of the system. (It depends on the 
 * temperature.) The basic idea is to monitor the energy strating from a 
 * far-from-equilibrium configuration.
 */
{
    double delta = 2*prec;
    double mean = measure_energy(ptr_arr, ptr_param);
    std::cout << mean << std::endl;
    int time = 0;
    while (delta>prec and time<t_max)
    {
        phys_quantities rec = compute_physical_quantities(ptr_arr, 
                                                average_time, ptr_param);
        delta = fabs(mean - rec.energy);
        mean = rec.energy;
        time += average_time;
    }
    return (time<t_max) ? time : -1;
}
