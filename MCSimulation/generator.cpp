/* generator.cpp
 * author: Coraline Letouzé
 * last revision: 21 dec 2020
 * goal : Monte-Carlo simulation of the Ising Model
 * compile with :
 * g++ -I ~/anaconda3/include/ generator.cpp -o generator.o
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include "functions.cpp"

/* ************************* MAIN ******************************* */

int main(int argc, char* argv[])
{    
    /* 
    // initialization
    std::cout << "Testing the array initialization" << std::endl;
  
    xt::xarray<int>res_u = init_simulation(3, 2, 'u');
    std::cout << "----Uniform array: " << std::endl;
    std::cout << res_u << std::endl;
  
    xt::xarray<int>res_r = init_simulation(3, 5, 'r');
    std::cout << "----Default array: " << std::endl;
    std::cout << res_r << std::endl;
    */

    // physical parameters
    double k_B = 1.0;
    double temperature = 4.0;
    double beta = 1/ k_B / temperature;
    double coupling_cst = 1.0;
    double mag_moment = 1.0;
  
    double param [3] = {beta, coupling_cst, mag_moment};
    print_parameters(param);
  
    /*
    // measurement
    double magnetization = measure_magnetization(&res_r, param);
    double energy = measure_energy(&res_r, param);
    double d_energy = delta_hamiltonian(res_u, 1, 1, param);
    std::cout << "Measurements" << std::endl;
    std::cout << "---- Magnetization: " << magnetization << std::endl;
    std::cout << "---- Energy: " << energy << std::endl;
    std::cout << "---- Delta in energy: " << d_energy << std::endl;
    
    // evolution
    std::cout << res_r << std::endl;
    bool isaccepted = make_offer(&res_r, param);
    std::cout << "Make an offer... " << isaccepted << std::endl; 
    std::cout << res_r << std::endl;
    
    std::cout << "Evolve... " << std::endl;
    evolve(&res_r, 50, param);
    std::cout << res_r << std::endl;

    std::cout << "New lattice: " << std::endl;
    xt::xarray<int> latt2 = init_simulation(3, 5, 'r');
    std::cout << latt2 << std::endl;
    
    phys_quantities physical = compute_physical_quantities(&latt2, 5, param);
    std::cout << latt2 << std::endl;
    std::cout << "energy: " << physical.energy << std::endl;
    std::cout << "magnetization: " << physical.magnetization << std::endl;
    *
    */
    
    // correlation time
    int rows = 40;
    int cols = rows;
    int size = rows * cols;
    xt::xarray<int> lattice = init_simulation(rows, cols, 'u');
    int duration = 10;
    int patience = 100;
    // BEWARE !!!
    // For hot systems, the correlation time should be evaluated starting
    // from 'up' configuration
    int t_corr = set_correlation_time(&lattice, param, duration, patience); 
    assert (t_corr > 0);
    std::cout << "correlation time: " << t_corr << std::endl;
    
    /* 
     * int nb_samples = ; // 50
     * xt::xarray<int> pile_array = xt::empty<int>({nb_samples, size});
     * 
     * for (int sample=0; sample<nb_samples; sample++)
     * {
     *      evolve(&lattice, t_corr, param);
     *      xt::row(pile_array, sample) = xt::flatten(lattice);
     * std::cout << sample << '/' << nb_samples << std::endl;
     * }
     * xt::dump_npy(, pile_array); //"T=025_save0.npy"
    */
    
    
    // description of the simulation in a text file
    /*
    std::ofstream descript;
    descript.open("simulation_T=025.txt");
    descript << "#Temperature" << std::endl << temperature << std::endl;
    descript << "#Beta factor" << std::endl << param[0] << std::endl;
    descript << "#Coupling constant" << std::endl << param[1] << std::endl;
    descript << "#Magnetic moment" << std::endl << param[2] << std::endl;
    descript << "#Number of Rows" << std::endl << rows << std::endl;
    descript << "#Number of Cols" << std::endl << cols << std::endl;
    descript << "#Correlation time" << std::endl << t_corr << std::endl;
    descript.close();
    */
    
    return 0;
}
