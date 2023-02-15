# Error-Estimates-for-MM-Progress-Curves
The repository contains the core 4 Python scripts used for the numerical simulations of paper:

"On the estimation errors of K<sup>M<\sup> and V from time-course experiments using the Michaelis–Menten equation"
Wylie Stroberg, Santiago Schnell 
Biophysical Chemistry
Volume 219, December 2016, Pages 17-27


* MM_error.py - This script contains all the functions needed for the analysis. There are a number of 
functions, which were not used in the analysis.

* MM_plot_conc_error.py - Uses the above functions to calculate errors in the concentration estimated 
via the MM equations/Schnell-Mendoza equation from time course data. 

* The other two calculate the condition number for the fit and compare the fitting of 
noisy (MM_data_noise_plot.py) versus deterministic data (MM_condition_number.py).

The scrips are not well-commented; this will be done in the future.

If you have any questions, please contact:

- Wylie Stroberg, University of Alberta, E-mail: stroberg@ualberta.ca
- Santiago Schnell, University of Notre Dame, E-mail: santiago.schnell@nd.edu
