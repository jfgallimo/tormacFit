# tormacFit
tormacFit fits TORMAC models to infrared light curves. Uses pydream for adaptive MCMC. 

For now, this is a script that runs in place. tormacFit.py looks for the infrared light curves: J_corrected.txt, H_corrected.txt, & K_corrected.txt; and the model grids (tar.gz files). The program performs an MCMC walk through parameter space, fitting the model light curves to the observed light curves. Results are sent to a "Results" subdirectory.

command line options

-a : use anisotropic AGN models (default: isotropic)
-n : specify the number iterations (default: 100000)
-b : specify the number of burn-in iterations (default: 10000)
-w : perform a wbic calculation; apply a thermodynamic MCMC run to calculate the Widely Applicable Bayes Criterion (wbic). WBIC is a thermodynamic estimate of the log evidence, which is stored under log_ps in the appropriate txt file.

Dependencies:
jackdream (a version of pydream modified to accept thermodynamic beta for WBIC calculations)
numpy
scipy
matplotlib
emcee (specifically the emcee.autocorr library)
