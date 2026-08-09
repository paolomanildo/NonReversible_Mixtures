# Figure 10

This folder contains the code required to reproduce the simulations and Figure 10 presented in Section K of the paper *A Fast Non-Reversible Sampler for Bayesian Mixture Models*.

The folder contains the following files:

* `nrMCtempmix.cpp`: C++ and R code implementing the parallel tempering version of the `nrMCmix` package. The implementation uses the deterministic even-odd parallel tempering scheme proposed by Okabe et al. (2001).

* `Figure_10.R`: R code used to generate Figure 10 from the simulation results.

* `create_temperature_ladder.R`: R code for constructing the temperature ladder used in the parallel tempering simulations. The temperature ladder is determined using the method proposed by Syed et al. (2022).

* `simulations.R`: R code used to run the simulations presented in Section K of the paper.

## References

Okabe, T., Kawata, M., Okamoto, Y., & Mikami, M. (2001). Replica-exchange Monte Carlo method for the isobaric-isothermal ensemble. *Chemical Physics Letters, 335*(5-6), 435-439.

Syed, S., Romaniello, M., Campbell, A., & Bouchard-Côté, A. (2022). Parallel tempering on optimized temperature ladders. *Journal of the American Statistical Association*.
