# NonReversible_Mixtures

This repository contains the R code required to reproduce the experiments presented in the paper *A Fast Non-Reversible Sampler for Bayesian Mixture Models* by Ascolani et al. (2026).

The repository is organized as follows:

- `nrMCmix/`: An Rcpp package implementing all the methods discussed in the paper.

- `nrMCmix.cpp`: A standalone file containing the C++ and R code of the package. It can be compiled and loaded directly in R using `Rcpp::sourceCpp("nrMCmix.cpp")`.

- `nrMCtempmix.cpp`: A standalone file containing the C++ and R code for a version of the package implementing the deterministic even-odd parallel tempering scheme of Okabe et al. (2001).

- `COMBO17/`: Code to reproduce the results of Section 6, *Application to Astronomical Data*, using the COMBO17 dataset.

- `Figure_10/`: Code to reproduce Figure 10. See Section K of the paper for details of the simulation setup.

- `Figure_4/`: Code to reproduce Figure 4. See Section 6.3 of the paper for details of the simulation setup.

- `Figure_5/`: Code to reproduce Figure 5. See Section E.1 of the paper for details of the simulation setup.

- `Figure_1_2_3_6_7_8_9/`: Code to reproduce Figures 1, 2, 3, 6, 7, 8, and 9.

# References

Ascolani, F., Manildo, P. & Zanella, G. (2026). *A fast non-reversible sampler for Bayesian mixture models*. arXiv. https://arxiv.org/abs/2510.03226

Okabe, T., Kawata, M., Okamoto, Y., & Mikami, M. (2001). Replica-exchange Monte Carlo method for the isobaric-isothermal ensemble. *Chemical Physics Letters, 335*(5-6), 435-439.
