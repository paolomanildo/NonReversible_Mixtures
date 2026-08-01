# NonReversible_Mixtures

This repository contains the R code required to reproduce the experiments presented in the paper *A Fast Non-Reversible Sampler for Bayesian Mixture Models* by F. Ascolani, P. Manildo, and G. Zanella.

The repository is organized as follows:

- `nrMCmix/`: An Rcpp package implementing all the methods discussed in the paper.

- `nrMCmix.cpp`: A standalone file containing both the C++ and R code of the package. It can be compiled and loaded directly in R using `Rcpp::sourceCpp("nrMCmix.cpp")`.

- `COMBO17/`: Code to reproduce the results of Section 7, *Application to Astronomical Data*, using the COMBO17 dataset.

- `Tempering/`: Code for studying the effects of non-reversible dynamics when incorporated into the local updates of tempering schemes.

- `Figure_4/`: Code to reproduce Figure 4. See Section 6.3 of the paper for details of the simulation setup.

- `Figure_7/`: Code to reproduce Figure 7. See Section D.1 of the paper for details of the simulation setup.

- `Figure_1_2_3_6_8_9_10/`: Code to reproduce Figures 1, 2, 3, 6, 8, 9, and 10.
