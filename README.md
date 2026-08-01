# NonReversible_Mixtures
The repository contains R code to reproduce the experiments of the paper 'A fast non-reversible sampler for Bayesian mixture models' by F.Ascolani, P.Manildo and G.Zanella

The 'nrMCmix' folder contains an Rcpp package that implements all methods discussed in the article.
The 'nrMCmix.cpp' file contains the C++ and R code of the package into a single file, which can be loaded directly by callin Rcpp::sourceCpp("nrMCmix.cpp")
