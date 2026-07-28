#ifndef SUPPORTIVE_FUNCTIONS_H
#define SUPPORTIVE_FUNCTIONS_H

#include <iostream>
#include <RcppArmadillo.h>

// FUNCTION TO CREATE THE TEMPORARY DIRECTORY TO SAVE DATA
std::string get_tempdir_cpp();

// FUNCTION THAT CREATES A VECTOR OF INDEXES WITH WHICH TO RETURN A MESSAGE ON THE CONSOLE
arma::uvec sequence(const unsigned int& N, const double& p);

// LOG SUM EXP  
double log_sum_exp(const double& a, const double& b);

double log_sum_exp(const arma::vec& x);

// g-balanced version
double log_sum_exp_g(const arma::vec& x, const unsigned int& g);

// fully informed version
double log_sum_exp_g(const arma::vec& x,
                     const arma::vec& log_n,
                     const unsigned int& g);

// FUNCTION FOR THE CALLING THE LOG BETA FUNCTION
arma::vec lbeta(const arma::vec& x, const arma::vec& y);

arma::vec lbeta(const double& x, const arma::vec& y);

arma::vec lbeta(const arma::vec& x, const double& y);

double lbeta(const double& x, const double& y);

arma::mat lbeta_mat(const arma::mat& x, const arma::mat& y);

// FUNCTION TO SIMULATE FROM A DIRICHLET DISTRIBUTION
arma::vec rdirichlet(const arma::vec& x);

// FUNCTION FOR THE GUMBEL-MAX TRICK
unsigned int gumbel_max(const arma::vec& log_prob);

// varying length vectors version
unsigned int gumbel_max(const std::vector<double>& log_prob,
                        const unsigned int& K);

// FUNCTION TO SELECT THE DIRECTION OF THE METROPOLIS-HASTINGS SCHEMES

// reversible variant
void set_directions(unsigned int& km,
                    unsigned int& kp,
                    const arma::uvec& c,
                    const unsigned int& K);

// non-reversible variant
void set_directions(unsigned int& km,
                    unsigned int& kp,
                    std::uint8_t& vel,
                    const std::vector<std::uint8_t>& dir,
                    unsigned int& idx_dir,
                    const unsigned int& n,
                    const arma::uvec& c,
                    const unsigned int& K);

// infinite mixture variant (reversible)
bool set_directions(unsigned int& old_lbl,
                    unsigned int& km,
                    unsigned int& kp,
                    const arma::uvec& c,
                    const unsigned int& K,
                    const std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                    const std::unordered_map<unsigned int, unsigned int>& atm2lbl);

// infinite mixture variant (non-reversible)
bool set_directions(unsigned int& old_lbl,
                    unsigned int& km,
                    unsigned int& kp,
                    std::uint8_t& vel,
                    const std::vector<std::uint8_t>& dir,
                    unsigned int& idx_dir,
                    const arma::uvec& c,
                    const unsigned int& K,
                    const std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                    const std::unordered_map<unsigned int, unsigned int>& atm2lbl);

// function that sample the new directional variables
void set_directions(std::vector<std::uint8_t>& dir,
                    std::uint8_t& vel,
                    unsigned int& idx_dir,
                    const unsigned int& old_lbl,
                    const unsigned int& new_lbl,
                    const unsigned int& K,
                    const std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                    const std::unordered_map<unsigned int, unsigned int>& atm2lbl);


// function that reshuffle at most m element of an element of the partition
void reshuffle_partition(unsigned int* c_k,
                         unsigned int& n_k,
                         const double& n_c_k,
                         const unsigned int& m);

// function that reshuffle an entire partition group
void shuffle_group(unsigned int* c_k,
                   const unsigned int& n_k,
                   const unsigned int& i,
                   const unsigned int& j);

// // function to perturb the momentum
// void perturb_momentum(std::vector<std::uint8_t>& dir,
//                       const double& theta);

// function to perturb the momentum
void perturb_momentum(std::vector<std::uint8_t>& dir,
                      const double& theta,
                      const unsigned int& K);

// FUNCTIONS FOR THE MULTIVARIATE GAUSSIAN KERNEL

// function that compute the determinant of a matrix given 
// its cholesky factor
double half_log_det_chol(const arma::vec& chol_Bm1,
                         const unsigned int& d);

// function that compute Rx, with R the right cholesky factor
// but always using the left unrolled version
arma::vec prod_Rchol_x(const arma::vec& x,
                       const arma::vec& chol_Bm1,
                       const arma::umat& idx_map,
                       const unsigned int& d);

// function that compute Lx, with K the left cholesky factor
// but always using the unrolled version
arma::vec prod_Lchol_x(const arma::vec& x,
                       const arma::vec& chol_Bm1,
                       const arma::umat& idx_map,
                       const unsigned int& d);

// function that update a cholesky vector
arma::vec chol_update(arma::vec chol_Bm1s,
                      arma::vec v,
                      const arma::umat& idx_map,
                      const bool& add);

// function to compute the inverse of the outer cholesky product
arma::mat inv_outer_chol(const arma::vec& chol_Bm1,
                         const arma::umat& idx_map,
                         const unsigned int& D);

// // hyperparameter full conditional
// double log_fc_alpha(const double& omega,
//                     const arma::vec& ns,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double& a,
//                     const double& b);
// 
// // slice sampler
// double slice_sampler_alpha(const double& alpha,
//                            const arma::vec& ns,
//                            const unsigned int& n,
//                            const unsigned int& K,
//                            const double& a,
//                            const double& b,
//                            const double& scale,
//                            const unsigned int& m);

// // concentration hyperparameter full conditional
// // under Unif(0,15) prior
// double log_fc_alpha(const double& omega,
//                     const double& delta,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double& lwr_bound,
//                     const double& upr_bound);
// 
// 
// // discount hyperparameter full conditional under Unif(0,1) prior
// double log_fc_delta(const double& omega,
//                     const double& alpha,
//                     const std::vector<double>& ns,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double& lwr_bound,
//                     const double& upr_bound);
// 
// // generic slice sampler
// double slice_sampler(double& log_post,
//                      const double& alpha,
//                      const double& delta,
//                      const bool& concentration,
//                      const std::vector<double>& ns,
//                      const unsigned int& n,
//                      const unsigned int& K,
//                      const double& lwr_bound,
//                      const double& upr_bound,
//                      const double& scale,
//                      const unsigned int& m);
// 
// // slice sampler for finite mixtures
// 
// // full conditional
// double log_fc_alpha(const double& omega,
//                     const arma::vec& ns,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double a0,
//                     const double b0);
// 
// // sampler
// double slice_sampler(double& log_post,
//                      const double& alpha,
//                      const arma::vec& ns,
//                      const unsigned int& n,
//                      const unsigned int& K,
//                      const double& a0,
//                      const double& b0,
//                      const double& scale,
//                      const unsigned int& m);

// function that convert a bit string into a decimal number
unsigned int binaryToDecimal(const arma::uvec& y);

// inverse transform
arma::uvec decimalToUvec(const unsigned int& sigma,
                         const unsigned int& D);

// FUNCTION THAT PRINT THE SAMPLER'S SPECIFICS
std::string describe_sampler(const std::string& kernel,
                             const unsigned int& K,
                             const unsigned int& N,
                             const bool& gibbs,
                             const bool& reversible,
                             const bool& informed,
                             const unsigned int& m);

// function for creating an aliasing table
void create_aliasing_table(arma::vec& log_prob,
                           arma::uvec& alias);

// function to sample from a fixed probability distribution through aliasing table
unsigned int sample_alias(const arma::vec& prob,
                          const arma::uvec& alias);

// URN SAMPLERS

// finite case
arma::uvec urn(const unsigned int& n,
               const arma::vec& alpha);

// infinite case
arma::uvec urn(const unsigned int& n,
               const double& alpha = 1.0,
               const double& delta = 0.0);

#endif // SUPPORTIVE_FUNCTIONS_H
