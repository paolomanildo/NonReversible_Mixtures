#ifndef MODELS_H
#define MODELS_H

#include <iostream>
#include <RcppArmadillo.h>
#include "supportive_functions.h"

// -----------------------------------------------------------------------------
// --------------------------- FINITE MIXTURE CASE -----------------------------
// -----------------------------------------------------------------------------


// ----------------------- UNIVARIATE GAUSSIAN KNWON VARIANCE ------------------

class Gauss1 {
public:
  
  // FIELDS
  
  // data
  const arma::vec& y;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  arma::vec alpha;
  double sigma2;
  double mu0;
  double lambda0;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  double log_post;
  
  // parameters
  arma::vec mus;
  arma::vec sigma2s;
  
  // current statistical unit
  double y_i;
  
  // METHODS
  
  // constructor
  Gauss1(std::vector<std::vector<unsigned int>>& partition,
         const arma::vec& y_,
         const unsigned int& K_,
         const arma::vec& alpha_,
         const double& sigma2_,
         const double& mu0_,
         const double& lambda0_,
         const arma::uvec& c,
         const bool& gibbs,
         const bool& hyperpar_alpha_,
         const bool& hyperpar_baseline_,
         const double& a0_,
         const double& b0_,
         const double& scale_,
         const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ---------------------- UNIVARIATE GAUSSIAN UNKNWON VARIANCE -----------------

class Gauss2 {
public:
  
  // FIELDS
  
  // data
  const arma::vec& y;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double M,V,a_l,b_l,a_a,b_a,a_b,b_b;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  arma::vec alpha;
  double mu0;
  double lambda0;
  double alpha0;
  double beta0;
  
  
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  arma::vec sums_y;
  arma::vec sums_y2;
  double log_post;
  
  // parameters
  arma::vec mus;
  arma::vec lambdas;
  arma::vec alphas;
  arma::vec betas;
  
  // current statistical unit
  double y_i;
  
  // METHODS
  
  // constructor
  Gauss2(std::vector<std::vector<unsigned int>>& partition,
         const arma::vec& y_,
         const unsigned int& K_,
         const arma::vec& alpha_,
         const double& mu0_,
         const double& lambda0_,
         const double& alpha0_,
         const double& beta0_,
         const arma::uvec& c,
         const bool& gibbs,
         const bool& hyperpar_alpha_,
         const bool& hyperpar_baseline_,
         const double& a0_,
         const double& b0_,
         const double& M_,
         const double& V_,
         const double& a_l_,
         const double& b_l_,
         const double& a_a_,
         const double& b_a_,
         const double& a_b_,
         const double& b_b_,
         const double& scale_,
         const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ---------------------------- MULTIVARIATE GAUSSIAN --------------------------

class MGauss {
public:
  
  // FIELDS
  
  // data
  const arma::mat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  unsigned int D;
  arma::vec alpha;
  arma::vec mu0;
  double lambda0;
  double alpha0;
  arma::vec chol_Bm10;
  arma::umat idx_map;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  double log_post;
  
  // parameters
  arma::mat mus;
  arma::vec lambdas;
  arma::vec alphas;
  arma::mat chol_Bm1s;
  
  // current statistical unit
  arma::vec y_i;
  
  // METHODS
  
  // constructor
  MGauss(std::vector<std::vector<unsigned int>>& partition,
         const arma::mat& y_,
         const unsigned int& K_,
         const arma::vec& alpha_,
         const arma::vec& mu0_,
         const double& lambda0_,
         const double& alpha0_,
         const arma::mat& B0_,
         const arma::uvec& c,
         const bool& gibbs,
         const bool& hyperpar_alpha_,
         const bool& hyperpar_baseline_,
         const double& a0_,
         const double& b0_,
         const double& scale_,
         const double& m_);
  
  // update parameters
  void update_par(const unsigned int& k,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ---------------- MULTIVARIATE GAUSSIAN KNWON SPHERICAL VARIANCE -------------

class MGauss1 {
public:
  
  // FIELDS
  
  // data
  const arma::mat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  unsigned int D;
  arma::vec alpha;
  double sigma2;
  arma::vec mu0;
  double lambda0;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  double log_post;
  
  // parameters
  arma::mat mus;
  arma::vec sigma2s;
  
  // current statistical unit
  arma::vec y_i;
  
  // METHODS
  
  // constructor
  MGauss1(std::vector<std::vector<unsigned int>>& partition,
          const arma::mat& y_,
          const unsigned int& K_,
          const arma::vec& alpha_,
          const double& sigma2_,
          const arma::vec& mu0_,
          const double& lambda0_,
          const arma::uvec& c,
          const bool& gibbs,
          const bool& hyperpar_alpha_,
          const bool& hyperpar_baseline_,
          const double& a0_,
          const double& b0_,
          const double& scale_,
          const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ------------ MULTIVARIATE GAUSSIAN UNKNOWN DIAGONAL COVARIANCE --------------

class MGauss2 {
public:
  
  // FIELDS
  
  // data
  const arma::mat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  unsigned int D;
  arma::vec alpha;
  arma::vec mu0;
  double lambda0;
  double alpha0;
  double beta0;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  arma::mat sums_y;
  arma::mat sums_y2;
  double log_post;
  
  // parameters
  arma::mat mus;
  arma::vec lambdas;
  arma::vec alphas;
  arma::mat betas;
  
  // current statistical unit
  arma::vec y_i;
  
  // METHODS
  
  // constructor
  MGauss2(std::vector<std::vector<unsigned int>>& partition,
          const arma::mat& y_,
          const unsigned int& K_,
          const arma::vec& alpha_,
          const arma::vec& mu0_,
          const double& lambda0_,
          const double& alpha0_,
          const double& beta0_,
          const arma::uvec& c,
          const bool& gibbs,
          const bool& hyperpar_alpha_,
          const bool& hyperpar_baseline_,
          const double& a0_,
          const double& b0_,
          const double& scale_,
          const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// --------------------------------- POISSON -----------------------------------

class Poiss {
public:
  
  // FIELDS
  
  // data
  const arma::uvec& y;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  arma::vec alpha;
  double alpha0;
  double beta0;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  double log_post;
  
  // parameters
  arma::vec alphas;
  arma::vec betas;
  
  // current statistical unit
  unsigned int y_i;
  
  // METHODS
  
  // constructor
  Poiss(std::vector<std::vector<unsigned int>>& partition,
        const arma::uvec& y_,
        const unsigned int& K_,
        const arma::vec& alpha_,
        const double& alpha0_,
        const double& beta0_,
        const arma::uvec& c,
        const bool& gibbs,
        const bool& hyperpar_alpha_,
        const bool& hyperpar_baseline_,
        const double& a0_,
        const double& b0_,
        const double& scale_,
        const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// -------------------------------- BINOMIAL -----------------------------------

class Binom {
public:
  
  // FIELDS
  
  // data
  const arma::uvec& y;
  const arma::uvec& n_trials;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  arma::vec alpha;
  double alpha0;
  double beta0;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  double log_post;
  
  // parameters
  arma::vec alphas;
  arma::vec betas;
  
  // current statistical unit
  unsigned int y_i, n_i;
  
  // METHODS
  
  // constructor
  Binom(std::vector<std::vector<unsigned int>>& partition,
        const arma::uvec& y_,
        const arma::uvec& n_trials_,
        const unsigned int& K_,
        const arma::vec& alpha_,
        const double& alpha0_,
        const double& beta0_,
        const arma::uvec& c,
        const bool& gibbs,
        const bool& hyperpar_alpha_,
        const bool& hyperpar_baseline_,
        const double& a0_,
        const double& b0_,
        const double& scale_,
        const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ----------------------------- BERNOULLI PRODUCT -----------------------------

class MBern {
public:
  
  // FIELDS
  
  // data
  const arma::umat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  unsigned int D;
  arma::vec alpha;
  double alpha0;
  double beta0;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  arma::mat n_dk1;
  double log_post;
  
  // current statistical unit
  arma::uvec y_i;
  
  // METHODS
  
  // constructor
  MBern(std::vector<std::vector<unsigned int>>& partition,
        const arma::umat& y_,
        const unsigned int& K_,
        const arma::vec& alpha_,
        const double& alpha0_,
        const double& beta0_,
        const arma::uvec& c,
        const bool& gibbs,
        const bool& hyperpar_alpha_,
        const bool& hyperpar_baseline_,
        const double& a0_,
        const double& b0_,
        const double& scale_,
        const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// -------------------------------- PARTITION ----------------------------------

class Partition {
public:
  
  // FIELDS
  
  // data
  
  // slice sampler stuff
  bool hyperpar_alpha;
  bool hyperpar_baseline;
  double a0, b0;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K;
  arma::vec alpha;
  
  // sufficient statistics
  unsigned int n;
  arma::vec ns;
  double log_post;
  
  // METHODS
  
  // constructor
  Partition(std::vector<std::vector<unsigned int>>& partition,
            const unsigned int& K_,
            const arma::vec& alpha_,
            const arma::uvec& c,
            const bool& gibbs,
            const bool& hyperpar_alpha_,
            const bool& hyperpar_baseline_,
            const double& a0_,
            const double& b0_,
            const double& scale_,
            const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// -----------------------------------------------------------------------------
// -------------------------- INFINITE MIXTURE CASE ----------------------------
// -----------------------------------------------------------------------------

// ----------------------- UNIVARIATE GAUSSIAN KNWON VARIANCE ------------------

class inf_Gauss1 {
public:
  
  // FIELDS
  
  // data
  const arma::vec& y;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  double alpha;
  double delta;
  double sigma2;
  double mu0;
  double lambda0;
  
  double sigma20;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  double log_post;
  
  // parameters
  std::vector<double> mus;
  std::vector<double> sigma2s;
  
  // current statistical unit
  double y_i;
  
  // METHODS
  
  // constructor
  inf_Gauss1(std::vector<std::vector<unsigned int>>& partition,
             std::unordered_map<unsigned int,unsigned int>& lbl2atm,
             std::unordered_map<unsigned int, unsigned int>& atm2lbl,
             std::vector<unsigned int>& labels,
             const arma::vec& y_,
             const double& alpha_,
             const double& delta_,
             const double& sigma2_,
             const double& mu0_,
             const double& lambda0_,
             arma::uvec& c,
             const bool& gibbs,
             const bool& hyperpar_alpha_,
             const bool& hyperpar_delta_,
             const bool& hyperpar_baseline_,
             const double& a0_,
             const double& b0_,
             const double& lwr_bound_,
             const double& upr_bound_,
             const double& scale_,
             const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ---------------------- UNIVARIATE GAUSSIAN UNKNWON VARIANCE -----------------

class inf_Gauss2 {
public:
  
  // FIELDS
  
  // data
  const arma::vec& y;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double M,V,a_l,b_l,a_a,b_a,a_b,b_b;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  double alpha;
  double delta;
  double mu0;
  double lambda0;
  double alpha0;
  double beta0;
  
  double prior_term1;
  double prior_term2;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  std::vector<double> sums_y;
  std::vector<double> sums_y2;
  double log_post;
  
  // parameters
  std::vector<double> mus;
  std::vector<double> lambdas;
  std::vector<double> alphas;
  std::vector<double> betas;
  
  // current statistical unit
  double y_i;
  
  // METHODS
  
  // constructor
  inf_Gauss2(std::vector<std::vector<unsigned int>>& partition,
             std::unordered_map<unsigned int,unsigned int>& lbl2atm,
             std::unordered_map<unsigned int, unsigned int>& atm2lbl,
             std::vector<unsigned int>& labels,
             const arma::vec& y_,
             const double& alpha_,
             const double& delta_,
             const double& mu0_,
             const double& lambda0_,
             const double& alpha0_,
             const double& beta0_,
             arma::uvec& c,
             const bool& gibbs,
             const bool& hyperpar_alpha_,
             const bool& hyperpar_delta_,
             const bool& hyperpar_baseline_,
             const double& a0_,
             const double& b0_,
             const double& lwr_bound_,
             const double& upr_bound_,
             const double& M_,
             const double& V_,
             const double& a_l_,
             const double& b_l_,
             const double& a_a_,
             const double& b_a_,
             const double& a_b_,
             const double& b_b_,
             const double& scale_,
             const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ---------------------------- MULTIVARIATE GAUSSIAN --------------------------

class inf_MGauss {
public:
  
  // FIELDS
  
  // data
  const arma::mat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, bb0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  unsigned int D;
  double alpha;
  double delta;
  arma::vec mu0;
  double lambda0;
  double alpha0;
  arma::mat B0;
  arma::vec chol_Bm10;
  arma::umat idx_map;
  double a10;
  double a20;
  double b0;
  double prior_term;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  double log_post;
  
  // parameters
  std::vector<arma::vec> mus;
  std::vector<double> lambdas;
  std::vector<double> alphas;
  std::vector<arma::vec> chol_Bm1s;
  
  // current statistical unit
  arma::vec y_i;
  
  // METHODS
  
  // constructor
  inf_MGauss(std::vector<std::vector<unsigned int>>& partition,
             std::unordered_map<unsigned int,unsigned int>& lbl2atm,
             std::unordered_map<unsigned int, unsigned int>& atm2lbl,
             std::vector<unsigned int>& labels,
             const arma::mat& y_,
             const double& alpha_,
             const double& delta_,
             const arma::vec& mu0_,
             const double& lambda0_,
             const double& alpha0_,
             const arma::mat& B0_,
             arma::uvec& c,
             const bool& gibbs,
             const bool& hyperpar_alpha_,
             const bool& hyperpar_delta_,
             const bool& hyperpar_baseline_,
             const double& a0_,
             const double& b0_,
             const double& lwr_bound_,
             const double& upr_bound_,
             const double& scale_,
             const double& m_);
  
  // update parameters
  void update_par(const unsigned int& k,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ---------------- MULTIVARIATE GAUSSIAN KNWON SPHERICAL VARIANCE -------------

class inf_MGauss1 {
public:
  
  // FIELDS
  
  // data
  const arma::mat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  unsigned int D;
  double alpha;
  double delta;
  double sigma2;
  arma::vec mu0;
  double lambda0;
  
  double sigma20;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  double log_post;
  
  // parameters
  std::vector<arma::vec> mus;
  std::vector<double> sigma2s;
  
  // current statistical unit
  arma::vec y_i;
  
  // METHODS
  
  // constructor
  inf_MGauss1(std::vector<std::vector<unsigned int>>& partition,
              std::unordered_map<unsigned int,unsigned int>& lbl2atm,
              std::unordered_map<unsigned int, unsigned int>& atm2lbl,
              std::vector<unsigned int>& labels,
              const arma::mat& y_,
              const double& alpha_,
              const double& delta_,
              const double& sigma2_,
              const arma::vec& mu0_,
              const double& lambda0_,
              arma::uvec& c,
              const bool& gibbs,
              const bool& hyperpar_alpha_,
              const bool& hyperpar_delta_,
              const bool& hyperpar_baseline_,
              const double& a0_,
              const double& b0_,
              const double& lwr_bound_,
              const double& upr_bound_,
              const double& scale_,
              const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ------------ MULTIVARIATE GAUSSIAN UNKNOWN DIAGONAL COVARIANCE --------------

class inf_MGauss2 {
public:
  
  // FIELDS
  
  // data
  const arma::mat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  unsigned int D;
  double alpha;
  double delta;
  arma::vec mu0;
  double lambda0;
  double alpha0;
  double beta0;
  double prior_term;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  std::vector<arma::vec> sums_y;
  std::vector<arma::vec> sums_y2;
  double log_post;
  
  // parameters
  std::vector<arma::vec> mus;
  std::vector<double> lambdas;
  std::vector<double> alphas;
  std::vector<arma::vec> betas;
  
  arma::vec beta0_vec;
  arma::vec zero_vec;
  
  // current statistical unit
  arma::vec y_i;
  
  // METHODS
  
  // constructor
  inf_MGauss2(std::vector<std::vector<unsigned int>>& partition,
              std::unordered_map<unsigned int,unsigned int>& lbl2atm,
              std::unordered_map<unsigned int, unsigned int>& atm2lbl,
              std::vector<unsigned int>& labels,
              const arma::mat& y_,
              const double& alpha_,
              const double& delta_,
              const arma::vec& mu0_,
              const double& lambda0_,
              const double& alpha0_,
              const double& beta0_,
              arma::uvec& c,
              const bool& gibbs,
              const bool& hyperpar_alpha_,
              const bool& hyperpar_delta_,
              const bool& hyperpar_baseline_,
              const double& a0_,
              const double& b0_,
              const double& lwr_bound_,
              const double& upr_bound_,
              const double& scale_,
              const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// --------------------------------- POISSON -----------------------------------

class inf_Poiss {
public:
  
  // FIELDS
  
  // data
  const arma::uvec& y;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  double alpha;
  double delta;
  double alpha0;
  double beta0;
  
  double prior_term1;
  double prior_term2;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  double log_post;
  
  // parameters
  std::vector<double> alphas;
  std::vector<double> betas;
  
  // current statistical unit
  unsigned int y_i;
  
  // METHODS
  
  // constructor
  inf_Poiss(std::vector<std::vector<unsigned int>>& partition,
            std::unordered_map<unsigned int,unsigned int>& lbl2atm,
            std::unordered_map<unsigned int, unsigned int>& atm2lbl,
            std::vector<unsigned int>& labels,
            const arma::uvec& y_,
            const double& alpha_,
            const double& delta_,
            const double& alpha0_,
            const double& beta0_,
            arma::uvec& c,
            const bool& gibbs,
            const bool& hyperpar_alpha_,
            const bool& hyperpar_delta_,
            const bool& hyperpar_baseline_,
            const double& a0_,
            const double& b0_,
            const double& lwr_bound_,
            const double& upr_bound_,
            const double& scale_,
            const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// -------------------------------- BINOMIAL -----------------------------------

class inf_Binom {
public:
  
  // FIELDS
  
  // data
  const arma::uvec& y;
  const arma::uvec& n_trials;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  double alpha;
  double delta;
  double alpha0;
  double beta0;
  
  double prior_term;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  double log_post;
  
  // parameters
  std::vector<double> alphas;
  std::vector<double> betas;
  
  // current statistical unit
  unsigned int y_i, n_i;
  
  // METHODS
  
  // constructor
  inf_Binom(std::vector<std::vector<unsigned int>>& partition,
            std::unordered_map<unsigned int,unsigned int>& lbl2atm,
            std::unordered_map<unsigned int, unsigned int>& atm2lbl,
            std::vector<unsigned int>& labels,
            const arma::uvec& y_,
            const arma::uvec& n_trials_,
            const double& alpha_,
            const double& delta_,
            const double& alpha0_,
            const double& beta0_,
            arma::uvec& c,
            const bool& gibbs,
            const bool& hyperpar_alpha_,
            const bool& hyperpar_delta_,
            const bool& hyperpar_baseline_,
            const double& a0_,
            const double& b0_,
            const double& lwr_bound_,
            const double& upr_bound_,
            const double& scale_,
            const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ----------------------------- BERNOULLI PRODUCT -----------------------------

class inf_MBern {
public:
  
  // FIELDS
  
  // data
  const arma::umat& y;
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  unsigned int D;
  double alpha;
  double delta;
  double alpha0;
  double beta0;
  
  double prior_term1, prior_term2, prior_term3;
  arma::vec n_dk10;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  std::vector<arma::vec> n_dk1;
  double log_post;
  
  // current statistical unit
  arma::uvec y_i;
  
  // METHODS
  
  // constructor
  inf_MBern(std::vector<std::vector<unsigned int>>& partition,
            std::unordered_map<unsigned int,unsigned int>& lbl2atm,
            std::unordered_map<unsigned int, unsigned int>& atm2lbl,
            std::vector<unsigned int>& labels,
            const arma::umat& y_,
            const double& alpha_,
            const double& delta_,
            const double& alpha0_,
            const double& beta0_,
            arma::uvec& c,
            const bool& gibbs,
            const bool& hyperpar_alpha_,
            const bool& hyperpar_delta_,
            const bool& hyperpar_baseline_,
            const double& a0_,
            const double& b0_,
            const double& lwr_bound_,
            const double& upr_bound_,
            const double& scale_,
            const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

// ------------------------------ PARTITION ------------------------------------

class inf_Partition {
public:
  
  // FIELDS
  
  // data
  
  // slice sampler stuff
  bool hyperpar_alpha, hyperpar_delta, hyperpar_baseline;
  double a0, b0, lwr_bound, upr_bound;
  double scale;
  unsigned int m;
  
  // hyperparameters
  unsigned int K_atm;
  double alpha;
  double delta;
  
  // sufficient statistics
  unsigned int n;
  std::vector<double> ns;
  double log_post;
  
  // METHODS
  
  // constructor
  inf_Partition(std::vector<std::vector<unsigned int>>& partition,
                std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                std::vector<unsigned int>& labels,
                const double& alpha_,
                const double& delta_,
                arma::uvec& c,
                const bool& gibbs,
                const bool& hyperpar_alpha_,
                const bool& hyperpar_delta_,
                const bool& hyperpar_baseline_,
                const double& a0_,
                const double& b0_,
                const double& lwr_bound_,
                const double& upr_bound_,
                const double& scale_,
                const double& m_);
  
  // update parameters
  void update_par(const unsigned int& c_i,
                  double sign);
  
  // log predictive function
  double log_pred(const unsigned int& k);
  
  // log prior predictive
  double log_pred_prior();
  
  // function that set the statistical unit
  void set(const unsigned int& i);
  
  // function to generate the collapsed parameters
  // and save them to file
  void save_parameters(const std::string& filename,
                       const unsigned int& chain_id);
  
  // function to swap the position of two atoms
  void copy_atom(const unsigned int& atm1,
                 const unsigned int& atm2);
  
  // function to delete one atom
  void delete_last_atom();
  
  // function to add an atom
  void add_new_atom();
  
  // function to merge two atoms into one
  void merge_atoms(const unsigned int& atm1,
                   const unsigned int& atm2);
  
  // function to compute the log marginal likelihood of a given group of data
  double log_marginal(const unsigned int& atm);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
};

#endif // MODELS_H