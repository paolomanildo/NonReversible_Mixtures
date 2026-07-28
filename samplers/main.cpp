#include <iostream>
#include <RcppArmadillo.h>
#include <variant>
#include "supportive_functions.h"
#include "samplers.h"
#include "models.h"

// [[Rcpp::depends(RcppArmadillo)]]

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

// FUNCTION TO COMPUTE THE NON PARAMETRIC PRIOR ON THE NUMBER OF CLUSTER
//' Prior distribution on the number of clusters
//'
//' Computes the prior distribution of the number of clusters \(K_n\)
//' induced by a two-parameter Bayesian nonparametric process
//' (e.g. Dirichlet process when \code{delta = 0}, or Pitman–Yor process
//' when \code{delta > 0}) for a sample of size \code{n}.
//'
//' The probabilities are obtained via a stable recursion on the log-scale,
//' and the final result is returned on the natural scale.
//'
//' @param alpha A positive concentration parameter.
//' @param delta A discount parameter. Must satisfy \eqn{0 \le \delta < 1}.
//'   When \code{delta = 0}, the distribution reduces to the Dirichlet
//'   process prior.
//' @param n Sample size (number of observations).
//'
//' @return A numeric vector of length \code{n}, where the \eqn{k}-th entry
//'   (in position \code{k}) corresponds to \eqn{P(K_n = k)} for
//'   \eqn{k = 1, \dots, n}.
//'
//' @details
//' The function uses a dynamic programming recursion to compute
//' \eqn{P(K_n = k)} for all \eqn{k}, storing intermediate results
//' on the log-scale to ensure numerical stability. The recursion
//' follows the predictive structure of Gibbs-type priors.
//'
//' @examples
//' \dontrun{
//' priorK(alpha = 1.0, delta = 0.0, n = 50)  # Dirichlet process case
//' priorK(alpha = 1.0, delta = 0.5, n = 50)  # Pitman–Yor process
//' }
//'
//' @export priorK
// [[Rcpp::export]]
arma::vec priorK(const double& alpha = 1.0,
                 const double& delta = 0.0,
                 const unsigned int& n = 100L){
  
  // -------------------- INPUT VALIDATION --------------------
  
  if(alpha <= 0.0){
    Rcpp::stop("'alpha' must be > 0.");
  }
  
  if(delta < 0.0 || delta >= 1.0){
    Rcpp::stop("'delta' must satisfy 0 <= delta < 1.");
  }
  
  if(n == 0){
    Rcpp::stop("'n' must be >= 1.");
  }
  
  // initialize the matrix with recursive probability
  arma::mat log_P(n,n);
  log_P(0,0) = 0.0;
  
  // loop over every row
  for(unsigned int i = 1; i < n; i++){
    
    // update the first column
    log_P(i,0) = std::log( 1 - (alpha + delta) / (alpha + i) ) + log_P(i-1,0);
    
    // update the rest until the last column
    for(unsigned int k = 1; k < i; k++){
      
      log_P(i,k) = log_sum_exp( std::log(1 - (alpha + delta * (k+1)) / (alpha + i) ) + log_P(i-1,k),
            std::log( (alpha + delta * k) / (alpha + i) ) + log_P(i-1,k-1));

    }
    
    // update the last column
    log_P(i,i) = std::log( (alpha + delta*i) / (alpha + i) ) + log_P(i-1,i-1);
    
  }
  
  // return the last row of the matrix
  return arma::exp(log_P.row(n-1).t());
}

// FUNCTION TO SIMULATE DATA FROM THE MODEL
//' Simulate data from Bayesian mixture models
//'
//' Generates synthetic data from either a finite mixture model or a
//' Bayesian nonparametric mixture (Dirichlet or Pitman–Yor process),
//' with several supported likelihood kernels.
//'
//' The function first samples cluster allocations and component
//' cardinalities, then generates observations sequentially from
//' cluster-specific parameters drawn from the corresponding prior.
//'
//' @param n Sample size.
//' @param K Number of mixture components. If \code{K = 0}, an infinite
//'   mixture model is used (Dirichlet or Pitman–Yor process).
//' @param alpha Concentration parameter of the mixing measure.
//' @param delta Discount parameter. Must satisfy \eqn{\delta < 1}.
//'   When \code{delta = 0}, the process reduces to a Dirichlet process.
//' @param kernel Likelihood model. One of:
//'   \code{"gaussian"}, \code{"poisson"}, \code{"binomial"},
//'   or \code{"MBernoulli"}.
//' @param d Dimension of the data (used for Gaussian and multivariate
//'   Bernoulli kernels).
//' @param mu0 Prior mean (Gaussian kernels). Scalar for univariate,
//'   vector for multivariate.
//' @param lambda0 Precision scaling parameter for Gaussian means.
//' @param alpha0 Shape parameter of the prior distribution.
//' @param xi Mean of the Poisson distribution for the number of trials
//'   in the binomial kernel.
//' @param beta0 Rate/scale parameter of the prior distribution.
//' @param sigma2 Known variance for the univariate Gaussian kernel.
//'   If not provided, variance is treated as unknown.
//'
//' @return A list containing:
//' \describe{
//'   \item{y}{Simulated observations. Type depends on the kernel:
//'     numeric vector, matrix, or integer vector/matrix.}
//'   \item{n_trials}{(Only for binomial kernel) Number of trials for
//'     each observation.}
//' }
//'
//' @details
//' If \code{K > 0}, a finite mixture model is used, where mixing weights
//' are sampled from a Dirichlet distribution.
//'
//' If \code{K = 0}, an infinite mixture is assumed, and cluster
//' assignments are generated via a stick-breaking construction of a
//' Dirichlet process (\eqn{\delta = 0}) or Pitman–Yor process
//' (\eqn{\delta \neq 0}).
//'
//' Cluster-specific parameters are drawn from the corresponding
//' conjugate priors:
//' \itemize{
//'   \item Gaussian: Normal–Inverse-Gamma (univariate) or
//'     Normal–Wishart (multivariate)
//'   \item Poisson: Gamma prior
//'   \item Binomial: Beta prior
//'   \item Multivariate Bernoulli: Beta prior (shared probability)
//' }
//'
//' Observations are generated sequentially according to the sampled
//' cluster structure.
//'
//' @examples
//' \dontrun{
//' # Dirichlet process Gaussian mixture
//' generate_data(n = 100, K = 0, kernel = "gaussian")
//'
//' # Finite Gaussian mixture with 3 components
//' generate_data(n = 200, K = 3, kernel = "gaussian")
//'
//' # Poisson mixture
//' generate_data(n = 100, kernel = "poisson")
//'
//' # Multivariate Bernoulli mixture
//' generate_data(n = 100, kernel = "MBernoulli", d = 5)
//' }
//'
//' @export generate_data
// [[Rcpp::export]]
Rcpp::List generate_data(const unsigned int& n = 100L,
                         const unsigned int& K = 0,
                         const Rcpp::Nullable<Rcpp::NumericVector>& alpha = R_NilValue,
                         double delta = 0.0,
                         const std::string& kernel = "gaussian",
                         const unsigned int& d = 1,
                         const Rcpp::RObject mu0 = R_NilValue,
                         const double& lambda0 = 0.01,
                         const double& alpha0 = 0.01,
                         const double& xi = 100,
                         const Rcpp::RObject beta0 = R_NilValue,
                         const Rcpp::RObject sigma2 = R_NilValue,
                         const Rcpp::RObject& a0 = R_NilValue,
                         const Rcpp::RObject& b0 = R_NilValue,
                         const Rcpp::RObject& lwr_bound = R_NilValue,
                         const Rcpp::RObject& upr_bound = R_NilValue){
  
  // initialize the output
  Rcpp::List out;
  
  // initialize the sample sizes
  arma::uvec ns;
  
  // initialize the map from labels to atoms
  std::unordered_map<unsigned int,unsigned int> lbl2atm;
  
  // initialize the hyperparameters
  arma::vec alpha_finite;
  double alpha_infinite;
  if(alpha.isNull()){
    if(K != 0){
      alpha_finite = arma::ones<arma::vec>(K);
    }else{
      alpha_infinite = 1.0;
    }
  }else{
    if(K != 0){
      alpha_finite = arma::ones<arma::vec>(K);
      arma::vec tmp = Rcpp::as<arma::vec>(alpha);
      if(tmp.n_elem == 1){
        for(unsigned int k = 0; k < K; k++){
          alpha_finite(k) = tmp(0);
        }
      }else{
        alpha_finite = tmp;
      }
    }else{
      alpha_infinite = Rcpp::as<double>(alpha);
    }
    
  }
  
  // finite or infinite mixture?
  if(K == 0){
    // infinite mixture case
    
    // hyperprior for alpha?
    if(!a0.isNULL() && !b0.isNULL()){
      // sample if from the prior
      alpha_infinite = R::runif(Rcpp::as<double>(a0),Rcpp::as<double>(b0));
    }
    
    // hyperprior for delta?
    if(!lwr_bound.isNULL() && !upr_bound.isNULL()){
      // sample if from the prior
      delta = R::runif(Rcpp::as<double>(lwr_bound),Rcpp::as<double>(upr_bound));
    }
    
    // sample the group sizes
    ns = urn(n,alpha_infinite,delta);
    
  }else{
    // finite mixture case
    
    // hyperprior for alpha?
    if(!a0.isNULL() && !b0.isNULL()){
      // sample if from the prior
      alpha_finite *= 0.0;
      alpha_finite += R::rgamma(Rcpp::as<double>(a0),1.0 / Rcpp::as<double>(b0));
    }
    
    // sample the group sizes
    ns = urn(n,alpha_finite);
    
  }
  
  // get the cumulative sums of the cardinalities
  for(unsigned int k = 1; k < ns.n_elem; k++){
    ns(k) += ns(k-1);
  }
  
  // simulate the data in sequence
  unsigned int k = 0;
  
  if(kernel == "gaussian"){
    
    if(d == 1){
      // univariate
      
      arma::vec y(n);
      
      if(sigma2.isNULL()){
        // unknown variance
        
        // sample the first atom
        double beta00;
        if(beta0.isNULL()){
          beta00 = 0.01;
        }else{
          beta00 = Rcpp::as<double>(beta0);
        }
        
        double mu00;
        if(mu0.isNULL()){
          mu00 = 0.0;
        }else{
          mu00 = Rcpp::as<double>(mu0);
        }
        double sigma = std::sqrt(1.0/R::rgamma(alpha0,1.0 / beta00));
        double mu = R::rnorm(mu00,sigma / std::sqrt(lambda0));
        
        for(unsigned int i = 0; i < n; i++){
          
          if(i == ns(k)){
            
            // increase the counter
            k++;
              
            // sample a new atom
            sigma = std::sqrt(1.0/R::rgamma(alpha0,1.0 / beta00));
            mu = R::rnorm(mu00,sigma / std::sqrt(lambda0));
          }
          
          // sample the data
          y(i) = R::rnorm(mu,sigma);
          
        }
        
      }else{
        // known variance
        
        // sample the first atom
        double mu00;
        if(mu0.isNULL()){
          mu00 = 0.0;
        }else{
          mu00 = Rcpp::as<double>(mu0);
        }
        
        double sigma = std::sqrt(Rcpp::as<double>(sigma2));
        double mu = R::rnorm(mu00,sigma / std::sqrt(lambda0));
        
        for(unsigned int i = 0; i < n; i++){
          
          if(i == ns(k)){
            
            // increase the counter
            k++;
            
            // sample a new atom
            mu = R::rnorm(mu00,sigma / std::sqrt(lambda0));
          }
          
          // sample the data
          y(i) = R::rnorm(mu,sigma);
          
        }
        
      }
      
      // save the results
      out["y"] = y;
      
    }else{
      // multivariate
      
      arma::mat y(n,d);
      
      arma::vec mu00;
      if(mu0.isNULL()){
        mu00 = arma::zeros<arma::vec>(d);
      }else{
        mu00 = Rcpp::as<arma::vec>(mu0);
        if(mu00.n_elem != d){
          Rcpp::stop("Wrong dimension in mu0!");
        }
      }
      
      if(sigma2.isNULL()){
        // unknown variance
        
        // sample the first atom
        arma::mat A = arma::eye(d,d);
        
        arma::mat chol_B0m1;
        if(beta0.isNULL()){
          chol_B0m1 = arma::eye(d,d);
        }else{
          chol_B0m1 = arma::chol(arma::inv(Rcpp::as<arma::mat>(beta0)));
          if(chol_B0m1.n_cols != d || chol_B0m1.n_rows != d){
            Rcpp::stop("Wrong dimenions in beta0");
          }
        }
        double alpha00;
        if(alpha0 < d){
          alpha00 = alpha0 + d;
        }else{
          alpha00 = alpha0;
        }
        
        arma::vec mu(d);
        arma::mat tLQ(d,d);
        arma::vec tmp(d);
        
        // sample the left precision from a Wishart
        for(unsigned int r = 0; r < d; r++){
          
          for(unsigned int c = 0; c <= r; c++){
            
            if(r == c){
              A(r,c) = std::sqrt(R::rgamma(0.5 * (alpha00 - r),2.0)); // 0.5 inverso
            }else{
              A(r,c) = R::rnorm(0.0,1.0);
            }
            
          }
          
        }
        tLQ = (chol_B0m1 * A).t();
        
        // solve the linear system to sample the mean
        for(unsigned int dd = 0; dd < d; dd++){
          tmp(dd) = R::rnorm(0.0,1.0);
        }
        mu = mu00 + arma::solve(arma::trimatl(std::sqrt(lambda0) * tLQ),tmp);
        
        for(unsigned int i = 0; i < n; i++){
          
          if(i == ns(k)){
            
            // increase the counter
            k++;
            
            // sample a new atom
            // sample the left precision from a Wishart
            for(unsigned int r = 0; r < d; r++){
              
              for(unsigned int c = 0; c <= r; c++){
                
                if(r == c){
                  A(r,c) = std::sqrt(R::rgamma(0.5 * (alpha00 - r),2.0)); // 0.5 inverso
                }else{
                  A(r,c) = R::rnorm(0.0,1.0);
                }
                
              }
              
            }
            tLQ = (chol_B0m1 * A).t();
            
            // solve the linear system to sample the mean
            for(unsigned int dd = 0; dd < d; dd++){
              tmp(dd) = R::rnorm(0.0,1.0);
            }
            mu = mu00 + arma::solve(arma::trimatl(std::sqrt(lambda0) * tLQ),tmp);
          }
          
          // sample the data
          for(unsigned int dd = 0; dd < d; dd++){
            tmp(dd) = R::rnorm(0.0,1.0);
          }
          y.row(i) = (mu + arma::solve(arma::trimatl(tLQ),tmp)).t();
          
        }
        
      }else{
        // spherical variance
        double sigma = std::sqrt(Rcpp::as<double>(sigma2));
        double sigma0 = sigma / std::sqrt(lambda0);
        // sample the first atom
        arma::vec mu(d);
        for(unsigned int dd = 0; dd < d; dd++){
          mu(dd) = R::rnorm(mu00(dd),sigma0);
        }
        
        for(unsigned int i = 0; i < n; i++){
          
          if(i == ns(k)){
            
            // increase the counter
            k++;
            
            // sample a new atom
            for(unsigned int dd = 0; dd < d; dd++){
              mu(dd) = R::rnorm(mu00(dd),sigma0);
            }
          }
          
          // sample the data
          for(unsigned int dd = 0; dd < d; dd++){
            y(i,dd) = R::rnorm(mu(dd),sigma);
          }

        }
        
      }
      
      // save the results
      out["y"] = y;
      
    }
    
  }else if(kernel == "poisson"){
    
    arma::uvec y(n);
    
    // sample the first atom
    double beta00;
    if(beta0.isNULL()){
      beta00 = 0.01;
    }else{
      beta00 = Rcpp::as<double>(beta0);
    }
    double lambda = R::rgamma(alpha0,1.0 / beta00);
    
    for(unsigned int i = 0; i < n; i++){
      
      if(i == ns(k)){
        
        // increase the counter
        k++;
        
        // sample a new atom
        lambda = R::rgamma(alpha0,1.0 / beta00);
      }
      
      // sample the data
      y(i) = R::rpois(lambda);
    }
    
    // save the results
    out["y"] = y;
    
  }else if(kernel == "binomial"){
    
    arma::uvec y(n);
    arma::uvec n_trials(n);
    
    // sample the first atom
    double beta00;
    if(beta0.isNULL()){
      beta00 = 0.01;
    }else{
      beta00 = Rcpp::as<double>(beta0);
    }
    double p = R::rbeta(alpha0,beta00);
    
    for(unsigned int i = 0; i < n; i++){
      
      if(i == ns(k)){
        
        // increase the counter
        k++;
        
        // sample a new atom
        p = R::rbeta(alpha0,beta00);
      }
      
      // sample the data
      n_trials(i) = R::rpois(xi);
      y(i) = R::rbinom(n_trials(i),p);
      
    }
    
    // save the results
    out["y"] = y;
    out["n_trials"] = n_trials;
    
  }else if(kernel == "MBernoulli"){
    
    arma::umat y(n,d);
    
    // sample the first atom
    double beta00;
    if(beta0.isNULL()){
      beta00 = 0.01;
    }else{
      beta00 = Rcpp::as<double>(beta0);
    }
    arma::vec p(d);
    for(unsigned int dd = 0; dd < d; dd++){
      p(dd) = R::rbeta(alpha0,beta00);
    }
    
    for(unsigned int i = 0; i < n; i++){
      
      if(i == ns(k)){
        
        // increase the counter
        k++;
        
        // sample a new atom
        for(unsigned int dd = 0; dd < d; dd++){
          p(dd) = R::rbeta(alpha0,beta00);
        }
      }
      
      // sample the data
      for(unsigned int dd = 0; dd < d; dd++){
        y(i,dd) = R::rbinom(1,p(dd));
      }
      
    }
    
    // save the results
    out["y"] = y;
    
  }else{
    // partition
    for(unsigned int k = ns.n_elem-1; k > 1; k--){
      ns(k) -= ns(k-1);
    }
    out["ns"] = ns;
  }
  
  // return the data
  return out;
  
}

//' Non-reversible MCMC sampler for Bayesian mixture models
//'
//' Runs a flexible Markov chain Monte Carlo sampler for a wide class of
//' finite and Bayesian nonparametric mixture models, including Gaussian,
//' Poisson, Binomial, and multivariate Bernoulli kernels.
//'
//' The sampler supports both finite mixtures (\code{K > 0}) and infinite
//' mixtures (\code{K = 0}) based on Dirichlet or Pitman–Yor process priors.
//' It can operate in Gibbs, reversible, or non-reversible mode.
//'
//' @param y Observed data. Can be a vector or matrix depending on the
//'   chosen kernel.
//' @param kernel Likelihood kernel. One of:
//'   \code{"gaussian"}, \code{"poisson"}, \code{"binomial"},
//'   or \code{"MBernoulli"}.
//' @param K Number of mixture components. If \code{K = 0}, an infinite
//'   mixture model is assumed.
//' @param alpha Concentration parameter of the mixing distribution.
//' @param delta Discount parameter of the Pitman–Yor process. Must satisfy
//'   \eqn{\delta < 1}. Set to 0 for a Dirichlet process.
//' @param mu0 Prior location parameter (Gaussian kernels). Scalar or vector.
//' @param lambda0 Precision scaling parameter for Gaussian means.
//' @param alpha0 Shape parameter of the base prior distribution.
//' @param beta0 Rate/scale parameter of the base prior distribution.
//' @param sigma2 Known variance for the univariate Gaussian kernel.
//'   If \code{NULL}, variance is inferred.
//' @param n_trials Number of trials for binomial observations.
//' @param init Optional initial cluster allocation vector.
//' @param N Number of MCMC iterations after burn-in.
//' @param warm_up Number of burn-in iterations.
//' @param thin Thinning interval. If 0, defaults to \code{n}.
//' @param reversible Logical; if TRUE uses reversible MCMC updates.
//' @param gibbs Logical; if TRUE uses Gibbs sampling updates.
//' @param xi Scaling parameter for auxiliary proposals in
//'   non-reversible samplers.
//' @param refresh Frequency of diagnostic or output refresh.
//' @param save_parameters Logical; if TRUE saves cluster parameters.
//' @param save_configurations Logical; if TRUE saves cluster allocations.
//' @param informed Logical; if TRUE uses informed proposal mechanisms.
//' @param m Auxiliary parameter controlling proposal structure.
//' @param g Auxiliary grouping parameter used in advanced samplers.
//' @param chain_id Identifier for parallel chains.
//'
//' @return A list containing:
//' \describe{
//'   \item{log_p}{Log-posterior trace of the Markov chain.}
//'   \item{W}{(Finite mixtures) Posterior mixture weights over iterations.}
//'   \item{K}{(Infinite mixtures) Number of active clusters per iteration.}
//'   \item{acceptance_rates}{Acceptance rates (non-Gibbs samplers only).}
//'   \item{max_excursions}{Maximum excursion statistics (non-reversible samplers).}
//'   \item{hyperpars}{List of hyperparameters used in the model.}
//'   \item{filename}{Temporary file path used for intermediate storage.}
//' }
//'
//' @details
//' The function automatically selects the appropriate model class based on
//' \code{kernel} and data structure:
//'
//' \itemize{
//'   \item Gaussian models: univariate or multivariate Normal mixtures
//'   \item Poisson models: Gamma–Poisson mixtures
//'   \item Binomial models: Beta–Binomial mixtures
//'   \item MBernoulli models: multivariate Bernoulli mixtures
//' }
//'
//' For \code{K = 0}, a Bayesian nonparametric prior is assumed:
//' a Dirichlet process when \eqn{\delta = 0}, or a Pitman–Yor process
//' otherwise.
//'
//' The sampler supports multiple inference schemes:
//' Gibbs sampling, reversible jump-like updates, and non-reversible
//' Markov chain dynamics with auxiliary proposals.
//'
//' @examples
//' \dontrun{
//' # Gaussian finite mixture
//' nrMCmix(y, kernel = "gaussian", K = 3, N = 1000)
//'
//' # Dirichlet process mixture
//' nrMCmix(y, kernel = "gaussian", K = 0, alpha = 1.0)
//'
//' # Poisson mixture
//' nrMCmix(y, kernel = "poisson", K = 0)
//'
//' # Multivariate Bernoulli
//' nrMCmix(y, kernel = "MBernoulli", K = 0, d = 5)
//' }
//'
//' @export nrMCmix
// [[Rcpp::export]]
Rcpp::List nrMCmix(const Rcpp::RObject& y = R_NilValue,
                  const std::string& kernel = "gaussian",
                  const bool& dense = false,
                  const unsigned int& K = 0,
                  const Rcpp::Nullable<Rcpp::NumericVector>& alpha = R_NilValue,
                  double delta = 0.0,
                  const bool& hyperpar_alpha = false,
                  const bool& hyperpar_delta = false,
                  const bool& hyperpar_baseline = false,
                  const double& a0 = 0.01,
                  const double& b0 = 15.0,
                  const double& lwr_bound = 0.0,
                  const double& upr_bound = 1.0,
                  const double& M = 0.0,
                  const double& V = 10.0,
                  const double& a_l = 0.1,
                  const double& b_l = 0.1, 
                  const double& a_a = 0.1,
                  const double& b_a = 0.1, 
                  const double& a_b = 0.1,
                  const double& b_b = 0.1, 
                  const double& scale = 0.5,
                  const unsigned int& slice_max_iter = 10,
                  const Rcpp::RObject mu0 = R_NilValue,
                  const double& lambda0 = 0.01,
                  const double& alpha0 = 0.01,
                  const Rcpp::RObject beta0 = R_NilValue,
                  const Rcpp::RObject sigma2 = R_NilValue,
                  const Rcpp::RObject n_trials = R_NilValue,
                  const Rcpp::Nullable<Rcpp::NumericVector>& init = R_NilValue,
                  const unsigned int& N = 1000L,
                  const unsigned int& warm_up = 100L,
                  const unsigned int thin = 1L,
                  unsigned int thin_scan = 0L,
                  const unsigned int thin_SAM = 1L,
                  const bool& NUSAMS = false,
                  const Rcpp::Nullable<Rcpp::NumericMatrix>& NU_weights = R_NilValue,
                  const bool& return_weights = false,
                  const unsigned int& n_restricted_steps = 0L,
                  const bool& reversible = false,
                  const bool& gibbs = false,
                  const bool& SAM = false,
                  const double& xi = 0.5,
                  const double& s = 0,
                  const double& refresh = 0.1,
                  const bool& save_parameters = false,
                  const bool& save_configurations = false,
                  const bool& informed = false,
                  const unsigned int& m = 20L,
                  const unsigned int& g = 0,
                  const unsigned int& chain_id = 1){
  
  // check there is some data
  if(y.isNULL()){
    Rcpp::stop("y is missing, no data provided!");
  }
  
  // R list
  Rcpp::List out;
  
  // create the hyperparameters list
  Rcpp::List hyperpars;
  
  // finite or infinite mixtures?
  bool finite = K > 0;
  
  // get the sample size
  unsigned int n;
  
  if (y.hasAttribute("dim")) {
    Rcpp::IntegerVector dim = y.attr("dim");
    n = dim[0];  
  } else {
    n = Rcpp::as<Rcpp::NumericVector>(y).size();
  }
  
  // check if thin is 0
  if(thin_scan == 0){
    // if so, set it to the sample size
    thin_scan = n;
  }
  
  // get the current configuration
  arma::uvec c;
  if(init.isNull()){
    
    // initialize the configuration vector at random
    c = arma::zeros<arma::uvec>(n);
    
    if(finite){
      for(unsigned int i = 0; i < n; i++){
        c(i) = static_cast<unsigned int>(R::runif(0.0,K));
      }
    }else{
      c.zeros();
    }
  }else{
    
    // set the configuration vector as the input one
    Rcpp::NumericVector tmp(init);
    arma::vec tmp2(tmp.begin(), tmp.size(), false);  
    c = arma::conv_to<arma::uvec>::from(tmp2) - 1;
  }
  
  arma::vec alpha_finite;
  double alpha_infinite;
  if(alpha.isNull()){
    if(K != 0){
      alpha_finite = arma::ones<arma::vec>(K);
    }else{
      alpha_infinite = 1.0;
    }
  }else{
    if(K != 0){
      alpha_finite = arma::ones<arma::vec>(K);
      arma::vec tmp = Rcpp::as<arma::vec>(alpha);
      if(tmp.n_elem == 1){
        for(unsigned int k = 0; k < K; k++){
          alpha_finite(k) = tmp(0);
        }
      }else{
        alpha_finite = tmp;
      }
    }else{
      alpha_infinite = Rcpp::as<double>(alpha);
    }

  }
  
  // current partition vector
  std::vector<std::vector<unsigned int>> partition;
  
  if(finite){
    partition.resize(K);
  }else{
    partition.resize(n);
  }
  
  // posterior markov chain
  arma::vec log_p(N);
  arma::uvec n_log_pred_calls(N);
  arma::vec entropy(N);

  // prior check quantities
  arma::mat W;
  arma::uvec K_atoms;
  if(finite){
    W = arma::zeros<arma::mat>(N,K);
  }else{
    K_atoms = arma::zeros<arma::uvec>(N);
    //entropy = arma::zeros<arma::vec>(N);
  }
  
  // MH summaries
  arma::mat acceptance_rates;
  arma::uvec max_excursions;
  
  // create the filename for the heavier stuff
  std::string filename = get_tempdir_cpp();
  
  if(gibbs && SAM){
    // reserve n element each to avoid resizing
    for(unsigned int k = 0; k < K; k++){
      
      partition[k].reserve(n);
      
    }
    acceptance_rates = arma::zeros<arma::mat>(N,2);
  }else if(SAM){
    
    // reserve n element each to avoid resizing
    for(unsigned int k = 0; k < K; k++){
      
      partition[k].reserve(n);
      
    }
    acceptance_rates = arma::zeros<arma::mat>(N,3);
  }else{
    
    // reserve n element each to avoid resizing
    for(unsigned int k = 0; k < K; k++){
      
      partition[k].reserve(n);
      
    }
    
    // reserve the excursion and acceptance rates vectors
    acceptance_rates = arma::zeros<arma::mat>(N,1);
  }
  
  if(!reversible){
    max_excursions = arma::zeros<arma::uvec>(N);
  }
  
  // alphas and deltas Markov chains
  arma::vec alphas;
  arma::vec deltas;
  if(hyperpar_alpha){
    alphas = arma::zeros<arma::vec>(N);
  }
  if(hyperpar_delta){
    deltas = arma::zeros<arma::vec>(N);
    
    // avoid the degenerate case
    if(delta == 0.0){
      delta = 0.1;
    }
  }
  
  // let's  construct the sequence of indices to check
  arma::uvec seq_idx;
  if(warm_up > 0){
    seq_idx = sequence(warm_up,refresh);
  }else{
    seq_idx = {0};
  }
  arma::uvec seq_idx_sampler = warm_up + sequence(N,refresh);
  
  // map to relabel the indeces
  std::unordered_map<unsigned int, unsigned int> lbl2atm;
  std::unordered_map<unsigned int, unsigned int> atm2lbl;
  // initialize the vector of possible new labels
  std::vector<unsigned int> labels;
  
  // NUSAMS weights
  arma::mat NUW;
  if(SAM && NUSAMS && return_weights){
    NUW = arma::zeros<arma::mat>(n,n);
  }
  
  // parse the right model
  if(kernel == "gaussian"){
    
    // get the dimension
    if(Rf_isMatrix(y)){
      
      // multivariate case
      
      // get the data
      arma::mat y_mat = Rcpp::as<arma::mat>(y);
      
      // multivariate case
      unsigned int D = y_mat.n_cols;
      arma::vec mu00;
      if(mu0.isNULL()){
        mu00 = arma::zeros<arma::vec>(D);
      }else{
        mu00 = Rcpp::as<arma::vec>(mu0);
        if(mu00.n_elem != D){
          Rcpp::stop("Wrong dimension in mu0!");
        }
      }
      
      if(lambda0 <= 0){
        Rcpp::stop("'lambda0' cannot be negative!");
      }
      
      if(dense){
        
        // uknown variance
        
        arma::mat B0;
        if(beta0.isNULL()){
          B0 = arma::eye(D,D);
        }else{
          B0 = Rcpp::as<arma::mat>(beta0);
          if(B0.n_cols != D || B0.n_rows != D){
            Rcpp::stop("Wrong dimenions in beta0");
          }
        }
        
        if(alpha0 < D){
          Rcpp::stop("'alpha0' cannot be smaller than the number of dimenions!");
        }
        
        hyperpars["B0"] = B0;
        hyperpars["alpha0"] = alpha0;
        
        if(finite){
          MGauss model(partition,y_mat,K,alpha_finite,mu00,lambda0,alpha0,B0,c,gibbs,
                       hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
          
          finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                         max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                         seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                         filename,gibbs,reversible,informed);
          
        }else{
          inf_MGauss model(partition,lbl2atm,atm2lbl,labels,y_mat,alpha_infinite,delta,mu00,lambda0,alpha0,B0,c,gibbs && !SAM,
                           hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          
          infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                           max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                           n_restricted_steps,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
        }
      }else if(sigma2.isNULL()){
        
        // uknonwn diagonal covariance
        
        double beta00;
        if(beta0.isNULL()){
          beta00 = 0.01;
        }else{
          beta00 = Rcpp::as<double>(beta0);
        }
        
        if(finite){
          MGauss2 model(partition,y_mat,K,alpha_finite,mu00,lambda0,alpha0,beta00,c,gibbs,
                        hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
          
          finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                         max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                         seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                         filename,gibbs,reversible,informed);
          
        }else{
          inf_MGauss2 model(partition,lbl2atm,atm2lbl,labels,y_mat,alpha_infinite,delta,mu00,lambda0,alpha0,beta00,c,gibbs && !SAM,
                            hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          
          infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                           max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                           n_restricted_steps,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
        }
        
        hyperpars["alpha"] = alpha;
        if(finite){
          hyperpars["K"] = K;
        }else{
          hyperpars["delta"] = delta;
        }
        hyperpars["mu0"] = mu00;
        hyperpars["lambda0"] = lambda0;
        hyperpars["alpha0"] = alpha0;
        hyperpars["beta0"] = beta00;
        
      }else{
        
        // known spherical variance
        
        double sigma20 = Rcpp::as<double>(sigma2);
        hyperpars["sigma2"] = sigma20;
        
        if(finite){
          MGauss1 model(partition,y_mat,K,alpha_finite,sigma20,mu00,lambda0,c,gibbs,
                        hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
          
          finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                         max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                         seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                         filename,gibbs,reversible,informed);
          
        }else{
          inf_MGauss1 model(partition,lbl2atm,atm2lbl,labels,y_mat,alpha_infinite,delta,sigma20,mu00,lambda0,c,gibbs && !SAM,
                            hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          
          infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                           max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                           n_restricted_steps,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
        }
      }
      
      hyperpars["alpha"] = alpha;
      if(finite){
        hyperpars["K"] = K;
      }else{
        hyperpars["delta"] = delta;
      }
      hyperpars["mu0"] = mu00;
      hyperpars["lambda0"] = lambda0;
      
    }else{
      
      // univariate case
      
      arma::vec y_vec = Rcpp::as<arma::vec>(y);
      
      if(sigma2.isNULL()){
        
        // unknown variance
        
        double beta00;
        if(beta0.isNULL()){
          beta00 = 0.01;
        }else{
          beta00 = Rcpp::as<double>(beta0);
        }
        
        double mu00;
        if(mu0.isNULL()){
          mu00 = 0.0;
        }else{
          mu00 = Rcpp::as<double>(mu0);
        }
        
        if(finite){
          Gauss2 model(partition,y_vec,K,alpha_finite,mu00,lambda0,alpha0,beta00,c,gibbs,
                       hyperpar_alpha,hyperpar_baseline,a0,b0,
                       M,V,a_l,b_l,a_a,b_a,a_b,b_b,
                       scale,slice_max_iter);
          
          finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                         max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                         seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                         filename,gibbs,reversible,informed);
        }else{
          inf_Gauss2 model(partition,lbl2atm,atm2lbl,labels,y_vec,alpha_infinite,delta,mu00,lambda0,alpha0,beta00,c,gibbs && !SAM,
                           hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,
                           M,V,a_l,b_l,a_a,b_a,a_b,b_b,
                           scale,slice_max_iter);
          
          infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates, entropy,alphas,deltas,
                           max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                           n_restricted_steps,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
        }
        
        hyperpars["alpha"] = alpha;
        if(finite){
          hyperpars["K"] = K;
        }else{
          hyperpars["delta"] = delta;
        }
        hyperpars["mu0"] = mu00;
        hyperpars["lambda0"] = lambda0;
        hyperpars["alpha0"] = alpha0;
        hyperpars["beta0"] = beta00;        
      }else{
        
        // known variance
        
        double mu00;
        if(mu0.isNULL()){
          mu00 = 0.0;
        }else{
          mu00 = Rcpp::as<double>(mu0);
        }
        
        double sigma20 = Rcpp::as<double>(sigma2);
        
        if(finite){
          Gauss1 model(partition,y_vec,K,alpha_finite,sigma20,mu00,lambda0,c,gibbs,
                       hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
          
          finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                         max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                         seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                         filename,gibbs,reversible,informed);
        }else{
          inf_Gauss1 model(partition,lbl2atm,atm2lbl,labels,y_vec,alpha_infinite,delta,sigma20,mu00,lambda0,c,gibbs && !SAM,
                           hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          
          infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                           max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                           n_restricted_steps,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
        }
        
        hyperpars["alpha"] = alpha;
        if(finite){
          hyperpars["K"] = K;
        }else{
          hyperpars["delta"] = delta;
        }
        hyperpars["mu0"] = mu00;
        hyperpars["lambda0"] = lambda0;
        hyperpars["sigma20"] = sigma20;
        
      }
      
    }
    
  }else if(kernel == "poisson"){
    
    // poisson
    
    arma::uvec y_uvec = Rcpp::as<arma::uvec>(y);
    
    double beta00;
    if(beta0.isNULL()){
      beta00 = 0.01;
    }else{
      beta00 = Rcpp::as<double>(beta0);
    }
    
    if(finite){
      Poiss model(partition,y_uvec,K,alpha_finite,alpha0,beta00,c,gibbs,
                  hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
      
      finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                     max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                     seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                     filename,gibbs,reversible,informed);
    }else{
      inf_Poiss model(partition,lbl2atm,atm2lbl,labels,y_uvec,alpha_infinite,delta,alpha0,beta00,c,gibbs && !SAM,
                      hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      
      infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                       max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                       n_restricted_steps,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
    }
    
    hyperpars["alpha"] = alpha;
    if(finite){
      hyperpars["K"] = K;
    }else{
      hyperpars["delta"] = delta;
    }
    hyperpars["alpha0"] = alpha0;
    hyperpars["beta0"] = beta00;
    
  }else if(kernel == "binomial"){
    
    // binomial
    
    double beta00;
    if(beta0.isNULL()){
      beta00 = 0.01;
    }else{
      beta00 = Rcpp::as<double>(beta0);
    }
    
    arma::uvec y_uvec = Rcpp::as<arma::uvec>(y);
    if(n_trials.isNULL()){
      Rcpp::stop("'n_trials' doesn't provided!");
    }
    arma::uvec n_trials_uvec = Rcpp::as<arma::uvec>(n_trials);
    
    if(finite){
      Binom model(partition,y_uvec,n_trials_uvec,K,alpha_finite,alpha0,beta00,c,gibbs,
                  hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
      
      finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                     max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                     seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                     filename,gibbs,reversible,informed);
    }else{
      inf_Binom model(partition,lbl2atm,atm2lbl,labels,y_uvec,n_trials_uvec,alpha_infinite,delta,alpha0,beta00,c,gibbs && !SAM,
                      hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      
      infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                       max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                       n_restricted_steps,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
    }
    
    hyperpars["alpha"] = alpha;
    if(finite){
      hyperpars["K"] = K;
    }else{
      hyperpars["delta"] = delta;
    }
    hyperpars["alpha0"] = alpha0;
    hyperpars["beta0"] = beta00;
    
  }else if(kernel == "MBernoulli"){
    
    // multivariate bernoulli
    
    double beta00;
    if(beta0.isNULL()){
      beta00 = 0.01;
    }else{
      beta00 = Rcpp::as<double>(beta0);
    }
    
    arma::umat y_umat = Rcpp::as<arma::umat>(y);
    
    if(finite){
      MBern model(partition,y_umat,K,alpha_finite,alpha0,beta00,c,gibbs,
                  hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
      
      finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                     max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                     seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                     filename,gibbs,reversible,informed);
    }else{
      inf_MBern model(partition,lbl2atm,atm2lbl,labels,y_umat,alpha_infinite,delta,alpha0,beta00,c,gibbs && !SAM,
                      hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      
      infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                       max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                       n_restricted_steps,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
    }
    
    hyperpars["alpha"] = alpha;
    if(finite){
      hyperpars["K"] = K;
    }else{
      hyperpars["delta"] = delta;
    }
    hyperpars["alpha0"] = alpha0;
    hyperpars["beta0"] = beta00;
    
  }else{
    
    // partition
    
    if(finite){
      Partition model(partition,K,alpha_finite,c,gibbs,
                      hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter);
      
      finite_mixture(model,partition,c,log_p,n_log_pred_calls,W,entropy,acceptance_rates,alphas,
                     max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                     seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                     filename,gibbs,reversible,informed);
    }else{
      inf_Partition model(partition,lbl2atm,atm2lbl,labels,alpha_infinite,delta,c,gibbs && !SAM,
                          hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      
      infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                       max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                       n_restricted_steps,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
    }
    
    hyperpars["alpha"] = alpha;
    if(finite){
      hyperpars["K"] = K;
    }else{
      hyperpars["delta"] = delta;
    }
    
  }
  
  // add the log posterior chain
  out["log_p"] = log_p;
  out["n_pred_calls"] = n_log_pred_calls;
  out["c"] = c;
  out["entropy"] = entropy;
  if(finite){
    out["W"] = W;
  }else{
    out["K"] = K_atoms;
  }
  
  if(SAM && NUSAMS && return_weights){
    out["NUW"] = NUW;
  }
  
  // add the non-reversible summaries
  if(gibbs && SAM && K == 0){
    out["acceptance_rates"] = acceptance_rates;
  }else if(reversible){
    out["acceptance_rates"] = acceptance_rates;
  }else if(!gibbs){
    out["acceptance_rates"] = acceptance_rates;
    out["max_excursions"] = max_excursions;
  }
  
  if(hyperpar_alpha){
    out["alpha"] = alphas;
  }
  
  if(hyperpar_delta){
    out["delta"] = deltas;
  }
  
  // add it to the output list
  out["hyperpars"] = hyperpars;
  
  // add also the filename
  out["filename"] = filename;
  
  // add the sampler specifics
  out.attr("specifics") = describe_sampler("gaussian (known variance)",
           K,N,gibbs,reversible,informed,m);
  
  // return the list
  return out;
  
}

// FUNCTION TO SIMULATE THE LATENT PROCESS

// start with only the univariate gaussian case with uknown variance

// [[Rcpp::export]]
Rcpp::List get_latent_process(const arma::uvec& c,
                              const arma::vec& y,
                              const unsigned int& K = 0L,
                              const double& alpha = 1.0,
                              const double& delta = 0.0,
                              const double& mu0 = 0.0,
                              const double& lambda0 = 0.01,
                              const double& alpha0 = 0.01,
                              const double& beta0 = 0.01,
                              const Rcpp::RObject& x_grid = R_NilValue,
                              const unsigned int& n_grid = 512L){
  
  // initialize the output list
  Rcpp::List out;
  
  // get the grid for density estimation
  arma::vec xx_grid;
  
  // initialize the density response vector
  arma::vec y_grid;
  
  if(x_grid.isNULL()){
    
    // get the data range
    double y_min = arma::min(y);
    double y_max = arma::max(y);
    double y_range = y_max - y_min;
    y_min -= 0.05 * y_range;
    y_max += 0.05 * y_range;
    double y_step = y_range * 1.1 / (n_grid-1);
    
    // create the grid
    xx_grid = arma::zeros<arma::vec>(n_grid);
    y_grid = arma::zeros<arma::vec>(n_grid);
    xx_grid(0) = y_min;
    for(unsigned int i = 1; i < n_grid; i++){
      xx_grid(i) = xx_grid(i-1) + y_step;
    }
    
  }else{
    
    // get the user defined grid
    xx_grid = Rcpp::as<arma::vec>(x_grid);
    
    // create the response
    y_grid = arma::zeros<arma::vec>(xx_grid.n_elem);
    
  }
  
  // distinguish the two cases
  if(K == 0){
    
    // infinite mixture

    // initialize the sufficient statistics
    std::vector<double> ns;
    ns.reserve(y.n_elem);
    
    std::vector<double> sums_y;
    sums_y.reserve(y.n_elem);
    
    std::vector<double> sums_y2;
    sums_y2.reserve(y.n_elem);
    
    // loop along the configuration vector
    unsigned int k = 0;
    double y_i = 0.0;
    unsigned int K_atm = 0;
    std::unordered_map<unsigned int, unsigned int> lbl2atm;
    for(unsigned int i = 0; i < y.n_elem; i++){
      
      // check that the current labels is not already present in the dictionary
      if(!lbl2atm.count(c(i))){
        
        // if the current label is not present, append it to the dictionary
        lbl2atm[c(i)] = K_atm;
        
        // increase the number of active components
        K_atm++;
        
        // append the new values to the sufficient statistics vectors
        ns.push_back(0.0);
        sums_y.push_back(0.0);
        sums_y2.push_back(0.0);
        
      }
      
      // get the current new label
      k = lbl2atm[c(i)];
      
      // get the datum
      y_i = y(i);
      
      // update the sufficient statistics
      ns[k]++;
      sums_y[k] += y_i;
      sums_y2[k] += y_i * y_i;
      
    }
    
    // compute the atoms
    arma::vec w(K_atm);
    arma::vec sigma2s(K_atm);
    arma::vec mus(K_atm);
    double lambda,mu;
    for(unsigned int k = 0; k < K_atm; k++){
      lambda = lambda0 + ns[k];
      mu = (lambda0*mu0 + sums_y[k] ) / lambda;
      w(k) = (ns[k] - delta) / (alpha + y.n_elem);
      sigma2s(k) = 1.0 / R::rgamma(alpha0 + 0.5*ns[k], 1.0 / (beta0 + 0.5 * (mu0 * mu0 * lambda0 + sums_y2[k] - lambda * mu * mu)));
      mus(k) = R::rnorm(mu,std::sqrt(sigma2s(k)/lambda));
    }
    
    // evaluate the density in the specified points
    arma::vec tmp(K_atm + 1);
    
    // save some recurrent quantities
    double prior_term1 =
      std::lgamma(alpha0 + 0.5)
      - std::lgamma(alpha0)
      - 0.5 * std::log(
          2.0 * arma::datum::pi * beta0 *
          (1.0 + 1.0 / lambda0)
      );
    
    double prior_term2 =
      1.0 /
        (2.0 * beta0 * (1.0 + 1.0 / lambda0));
    
    double const_term = -0.5 * std::log(2*arma::datum::pi);
    
    for(unsigned int i = 0; i < y_grid.n_elem; i++){
      
      // get the value
      y_i = xx_grid(i);
      
      // compute the log contributes from each atom
      for(unsigned int k = 0; k < K_atm; k++){
        tmp(k) = const_term + std::log(w(k))-
          0.5 * std::log(sigma2s(k)) - 
          0.5 * (y_i - mus(k)) * (y_i - mus(k)) / sigma2s(k);
      }
      
      // add the baseline contribution
      tmp(K_atm) = std::log( alpha + K_atm * delta ) - std::log(alpha + y.n_elem) +
        prior_term1 - (alpha0 + 0.5) * std::log(1.0 + prior_term2 * (y_i - mu0) * (y_i - mu0));
      
      // normalize the contributes
      y_grid(i) = std::exp(log_sum_exp(tmp));
      
    }
    
    // return the quantities of interest
    out["w"] = w;
    out["mu"] = mus;
    out["sigma2"] = sigma2s;
    out["x"] = xx_grid;
    out["y"] = y_grid;
    
  }else{
    
    // finite mixture
    
    // compute the sufficient statistics
    arma::vec ns = arma::zeros<arma::vec>(K);
    arma::vec sums_y = arma::zeros<arma::vec>(K);
    arma::vec sums_y2 = arma::zeros<arma::vec>(K);
    
    unsigned int k = 0;
    double y_i = 0.0;
    for(unsigned int i = 0; i < y.n_elem; i++){
      
      // get the current label
      k = c(i);
      if(k >= K){
        Rcpp::stop("Labels greater than the number of cluster specified!");
      }
      
      // get the unit
      y_i = y(i);
      
      // update the sufficient statistcs
      ns(k)++;
      sums_y(k) += y_i;
      sums_y2(k) += y_i * y_i;
      
    }
    
    // compute the atoms
    arma::vec w = rdirichlet(alpha + ns);
    arma::vec sigma2s(K);
    arma::vec mus(K);
    double lambda,mu;
    for(unsigned int k = 0; k < K; k++){
      lambda = lambda0 + ns(k);
      mu = (lambda0*mu0 + sums_y(k) ) / lambda;
      sigma2s(k) = 1.0 / R::rgamma(alpha0 + 0.5*ns(k), 1.0 / (beta0 + 0.5 * (mu0 * mu0 * lambda0 + sums_y2(k) - lambda * mu * mu)));
      mus(k) = R::rnorm(mu,std::sqrt(sigma2s(k)/lambda));
    }
    
    // evaluate the density in the specified points
    arma::vec tmp(K);
    double const_term = -0.5 * std::log(2*arma::datum::pi) - std::log(alpha + y.n_elem);
    for(unsigned int i = 0; i < y_grid.n_elem; i++){
      
      // get the value
      y_i = xx_grid(i);
      
      // compute the log contributes from each atom
      for(unsigned int k = 0; k < K; k++){
        tmp(k) = const_term + std::log(alpha + ns(k))-
          0.5 * std::log(sigma2s(k)) - 
          0.5 * (y_i - mus(k)) * (y_i - mus(k)) / sigma2s(k);
      }
      
      // normalize the contributes
      y_grid(i) = std::exp(log_sum_exp(tmp));
      
    }

    // return the quantities of interest
    out["w"] = w;
    out["mu"] = mus;
    out["sigma2"] = sigma2s;
    out["x"] = xx_grid;
    out["y"] = y_grid;
  }
  
  // return the output list
  return out;
  
}

// [[Rcpp::export]]
arma::mat compute_similarity_matrix(const arma::umat& c){
  
  // get the number of units
  unsigned int n = c.n_cols;
  
  // get the number of iterations
  unsigned int N = c.n_rows;
  
  // initialize the similarity matrix
  arma::mat out = arma::zeros<arma::mat>(n,n);
  
  // add the counts
  for(unsigned int iter = 0; iter < N; iter++){
    
    // loop over each pair
    for(unsigned int i = 0; i < n; i++){
      for(unsigned int j = 0; j < i; j++){
        
        // check the occurrence
        if(c(iter,i) == c(iter,j)){
          
          out(i,j)++;
          out(j,i)++;
          
        }
          
      }
    }
    
  }
  
  // normalize the counts
  out /= N;
  
  // return the similarity matrix
  return out;
  
}
