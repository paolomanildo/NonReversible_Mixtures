#include <iostream>
#include <RcppArmadillo.h>
#include "supportive_functions.h"
#include "models.h"
#include "hyperpar_updates.h"

// -----------------------------------------------------------------------------
// --------------------------- FINITE MIXTURE CASE -----------------------------
// -----------------------------------------------------------------------------


// ----------------------- UNIVARIATE GAUSSIAN KNWON VARIANCE ------------------

// constructor
Gauss1::Gauss1(std::vector<std::vector<unsigned int>>& partition,
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
               const double& m_) : y(y_), 
               hyperpar_alpha(hyperpar_alpha_), 
               a0(a0_),
               b0(b0_),
               scale(scale_),
               m(m_){
  
  // save the hyperparameters
  K = K_;
  alpha = alpha_;
  sigma2 = sigma2_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  
  // set the statistical unit
  y_i = 0.0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  
  mus = arma::zeros<arma::vec>(K);
  sigma2s = arma::zeros<arma::vec>(K);
  
  // initialize the group sums
  arma::vec sums_y = arma::zeros<arma::vec>(K);
  
  unsigned int c_i = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // update the sufficient statistics
    ns(c_i)++;
    sums_y(c_i) += y(i);
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  // compute the parameter for the full conditional
  
  // means
  mus = (lambda0*mu0 + sums_y ) / (lambda0 + ns);
  
  // variances
  sigma2s = sigma2 * (1.0 + 1.0 / (lambda0 + ns));
  
  // compute the unnormalized log posterior
  log_post = arma::sum(arma::lgamma(alpha + ns) - 
    0.5*arma::log(lambda0 + ns) + 
    0.5 * (mus % mus % (lambda0 + ns) / sigma2));
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
}

// update parameters
void Gauss1::update_par(const unsigned int& c_i,
                        double sign){
  
  // update the dimension of the target group
  ns(c_i) += sign;
  
  // update parameters
  
  // mu
  mus(c_i) = ( (lambda0 + ns(c_i) - sign) * mus(c_i) + sign*y_i) / (lambda0 + ns(c_i));
  
  // sigma2
  sigma2s(c_i) = ( (lambda0 + ns(c_i) - sign) * 
    sigma2s(c_i) + sign*sigma2  ) / ( lambda0 + ns(c_i) );
  
}

// log predictive function
double Gauss1::log_pred(const unsigned int& k){
  
  return -0.5 * std::log(sigma2s(k)) - 
    0.5 * (y_i - mus(k)) * (y_i - mus(k)) / sigma2s(k);
  
}

// function that set the statistical unit
void Gauss1::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void Gauss1::save_parameters(const std::string& filename,
                             const unsigned int& chain_id){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + ".csv", std::ios::app);
  
  // sample the mixing weights
  arma::vec mix = rdirichlet(alpha + ns);
  
  // save the values
  for(unsigned int k = 0; k < K-1; k++){
    
    file_pars << mix(k) << "," << R::rnorm(mus(k),std::sqrt( sigma2s(k) - sigma2)) << ",";
  }
  
  // add the last one with an endline
  file_pars << mix(K-1) << "," << R::rnorm(mus(K-1),std::sqrt( sigma2s(K-1) - sigma2)) << "\n";
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double Gauss1::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void Gauss1::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void Gauss1::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// ---------------------- UNIVARIATE GAUSSIAN UNKNWON VARIANCE -----------------

// constructor
Gauss2::Gauss2(std::vector<std::vector<unsigned int>>& partition,
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
               const double& m_) : y(y_), 
               hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
               a0(a0_),
               b0(b0_),
               M(M_), V(V_),a_l(a_l_),b_l(b_l_),a_a(a_a_),b_a(b_a_),a_b(a_b_),b_b(b_b_),
               scale(scale_),
               m(m_){
  
  // save the hyperparameters
  K = K_;
  alpha = alpha_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // hyperpriors parameters
  // M = 0.0;
  // V = 100;
  // a_l = b_l = a_a = b_a = a_b = b_b = 0.01;
  
  // set the statistical unit
  y_i = 0.0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  sums_y = arma::zeros<arma::vec>(K);
  sums_y2 = arma::zeros<arma::vec>(K);
  
  mus = arma::zeros<arma::vec>(K);
  lambdas = arma::zeros<arma::vec>(K);
  alphas = arma::zeros<arma::vec>(K);
  betas = arma::zeros<arma::vec>(K);
  
  unsigned int c_i = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // update the sufficient statistics
    ns(c_i)++;
    sums_y(c_i) += y(i);
    sums_y2(c_i) += y(i) * y(i);
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  // compute the parameter for the full conditional
  
  // compute the parameter for the full conditional
  lambdas = lambda0 + ns;
  alphas = alpha0 + 0.5 * ns;
  mus = (lambda0*mu0 + sums_y ) / lambdas;
  betas = beta0 + 0.5 * (mu0 * mu0 * lambda0 + sums_y2 - lambdas % mus % mus);
  
  // compute the unnormalized log posterior
  log_post = arma::sum(arma::lgamma(alpha + ns) -
    0.5 * arma::log(lambdas) -
    alphas % arma::log(betas) + 
    arma::lgamma(alphas)) + 
    K * (0.5 * std::log(lambda0) + 
    alpha0 * std::log(beta0) - 
    std::lgamma(alpha0));
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
  
  // add the hyperprior's prior terms
  if(hyperpar_baseline){
    log_post += -0.5 * (mu0 - M) * (mu0 - M) / V + 
      a_l * std::log(lambda0) - b_l*lambda0 + 
      a_a * std::log(alpha0) - b_a*alpha0 + 
      a_b * std::log(beta0) - b_b*beta0 ;
  }
}

// update parameters
void Gauss2::update_par(const unsigned int& c_i,
                        double sign){
  
  // update the dimension of the target group
  ns(c_i) += sign;
  
  // update the sufficient statistics
  sums_y(c_i) += sign*y_i;
  sums_y2(c_i) += sign*y_i * y_i;
  
  // update parameter
  
  // alpha
  alphas(c_i) += sign*0.5;
  
  // mu
  mus(c_i) = (lambdas(c_i) * mus(c_i) + sign*y_i) / (lambdas(c_i) + sign);
  
  // lambda
  lambdas(c_i) += sign;
  
  // betas
  betas(c_i) = beta0 + 0.5 * (mu0*mu0*lambda0 + sums_y2(c_i) - mus(c_i) * mus(c_i) * lambdas(c_i));
  
}

// log predictive function
double Gauss2::log_pred(const unsigned int& k){
  
  return std::lgamma(alphas(k) + 0.5) - std::lgamma(alphas(k)) - 
    0.5 * std::log(2.0 * betas(k) * (1.0 + 1.0 / lambdas(k))) - 
    (alphas(k) + 0.5) * std::log(1.0 + 0.5 * 
    (y_i - mus(k)) * (y_i - mus(k)) / (1.0 + 1.0 / lambdas(k)) / betas(k) );
  
}

// function that set the statistical unit
void Gauss2::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void Gauss2::save_parameters(const std::string& filename,
                             const unsigned int& chain_id){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + ".csv", std::ios::app);
  
  // // sample the mixing weights
  // arma::vec mix = rdirichlet(alpha + ns);
  // 
  // double sigma2;
  // // save the values
  // for(unsigned int k = 0; k < K-1; k++){
  //   
  //   // sample the variance
  //   sigma2 = 1.0 / R::rgamma(alphas(k), 1.0 / betas(k) );
  //   
  //   file_pars << mix(k) << "," << R::rnorm(mus(k),std::sqrt( sigma2 / lambdas(k) )) << "," << sigma2 << ",";
  // }
  // 
  // // add the last one with an endline
  // sigma2 = 1.0 / R::rgamma(alphas(K-1), 1.0 / betas(K-1) );
  // file_pars << mix(K-1) << "," << R::rnorm(mus(K-1),std::sqrt( sigma2 / lambdas(K-1) )) << "," << sigma2 << "\n";
  // 
  
  // save the hyperparameters
  file_pars << mu0 << "," << lambda0 << "," << alpha0 << "," << beta0 << "\n";
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double Gauss2::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else if(which == 1){
    // mu0 (with normal prior)
    
    // initialize the output with the hyperprior's log density
    double out = -0.5 * (x - M) * (x - M) / V;
    
    // add the posterior contributes
    for(unsigned int k=0; k < K; k++){
      
      out -= alphas(k) * std::log( beta0 + 0.5 * ( x*x*lambda0 + sums_y2(k) - 
        (x*lambda0 + sums_y(k)) * (x*lambda0 + sums_y(k)) / lambdas(k) ) );
      
    }
    
    // return the log density
    return out;
    
  }else if(which == 2){
    
    // lambda0 (log scale with gamma prior)
    double ll = std::exp(x);
    
    // initialize the output with the hyper prior contribute
    // as well with all the constant terms
    double out = (0.5*K + a_l) * x - b_l * ll;
    
    // add the other posterior contributes
    for(unsigned int k = 0; k < K; k++){
      
      out -= 0.5 * std::log(ll + ns(k)) + 
        alphas(k) * std::log(beta0 + 0.5 * ( mu0*mu0*ll + sums_y2(k) - 
        (ll*mu0 + sums_y(k)) * (ll*mu0 + sums_y(k)) / (ll + ns(k)) ));
      
    }
    
    // return the log density
    return out;
    
  }else if(which == 3){
    
    // alpha0 (log scale with gamma prior)
    double aa = std::exp(x);
    
    // initialize the output with the hyper prior contribute
    // as well with all the constant terms
    double out = a_a * x - b_a * aa + 
      K * ( aa * std::log(beta0) - std::lgamma(aa) );
    
    // add the other posterior contributes
    for(unsigned int k = 0; k < K; k++){
      
      out += std::lgamma(aa + 0.5 * ns(k)) - 
        (aa + 0.5*ns(k)) * std::log(betas(k));
    }
    
    // return the log density
    return out;
    
  }else if(which == 4){
    
    // beta0 (log scale with gamma prior)
    double bb = std::exp(x);
    
    // initialize the output with the hyper prior contribute
    // as well with all the constant terms
    double out = a_b * x - b_b * bb + 
      K * alpha0 * std::log(bb);
    
    // add the other posterior contributes
    for(unsigned int k = 0; k < K; k++){
      
      out -= alphas(k) * std::log(betas(k) - beta0 + bb);
    }
    
    // return the log density
    return out;
    
  }
  
  return 0.0;
  
}

// function to update the atoms given a change in the hyperparameters
void Gauss2::update_atoms(double new_value, unsigned int which){
  
  if(which == 0){
    
    // mu0
    for(unsigned int k = 0; k < K; k++){
      
      mus(k) += lambda0 * (new_value - mu0) / lambdas(k);
      
      betas(k) = beta0 + 0.5 * ( new_value*new_value*lambda0 + sums_y2(k) - mus(k)*mus(k)*lambdas(k) );
      
    }
    
    mu0 = new_value;
    
  }else if(which == 1){
    
    // lambda0
    
    for(unsigned int k = 0; k < K; k++){
      
      lambdas(k) += (new_value - lambda0);
      mus(k) = ( (lambda0 + ns(k)) * mus(k) + mu0 * (new_value - lambda0) ) / lambdas(k);
      betas(k) = beta0 + 0.5 * ( mu0*mu0*new_value + sums_y2(k) - mus(k)*mus(k)*lambdas(k) );
      
    }
    
    lambda0 = new_value;
    
  }else if(which == 2){
    
    // alpha0
    
    for(unsigned int k = 0; k < K; k++){
      
      alphas(k) += (new_value - alpha0);
      
    }
    
    alpha0 = new_value;
    
  }else if(which == 3){
    
    // beta0
    
    for(unsigned int k = 0; k < K; k++){
      
      betas(k) += (new_value - beta0);
      
    }
    
    beta0 = new_value;
  }
  
}

// function to update the hyperparameters
void Gauss2::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
  // baseline distribution
  if(hyperpar_baseline){
    
    // initialize the temporary new value
    double tmp;
    
    // mu0
    tmp = slice_sampler(*this, log_post2, mu0, 1, false);
    update_atoms(tmp,0);
    
    // lambda0
    tmp = slice_sampler(*this, log_post2, lambda0, 2, true);
    update_atoms(tmp,1);
    
    // alpha0
    tmp = slice_sampler(*this, log_post2, alpha0, 3, true);
    update_atoms(tmp,2);
    
    // beta0
    tmp = slice_sampler(*this, log_post2, beta0, 4, true);
    update_atoms(tmp,3);
    
    
  }
  
}

// ---------------------------- MULTIVARIATE GAUSSIAN --------------------------

// constructor
MGauss::MGauss(std::vector<std::vector<unsigned int>>& partition,
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
               const double& m_) : y(y_), 
               hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
               a0(a0_),
               b0(b0_),
               scale(scale_),
               m(m_){
  
  // save the hyperparameters
  K = K_;
  D = y.n_cols;
  alpha = alpha_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  alpha0 = alpha0_;
  chol_Bm10 = arma::zeros<arma::vec>(D*(D+1)/2);
  
  arma::mat chol_Bm10_mat = arma::chol(arma::inv(B0_));
  
  // set the statistical unit
  y_i = arma::zeros<arma::vec>(D);
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  
  mus = arma::zeros<arma::mat>(D,K);
  lambdas = arma::zeros<arma::vec>(K);
  alphas = arma::zeros<arma::vec>(K);
  chol_Bm1s = arma::zeros<arma::mat>(D*(D+1)/2,K);
  idx_map = arma::zeros<arma::umat>(D,D);
  
  // compute the idx map
  unsigned int conta = 0;
  for(unsigned int j = 0; j < D; j++){
    for(unsigned int i = 0; i < D; i++){
      
      if(j <= i){
        idx_map(i,j) = idx_map(j,i) = conta;
        conta++;
      }
      
    }
  }
  
  // fill the transpose of the right cholesky
  for(unsigned int i = 0; i < D; i++){
    for(unsigned int j = 0; j <= i; j++){
      
      chol_Bm10(idx_map(i,j)) = chol_Bm10_mat(j,i);
      
    }
  }
  
  double tmp_k = 0;
  arma::mat y_bar = arma::zeros<arma::mat>(D,K);
  arma::cube Bs = arma::zeros<arma::cube>(D,D,K);
  for(unsigned int i = 0; i < n; i++ ){
    
    // get the cluster membership
    tmp_k = c(i);
    
    // cluster membership counter
    ns(tmp_k)++;
    
    // update the deviance
    Bs.slice(tmp_k) += ( ns(tmp_k)-1.0) / ns(tmp_k) *
      (y.row(i).t() - y_bar.col(tmp_k)) * (y.row(i).t() - y_bar.col(tmp_k)).t();
    
    // update the means
    for(unsigned int dd = 0; dd < D; dd++){
      y_bar(dd,tmp_k) += ( y(i,dd) - y_bar(dd,tmp_k) ) / ns(tmp_k);
    }
    
    // add to the partition
    if(!gibbs){
      partition[tmp_k].push_back(i);
    }
    
  }
  
  // lambdas
  lambdas = lambda0 + ns;
  
  // as
  alphas = alpha0 + ns;
  
  for(unsigned int k = 0; k < K; k++){
    
    // mus
    mus.col(k) = ( lambda0*mu0 + ns(k) * y_bar.col(k) ) / lambdas(k);
    
    // Bs
    Bs.slice(k) += B0_ + (lambda0 * ns(k)) / (lambda0 + ns(k)) * 
      (y_bar.col(k) - mu0) * (y_bar.col(k) - mu0).t();
    
    // solve this matrix and compute the cholesky
    Bs.slice(k) = arma::chol(arma::inv(Bs.slice(k)));
    
    // fill the transpose of the right cholesky
    for(unsigned int i = 0; i < D; i++){
      for(unsigned int j = 0; j <= i; j++){
        
        chol_Bm1s(idx_map(i,j),k) = Bs(j,i,k);
        
      }
    }
    
  }
  // initialize the unnormalized log posterior
  for(unsigned int k = 0; k < K; k++){
    
    log_post += std::lgamma(alpha(k) + ns(k)) + 
      alphas(k) * half_log_det_chol(chol_Bm1s.col(k),D) -
      0.5 * D * std::log(lambdas(k));
    
    // add the multivariate gamma function term
    for(unsigned int dd = 0; dd < D; dd++){
      log_post += std::lgamma(0.5 * (alphas(k) - dd));
    }
    
  }
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
  
}

// update parameters
void MGauss::update_par(const unsigned int& k,
                        double sign){
  
  // update the IW precision matrix
  arma::vec v(mus.n_rows);
  arma::vec u(mus.n_rows);
  if(sign == 1.0){
    
    // update the cholesky factor of the inverse IW precision
    u = std::sqrt( lambdas(k) /  ( lambdas(k) + 1.0 ) ) * (y_i - mus.col(k));
    
    v = prod_Rchol_x(u,chol_Bm1s.col(k),idx_map,D);
    
    v = prod_Lchol_x(v,chol_Bm1s.col(k),idx_map,D) / std::sqrt(1.0 + arma::dot(v,v));
    
    chol_Bm1s.col(k) = chol_update(chol_Bm1s.col(k),v,idx_map,false);
    
    // update the mean
    mus.col(k) = ( lambdas(k)*mus.col(k) + sign* y_i ) / (lambdas(k) + sign);
    
  }else{
    
    // update the mean
    mus.col(k) = ( lambdas(k)*mus.col(k) + sign* y_i ) / (lambdas(k) + sign);
    
    // update the cholesky factor of the inverse IW precision
    u = std::sqrt( (lambdas(k)-1.0) / lambdas(k) ) * (y_i - mus.col(k));
    
    v = prod_Rchol_x(u,chol_Bm1s.col(k),idx_map,D);
    
    double c = 1.0 - arma::dot(v,v);
    if(c == 0){
      c += 1e-16;
    }
    
    v = prod_Lchol_x(v,chol_Bm1s.col(k),idx_map,D) / std::sqrt(std::abs(c));
    
    chol_Bm1s.col(k) = chol_update(chol_Bm1s.col(k),v,idx_map,c > 0);
    
  }
  
  // update the topic counter
  ns(k) += sign;
  
  // update the variance inflater
  lambdas(k) += sign;
  
  // update the IW d.o.f.
  alphas(k) += sign;
}

// log predictive function
double MGauss::log_pred(const unsigned int& k){
  
  // get the vector to square
  arma::vec tmp = prod_Rchol_x(y_i - mus.col(k),chol_Bm1s.col(k),idx_map,D);
  
  // compute some recurrent quantities
  double a1 = 0.5 * (alphas(k) + 1.0);
  double a2 = 0.5 * (alphas(k) - D + 1.0);
  double b = lambdas(k) / (lambdas(k) + 1.0);
  
  // return the unnormalized log predictive density
  return std::lgamma(a1) + 
    half_log_det_chol(chol_Bm1s.col(k),D) - 
    std::lgamma(a2) + 
    0.5 * D * std::log(b) - 
    a1 * std::log(1.0 + b * arma::dot(tmp,tmp));
  
}

// function that set the statistical unit
void MGauss::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MGauss::save_parameters(const std::string& filename,
                             const unsigned int& chain_id){
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double MGauss::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void MGauss::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void MGauss::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// ---------------- MULTIVARIATE GAUSSIAN KNWON SPHERICAL VARIANCE -------------

// constructor
MGauss1::MGauss1(std::vector<std::vector<unsigned int>>& partition,
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
               const double& m_) : y(y_), 
               hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
               a0(a0_),
               b0(b0_),
               scale(scale_),
               m(m_){
  
  // save the hyperparameters
  K = K_;
  D = y_.n_cols;
  alpha = alpha_;
  sigma2 = sigma2_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  
  // set the statistical unit
  y_i = arma::zeros<arma::vec>(D);
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  
  mus = arma::zeros<arma::mat>(D,K);
  sigma2s = arma::zeros<arma::vec>(K);
  
  // initialize the group sums
  arma::mat sums_y = arma::zeros<arma::mat>(D,K);
  
  unsigned int c_i = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // update the sufficient statistics
    ns(c_i)++;
    sums_y.col(c_i) += y.row(i).t();
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  // compute the parameter for the full conditional
  
  // means
  for(unsigned int k = 0; k < K; k++){
    mus.col(k) = (sums_y.col(k) + lambda0 * mu0) / (lambda0 + ns(k));
  }

  // variances
  sigma2s = sigma2 * (1.0 + 1.0 / (lambda0 + ns));
  
  // compute the unnormalized log posterior
  log_post = arma::sum(arma::lgamma(alpha + ns) - 
    0.5 * D * arma::log(lambda0 + ns));
  for(unsigned int k = 0; k < K; k++){
    log_post += 0.5 * arma::sum(mus.col(k) % mus.col(k) * 
      (lambda0 + ns(k)) / sigma2);
  }
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
}

// update parameters
void MGauss1::update_par(const unsigned int& c_i,
                        double sign){
  
  // update the dimension of the target group
  ns(c_i) += sign;
  
  // update parameters
  
  // mu
  mus.col(c_i) = ( (lambda0 + ns(c_i) - sign) * mus.col(c_i) + sign*y_i) / (lambda0 + ns(c_i));
  
  // sigma2
  sigma2s(c_i) = ( (lambda0 + ns(c_i) - sign) * 
    sigma2s(c_i) + sign*sigma2  ) / ( lambda0 + ns(c_i) );
  
}

// log predictive function
double MGauss1::log_pred(const unsigned int& k){
  
  arma::vec tmp = y_i - mus.col(k);
  
  return -0.5 * D * std::log(sigma2s(k)) - 
    0.5 * arma::dot(tmp,tmp) / sigma2s(k);
  
}

// function that set the statistical unit
void MGauss1::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MGauss1::save_parameters(const std::string& filename,
                             const unsigned int& chain_id){
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double MGauss1::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void MGauss1::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void MGauss1::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// ------------ MULTIVARIATE GAUSSIAN UNKNOWN DIAGONAL COVARIANCE --------------

// constructor
MGauss2::MGauss2(std::vector<std::vector<unsigned int>>& partition,
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
                 const double& m_) : y(y_), 
                 hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
                 a0(a0_),
                 b0(b0_),
                 scale(scale_),
                 m(m_){
  
  // save the hyperparameters
  K = K_;
  D = y_.n_cols;
  alpha = alpha_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // set the statistical unit
  y_i = arma::zeros<arma::vec>(D);
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  sums_y = arma::zeros<arma::mat>(D,K);
  sums_y2 = arma::zeros<arma::mat>(D,K);
  
  mus = arma::zeros<arma::mat>(D,K);
  lambdas = arma::zeros<arma::vec>(K);
  alphas = arma::zeros<arma::vec>(K);
  betas = arma::zeros<arma::mat>(D,K);
  
  unsigned int c_i = 0;
  arma::vec y_i = arma::zeros<arma::vec>(D);
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // get the datum
    y_i = y.row(i).t();
    
    // update the sufficient statistics
    ns(c_i)++;
    sums_y.col(c_i) += y_i;
    sums_y2.col(c_i) += y_i % y_i;
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  // compute the parameter for the full conditional
  
  // compute the parameter for the full conditional
  lambdas = lambda0 + ns;
  alphas = alpha0 + 0.5 * ns;
  for(unsigned int k = 0; k < K; k++){
    mus.col(k) = (sums_y.col(k) + lambda0 * mu0) / (lambda0 + ns(k));
    betas.col(k) = beta0 + 0.5 * (mu0 % mu0 * lambda0 + sums_y2.col(k) - lambdas(k) * mus.col(k) % mus.col(k));
  }

  // compute the unnormalized log posterior
  log_post = arma::sum(arma::lgamma(alpha + ns) -
    0.5 * D * arma::log(lambdas) + D * arma::lgamma(alphas));
  for(unsigned int k = 0; k < K; k++){
    log_post -= alphas(k) * arma::sum(arma::log(betas.col(k))); 
  }
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }

}

// update parameters
void MGauss2::update_par(const unsigned int& c_i,
                         double sign){
  
  // update the dimension of the target group
  ns(c_i) += sign;
  
  // update the sufficient statistics
  sums_y.col(c_i) += sign*y_i;
  sums_y2.col(c_i) += sign* y_i % y_i;
  
  // update parameter
  
  // alpha
  alphas(c_i) += sign*0.5;
  
  // mu
  mus.col(c_i) = (lambdas(c_i) * mus.col(c_i) + sign*y_i) / (lambdas(c_i) + sign);
  
  // lambda
  lambdas(c_i) += sign;
  
  // betas
  betas.col(c_i) = beta0 + 0.5 * (mu0%mu0*lambda0 + sums_y2.col(c_i) - mus.col(c_i) % mus.col(c_i) * lambdas(c_i));
  
}

// log predictive function
double MGauss2::log_pred(const unsigned int& k){
  
  arma::vec tmp = y_i - mus.col(k);
  
  return (std::lgamma(alphas(k) + 0.5) - std::lgamma(alphas(k))) * D -
    arma::sum( 0.5 * arma::log(2.0 * betas.col(k) * (1.0 + 1.0 / lambdas(k))) +
    (alphas(k) + 0.5) * arma::log(1.0 + 0.5 * 
    tmp % tmp / (1.0 + 1.0 / lambdas(k)) / betas.col(k) ));
  
  
}

// function that set the statistical unit
void MGauss2::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MGauss2::save_parameters(const std::string& filename,
                              const unsigned int& chain_id){
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double MGauss2::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void MGauss2::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void MGauss2::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// --------------------------------- POISSON -----------------------------------

// constructor
Poiss::Poiss(std::vector<std::vector<unsigned int>>& partition,
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
             const double& m_) : y(y_), 
             hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
             a0(a0_),
             b0(b0_),
             scale(scale_),
             m(m_){
  
  // save the hyperparameters
  K = K_;
  alpha = alpha_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // set the statistical unit
  y_i = 0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  
  alphas = arma::zeros<arma::vec>(K);
  betas = arma::zeros<arma::vec>(K);
  
  unsigned int c_i = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // update the sufficient statistics
    ns(c_i)++;
    alphas(c_i) += y(i);
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  // compute the parameter for the full conditional
  
  // add the prior for alphas
  alphas += alpha0;
  
  // betas
  betas = beta0 + ns;
  
  // compute the unnormalized log posterior
  log_post = arma::sum(arma::lgamma(alpha + ns) + 
    arma::lgamma(alphas) - 
    alphas % arma::log(betas));
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
  
}

// update parameters
void Poiss::update_par(const unsigned int& c_i,
                       double sign){
  
  // update the dimension of the target group
  ns(c_i) += sign;
  
  // update parameters
  
  // alphas
  alphas(c_i) += sign * y_i;
  
  // betas
  betas(c_i) += sign;
  
}

// log predictive function
double Poiss::log_pred(const unsigned int& k){
  
  return alphas(k) * std::log(betas(k)) -
    (alphas(k) + y_i) * std::log(1.0 + betas(k)) +
    std::lgamma(alphas(k) + y_i) - 
    std::lgamma(alphas(k));
  
}

// function that set the statistical unit
void Poiss::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void Poiss::save_parameters(const std::string& filename,
                            const unsigned int& chain_id){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + ".csv", std::ios::app);
  
  // sample the mixing weights
  arma::vec mix = rdirichlet(alpha + ns);
  
  // save the values
  for(unsigned int k = 0; k < K-1; k++){
    
    file_pars << mix(k) << "," <<  R::rgamma(alphas(k),1.0 / betas(k)) << ",";
  }
  
  // add the last one with an endline
  file_pars << mix(K-1) << "," <<  R::rgamma(alphas(K-1),1.0 / betas(K-1)) << "\n";
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double Poiss::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void Poiss::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void Poiss::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// -------------------------------- BINOMIAL -----------------------------------

// constructor
Binom::Binom(std::vector<std::vector<unsigned int>>& partition,
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
             const double& m_) : y(y_), n_trials(n_trials_), 
             hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
             a0(a0_),
             b0(b0_),
             scale(scale_),
             m(m_){
  
  // save the hyperparameters
  K = K_;
  alpha = alpha_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // set the statistical unit
  y_i = 0;
  n_i = 0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  
  alphas = arma::zeros<arma::vec>(K);
  betas = arma::zeros<arma::vec>(K);
  
  unsigned int c_i = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // update the sufficient statistics
    ns(c_i)++;
    alphas(c_i) += y(i);
    betas(c_i) += n_trials(i) - y(i);
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  // compute the parameter for the full conditional
  
  // add the prior for alphas
  alphas += alpha0;
  
  // betas
  betas += beta0;
  
  // compute the unnormalized log posterior
  log_post = arma::sum(arma::lgamma(alpha + ns) + 
    lbeta(alphas,betas));
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
  
}

// update parameters
void Binom::update_par(const unsigned int& c_i,
                       double sign){
  
  // update the dimension of the target group
  ns(c_i) += sign;
  
  // update parameters
  
  // alphas
  alphas(c_i) += sign * y_i;
  
  // betas
  betas(c_i) += sign * ( n_i - y_i );
  
}

// log predictive function
double Binom::log_pred(const unsigned int& k){
  
  return lbeta(alphas(k) + y_i, betas(k) + n_i - y_i) -
    lbeta(alphas(k),betas(k));
  
}

// function that set the statistical unit
void Binom::set(const unsigned int& i){
  y_i = y(i);
  n_i = n_trials(i);
}

// function to generate the collapsed parameters
// and save them to file
void Binom::save_parameters(const std::string& filename,
                            const unsigned int& chain_id){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + ".csv", std::ios::app);
  
  // sample the mixing weights
  arma::vec mix = rdirichlet(alpha + ns);
  
  // save the values
  for(unsigned int k = 0; k < K-1; k++){
    
    file_pars << mix(k) << "," << R::rbeta(alphas(k),betas(k)) << ",";
  }
  
  // add the last one with an endline
  file_pars << mix(K-1) << "," << R::rbeta(alphas(K-1),betas(K-1)) << "\n";
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double Binom::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void Binom::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void Binom::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// ----------------------------- BERNOULLI PRODUCT -----------------------------

// constructor
MBern::MBern(std::vector<std::vector<unsigned int>>& partition,
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
             const double& m_) : y(y_), 
             hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
             a0(a0_),
             b0(b0_),
             scale(scale_),
             m(m_){
  
  // save the hyperparameters
  K = K_;
  D = y.n_cols;
  alpha = alpha_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // set the statistical unit
  y_i = arma::zeros<arma::uvec>(D);
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  
  n_dk1 = arma::zeros<arma::mat>(D,K);
  
  unsigned int c_i = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // update the sufficient statistics
    ns(c_i)++;
    
    // success counter
    for(unsigned int d = 0; d < D; d++){
      
      if(y(i,d) == 1){
        n_dk1(d,c_i)++;
      }
      
    }
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  log_post = 0.0;
  // initialize the unnormalized log posterior
  for(unsigned int k = 0; k < K; k++){
    log_post += std::lgamma(alpha(k) + ns(k)) - 
      D*std::lgamma(alpha0+beta0+ns(k));
    for(unsigned int d = 0; d < D; d++){
      log_post += std::lgamma(alpha0 + n_dk1(d,k)) + 
        std::lgamma(beta0 + ns(k) - n_dk1(d,k));
    }
  }
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
  
}

// update parameters
void MBern::update_par(const unsigned int& c_i,
                       double sign){
  
  // update the cluster counts
  ns(c_i) += sign;
  
  // update each coordinate
  for(unsigned int d = 0; d < D; d++){
    if(y_i(d) == 1){
      n_dk1(d,c_i) += sign;
    }
  }
  
}

// log predictive function
double MBern::log_pred(const unsigned int& k){
  
  // initialize the output
  double out = D * -std::log(alpha0 + beta0 + ns(k));
  
  // add the dimension varying part
  for(unsigned int d = 0; d < D; d++){
    if(y_i(d) == 1){
      out += std::log(alpha0 + n_dk1(d,k));
    }else{
      out += std::log(beta0 + ns(k) - n_dk1(d,k));
    }
  }
  
  // return the output
  return out;
  
}

// function that set the statistical unit
void MBern::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MBern::save_parameters(const std::string& filename,
                            const unsigned int& chain_id){
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double MBern::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void MBern::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void MBern::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// ------------------------------ PARTITION ------------------------------------

// constructor
Partition::Partition(std::vector<std::vector<unsigned int>>& partition,
                     const unsigned int& K_,
                     const arma::vec& alpha_,
                     const arma::uvec& c,
                     const bool& gibbs,
                     const bool& hyperpar_alpha_,
                     const bool& hyperpar_baseline_,
                     const double& a0_,
                     const double& b0_,
                     const double& scale_,
                     const double& m_) : hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
                     a0(a0_),
                     b0(b0_),
                     scale(scale_),
                     m(m_){
  
  // save the hyperparameters
  K = K_;
  alpha = alpha_;
  
  // get the dimension
  n = c.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns = arma::zeros<arma::vec>(K);
  
  unsigned int c_i = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < c.n_elem; i++){
    
    // get the label
    c_i = c(i);
    
    // update the sufficient statistics
    ns(c_i)++;
    
    // update the partition
    if(!gibbs){
      partition[c_i].push_back(i);
    }
    
  }
  
  // initialize the unnormalized log posterior
  for(unsigned int k = 0; k < K; k++){
    log_post += std::lgamma(alpha(k) + ns(k));
  }
  
  // add the concentration hyperprior terms
  if(hyperpar_alpha){
    log_post += std::lgamma(K*alpha(0)) - 
      std::lgamma(K*alpha(0) + n) - 
      K*std::lgamma(alpha(0));
    
    log_post += a0 * std::log(alpha(0)) - b0 * alpha(0);
  }
  
}

// update parameters
void Partition::update_par(const unsigned int& c_i,
                           double sign){
  
  // update the cluster counts
  ns(c_i) += sign;
  
}

// log predictive function
double Partition::log_pred(const unsigned int& k){
  
  return 0.0;
  
}

// function that set the statistical unit
void Partition::set(const unsigned int& i){
  
}

// function to generate the collapsed parameters
// and save them to file
void Partition::save_parameters(const std::string& filename,
                                const unsigned int& chain_id){
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double Partition::hyperpar_lfc(const double& x,
                            unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with gamma prior)
    
    // get alpha from omega
    double alpha = std::exp(x);
    
    // return the full conditional
    return a0 * x - b0 * alpha + 
      std::lgamma(alpha*K) - 
      K*std::lgamma(alpha) - 
      std::lgamma(alpha*K + n) + 
      arma::sum(arma::lgamma(alpha + ns));
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void Partition::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void Partition::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    alpha.zeros();
    alpha += slice_sampler(*this,log_post2,alpha(0),0,true);
  }
  
}

// -----------------------------------------------------------------------------
// -------------------------- INFINITE MIXTURE CASE ----------------------------
// -----------------------------------------------------------------------------

// ----------------------- UNIVARIATE GAUSSIAN KNWON VARIANCE ------------------

// constructor
inf_Gauss1::inf_Gauss1(std::vector<std::vector<unsigned int>>& partition,
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
                       const double& m_) : y(y_), 
                       hyperpar_alpha(hyperpar_alpha_), 
                       hyperpar_delta(hyperpar_delta_),
                       hyperpar_baseline(hyperpar_baseline_),
                       a0(a0_),
                       b0(b0_),
                       lwr_bound(lwr_bound_),
                       upr_bound(upr_bound_),
                       scale(scale_),
                       m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  alpha = alpha_;
  delta = delta_;
  sigma2 = sigma2_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  
  sigma20 = sigma2* (1.0 + 1.0/lambda0);
  
  // set the statistical unit
  y_i = 0.0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  
  mus.reserve(n);
  sigma2s.reserve(n);
  
  // initialize the group sums
  std::vector<double> sums_y;
  sums_y.reserve(n);
  
  // loop over each observation
  
  // initialize the number of active components
  unsigned int k = 0;
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      sums_y.push_back(0.0);
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    sums_y[k] += y(i);
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // update the k-th mean
    mus.push_back( (lambda0*mu0 + sums_y[k]) / (lambda0 + ns[k]) );
    
    // update the k-th variance
    sigma2s.push_back( sigma2 * (1.0 + 1.0 / (lambda0 + ns[k])) );
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += -0.5 * std::log(lambda0 + ns[k]) + 
      0.5 * std::log(lambda0) + 
      0.5 * mus[k] * mus[k] * (lambda0 + ns[k]) / sigma2 - 
      0.5 * mu0 * mu0 * lambda0 / sigma2; 
    
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
}

// update parameters
void inf_Gauss1::update_par(const unsigned int& c_i,
                            double sign){
  
  // update the dimension of the target group
  ns[c_i] += sign;
  
  // update parameters
  
  // mu
  mus[c_i] = ( (lambda0 + ns[c_i] - sign) * mus[c_i] + sign*y_i) / (lambda0 + ns[c_i]);
  
  // sigma2
  sigma2s[c_i] = ( (lambda0 + ns[c_i] - sign) * 
    sigma2s[c_i] + sign*sigma2  ) / ( lambda0 + ns[c_i] );
  
}

// log predictive function
double inf_Gauss1::log_pred(const unsigned int& k){
  
  return -0.5 * std::log(sigma2s[k]) - 
    0.5 * (y_i - mus[k]) * (y_i - mus[k]) / sigma2s[k];
  
}

// log prior predictive
double inf_Gauss1::log_pred_prior(){
  
  return -0.5 * std::log(sigma20) - 
    0.5 * (y_i - mu0) * (y_i - mu0) / sigma20;
  
}

// function that set the statistical unit
void inf_Gauss1::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void inf_Gauss1::save_parameters(const std::string& filename,
                                 const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_Gauss1::copy_atom(const unsigned int& atm1,
                           const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  mus[atm1] = mus[atm2];
  sigma2s[atm1] = sigma2s[atm2];
  
}

// function to delete one atom
void inf_Gauss1::delete_last_atom(){
  
  ns.pop_back();
  mus.pop_back();
  sigma2s.pop_back();
  
}

// funciton to add an atom
void inf_Gauss1::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  mus.push_back( mu0 );
  sigma2s.push_back( sigma20 );
  
}

// function to merge two atoms into one
void inf_Gauss1::merge_atoms(const unsigned int& atm1,
                             const unsigned int& atm2){
  
  // get the new atom index
  unsigned int idx = ns.size()-1;
  
  // merge the sample size
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the means
  mus[idx] = ( (lambda0 + ns[atm1]) * mus[atm1] + 
    (lambda0 + ns[atm2]) * mus[atm2] - mu0*lambda0 ) / 
    (lambda0 + ns[idx]);
  
  // merge the variances
  // sigma2s[idx] = ( (lambda0 + ns[atm1]) * sigma2s[atm1] + 
  //   (lambda0 + ns[atm2]) * sigma2s[atm2] - sigma2 ) / 
  //   (lambda0 + ns[idx]);
  sigma2s[idx] = sigma2 * (1.0 + 1.0 / (lambda0 + ns[idx]));
  
}

// function to compute the log marginal likelihood of a given group of data
double inf_Gauss1::log_marginal(const unsigned int& atm){
  
  return -0.5 * std::log(lambda0 + ns[atm]) + 0.5 * std::log(lambda0) + 
    0.5 * mus[atm] * mus[atm] * (lambda0 + ns[atm]) / sigma2 - 
    0.5 * mu0 * mu0 * lambda0 / sigma2; 
  
}

double inf_Gauss1::hyperpar_lfc(const double& x,
                                unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_Gauss1::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_Gauss1::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}

// ---------------------- UNIVARIATE GAUSSIAN UNKNWON VARIANCE -----------------

// constructor
inf_Gauss2::inf_Gauss2(std::vector<std::vector<unsigned int>>& partition,
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
                       const double& m_) : y(y_), 
                       hyperpar_alpha(hyperpar_alpha_), 
                       hyperpar_delta(hyperpar_delta_),
                       hyperpar_baseline(hyperpar_baseline_),
                       a0(a0_),
                       b0(b0_),
                       lwr_bound(lwr_bound_),
                       upr_bound(upr_bound_),
                       M(M_), V(V_),a_l(a_l_),b_l(b_l_),a_a(a_a_),b_a(b_a_),a_b(a_b_),b_b(b_b_),
                       scale(scale_),
                       m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  alpha = alpha_;
  delta = delta_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // hyperpriors parameters
  // M = 0.0;
  // V = 10;
  // a_l = b_l = a_a = b_a = a_b = b_b = 1;
  
  prior_term1 = std::lgamma(alpha0 + 0.5) - std::lgamma(alpha0) - 
    0.5 * std::log(2.0 * beta0 * (1.0 + 1.0 / lambda0));
  
  prior_term2 = 0.5 / (1.0 + 1.0 / lambda0) / beta0;
  
  // set the statistical unit
  y_i = 0.0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  sums_y.reserve(n);
  sums_y2.reserve(n);
  
  mus.reserve(n);
  lambdas.reserve(n);
  alphas.reserve(n);
  betas.reserve(n);
  
  unsigned int k = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < n; i++){
    
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
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    sums_y[k] += y(i);
    sums_y2[k] += y(i) * y(i);
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // update the k-th precision
    lambdas.push_back( lambda0 + ns[k] );
    
    // update the k-th mean
    mus.push_back( (lambda0*mu0 + sums_y[k]) / (lambdas[k]) );
    
    // update the k-th variance parameters
    alphas.push_back( alpha0 + 0.5 * ns[k] );
    
    betas.push_back( beta0 + 0.5 * (mu0 * mu0 * lambda0 + sums_y2[k] - lambdas[k] * mus[k] * mus[k]) );
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n) - 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += -0.5 * std::log(lambdas[k]) + 0.5 * std::log(lambda0) - 
      alphas[k] * std::log(betas[k]) + alpha0 * std::log(beta0) + 
      std::lgamma(alphas[k]) - std::lgamma(alpha0);
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
  
  // add the hyperprior's prior terms
  if(hyperpar_baseline){
    log_post += -0.5 * (mu0 - M) * (mu0 - M) / V + 
      a_l * std::log(lambda0) - b_l*lambda0 + 
      a_a * std::log(alpha0) - b_a*alpha0 + 
      a_b * std::log(beta0) - b_b*beta0 ;
  }
}

// update parameters
void inf_Gauss2::update_par(const unsigned int& c_i,
                            double sign){
  
  // update the dimension of the target group
  ns[c_i] += sign;
  
  // update the sufficient statistics
  sums_y[c_i] += sign*y_i;
  sums_y2[c_i] += sign*y_i * y_i;
  
  // update parameter
  
  // alpha
  alphas[c_i] += sign*0.5;
  
  // mu
  mus[c_i] = (lambdas[c_i] * mus[c_i] + sign*y_i) / (lambdas[c_i] + sign);
  
  // lambda
  lambdas[c_i] += sign;
  
  // betas
  betas[c_i] = beta0 + 0.5 * (mu0*mu0*lambda0 + sums_y2[c_i] - mus[c_i] * mus[c_i] * lambdas[c_i]);
  
}

// log predictive function
double inf_Gauss2::log_pred(const unsigned int& k){
  
  return std::lgamma(alphas[k] + 0.5) - std::lgamma(alphas[k]) - 
    0.5 * std::log(2.0 * betas[k] * (1.0 + 1.0 / lambdas[k])) - 
    (alphas[k] + 0.5) * std::log(1.0 + 0.5 * 
    (y_i - mus[k]) * (y_i - mus[k]) / (1.0 + 1.0 / lambdas[k]) / betas[k] );
  
}

// log prior predictive
double inf_Gauss2::log_pred_prior(){
  
  return prior_term1 - 
    (alpha0 + 0.5) * std::log(1.0 + prior_term2 * (y_i - mu0) * (y_i - mu0));
  
}

// function that set the statistical unit
void inf_Gauss2::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void inf_Gauss2::save_parameters(const std::string& filename,
                                 const unsigned int& chain_id){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + ".csv", std::ios::app);
  
  // save the hyperparameters
  file_pars << mu0 << "," << lambda0 << "," << alpha0 << "," << beta0 << "\n";
  
  
  
}

// function to copy one atom into the other
void inf_Gauss2::copy_atom(const unsigned int& atm1,
                           const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  sums_y[atm1] = sums_y[atm2];
  sums_y2[atm1] = sums_y2[atm2];
  mus[atm1] = mus[atm2];
  lambdas[atm1] = lambdas[atm2];
  alphas[atm1] = alphas[atm2];
  betas[atm1] = betas[atm2];
  
}

// function to delete one atom
void inf_Gauss2::delete_last_atom(){
  
  ns.pop_back();
  sums_y.pop_back();
  sums_y2.pop_back();
  mus.pop_back();
  lambdas.pop_back();
  alphas.pop_back();
  betas.pop_back();
  
}

// funciton to add an atom
void inf_Gauss2::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  sums_y.push_back(0.0);
  sums_y2.push_back(0.0);
  mus.push_back( mu0 );
  lambdas.push_back( lambda0 );
  alphas.push_back( alpha0 );
  betas.push_back( beta0 );
  
  
}

// function to merge two atoms into one
void inf_Gauss2::merge_atoms(const unsigned int& atm1,
                             const unsigned int& atm2){
  
  // get the merged group index
  unsigned int idx = ns.size()-1;
  
  // merge the sample sizes
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the sufficient statistics
  sums_y[idx] = sums_y[atm1] + sums_y[atm2];
  sums_y2[idx] = sums_y2[atm1] + sums_y2[atm2];
  
  // merge the parameters
  
  // alpha
  alphas[idx] = alphas[atm1] + alphas[atm2] - alpha0;
  
  // lambda
  lambdas[idx] = lambdas[atm1] + lambdas[atm2] - lambda0;
  
  // mu
  mus[idx] = (lambdas[atm1] * mus[atm1] + lambdas[atm2] * mus[atm2] -lambda0*mu0 ) / lambdas[idx];
  
  // betas
  betas[idx] = beta0 + 0.5 * (mu0*mu0*lambda0 + sums_y2[idx] - mus[idx] * mus[idx] * lambdas[idx]);
  
  
}

// function to compute the log marginal likelihood of a given group of data
double inf_Gauss2::log_marginal(const unsigned int& atm){

  return -0.5 * std::log(lambdas[atm]) + 
    0.5 * std::log(lambda0) - 
    alphas[atm] * std::log(betas[atm]) + 
    alpha0 * std::log(beta0) + 
    std::lgamma(alphas[atm]) - 
    std::lgamma(alpha0);  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_Gauss2::hyperpar_lfc(const double& x,
                                unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else if(which == 2){
    
    // mu0 (with normal prior)
    
    // initialize the output with the hyperprior's log density
    double out = -0.5 * (x - M) * (x - M) / V;
    
    // add the posterior contributes
    for(unsigned int k=0; k < ns.size(); k++){
      
      out -= alphas[k] * std::log( beta0 + 0.5 * ( x*x*lambda0 + sums_y2[k] - 
        (x*lambda0 + sums_y[k]) * (x*lambda0 + sums_y[k]) / lambdas[k] ) );
      
    }
    
    // return the log density
    return out;
    
  }else if(which == 3){
    
    // lambda0 (log scale with gamma prior)
    double ll = std::exp(x);
    
    // initialize the output with the hyper prior contribute
    // as well with all the constant terms
    double out = (0.5*ns.size() + a_l) * x - b_l * ll;
    
    // add the other posterior contributes
    for(unsigned int k = 0; k < ns.size(); k++){
      
      out -= 0.5 * std::log(ll + ns[k]) + 
        alphas[k] * std::log(beta0 + 0.5 * ( mu0*mu0*ll + sums_y2[k] - 
        (ll*mu0 + sums_y[k]) * (ll*mu0 + sums_y[k]) / (ll + ns[k]) ));
      
    }
    
    // return the log density
    return out;
    
  }else if(which == 4){
    
    // alpha0 (log scale with gamma prior)
    double aa = std::exp(x);
    
    // initialize the output with the hyper prior contribute
    // as well with all the constant terms
    double out = a_a * x - b_a * aa + 
      ns.size() * ( aa * std::log(beta0) - std::lgamma(aa) );
    
    // add the other posterior contributes
    for(unsigned int k = 0; k < ns.size(); k++){
      
      out += std::lgamma(aa + 0.5 * ns[k]) - 
        (aa + 0.5*ns[k]) * std::log(betas[k]);
    }
    
    // return the log density
    return out;
    
  }else if(which == 5){
    
    // beta0 (log scale with gamma prior)
    double bb = std::exp(x);
    
    // initialize the output with the hyper prior contribute
    // as well with all the constant terms
    double out = a_b * x - b_b * bb + 
      ns.size() * alpha0 * std::log(bb);
    
    // add the other posterior contributes
    for(unsigned int k = 0; k < ns.size(); k++){
      
      out -= alphas[k] * std::log(betas[k] - beta0 + bb);
    }
    
    // return the log density
    return out;
    
  }
  
  return 0.0;
  
}

// function to update the atoms given a change in the hyperparameters
void inf_Gauss2::update_atoms(double new_value, unsigned int which){
  
  if(which == 0){
    
    // mu0
    for(unsigned int k = 0; k < ns.size(); k++){
      
      mus[k] += lambda0 * (new_value - mu0) / lambdas[k];
      
      betas[k] = beta0 + 0.5 * ( new_value*new_value*lambda0 + sums_y2[k] - mus[k]*mus[k]*lambdas[k] );
      
    }
    
    mu0 = new_value;
    
    // Rcpp::Rcout << "mu0: " << mu0 << std::endl;
    
  }else if(which == 1){
    
    // lambda0
    
    for(unsigned int k = 0; k < ns.size(); k++){
      
      lambdas[k] += (new_value - lambda0);
      mus[k] = ( (lambda0 + ns[k]) * mus[k] + mu0 * (new_value - lambda0) ) / lambdas[k];
      betas[k] = beta0 + 0.5 * ( mu0*mu0*new_value + sums_y2[k] - mus[k]*mus[k]*lambdas[k] );
      
    }
    
    lambda0 = new_value;
    
    // Rcpp::Rcout << "lambda0: " << lambda0 << std::endl;
    
  }else if(which == 2){
    
    // alpha0
    
    for(unsigned int k = 0; k < ns.size(); k++){
      
      alphas[k] += (new_value - alpha0);
      
    }
    
    alpha0 = new_value;
    
    // Rcpp::Rcout << "alpha0: " << alpha0 << std::endl;
    
  }else if(which == 3){
    
    // beta0
    
    for(unsigned int k = 0; k < ns.size(); k++){
      
      betas[k] += (new_value - beta0);
      
    }
    
    beta0 = new_value;
    
    // Rcpp::Rcout << "beta0: " << beta0 << std::endl;
    
  }
  
}

// function to update the hyperparameters
void inf_Gauss2::update_hyperpars(double& log_post2){
  
  // concentration parameter
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  // discount parameter
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
  // baseline distribution
  if(hyperpar_baseline){
    
    // initialize the temporary new value
    double tmp;
    
    // mu0
    tmp = slice_sampler(*this, log_post2, mu0, 2, false);
    update_atoms(tmp,0);
    
    // lambda0
    tmp = slice_sampler(*this, log_post2, lambda0, 3, true);
    update_atoms(tmp,1);
    
    // alpha0
    tmp = slice_sampler(*this, log_post2, alpha0, 4, true);
    update_atoms(tmp,2);
    
    // beta0
    tmp = slice_sampler(*this, log_post2, beta0, 5, true);
    update_atoms(tmp,3);
    
    // update the prior terms
    prior_term1 = std::lgamma(alpha0 + 0.5) - std::lgamma(alpha0) -
      0.5 * std::log(2.0 * beta0 * (1.0 + 1.0 / lambda0));
    
    prior_term2 = 0.5 / (1.0 + 1.0 / lambda0) / beta0;
  }
  
}

// ---------------------------- MULTIVARIATE GAUSSIAN --------------------------

// constructor
inf_MGauss::inf_MGauss(std::vector<std::vector<unsigned int>>& partition,
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
                       const double& bb0_,
                       const double& lwr_bound_,
                       const double& upr_bound_,
                       const double& scale_,
                       const double& m_) : y(y_), 
                       hyperpar_alpha(hyperpar_alpha_), 
                       hyperpar_delta(hyperpar_delta_),
                       hyperpar_baseline(hyperpar_baseline_),
                       a0(a0_),
                       bb0(bb0_),
                       lwr_bound(lwr_bound_),
                       upr_bound(upr_bound_),
                       scale(scale_),
                       m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  D = y.n_cols;
  alpha = alpha_;
  delta = delta_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  alpha0 = alpha0_;
  B0 = B0_;
  chol_Bm10 = arma::zeros<arma::vec>(D*(D+1)/2);
  
  arma::mat chol_Bm10_mat = arma::chol(arma::inv(B0_));
  
  // set the statistical unit
  y_i = arma::zeros<arma::vec>(D);
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  
  mus.reserve(n);
  lambdas.reserve(n);
  alphas.reserve(n);
  chol_Bm1s.reserve(n);
  idx_map = arma::zeros<arma::umat>(D,D);
  
  // compute the idx map
  unsigned int conta = 0;
  for(unsigned int j = 0; j < D; j++){
    for(unsigned int i = 0; i < D; i++){
      
      if(j <= i){
        idx_map(i,j) = idx_map(j,i) = conta;
        conta++;
      }
      
    }
  }
  
  // fill the transpose of the right cholesky
  for(unsigned int i = 0; i < D; i++){
    for(unsigned int j = 0; j <= i; j++){
      
      chol_Bm10(idx_map(i,j)) = chol_Bm10_mat(j,i);
      
    }
  }
  
  std::vector<arma::vec> y_bar;
  y_bar.reserve(n);
  
  std::vector<arma::mat> Bs;
  Bs.reserve(n);
  
  unsigned int k = 0;
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      
      y_bar.push_back(arma::zeros<arma::vec>(D));
      Bs.push_back(arma::zeros<arma::mat>(D, D));
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    
    // update the deviance
    Bs[k] += ( ns[k]-1.0) / ns[k] *
      (y.row(i).t() - y_bar[k]) * (y.row(i).t() - y_bar[k]).t();
    
    // update the means
    for(unsigned int dd = 0; dd < D; dd++){
      y_bar[k](dd) += ( y(i,dd) - y_bar[k](dd) ) / ns[k];
    }
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // update the k-th precision
    lambdas.push_back( lambda0 + ns[k] );
    
    // update the k-th mean
    mus.push_back( ( lambda0*mu0 + ns[k] * y_bar[k] ) / lambdas[k] );    
    
    // update the k-th variance parameters
    alphas.push_back( alpha0 + ns[k] );
    
    // Bs
    Bs[k] += B0_ + (lambda0 * ns[k]) / (lambda0 + ns[k]) * 
      (y_bar[k] - mu0) * (y_bar[k] - mu0).t(); 
    
    // solve this matrix and compute the cholesky
    Bs[k] = arma::chol(arma::inv(Bs[k]));
    
    arma::vec tmp = arma::zeros<arma::vec>(D*(D+1)/2);
    // fill the transpose of the right cholesky
    for(unsigned int i = 0; i < D; i++){
      for(unsigned int j = 0; j <= i; j++){
        
        tmp(idx_map(i,j)) = Bs[k](j,i);
        
      }
    }
    
    chol_Bm1s.push_back(tmp);
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += alphas[k] * half_log_det_chol(chol_Bm1s[k],D) -
      alpha0 * half_log_det_chol(chol_Bm10,D) -
      0.5 * D * std::log(lambdas[k]) + 
      0.5 * D * std::log(lambda0);
    
    // add the multivariate gamma function term
    for(unsigned int dd = 0; dd < D; dd++){
      log_post += std::lgamma(0.5 * (alphas[k] - dd)) - 
        std::lgamma(0.5 * (alpha0 - dd));
    }
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(bb0 - alpha) - 
      std::log(bb0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
  
  // recurrent prior predictive terms
  a10 = 0.5 * (alpha0 + 1.0);
  a20 = 0.5 * (alpha0 - D + 1.0);
  b0 = lambda0 / (lambda0 + 1.0);
  
  prior_term = std::lgamma(a10) + 
    half_log_det_chol(chol_Bm10,D) - 
    std::lgamma(a20) + 
    0.5 * D * std::log( b0 );
}

// update parameters
void inf_MGauss::update_par(const unsigned int& k,
                            double sign){
  
  // update the IW precision matrix
  arma::vec v(mus[0].n_elem);
  arma::vec u(mus[0].n_elem);
  if(sign == 1.0){
    
    // update the cholesky factor of the inverse IW precision
    u = std::sqrt( lambdas[k] /  ( lambdas[k] + 1.0 ) ) * (y_i - mus[k]);
    
    v = prod_Rchol_x(u,chol_Bm1s[k],idx_map,D);
    
    v = prod_Lchol_x(v,chol_Bm1s[k],idx_map,D) / std::sqrt(1.0 + arma::dot(v,v));
    
    chol_Bm1s[k] = chol_update(chol_Bm1s[k],v,idx_map,false);
    
    // update the mean
    mus[k] = ( lambdas[k]*mus[k] + sign* y_i ) / (lambdas[k] + sign);
    
  }else{
    
    // update the mean
    mus[k] = ( lambdas[k]*mus[k] + sign* y_i ) / (lambdas[k] + sign);
    
    // update the cholesky factor of the inverse IW precision
    u = std::sqrt( (lambdas[k]-1.0) / lambdas[k] ) * (y_i - mus[k]);
    
    v = prod_Rchol_x(u,chol_Bm1s[k],idx_map,D);
    
    double c = 1.0 - arma::dot(v,v);
    if(c == 0){
      c += 1e-16;
    }
    
    v = prod_Lchol_x(v,chol_Bm1s[k],idx_map,D) / std::sqrt(std::abs(c));
    
    chol_Bm1s[k] = chol_update(chol_Bm1s[k],v,idx_map,c > 0);
    
  }
  
  // update the topic counter
  ns[k] += sign;
  
  // update the variance inflater
  lambdas[k] += sign;
  
  // update the IW d.o.f.
  alphas[k] += sign;
}

// log predictive function
double inf_MGauss::log_pred(const unsigned int& k){
  
  // get the vector to square
  arma::vec tmp = prod_Rchol_x(y_i - mus[k],chol_Bm1s[k],idx_map,D);
  
  // compute some recurrent quantities
  double a1 = 0.5 * (alphas[k] + 1.0);
  double a2 = 0.5 * (alphas[k] - D + 1.0);
  double b = lambdas[k] / (lambdas[k] + 1.0);
  
  // return the unnormalized log predictive density
  return std::lgamma(a1) + 
    half_log_det_chol(chol_Bm1s[k],D) - 
    std::lgamma(a2) + 
    0.5 * D * std::log(b) - 
    a1 * std::log(1.0 + b * arma::dot(tmp,tmp));
  
}

// log prior predictive
double inf_MGauss::log_pred_prior(){
  
  // get the vector to square
  arma::vec tmp = prod_Rchol_x(y_i - mu0,chol_Bm10,idx_map,D);
  
  // return the unnormalized log predictive density
  return prior_term - 
    a10 * std::log(1.0 + b0 * arma::dot(tmp,tmp));
  
  
}

// function that set the statistical unit
void inf_MGauss::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void inf_MGauss::save_parameters(const std::string& filename,
                                 const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_MGauss::copy_atom(const unsigned int& atm1,
                           const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  mus[atm1] = mus[atm2];
  lambdas[atm1] = lambdas[atm2];
  alphas[atm1] = alphas[atm2];
  chol_Bm1s[atm1] = chol_Bm1s[atm2];
  
}

// function to delete one atom
void inf_MGauss::delete_last_atom(){
  
  ns.pop_back();
  mus.pop_back();
  lambdas.pop_back();
  alphas.pop_back();
  chol_Bm1s.pop_back();
  
}

// funciton to add an atom
void inf_MGauss::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  mus.push_back( mu0 );
  lambdas.push_back( lambda0 );
  alphas.push_back( alpha0 );
  chol_Bm1s.push_back( chol_Bm10 );
  
}

// function to merge two atoms into one
void inf_MGauss::merge_atoms(const unsigned int& atm1,
                             const unsigned int& atm2){
  
  // get the merged group index
  unsigned int idx = ns.size()-1;
  
  // merge the sample sizes
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the variance inflater
  lambdas[idx] = lambdas[atm1] + lambdas[atm2] - lambda0;
  
  // merge the means
  mus[idx] = ( lambdas[atm1] * mus[atm1] + 
    lambdas[atm2] * mus[atm2] - mu0*lambda0 ) / 
    lambdas[idx];
  
  // update the IW d.o.f.
  alphas[idx] = alphas[atm1] + alphas[atm2] - alpha0;
  
  // merge the cholesky factors
  
  // get the sum of the inverses of the outer cholesky product
  // of the two matrices and subtract the prior
  arma::mat tmp = inv_outer_chol(chol_Bm1s[atm1],idx_map,D) + 
    inv_outer_chol(chol_Bm1s[atm2],idx_map,D) - B0 + 
    lambdas[atm1]*lambdas[atm2]/lambdas[idx] * 
    (mus[atm1] - mus[atm2]) * (mus[atm1] - mus[atm2]).t() - 
    lambda0 * (mu0 - mus[idx]) * (mu0 - mus[idx]).t(); 
  
  // get the cholesky of the inverse
  tmp = arma::chol(arma::inv(tmp));
  
  // fill the merge cholesky vector
  for(unsigned int i = 0; i < D; i++){
    for(unsigned int j = 0; j <= i; j++){
      
      chol_Bm1s[idx](idx_map(i,j)) = tmp(j,i);
      
    }
  }
}

// function to compute the log marginal likelihood of a given group of data
double inf_MGauss::log_marginal(const unsigned int& atm){
  
  // initialize the output
  double out = alphas[atm] * half_log_det_chol(chol_Bm1s[atm],D) -
    alpha0 * half_log_det_chol(chol_Bm10,D) -
    0.5 * D * std::log(lambdas[atm]) + 
    0.5 * D * std::log(lambda0);
  
  // add the multivariate gamma function term
  for(unsigned int dd = 0; dd < D; dd++){
    out += std::lgamma(0.5 * (alphas[atm] - dd)) - 
      std::lgamma(0.5 * (alpha0 - dd));
  }
  
  // return the output
  return out;
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_MGauss::hyperpar_lfc(const double& x,
                                unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (bb0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(bb0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_MGauss::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_MGauss::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(bb0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (bb0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}

// ---------------- MULTIVARIATE GAUSSIAN KNWON SPHERICAL VARIANCE -------------

// constructor
inf_MGauss1::inf_MGauss1(std::vector<std::vector<unsigned int>>& partition,
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
                         const double& m_) : y(y_), 
                         hyperpar_alpha(hyperpar_alpha_), 
                         hyperpar_delta(hyperpar_delta_),
                         hyperpar_baseline(hyperpar_baseline_),
                         a0(a0_),
                         b0(b0_),
                         lwr_bound(lwr_bound_),
                         upr_bound(upr_bound_),
                         scale(scale_),
                         m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  D = y_.n_cols;
  alpha = alpha_;
  delta = delta_;
  sigma2 = sigma2_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  
  sigma20 = sigma2* (1.0 + 1.0/lambda0);
  
  // set the statistical unit
  y_i = 0.0;
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  
  mus.reserve(n);
  sigma2s.reserve(n);
  
  // initialize the group sums
  std::vector<arma::vec> sums_y;
  sums_y.reserve(n);
  
  // loop over each observation
  
  arma::vec tmp = arma::zeros<arma::vec>(D);
  // initialize the number of active components
  unsigned int k = 0;
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      sums_y.push_back(tmp);
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    sums_y[k] += y.row(i).t();
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // update the k-th mean
    mus.push_back( (lambda0*mu0 + sums_y[k]) / (lambda0 + ns[k]) );
    
    // update the k-th variance
    sigma2s.push_back( sigma2 * (1.0 + 1.0 / (lambda0 + ns[k])) );
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += -0.5 * D * std::log(lambda0 + ns[k]) + 
      0.5 * D * std::log(lambda0) + 
      0.5 * arma::dot(mus[k],mus[k]) * (lambda0 + ns[k]) / sigma2 - 
      0.5 * arma::dot(mu0,mu0) * lambda0 / sigma2; 
    
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
  
  
}

// update parameters
void inf_MGauss1::update_par(const unsigned int& c_i,
                            double sign){
  
  // update the dimension of the target group
  ns[c_i] += sign;
  
  // update parameters
  
  // mu
  mus[c_i] = ( (lambda0 + ns[c_i] - sign) * mus[c_i] + sign*y_i) / (lambda0 + ns[c_i]);
  
  // sigma2
  sigma2s[c_i] = ( (lambda0 + ns[c_i] - sign) * 
    sigma2s[c_i] + sign*sigma2  ) / ( lambda0 + ns[c_i] );
  
}

// log predictive function
double inf_MGauss1::log_pred(const unsigned int& k){
  
  arma::vec tmp = y_i - mus[k];
  
  return -0.5 * D * std::log(sigma2s[k]) - 
    0.5 * arma::dot(tmp,tmp) / sigma2s[k];
  
}

// log prior predictive
double inf_MGauss1::log_pred_prior(){
  
  arma::vec tmp = y_i - mu0;
  
  return -0.5 * D * std::log(sigma20) - 
    0.5 * arma::dot(tmp,tmp) / sigma20;
  
}

// function that set the statistical unit
void inf_MGauss1::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void inf_MGauss1::save_parameters(const std::string& filename,
                                 const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_MGauss1::copy_atom(const unsigned int& atm1,
                           const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  mus[atm1] = mus[atm2];
  sigma2s[atm1] = sigma2s[atm2];
  
}

// function to delete one atom
void inf_MGauss1::delete_last_atom(){
  
  ns.pop_back();
  mus.pop_back();
  sigma2s.pop_back();
  
}

// funciton to add an atom
void inf_MGauss1::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  mus.push_back( mu0 );
  sigma2s.push_back( sigma20 );
  
}

// function to merge two atoms into one
void inf_MGauss1::merge_atoms(const unsigned int& atm1,
                             const unsigned int& atm2){
  
  // get the new atom index
  unsigned int idx = ns.size()-1;
  
  // merge the sample size
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the means
  mus[idx] = ( (lambda0 + ns[atm1]) * mus[atm1] + 
    (lambda0 + ns[atm2]) * mus[atm2] - mu0*lambda0 ) / 
    (lambda0 + ns[idx]);
  
  // merge the variances
  sigma2s[idx] = ( (lambda0 + ns[atm1]) * sigma2s[atm1] + 
    (lambda0 + ns[atm2]) * sigma2s[atm2] - sigma2 ) / 
    (lambda0 + ns[idx]);
  
}

// function to compute the log marginal likelihood of a given group of data
double inf_MGauss1::log_marginal(const unsigned int& atm){
  
  return -0.5 * D * std::log(lambda0 + ns[atm]) + 0.5 * D * std::log(lambda0) + 
    0.5 * arma::dot(mus[atm],mus[atm]) * (lambda0 + ns[atm]) / sigma2 - 
    0.5 * arma::dot(mu0,mu0) * lambda0 / sigma2; 
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_MGauss1::hyperpar_lfc(const double& x,
                                unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_MGauss1::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_MGauss1::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}

// ------------ MULTIVARIATE GAUSSIAN UNKNOWN DIAGONAL COVARIANCE --------------

// constructor
inf_MGauss2::inf_MGauss2(std::vector<std::vector<unsigned int>>& partition,
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
                         const double& m_) : y(y_), 
                         hyperpar_alpha(hyperpar_alpha_), 
                         hyperpar_delta(hyperpar_delta_),
                         hyperpar_baseline(hyperpar_baseline_),
                         a0(a0_),
                         b0(b0_),
                         lwr_bound(lwr_bound_),
                         upr_bound(upr_bound_),
                         scale(scale_),
                         m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  D = y_.n_cols;
  alpha = alpha_;
  delta = delta_;
  mu0 = mu0_;
  lambda0 = lambda0_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  beta0_vec = arma::ones<arma::vec>(D) * beta0;
  
  prior_term = (std::lgamma(alpha0 + 0.5) - std::lgamma(alpha0)) * D -
    arma::sum( 0.5 * arma::log(2.0 * beta0_vec * (1.0 + 1.0 / lambda0)));
  
  // set the statistical unit
  y_i = arma::zeros<arma::vec>(D);
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  sums_y.reserve(n);
  sums_y2.reserve(n);
  
  mus.reserve(n);
  lambdas.reserve(n);
  alphas.reserve(n);
  betas.reserve(n);
  
  // loop over each observation
  
  zero_vec = arma::zeros<arma::vec>(D);
  arma::vec y_i = arma::zeros<arma::vec>(D);
  // initialize the number of active components
  unsigned int k = 0;
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      sums_y.push_back(zero_vec);
      sums_y2.push_back(zero_vec);
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // get the datum
    y_i = y.row(i).t();
    
    // update the sufficient statistics
    ns[k]++;
    sums_y[k] += y_i;
    sums_y2[k] += y_i % y_i;
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // update the k-th mean parameters
    mus.push_back( (lambda0*mu0 + sums_y[k]) / (lambda0 + ns[k]) );

    lambdas.push_back( lambda0 + ns[k]);
    
    // update the k-th variance parameters
    alphas.push_back(alpha0 + 0.5 * ns[k]);
    
    betas.push_back( beta0 + 0.5 * (mu0 % mu0 * lambda0 + sums_y2[k] - lambdas[k] * mus[k] % mus[k] ));

    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  // constant term of the marginal likelihood
  log_post += 0.5 * D * K_atm * std::log(lambda0) + 
    K_atm * D * (alpha0 * std::log(beta0) - std::lgamma(alpha0));
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += 0.5 * D * std::log(lambdas[k]) + 
      std::lgamma(alphas[k]) * D - 
      alphas[k] * arma::sum(arma::log(betas[k])); 
    
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
  
}

// update parameters
void inf_MGauss2::update_par(const unsigned int& c_i,
                             double sign){
  
  // update the dimension of the target group
  ns[c_i] += sign;
  
  // update the sufficient statistics
  sums_y[c_i] += sign*y_i;
  sums_y2[c_i] += sign* y_i % y_i;
  
  // update parameter
  
  // alpha
  alphas[c_i] += sign*0.5;
  
  // mu
  mus[c_i] = (lambdas[c_i] * mus[c_i] + sign*y_i) / (lambdas[c_i] + sign);
  
  // lambda
  lambdas[c_i] += sign;
  
  // betas
  betas[c_i] = beta0 + 0.5 * (mu0%mu0*lambda0 + sums_y2[c_i] - mus[c_i] % mus[c_i] * lambdas[c_i]);
  
}

// log predictive function
double inf_MGauss2::log_pred(const unsigned int& k){
  
  arma::vec tmp = y_i - mus[k];
  
  return (std::lgamma(alphas[k] + 0.5) - std::lgamma(alphas[k])) * D -
    arma::sum( 0.5 * arma::log(2.0 * betas[k] * (1.0 + 1.0 / lambdas[k])) +
    (alphas[k] + 0.5) * arma::log(1.0 + 0.5 * 
    tmp % tmp / (1.0 + 1.0 / lambdas[k]) / betas[k] ));
  
}

// log prior predictive
double inf_MGauss2::log_pred_prior(){
  
  arma::vec tmp = y_i - mu0;
  
  return prior_term - arma::sum(
      (alpha0 + 0.5) * arma::log(1.0 + 0.5 * 
        tmp % tmp / (1.0 + 1.0 / lambda0) / beta0 ));
  
}

// function that set the statistical unit
void inf_MGauss2::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void inf_MGauss2::save_parameters(const std::string& filename,
                              const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_MGauss2::copy_atom(const unsigned int& atm1,
                            const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  sums_y[atm1] = sums_y[atm2];
  sums_y2[atm1] = sums_y2[atm2];
  
  mus[atm1] = mus[atm2];
  lambdas[atm1] = lambdas[atm2];
  alphas[atm1] = alphas[atm2];
  betas[atm1] = betas[atm2];
  
}

// function to delete one atom
void inf_MGauss2::delete_last_atom(){
  
  ns.pop_back();
  sums_y.pop_back();
  sums_y2.pop_back();
  
  mus.pop_back();
  lambdas.pop_back();
  alphas.pop_back();
  betas.pop_back();
  
}

// funciton to add an atom
void inf_MGauss2::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  sums_y.push_back(zero_vec);
  sums_y2.push_back(zero_vec);
  
  mus.push_back( mu0 );
  lambdas.push_back( lambda0 );
  alphas.push_back( alpha0 );
  betas.push_back( beta0_vec );
  
}

// function to merge two atoms into one
void inf_MGauss2::merge_atoms(const unsigned int& atm1,
                              const unsigned int& atm2){
  
  // get the merged group index
  unsigned int idx = ns.size()-1;
  
  // merge the sample sizes
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the sufficient statistics
  sums_y[idx] = sums_y[atm1] + sums_y[atm2];
  sums_y2[idx] = sums_y2[atm1] + sums_y2[atm2];
  
  // merge the parameters
  
  // alpha
  alphas[idx] = alphas[atm1] + alphas[atm2] - alpha0;
  
  // lambda
  lambdas[idx] = lambdas[atm1] + lambdas[atm2] - lambda0;
  
  // mu
  mus[idx] = (lambdas[atm1] * mus[atm1] + lambdas[atm2] * mus[atm2]  - lambda0 * mu0 ) / lambdas[idx];
  
  // betas
  betas[idx] = beta0 + 0.5 * (mu0%mu0*lambda0 + sums_y2[idx] - mus[idx] % mus[idx] * lambdas[idx]);
  
  
}

// function to compute the log marginal likelihood of a given group of data
double inf_MGauss2::log_marginal(const unsigned int& atm){
  
  return -0.5 * D * std::log(lambdas[atm]) + 
    0.5 * D * std::log(lambda0) - 
    arma::sum(alphas[atm] * arma::log(betas[atm]) - 
    alpha0 * std::log(beta0)) + 
    D * (std::lgamma(alphas[atm]) - std::lgamma(alpha0));
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_MGauss2::hyperpar_lfc(const double& x,
                                 unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_MGauss2::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_MGauss2::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}

// --------------------------------- POISSON -----------------------------------

// constructor
inf_Poiss::inf_Poiss(std::vector<std::vector<unsigned int>>& partition,
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
                     const double& m_) : y(y_), 
                     hyperpar_alpha(hyperpar_alpha_), 
                     hyperpar_delta(hyperpar_delta_),
                     hyperpar_baseline(hyperpar_baseline_),
                     a0(a0_),
                     b0(b0_),
                     lwr_bound(lwr_bound_),
                     upr_bound(upr_bound_),
                     scale(scale_),
                     m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  alpha = alpha_;
  delta = delta_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // set the statistical unit
  y_i = 0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  
  alphas.reserve(n);
  betas.reserve(n);
  
  unsigned int k = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      alphas.push_back(alpha0);
      betas.push_back(beta0);
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    alphas[k] += y(i);
    betas[k]++;
    
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += std::lgamma(alphas[k]) -
      std::lgamma(alpha0) - 
      alphas[k] * std::log(betas[k]) + 
      alpha0 * std::log(beta0);
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
  
  // save recurrent prior term
  prior_term1 = alpha0 * std::log(beta0) - std::lgamma(alpha0);
  prior_term2 = std::log(1.0 + beta0);
  
}

// update parameters
void inf_Poiss::update_par(const unsigned int& c_i,
                           double sign){
  
  // update the dimension of the target group
  ns[c_i] += sign;
  
  // update parameters
  
  // alphas
  alphas[c_i] += sign * y_i;
  
  // betas
  betas[c_i] += sign;
  
}

// log predictive function
double inf_Poiss::log_pred(const unsigned int& k){
  
  return alphas[k] * std::log(betas[k]) -
    (alphas[k] + y_i) * std::log(1.0 + betas[k]) +
    std::lgamma(alphas[k] + y_i) - 
    std::lgamma(alphas[k]);
  
}

// log prior predictive
double inf_Poiss::log_pred_prior(){
  
  return prior_term1 - prior_term2 * (alpha0 + y_i) + 
    std::lgamma(alpha0 + y_i);
  
}

// function that set the statistical unit
void inf_Poiss::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void inf_Poiss::save_parameters(const std::string& filename,
                                const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_Poiss::copy_atom(const unsigned int& atm1,
                          const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  alphas[atm1] = alphas[atm2];
  betas[atm1] = betas[atm2];
  
}

// function to delete one atom
void inf_Poiss::delete_last_atom(){
  
  ns.pop_back();
  alphas.pop_back();
  betas.pop_back();
  
}

// funciton to add an atom
void inf_Poiss::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  alphas.push_back( alpha0 );
  betas.push_back( beta0 );
  
}

// function to merge two atoms into one
void inf_Poiss::merge_atoms(const unsigned int& atm1,
                            const unsigned int& atm2){
  
  // get the merged group index
  unsigned int idx = ns.size()-1;
  
  // merge the sample sizes
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the gamma parameters
  
  // alphas
  alphas[idx] = alphas[atm1] + alphas[atm2] - alpha0;
  
  // betas
  betas[idx] = betas[atm1] + betas[atm2] - beta0;
  
}

// function to compute the log marginal likelihood of a given group of data
double inf_Poiss::log_marginal(const unsigned int& atm){
  
  return prior_term1 + std::lgamma(alphas[atm]) -
    alphas[atm] * std::log(betas[atm]);
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_Poiss::hyperpar_lfc(const double& x,
                                unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_Poiss::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_Poiss::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}

// -------------------------------- BINOMIAL -----------------------------------

// constructor
inf_Binom::inf_Binom(std::vector<std::vector<unsigned int>>& partition,
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
                     const double& m_) : y(y_), n_trials(n_trials_),
                     hyperpar_alpha(hyperpar_alpha_), 
                     hyperpar_delta(hyperpar_delta_),
                     hyperpar_baseline(hyperpar_baseline_),
                     a0(a0_),
                     b0(b0_),
                     lwr_bound(lwr_bound_),
                     upr_bound(upr_bound_),
                     scale(scale_),
                     m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  alpha = alpha_;
  delta = delta_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // set the statistical unit
  y_i = 0;
  n_i = 0;
  
  // get the dimension
  n = y.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  
  alphas.reserve(n);
  betas.reserve(n);
  
  unsigned int k = 0;
  // loop along the configuration vector
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      alphas.push_back(alpha0);
      betas.push_back(beta0);
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    alphas[k] += y(i);
    betas[k] += n_trials(i) - y(i);
    
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += lbeta(alphas[k],betas[k]) - lbeta(alpha0,beta0);
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
  
  // save the recurrent prior term
  prior_term = lbeta(alpha0,beta0);
  
}

// update parameters
void inf_Binom::update_par(const unsigned int& c_i,
                           double sign){
  
  // update the dimension of the target group
  ns[c_i] += sign;
  
  // update parameters
  
  // alphas
  alphas[c_i] += sign * y_i;
  
  // betas
  betas[c_i] += sign * ( n_i - y_i );
  
}

// log predictive function
double inf_Binom::log_pred(const unsigned int& k){
  
  return lbeta(alphas[k] + y_i, betas[k] + n_i - y_i) -
    lbeta(alphas[k],betas[k]);
  
}

// log prior predictive
double inf_Binom::log_pred_prior(){
  
  return lbeta(alpha0 + y_i, beta0 + n_i - y_i) -
    prior_term;
  
}

// function that set the statistical unit
void inf_Binom::set(const unsigned int& i){
  y_i = y(i);
  n_i = n_trials(i);
}

// function to generate the collapsed parameters
// and save them to file
void inf_Binom::save_parameters(const std::string& filename,
                                const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_Binom::copy_atom(const unsigned int& atm1,
                          const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  alphas[atm1] = alphas[atm2];
  betas[atm1] = betas[atm2];
  
}

// function to delete one atom
void inf_Binom::delete_last_atom(){
  
  ns.pop_back();
  alphas.pop_back();
  betas.pop_back();
  
}

// funciton to add an atom
void inf_Binom::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  alphas.push_back( alpha0 );
  betas.push_back( beta0 );
  
}

// function to merge two atoms into one
void inf_Binom::merge_atoms(const unsigned int& atm1,
                            const unsigned int& atm2){
  
  // get the merged group index
  unsigned int idx = ns.size()-1;
  
  // merge the sample sizes
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the beta parameters
  
  // alphas
  alphas[idx] = alphas[atm1] + alphas[atm2] - alpha0;
  
  // betas
  betas[idx] = betas[atm1] + betas[atm2] - beta0;
}

// function to compute the log marginal likelihood of a given group of data
double inf_Binom::log_marginal(const unsigned int& atm){
  
  return lbeta(alphas[atm],betas[atm]) - prior_term;
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_Binom::hyperpar_lfc(const double& x,
                                unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_Binom::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_Binom::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}

// ----------------------------- BERNOULLI PRODUCT -----------------------------

// constructor
inf_MBern::inf_MBern(std::vector<std::vector<unsigned int>>& partition,
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
                     const double& m_) : y(y_), 
                     hyperpar_alpha(hyperpar_alpha_), 
                     hyperpar_delta(hyperpar_delta_),
                     hyperpar_baseline(hyperpar_baseline_),
                     a0(a0_),
                     b0(b0_),
                     lwr_bound(lwr_bound_),
                     upr_bound(upr_bound_),
                     scale(scale_),
                     m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  D = y.n_cols;
  alpha = alpha_;
  delta = delta_;
  alpha0 = alpha0_;
  beta0 = beta0_;
  
  // set the statistical unit
  y_i = arma::zeros<arma::uvec>(D);
  
  // get the dimension
  n = y.n_rows;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  
  n_dk1.reserve(n);
  
  unsigned int k = 0;
  arma::vec tmp = arma::zeros<arma::vec>(D);
  // loop along the configuration vector
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      n_dk1.push_back(tmp);
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    
    // success counter
    for(unsigned int d = 0; d < D; d++){
      
      if(y(i,d) == 1){
        n_dk1[k](d)++;
      }
      
    }
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
    // marginal likelihood term
    log_post += D*std::lgamma(alpha0 + beta0) -
      D*std::lgamma(alpha0+beta0+ns[k]);
    for(unsigned int d = 0; d < D; d++){
      log_post += std::lgamma(alpha0 + n_dk1[k](d)) + 
        std::lgamma(beta0 + ns[k] - n_dk1[k](d)) - 
        std::lgamma(alpha0) - std::lgamma(beta0);
    }
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
  
  // save the recurrent prior terms
  prior_term1 = D * -std::log(alpha0 + beta0);
  prior_term2 = std::log(alpha0);
  prior_term3 = std::log(beta0);
  n_dk10 = arma::zeros<arma::vec>(D);
  
}

// update parameters
void inf_MBern::update_par(const unsigned int& c_i,
                           double sign){
  
  // update the cluster counts
  ns[c_i] += sign;
  
  // update each coordinate
  for(unsigned int d = 0; d < D; d++){
    if(y_i(d) == 1){
      n_dk1[c_i](d) += sign;
    }
  }
  
}

// log predictive function
double inf_MBern::log_pred(const unsigned int& k){
  
  // initialize the output
  double out = D * -std::log(alpha0 + beta0 + ns[k]);
  
  // add the dimension varying part
  for(unsigned int d = 0; d < D; d++){
    if(y_i(d) == 1){
      out += std::log(alpha0 + n_dk1[k](d));
    }else{
      out += std::log(beta0 + ns[k] - n_dk1[k](d));
    }
  }
  
  // return the output
  return out;
  
}

// log prior predictive
double inf_MBern::log_pred_prior(){
  
  // initialize the output
  double out = prior_term1;
  
  // add the dimension varying part
  for(unsigned int d = 0; d < D; d++){
    if(y_i(d) == 1){
      out += prior_term2;
    }else{
      out += prior_term3;
    }
  }
  
  // return the output
  return out;
  
  
}

// function that set the statistical unit
void inf_MBern::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void inf_MBern::save_parameters(const std::string& filename,
                                const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_MBern::copy_atom(const unsigned int& atm1,
                          const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  n_dk1[atm1] = n_dk1[atm2];
  
}

// function to delete one atom
void inf_MBern::delete_last_atom(){
  
  ns.pop_back();
  n_dk1.pop_back();
  
}

// funciton to add an atom
void inf_MBern::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  n_dk1.push_back( n_dk10 );
  
}

// function to merge two atoms into one
void inf_MBern::merge_atoms(const unsigned int& atm1,
                            const unsigned int& atm2){
  
  // get the merged group index
  unsigned int idx = ns.size()-1;
  
  // merge the sample sizes
  ns[idx] = ns[atm1] + ns[atm2];
  
  // merge the two column of sums
  n_dk1[idx] = n_dk1[atm1] + n_dk1[atm2];
  
}

// function to compute the log marginal likelihood of a given group of data
double inf_MBern::log_marginal(const unsigned int& atm){
  
  // initialize the output
  double out = D*std::lgamma(alpha0+beta0) - 
    D*std::lgamma(alpha0+beta0+ns[atm]);
  
  for(unsigned int d = 0; d < D; d++){
    out += std::lgamma(alpha0 + n_dk1[atm](d)) + 
      std::lgamma(beta0 + ns[atm] - n_dk1[atm](d)) - 
      std::lgamma(alpha0) - std::lgamma(beta0);
  }
  
  // return the output
  return out;
  
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_MBern::hyperpar_lfc(const double& x,
                               unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_MBern::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_MBern::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}

// ------------------------------ PARTITION ------------------------------------

// constructor
inf_Partition::inf_Partition(std::vector<std::vector<unsigned int>>& partition,
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
                             const double& m_) : hyperpar_alpha(hyperpar_alpha_), 
                             hyperpar_delta(hyperpar_delta_),
                             hyperpar_baseline(hyperpar_baseline_),
                             a0(a0_),
                             b0(b0_),
                             lwr_bound(lwr_bound_),
                             upr_bound(upr_bound_),
                             scale(scale_),
                             m(m_){
  
  // save the hyperparameters
  K_atm = 0;
  alpha = alpha_;
  delta = delta_;
  
  // get the dimension
  n = c.n_elem;
  
  // initialize the sufficient statistics and parameters
  ns.reserve(n);
  
  // loop over each observation
  
  // initialize the number of active components
  unsigned int k = 0;
  for(unsigned int i = 0; i < n; i++){
    
    // check that the current labels is not already present in the dictionary
    if(!lbl2atm.count(c(i))){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[c(i)] = K_atm;
      
      // increase the number of active components
      K_atm++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0.0);
      
    }
    
    // get the current new label
    k = lbl2atm[c(i)];
    
    // relabel the old one
    c(i) = k;
    
    // update the sufficient statistics
    ns[k]++;
    
    // update the partition
    if(!gibbs){
      partition[k].push_back(i);
    }
    
  }
  
  // clear the dictionary
  lbl2atm.clear();
  
  // compute the parameter for the full conditional
  // and reset the dictionary
  for(k = 0; k < K_atm; k++){
    
    // reset the dictionary
    lbl2atm[k] = k;
    atm2lbl[k] = k;
    
  }
  
  // initialize the vector of possible new labels
  labels.reserve(n);
  for(unsigned int i = n-1; i >= K_atm; i--){
    labels.push_back(i);
  }
  
  // compute the unnormalized log posterior
  
  // EPPF term
  log_post = std::lgamma(alpha) - 
    std::lgamma(alpha + n)- 
    K_atm * std::lgamma(1-delta);
  
  for(k = 0; k < K_atm; k++){
    
    // EPPF term dependent on k
    log_post += std::log(alpha + k*delta) + 
      std::lgamma(ns[k] - delta);
    
  }
  
  // add the prior on the hyperparameters
  if(hyperpar_alpha){
    log_post += std::log(alpha - a0) + 
      std::log(b0 - alpha) - 
      std::log(b0 - a0);
  }
  if(hyperpar_delta){
    log_post += std::log(delta - lwr_bound) + 
      std::log(upr_bound - delta) - 
      std::log(upr_bound - lwr_bound);
  }
}

// update parameters
void inf_Partition::update_par(const unsigned int& c_i,
                               double sign){
  
  // update the dimension of the target group
  ns[c_i] += sign;
  
}

// log predictive function
double inf_Partition::log_pred(const unsigned int& k){
  
  return 0.0;
  
}

// log prior predictive
double inf_Partition::log_pred_prior(){
  
  return 0.0;
  
}

// function that set the statistical unit
void inf_Partition::set(const unsigned int& i){
  
}

// function to generate the collapsed parameters
// and save them to file
void inf_Partition::save_parameters(const std::string& filename,
                                    const unsigned int& chain_id){
  
}

// function to copy one atom into the other
void inf_Partition::copy_atom(const unsigned int& atm1,
                              const unsigned int& atm2){
  
  ns[atm1] = ns[atm2];
  
}

// function to delete one atom
void inf_Partition::delete_last_atom(){
  
  ns.pop_back();
  
}

// funciton to add an atom
void inf_Partition::add_new_atom(){
  
  // add the new sufficient statistics (only the prior term)
  ns.push_back(0.0);
  
}

// function to merge two atoms into one
void inf_Partition::merge_atoms(const unsigned int& atm1,
                                const unsigned int& atm2){
  
  // get the merged group index
  unsigned int idx = ns.size()-1;
  
  // merge the sample sizes
  ns[idx] = ns[atm1] + ns[atm2];
  
}

// function to compute the log marginal likelihood of a given group of data
double inf_Partition::log_marginal(const unsigned int& atm){
  
  return 0.0;
}

// function to compute the log full conditional w.r.t. an hyperparameter
double inf_Partition::hyperpar_lfc(const double& x,
                                   unsigned int which){
  
  if(which == 0){
    
    // concentration hyperparameter (log transformed with truncated uniform prior)
    
    // get alpha from the input
    double alpha = a0 + (b0-a0) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(b0 - a0) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += std::lgamma(alpha) - std::lgamma(alpha + n);
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta);
    }
    
    return out;
    
  }else if(which == 1){
    
    // discount hyperparameter (log transformed with uniform prior)
    
    // get alpha from the input
    double delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-x));
    
    // initialize the output with the prior jacobian contribute
    double out = std::log(upr_bound - lwr_bound) - x - 
      2.0 * std::log(1.0 + std::exp(-x));
    
    // add the posterior term (partition contribute)
    out += -std::lgamma(1.0 - delta) * ns.size();
    for(unsigned int k = 0; k < ns.size(); k++){
      out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
    }
    
    return out;
    
  }else{
    return 0.0;
  }
  
}

// function to update the atoms given a change in the hyperparameters
void inf_Partition::update_atoms(double new_value, unsigned int which){
  
}

// function to update the hyperparameters
void inf_Partition::update_hyperpars(double& log_post2){
  
  if(hyperpar_alpha){
    double tmp = std::log(alpha - a0) - std::log(b0 - alpha);
    tmp = slice_sampler(*this,log_post2,tmp,0,false);
    alpha = a0 + (b0-a0) / (1.0 + std::exp(-tmp));
  }
  
  if(hyperpar_delta){
    double tmp = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
    tmp = slice_sampler(*this,log_post2,tmp,1,false);
    delta = lwr_bound + (upr_bound-lwr_bound) / (1.0 + std::exp(-tmp));  
  }
  
}
