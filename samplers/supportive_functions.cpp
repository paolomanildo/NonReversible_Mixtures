#include <iostream>
#include <RcppArmadillo.h>
#include "supportive_functions.h"

// SUPPORTIVE FUNCTIONS

// FUNCTION TO CREATE THE TEMPORARY DIRECTORY TO SAVE DATA
std::string get_tempdir_cpp(){
  static Rcpp::Function tempdir("tempdir");
  return Rcpp::as<std::string>(tempdir());
}

// FUNCTION THAT CREATES A VECTOR OF INDEXES WITH WHICH TO RETURN A MESSAGE ON THE CONSOLE
arma::uvec sequence(const unsigned int& N, const double& p){
  
  //divide N by p and take the smallest integer
  unsigned int n = std::floor(N * p);
  
  //check that it is admissible
  if(n == 0){
    arma::uvec x = {N+1};
    return x;
  }
  
  //calculate the length of the vector
  unsigned int K = std::floor(N / n);
  
  //check that it is admissible
  if(K == 0){
    arma::uvec x = {N+1};
    return x;
  }
  
  //create a vector of length K
  arma::uvec idx(K+1);
  
  //put the values inside
  for(unsigned int i = 0; i < K; i++){
    idx(i) = n*(i+1);
  }
  
  //return the sequence
  return idx;
}

// LOG SUM EXP  
double log_sum_exp(const double& a, const double& b){
  
  // get the maximum
  double max_val = std::max(a,b);
  
  // return the log sum exp
  return max_val + std::log(std::exp(a-max_val) + std::exp(b - max_val));
}

double log_sum_exp(const arma::vec& x){
  
  // get the maximum
  double max_val = arma::max(x);
  
  // return the log sum exp
  return max_val + std::log(arma::sum(arma::exp(x - max_val)));
  
}

// g-balanced version
double log_sum_exp_g(const arma::vec& x, const unsigned int& g){
  
  // distinguish all the different cases
  if(g == 0){
    // square root
    
    // get the maximum
    double max_val = 0.5 * arma::max(x);
    
    // return the log sum exp
    return max_val + std::log(arma::sum(arma::exp(0.5 * x - max_val)));
    
  }else if(g == 1){
    // Metropolis
    
    double out = 0;
    for(unsigned int i = 0; i<x.n_elem; i++){
      if(x(i) < 0){
        out += std::exp(x(i));
      }else{
        out++;
      }
    }
    return std::log(out);
    
  }else{
    // Barker
    return log_sum_exp(x - arma::log1p(arma::exp(x)));
  }
  
}

// fully informed version
double log_sum_exp_g(const arma::vec& x,
                     const arma::vec& log_n,
                     const unsigned int& g){
  
  // distinguish all the different cases
  if(g == 0){
    // square root
    
    return log_sum_exp(0.5 * x + log_n);
    
  }else if(g == 1){
    // Metropolis
    
    double out = 0;
    for(unsigned int i = 0; i<x.n_elem; i++){
      if(x(i) < 0){
        out += std::exp(x(i) + log_n(i));
      }else{
        out += std::exp(log_n(i));
      }
    }
    return std::log(out);
    
  }else{
    // Barker
    
    arma::vec tmp = x - arma::log1p(arma::exp(x)) + log_n;
    
    return log_sum_exp(x - arma::log1p(arma::exp(x)) + log_n);
  }
  
}

// FUNCTION FOR THE CALLING THE LOG BETA FUNCTION
arma::vec lbeta(const arma::vec& x, const arma::vec& y){
  return arma::lgamma(x) + arma::lgamma(y) - arma::lgamma(x+y);
}

arma::vec lbeta(const double& x, const arma::vec& y){
  return std::lgamma(x) + arma::lgamma(y) - arma::lgamma(x+y);
}

arma::vec lbeta(const arma::vec& x, const double& y){
  return arma::lgamma(x) + std::lgamma(y) - arma::lgamma(x+y);
}

double lbeta(const double& x, const double& y){
  return std::lgamma(x) + std::lgamma(y) - std::lgamma(x+y);
}

arma::mat lbeta_mat(const arma::mat& x, const arma::mat& y){
  return arma::lgamma(x) + arma::lgamma(y) - arma::lgamma(x+y);
}

// FUNCTION TO SIMULATE FROM A DIRICHLET DISTRIBUTION
arma::vec rdirichlet(const arma::vec& x){
  
  // initialize the output
  arma::vec out(x.n_elem);
  
  // sample each value from the corresponding exponential
  for(unsigned int i = 0; i < x.n_elem; i++){
    //out(i) = -std::log(arma::randu()) * x(i);
    out(i) = R::rgamma(x(i),1.0);
  }
  
  // normalize the vector
  out /= arma::sum(out);
  
  // return it
  return out;
}

// FUNCTION FOR THE GUMBEL-MAX TRICK
unsigned int gumbel_max(const arma::vec& log_prob) {
  
  // generate the perturbation of the log probability vector
  arma::vec tmp = log_prob  - arma::log(-arma::log(arma::randu<arma::vec>(log_prob.n_elem)));
  
  // return the maximum index
  return tmp.index_max();
}

// varying length vectors version
unsigned int gumbel_max(const std::vector<double>& log_prob,
                        const unsigned int& K){
  
  // initialize the vector
  arma::vec tmp(K+1);
  for(unsigned int k = 0; k <= K; k++){
    tmp(k) = log_prob[k] - std::log(-std::log(arma::randu()));
  }
  
  // return the maximum index
  return tmp.index_max();
}

// FUNCTION TO SELECT THE DIRECTION OF THE METROPOLIS-HASTINGS SCHEMES

// reversible variant
void set_directions(unsigned int& km,
                    unsigned int& kp,
                    const arma::uvec& c,
                    const unsigned int& K){
  
  // sample the first index uniformly from the data
  km = c(static_cast<unsigned int>(R::runif(0.0,c.n_elem)));
  
  // sample the second uniformly from the set of topic
  kp = static_cast<int>(R::runif(0.0, K-1));
  
  // if k2 >= k1 add one to guarantee a uniform sampling for the second
  if(kp >= km){
    kp++;
  }
  
  // choose a direction randomly
  if(arma::randu() < 0.5){
    
    unsigned int k_tmp = km;
    km = kp;
    kp = k_tmp;
  }
  
}

// non-reversible variant
void set_directions(unsigned int& km,
                    unsigned int& kp,
                    std::uint8_t& vel,
                    const std::vector<std::uint8_t>& dir,
                    unsigned int& idx_dir,
                    const unsigned int& n,
                    const arma::uvec& c,
                    const unsigned int& K){
  
  // define the two indexes
  unsigned int k1,k2;
  
  // sample the first index
  k1 = c(static_cast<unsigned int>(R::runif(0.0,n)));
  
  // sample the second uniformly 
  k2 = static_cast<unsigned int>(R::runif(0.0, K-1));
  
  // if k2 >= k1 add one to guarantee a uniform sampling for the second
  if(k2 >= k1){
    k2++;
  }
  
  // set the velocity
  if(k1 > k2){
    
    // get the index inside the unrolled lower triangular matrix
    idx_dir = k2 * (2*K - k2 - 1) / 2 + (k1 - k2 - 1);
    
    // get the corresponding entry
    vel = dir[idx_dir];
    
    // reorder the velocities
    if(vel){
      kp = k1;
      km = k2;
    }else{
      km = k1;
      kp = k2;
    }
    
  }else{
    
    // get the index inside the unrolled lower triangular matrix
    idx_dir = k1 * (2*K - k1 - 1) / 2 + (k2 - k1 - 1);
    
    // get the corresponding entry
    vel = dir[idx_dir];
    
    // reorder the velocities
    if(vel){
      kp = k2;
      km = k1;
    }else{
      km = k2;
      kp = k1;
    }
  }
  
}

// infinite mixture variant (reversible)
bool set_directions(unsigned int& old_lbl,
                    unsigned int& km,
                    unsigned int& kp,
                    const arma::uvec& c,
                    const unsigned int& K,
                    const std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                    const std::unordered_map<unsigned int, unsigned int>& atm2lbl){
  
  // compute the probability for the lazy move
  double prob = 1.0 / (K + 1.0) / K;
  
  // adjust for the case K = n
  if(K == c.n_elem){
    prob += K / (K+1.0) / K; 
  }
  
  // lazy move?
  if(arma::randu() <= prob){
    return true;
  }
  
  // sample the first lbl uniformly from the data
  old_lbl = c(static_cast<unsigned int>(R::runif(0.0,c.n_elem)));
  
  // get the corresponding atom
  km = lbl2atm.at(old_lbl);
  
  // new atom?
  if(arma::randu() <= 1/(1-prob)/(K+1.0) && K < c.n_elem){
    kp = km;
  }else{
    // sample the second uniformly from the set of topic
    kp = static_cast<int>(R::runif(0.0, K-1));
    
    // if k2 >= k1 add one to guarantee a uniform sampling for the second
    if(kp >= km){
      kp++;
    }
  }
  
  // choose a direction randomly
  if(arma::randu() < 0.5){
    
    // create a new group only half of the times
    if(km == kp){
      return true;
    }
    
    unsigned int k_tmp = km;
    km = kp;
    kp = k_tmp;
    old_lbl = atm2lbl.at(km);
    
  }
  
  return false;
  
}

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
                    const std::unordered_map<unsigned int, unsigned int>& atm2lbl){
  
  // define the two indexes
  unsigned int lbl1,lbl2,k_tmp;
  
  // compute the probability for the lazy move
  double prob = 1.0 / (K + 1.0) / K;
  
  // adjust for the case K = n
  if(K == c.n_elem){
    prob += K / (K+1.0) / K; 
  }
  
  // lazy move?
  if(arma::randu() <= prob){
    return true;
  }
  
  // sample the first lbl uniformly from the data
  old_lbl = c(static_cast<unsigned int>(R::runif(0.0,c.n_elem)));
  
  // get the corresponding atom
  km = lbl2atm.at(old_lbl);
  
  // new atom?
  if(arma::randu() <= 1/(1-prob)/(K+1.0) && K < c.n_elem){
    kp = km;
    
    // create one group only half of the time
    if(arma::randu() < 0.5){
      return true;
    }
    
  }else{
    // sample the second uniformly from the set of topic
    kp = static_cast<int>(R::runif(0.0, K-1));
    
    // if k2 >= k1 add one to guarantee a uniform sampling for the second
    if(kp >= km){
      kp++;
    }
    
    // SET THE ORIENTATION
    
    // get the labels
    lbl1 = old_lbl;
    lbl2 = atm2lbl.at(kp);
    
    // distinguish the two cases
    if(lbl1 > lbl2){
      
      // get the index inside the unrolled lower triangular matrix
      idx_dir = lbl1*(lbl1-1)/2 + lbl2;
      
      // get the corresponding entry
      vel = dir[idx_dir];
      
      // re order the velocities
      if(vel){
        k_tmp = kp;
        kp = km;
        km = k_tmp;
        old_lbl = atm2lbl.at(km);
      }
      
    }else{
      
      // get the index inside the unrolled lower triangular matrix
      idx_dir = lbl2*(lbl2-1)/2 + lbl1;
      
      // get the corresponding entry
      vel = dir[idx_dir];
      
      // re order the velocities
      if(vel == 0){
        k_tmp = km;
        km = kp;
        kp = k_tmp;
        old_lbl = atm2lbl.at(km);
      }
      
    }
    
  }
  
  return false;
  
}

// function that sample the new directional variables
void set_directions(std::vector<std::uint8_t>& dir,
                    std::uint8_t& vel,
                    unsigned int& idx_dir,
                    const unsigned int& old_lbl,
                    const unsigned int& new_lbl,
                    const unsigned int& K,
                    const std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                    const std::unordered_map<unsigned int, unsigned int>& atm2lbl){
  
  // loop over all the current atom labels
  unsigned int lbl = 0;
  unsigned int start = new_lbl*(new_lbl-1)/2;
  for(unsigned int atm = 0; atm < K; atm++){
    
    // get the current label
    lbl = atm2lbl.at(atm);
    
    // if this is equal to the old label
    if(lbl == old_lbl){
      
      // set the velocity
      if(old_lbl < new_lbl){
        
        // get the index of the directional variable
        idx_dir = new_lbl*(new_lbl-1)/2 + old_lbl;
        
        // set it to 1
        dir[idx_dir] = 1;
        
        // flip the velocity w.r.t. this orientation
        vel = 0;
        
      }else{
        
        // get the index of the directional variable
        idx_dir = old_lbl*(old_lbl-1)/2 + new_lbl;
        
        // set it to 1
        dir[idx_dir] = 0;
        
        // flip the velocity w.r.t. this orientation
        vel = 1;
      }
      
      
    }else{
      // sample a direction at random
      if(lbl < new_lbl){
        
        dir[start + lbl] = (R::runif(0,1) < 0.5) ? 1 : 0;
        
      }else if(lbl > new_lbl){
        
        dir[lbl*(lbl-1)/2 + new_lbl] = (R::runif(0,1) < 0.5) ? 1 : 0;
        
      }
      
    }
    
  }
  
}

// function that reshuffle at most m element of an element of the partition
void reshuffle_partition(unsigned int* c_k,
                         unsigned int& n_k,
                         const double& n_c_k,
                         const unsigned int& m){
  
  // get the effective cardinality of the informed subsets
  n_k = std::min(m,static_cast<unsigned int>( n_c_k ));
  
  // reorder the terms in the partition as a resampling just happend
  unsigned int tmp, r;
  if(n_k > 0){
    
    for(unsigned int j = 0; j <  n_k; j++){
      
      // sample an index among the available ones
      r = j + static_cast<unsigned int>( (  n_c_k - j ) * arma::randu() );
      
      // swap this index with the j-th one
      tmp = c_k[j];
      
      c_k[j] = c_k[r];
      
      c_k[r] = tmp;
      
    }
    
  }
  
}

// function that reshuffle an entire partition group
void shuffle_group(unsigned int* c_k,
                   const unsigned int& n_k,
                   const unsigned int& i,
                   const unsigned int& j){
  
  // reorder the terms in the partition as a resampling just happend
  unsigned int tmp, r;
  
  for(unsigned int k = 0; k < n_k; k++){
    
    // sample a new index among the available ones
    r = k + static_cast<unsigned int>( (  n_k - k ) * arma::randu() );
    
    // get the old index
    tmp = c_k[k];

    // swap it with the new one
    c_k[k] = c_k[r];
    
    c_k[r] = tmp;
    
    // check that the c_k is not equal to either i or j
    tmp = c_k[k];
    
    if(tmp == i){
      // swap it with the first one
      c_k[k] = c_k[0];
      c_k[0] = tmp;
    }
    
    if(tmp == j){
      // swap it with the second one
      c_k[k] = c_k[1];
      c_k[1] = tmp;
    }
    
  }
  
}

// // function to perturb the momentum
// void perturb_momentum(std::vector<std::uint8_t>& dir,
//                       const double& theta){
//   
//   // with probability theta
//   if(arma::randu() < theta){
//     
//     // select one directional entry
//     unsigned int idx = static_cast<unsigned int>(R::runif(0.0,dir.size()));
//     
//     // flip its sign
//     dir[idx] ^= 1;
//     
//   }
//   
// }

// function to perturb the momentum
void perturb_momentum(std::vector<std::uint8_t>& dir,
                      const double& theta,
                      const unsigned int& K){
  
  // with probability theta
  if(arma::randu() < theta){
    
    // sample all known directions
    for(unsigned int k = 0; k < K*(K-1)/2; k++){
      dir[k] = arma::randu() < 0.5 ? 0 : 1;
    }
    
  }
  
}

// FUNCTIONS FOR THE MULTIVARIATE GAUSSIAN KERNEL

// function that compute the determinant of a matrix given 
// its cholesky factor
double half_log_det_chol(const arma::vec& chol_Bm1,
                         const unsigned int& d){
  
  // initialize the output
  double out = 0;
  
  // initialize the diagonal index
  unsigned int idx = 0;
  
  // loop over each diagonal elements of the matrix in vector format
  for(unsigned int i = 0; i < d; i++){
    
    // update the squared rooted log determinant
    out += std::log(chol_Bm1(idx));
    
    // update the diagonal idx
    idx += d - i;
  }
  
  // return the half log square
  return out;
}

// function that compute Rx, with R the right cholesky factor
// but always using the left unrolled version
arma::vec prod_Rchol_x(const arma::vec& x,
                       const arma::vec& chol_Bm1,
                       const arma::umat& idx_map,
                       const unsigned int& d){
  
  // create a temporary vector
  arma::vec out = arma::zeros<arma::vec>(d);
  
  // fill its entries
  for(unsigned int i = 0; i < d; i++){
    
    for(unsigned int j = i; j < d; j++){
      
      out(i) += chol_Bm1(idx_map(j,i)) * x(j);
      
    }
    
  }
  
  // return the vector
  return out;
  
}

// function that compute Lx, with K the left cholesky factor
// but always using the unrolled version
arma::vec prod_Lchol_x(const arma::vec& x,
                       const arma::vec& chol_Bm1,
                       const arma::umat& idx_map,
                       const unsigned int& d){
  
  // create a temporary vector
  arma::vec out = arma::zeros<arma::vec>(d);
  
  // fill its entries
  for(unsigned int i = 0; i < d; i++){
    
    for(unsigned int j = 0; j <= i; j++){
      
      // update the product
      out(i) += chol_Bm1(idx_map(i,j)) * x(j);
      
    }
    
  }
  
  // return the vector
  return out;
  
}

// function that update a cholesky vector
arma::vec chol_update(arma::vec chol_Bm1s,
                      arma::vec v,
                      const arma::umat& idx_map,
                      const bool& add){
  
  // get the dimension of the matrix
  unsigned int d = v.n_elem;
  
  for(unsigned int i = 0; i < d; i++) {
    
    double Rii = chol_Bm1s(idx_map(i,i));
    double vi  = v(i);
    
    double r;
    
    if(!add){
      double tmp = Rii*Rii - vi*vi;
      if(tmp <= 0)
        Rcpp::stop("Downdate would destroy positive definiteness.");
      r = std::sqrt(tmp);
    } else {
      r = std::sqrt(Rii*Rii + vi*vi);
    }
    
    double c = r / Rii;
    double s = vi / Rii;
    
    chol_Bm1s(idx_map(i,i)) = r;
    
    if(i < d-1){
      
      arma::vec Rrow = chol_Bm1s.subvec(idx_map(i,i+1),idx_map(d-1,i)); 
      arma::vec vsub   = v.subvec(i+1, d-1);
      
      if(!add)
        Rrow = (Rrow - s * vsub) / c;
      else
        Rrow = (Rrow + s * vsub) / c;
      
      vsub = c * vsub - s * Rrow;
      
      chol_Bm1s.subvec(idx_map(i,i+1),idx_map(d-1,i)) = Rrow;
      
      v.subvec(i+1, d-1) = vsub;
    }
  }
  
  return chol_Bm1s;
  
}

// function to compute the inverse of the outer cholesky product
arma::mat inv_outer_chol(const arma::vec& chol_Bm1,
                         const arma::umat& idx_map,
                         const unsigned int& D){
  
  // initialize the output matrix
  arma::mat out = arma::zeros<arma::mat>(D,D);
  
  // de-unrolled the cholesky
  for(unsigned int i = 0; i < D; i++){
    for(unsigned int j = 0; j <= i; j++){
      
      out(j,i) = chol_Bm1(idx_map(i,j));
      
    }
  }
  
  // compute the inverse of the cholesky
  out = arma::inv(arma::trimatu(out));
  
  // return the inverse ot the outer product
  return out * out.t();
}

// // concentration hyperparameter full conditional
// // under gamma prior
// double log_fc_alpha(const double& omega,
//                     const arma::vec& ns,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double& a,
//                     const double& b){
//   
//   // get alpha from omega = log alpha
//   double alpha = std::exp(omega);
//   
//   return a*omega - 
//     b*alpha + 
//     std::lgamma(alpha * K) + 
//     arma::sum(arma::lgamma(alpha + ns)) - 
//     K*std::lgamma(alpha) - 
//     std::lgamma(alpha*K + n);
//   
// }

// // concentration hyperparameter full conditional
// // under Unif(0,15) prior
// double log_fc_alpha(const double& omega,
//                     const double& delta,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double& lwr_bound,
//                     const double& upr_bound){
//   
//   // get alpha from omega
//   double alpha = lwr_bound + upr_bound / (1.0 + std::exp(-omega));
//   
//   // initialize the output with the prior jacobian contribute
//   double out = std::log(upr_bound - lwr_bound) - omega - 
//     2.0 * std::log(1.0 + std::exp(-omega));
//   
//   // add the posterior term (partition contribute)
//   out += std::lgamma(alpha) - std::lgamma(alpha + n);
//   for(unsigned int k = 0; k < K; k++){
//     out += std::log(alpha + k*delta);
//   }
//   
//   return out;
// }
// 
// 
// // discount hyperparameter full conditional under Unif(0,1) prior
// double log_fc_delta(const double& omega,
//                     const double& alpha,
//                     const std::vector<double>& ns,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double& lwr_bound,
//                     const double& upr_bound){
//   
//   // get alpha from omega
//   double delta = lwr_bound + upr_bound / (1.0 + std::exp(-omega));
//   
//   // initialize the output with the prior jacobian contribute
//   double out = std::log(upr_bound - lwr_bound) - omega - 
//     2.0 * std::log(1.0 + std::exp(-omega));
//   
//   // add the posterior term (partition contribute)
//   out += -std::lgamma(1.0 - delta) * K;
//   for(unsigned int k = 0; k < K; k++){
//     out += std::log(alpha + k*delta) + std::lgamma(ns[k] - delta);
//   }
//   
//   return out;
// }
// 
// // generic slice sampler for infinite mixtures
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
//                      const unsigned int& m){
//   
//   // define omega the current full conditional and the density height
//   double omega = 0.0, log_pi = 0.0, log_y = 0.0;
//   if(concentration){
//     omega = std::log(alpha - lwr_bound) - std::log(upr_bound - alpha);
//     
//     log_pi = log_fc_alpha(omega,delta,n,K,lwr_bound,upr_bound);
//     
//     log_y = log_pi + std::log(arma::randu());
//     
//   }else{
//     omega = std::log(delta - lwr_bound) - std::log(upr_bound - delta);
//     
//     log_pi = log_fc_delta(omega,alpha,ns,n,K,lwr_bound,upr_bound);
//     
//     log_y = log_pi + std::log(arma::randu());
//   }
//   
//   // create the interval
//   double L = omega - scale*arma::randu();
//   double R = L + scale;
//   
//   // expand it on the left
//   unsigned int Lcount = static_cast<unsigned int>(R::runif(0.0,m));
//   unsigned int Rcount = m - 1 - Lcount;
//   bool exit = false;
//   while(!exit && Lcount > 0){
//     
//     // decrease the lower end
//     L -= scale;
//     
//     // check if it is out of the slice
//     if(concentration){
//       if(log_y >= log_fc_alpha(L,delta,n,K,lwr_bound,upr_bound)){
//         exit = true;
//       }
//     }else{
//       if(log_y >= log_fc_delta(L,alpha,ns,n,K,lwr_bound,upr_bound)){
//         exit = true;
//       }
//     }
//     
//     // decrease the counter of lower end iterations
//     Lcount--;
//     
//   }
//   
//   // expand it on the right
//   exit = false;
//   while(!exit && Rcount > 0){
//     
//     // increase the upper end
//     R += scale;
//     
//     // check if it is out of the slice
//     if(concentration){
//       if(log_y >= log_fc_alpha(R,delta,n,K,lwr_bound,upr_bound)){
//         exit = true;
//       }
//     }else{
//       if(log_y >= log_fc_delta(R,alpha,ns,n,K,lwr_bound,upr_bound)){
//         exit = true;
//       }
//     }
//     
//     // decrease the counter of upper end iterations
//     Rcount--;
//     
//   }
//   
//   // sample from the shrinking interval
//   exit = false;
//   double tmp = omega;
//   double log_pi_prime = log_pi;
//   while(!exit){
//     
//     // sample uniformly from the interval a proposal
//     tmp = L + (R-L)*arma::randu();
//     
//     // check if it's outside the slice
//     if(concentration){
//       
//       log_pi_prime = log_fc_alpha(tmp,delta,n,K,lwr_bound,upr_bound);
//       
//       if(log_y >= log_pi_prime){
//         // if it is outside, reduce the interval
//         if(tmp < omega){
//           L = tmp;
//         }else{
//           R = tmp;
//         }
//       }else{
//         // otherwise
//         exit = true;
//       }
//       
//     }else{
//       
//       log_pi_prime = log_fc_delta(tmp,alpha,ns,n,K,lwr_bound,upr_bound);
//       if(log_y >= log_pi_prime){
//         // if it is outside, reduce the interval
//         if(tmp < omega){
//           L = tmp;
//         }else{
//           R = tmp;
//         }
//       }else{
//         // otherwise
//         exit = true;
//       }
//     }
//     
//   }
//   
//   // update the log posterior by subtracting the
//   // the old log full conditional and adding the new one
//   log_post += log_pi_prime - log_pi;
//   
//   // return the last proposed value
//   return lwr_bound + (upr_bound - lwr_bound) * 1.0 / (1.0 + std::exp(-tmp));
//   
// }
// 
// // slice sampler for finite mixtures
// 
// // full conditional
// double log_fc_alpha(const double& omega,
//                     const arma::vec& ns,
//                     const unsigned int& n,
//                     const unsigned int& K,
//                     const double a0,
//                     const double b0){
//   
//   // get alpha from omega
//   double alpha = std::exp(omega);
//   
//   // return the full conditional
//   return a0 * omega - b0 * alpha + 
//     std::lgamma(alpha*K) - 
//     K*std::lgamma(alpha) - 
//     std::lgamma(alpha*K + n) + 
//     arma::sum(arma::lgamma(alpha + ns));
//   
// }
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
//                      const unsigned int& m){
//   
//   // define omega
//   double omega = std::log(alpha);
//   
//   // get the current full conditional
//   double log_pi = log_fc_alpha(omega,ns,n,K,a0,b0);
//     
//   // define the density height
//   double log_y = log_pi + std::log(arma::randu());
//   
//   // create the interval
//   double L = omega - scale*arma::randu();
//   double R = L + scale;
//   
//   // expand it on the left
//   unsigned int Lcount = static_cast<unsigned int>(R::runif(0.0,m));
//   unsigned int Rcount = m - 1 - Lcount;
//   bool exit = false;
//   while(!exit && Lcount > 0){
//     
//     // decrease the lower end
//     L -= scale;
//     
//     // check if it is out of the slice
//     if(log_y >= log_fc_alpha(L,ns,n,K,a0,b0)){
//       exit = true;
//     }
//     
//     // decrease the counter of lower end iterations
//     Lcount--;
//     
//   }
//   
//   // expand it on the right
//   exit = false;
//   while(!exit && Rcount > 0){
//     
//     // increase the upper end
//     R += scale;
//     
//     // check if it is out of the slice
//     if(log_y >= log_fc_alpha(R,ns,n,K,a0,b0)){
//       exit = true;
//     }
//     
//     // decrease the counter of upper end iterations
//     Rcount--;
//     
//   }
//   
//   // sample from the shrinking interval
//   exit = false;
//   double tmp = omega;
//   double log_pi_prime = log_pi;
//   while(!exit){
//     
//     // sample uniformly from the interval a proposal
//     tmp = L + (R-L)*arma::randu();
//     
//     log_pi_prime = log_fc_alpha(tmp,ns,n,K,a0,b0);
//     
//     // check if it's outside the slice
//     if(log_y >= log_pi_prime){
//       // if it is outside, reduce the interval
//       if(tmp < omega){
//         L = tmp;
//       }else{
//         R = tmp;
//       }
//     }else{
//       // otherwise
//       exit = true;
//     }
//     
//     
//   }
//   
//   // update the log posterior by subtracting the
//   // the old log full conditional and adding the new one
//   log_post += log_pi_prime - log_pi;
//   
//   // return the last proposed value
//   return std::exp(tmp);
//   
// }

// slice sampler
// double slice_sampler_alpha(const double& alpha,
//                            const arma::vec& ns,
//                            const unsigned int& n,
//                            const unsigned int& K,
//                            const double& a,
//                            const double& b,
//                            const double& scale,
//                            const unsigned int& m){
//   
//   // define omega = log alpha
//   double omega = std::log(alpha);
//   
//   // sample the log density height
//   double log_y = log_fc_alpha(omega,ns,n,K,a,b) + 
//                       std::log(arma::randu());
//   
//   // create the interval
//   double L = omega - scale*arma::randu();
//   double R = L + scale;
//   
//   // expand it on the left
//   unsigned int Lcount = static_cast<unsigned int>(R::runif(0.0,m));
//   unsigned int Rcount = m - 1 - Lcount;
//   bool exit = false;
//   while(!exit && Lcount > 0){
//     
//     // decrease the lower end
//     L -= scale;
//     
//     // check if it is out of the slice
//     if(log_y >= log_fc_alpha(L,ns,n,K,a,b)){
//       exit = true;
//     }
//     
//     // decrease the counter of lower end iterations
//     Lcount--;
//     
//   }
//   
//   // expand it on the right
//   exit = false;
//   while(!exit && Rcount > 0){
//     
//     // increase the upper end
//     R += scale;
//     
//     // check if it is out of the slice
//     if(log_y >= log_fc_alpha(R,ns,n,K,a,b)){
//       exit = true;
//     }
//     
//     // decrease the counter of upper end iterations
//     Rcount--;
//     
//   }
//   
//   // sample from the shrinking interval
//   exit = false;
//   double tmp = omega;
//   while(!exit){
//     
//     // sample uniformly from the interval a proposal
//     tmp = L + (R-L)*arma::randu();
//     
//     // check if it's outside the slice
//     if(log_y >= log_fc_alpha(tmp,ns,n,K,a,b)){
//       
//       // if it is outside, reduce the interval
//       if(tmp < omega){
//         L = tmp;
//       }else{
//         R = tmp;
//       }
//     }else{
//       
//       // otherwise
//       exit = true;
//     }
//   }
//   
//   // return the last proposed value
//   return std::exp(tmp);
//   
// }

// function that convert a bit string into a decimal number
unsigned int binaryToDecimal(const arma::uvec& y){
  
  // initilialize the output
  unsigned int out = 0;
  
  // loop over the elements
  for(unsigned int i = 0; i < y.n_elem; i++){
    
    // check if the i-th power of two is present in the vector
    out = (out << 1) | y(i);
  }
  
  return out;
}

// inverse transform
arma::uvec decimalToUvec(const unsigned int& sigma,
                         const unsigned int& D){
  
  // initialize the output
  arma::uvec out(D, arma::fill::zeros);
  
  // loop over each coordinate
  for(unsigned int d = 0; d < D; d++){
    
    // set the elements from top
    out(D - 1 - d) = (sigma >> d) & 1;
  }
  
  // return the output
  return out;
}

// FUNCTION THAT PRINT THE SAMPLER'S SPECIFICS
std::string describe_sampler(const std::string& kernel,
                             const unsigned int& K,
                             const unsigned int& N,
                             const bool& gibbs,
                             const bool& reversible,
                             const bool& informed,
                             const unsigned int& m){
  
  // initialize the output
  std::ostringstream out;
  if(K == 0){
    
    out << kernel << "infinite-mixture model, "
        << N << " draws obtained using a ";
  }else{
    
    out << K << "-components " << kernel << " mixture model, "
        << N << " draws obtained using a ";
  }
  
  // check the different sampler type
  if(gibbs){
    if(informed){
      out << "informed gibbs sampler with neighborhood size " << m;
    }else{
      out << "gibbs sampler";
    }
  }else if(reversible){
    if(informed){
      out << "informed reversible Metropolis-Hastings sampler with neighborhood size " << m;
    }else{
      out << "reversible Metropolis-Hastings sampler";
    }
  }else{
    if(informed){
      out << "informed non-reversible Metropolis-Hastings sampler with neighborhood size " << m;
    }else{
      out << "non-reversible Metropolis-Hastings sampler";
    }
  }
  
  return out.str();
}

// function for creating an aliasing table
void create_aliasing_table(arma::vec& log_prob,
                           arma::uvec& alias){
  
  // get the space dimension
  int D = log_prob.n_elem;
  
  // normalize the probability vector
  log_prob = arma::exp(log_prob - log_sum_exp(log_prob));
  
  // define the scaled probability vector
  arma::vec scaled = log_prob * D;
  
  // define the vectors of large and small scaled probabilities
  std::vector<int> small, large;
  
  // loop over the scaled probabilities and classify them as either small or large
  for (int i = 0; i < D; i++) {
    if (scaled(i) < 1.0) small.push_back(i);
    else large.push_back(i);
  }
  
  // loop over them until there are no more small or large scaled probabilities left
  while (!small.empty() && !large.empty()) {
    
    // get the last small and large scaled probabilities
    int s = small.back(); small.pop_back();
    int l = large.back(); large.pop_back();
    
    // set the small probability to the scaled one
    log_prob(s) = scaled(s);
    
    // set the alias to the large one
    alias(s) = l;
    
    // subtract to the large scaled probability the reminder of the small one
    scaled(l) -= (1.0 - scaled(s));
    
    // if the new scaled probability has becomed small
    if (scaled(l) < 1.0){
      // add it to the small list
      small.push_back(l);
    }else{
      // otherwise reput it in the large list
      large.push_back(l);
    }
  }
  
  // set to one the probability of the remaining large and small sets
  for (int i : large) log_prob(i) = 1.0;
  for (int i : small) log_prob(i) = 1.0;
  
}

// function to sample from a fixed probability distribution through aliasing table
unsigned int sample_alias(const arma::vec& prob,
                          const arma::uvec& alias){
  
  // sample an index at random
  unsigned int idx = static_cast<unsigned int>(R::runif(0, prob.n_elem));
  
  // if a random number is smaller than it's associated probability
  // return that index, otherwise its alias 
  return (R::runif(0,1) < prob(idx)) ? idx : alias(idx);
  
}

// URN SAMPLERS

// finite case
arma::uvec urn(const unsigned int& n,
               const arma::vec& alpha){
  
  // initialize the sample sizes
  std::vector<unsigned int> ns;
  ns.reserve(n);
  
  // get the number of current clusters
  unsigned int K_star = 0;
  
  // initialize the map from labels to atoms
  std::unordered_map<unsigned int,unsigned int> lbl2atm;
  
  // sample from the unnormalized Dirichlet distribution
  arma::vec log_pi(alpha.n_elem);
  for(unsigned int k = 0; k < alpha.n_elem; k++){
    log_pi(k) = std::log(R::rgamma(alpha(k),1));
  }
  
  // sample the cardinalities
  unsigned int k = 0;
  for(unsigned int i = 0; i < n; i++){
    
    // simulate one component
    k = gumbel_max(log_pi);
    
    // check if the atom is already exists
    if(!lbl2atm.count(k)){
      
      // if the current label is not present, append it to the dictionary
      lbl2atm[k] = K_star;
      
      // increase the number of active components
      K_star++;
      
      // append the new values to the sufficient statistics vectors
      ns.push_back(0);
    }
    
    // update the counts
    ns[lbl2atm[k]]++;
    
  }
  
  // return the numerosity vector
  arma::uvec out(K_star);
  for(unsigned int k = 0; k < K_star; k++){
    out(k) = ns[k];
  }
  return out;
  
}

// infinite case
arma::uvec urn(const unsigned int& n,
               const double& alpha,
               const double& delta){
  
  // initialize the urn with only one observation in the first group
  std::vector<unsigned int> ns;
  ns.reserve(n);
  ns.push_back(1);
  
  // initialize the number of groups at one
  unsigned int K = 1;
  
  // initialize the urn weights
  std::vector<double> log_prob;
  log_prob.reserve(n);
  log_prob.push_back(std::log(1.0 - delta));
  log_prob.push_back(std::log(alpha + K*delta));
  
  // initialize the i-th label
  unsigned int c_i = 0;
  for(unsigned int i = 1; i < n; i++){
    
    // sample an index with probability proportional 
    // to existing groups size
    c_i = gumbel_max(log_prob,K);
    
    // do we have need a new group?
    if(c_i == K){
      // yes
      
      // increase the number of groups
      K++;
      ns.push_back(1);
      // update the urn weights
      log_prob[K-1] = std::log(1.0 - delta);
      log_prob.push_back(std::log(alpha + K*delta));
      
    }else{
      
      // increase the other existing count
      ns[c_i]++;
      
      // and the corresponding probability
      log_prob[c_i] = std::log(ns[c_i] - delta);
      
    }
  }
  
  // return the groups size
  arma::uvec out(K);
  for(unsigned int k = 0; k < K; k++){
    out(k) = ns[k];
  }
  return out;
  
}



