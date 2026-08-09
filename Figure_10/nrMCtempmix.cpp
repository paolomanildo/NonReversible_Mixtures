#include <iostream>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

// FUNCTION TO CREATE THE TEMPORARY DIRECTORY TO SAVE DATA
std::string get_tempdir_cpp(){
  static Rcpp::Function tempdir("tempdir");
  return Rcpp::as<std::string>(tempdir());
}

// function that return the permutation index of a vector
unsigned int permutation_rank(const arma::uvec& sigma){
  
  // get the numer of elements
  unsigned int K = sigma.n_elem;
  
  // compute all the possible factorials
  arma::uvec fact = arma::ones<arma::uvec>(K + 1);
  for(unsigned int i = 1; i <= K; i++){
    fact(i) = fact(i - 1) * i;
  }
  
  // initailize the output
  unsigned int idx = 0;
  std::vector<bool> used(K, false);
  
  // loop over each index
  for(unsigned int i = 0; i < K; i++) {
    unsigned int smaller_unused = 0;
    for(unsigned int j = 0; j < sigma(i); j++)
      if(!used[j]){
        smaller_unused++;
      }
      
      // update the index
      idx += smaller_unused * fact[K - 1 - i];
      used[sigma(i)] = true;
  }
  
  // return the index
  return idx;
}

// function that compute the prior contribute of a configuration vector
double prior_contribute(const arma::vec& ns,
                        const arma::vec& alpha){
  
  return arma::sum(arma::lgamma(alpha + ns));
  
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
  double inv_temp;
  
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
         const double& m_,
         const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
         const double& m_,
         const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
         const double& m_,
         const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
          const double& m_,
          const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
          const double& m_,
          const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
        const double& m_,
        const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
        const double& m_,
        const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
        const double& m_,
        const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
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
  double inv_temp;
  
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
            const double& m_,
            const double& inv_temp_);
  
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
                       const unsigned int& chain_id,
                       const unsigned int& temperature);
  
  // function to compute the log full conditional w.r.t. an hyperparameter
  double hyperpar_lfc(const double& x,
                      unsigned int which);
  
  // function to update the atoms given a change in the hyperparameters
  void update_atoms(double new_value, unsigned int which);
  
  // function to update the hyperparameters
  void update_hyperpars(double& log_post);
  
  // function that compute the unpowered log likelihood
  double log_lik();
  
  // function that return the permutation index of the current state
  unsigned int permutation();
  
};

// TEMPLATE FOR SLICE SAMPLING OF THE HYPERPARAMETERS
template <typename Model>
double slice_sampler(Model& model,
                     double& log_post,
                     const double& parameter,
                     unsigned int which,
                     bool log_transform){
  
  // initialize the algorithm
  double tmp;
  if(log_transform){
    tmp = std::log(parameter);
  }else{
    tmp = parameter;
  }
  
  // get the current full conditional
  double log_pi = model.hyperpar_lfc(tmp,which);
  
  // define the density height
  double log_y = log_pi + std::log(arma::randu());
  
  // create the interval
  double L = tmp - model.scale*arma::randu();
  double R = L + model.scale;
  
  // expand it on the left
  unsigned int Lcount = static_cast<unsigned int>(R::runif(0.0,model.m));
  unsigned int Rcount = model.m - 1 - Lcount;
  bool exit = false;
  while(!exit && Lcount > 0){
    
    // decrease the lower end
    L -= model.scale;
    
    // check if it is out of the slice
    if(log_y >= model.hyperpar_lfc(L,which)){
      exit = true;
    }
    
    // decrease the counter of lower end iterations
    Lcount--;
    
  }
  
  // expand it on the right
  exit = false;
  while(!exit && Rcount > 0){
    
    // increase the upper end
    R += model.scale;
    
    // check if it is out of the slice
    if(log_y >= model.hyperpar_lfc(R,which)){
      exit = true;
    }
    
    // decrease the counter of upper end iterations
    Rcount--;
    
  }
  
  // sample from the shrinking interval
  exit = false;
  double log_pi_prime = log_pi;
  double old_value = tmp;
  while(!exit){
    
    // sample uniformly from the interval a proposal
    tmp = L + (R-L)*arma::randu();
    
    log_pi_prime = model.hyperpar_lfc(tmp,which);
    
    // check if it's outside the slice
    if(log_y >= log_pi_prime){
      // if it is outside, reduce the interval
      if(tmp < old_value){
        L = tmp;
      }else{
        R = tmp;
      }
    }else{
      // otherwise
      exit = true;
    }
    
    
  }
  
  // update the log posterior by subtracting the
  // the old log full conditional and adding the new one
  log_post += log_pi_prime - log_pi;
  
  // return the last proposed value
  if(log_transform){
    return std::exp(tmp);
  }else{
    return tmp;
  }
}



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
               const double& m_,
               const double& inv_temp_) : y(y_), 
               hyperpar_alpha(hyperpar_alpha_), 
               a0(a0_),
               b0(b0_),
               scale(scale_),
               m(m_),
               inv_temp(inv_temp_){
  
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
  log_post = arma::sum(arma::lgamma(alpha + ns) + 
    inv_temp * (-0.5*arma::log(lambda0 + ns) + 
    0.5 * (mus % mus % (lambda0 + ns) / sigma2)));
  
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
  
  return inv_temp * (-0.5 * std::log(sigma2s(k)) - 
                     0.5 * (y_i - mus(k)) * (y_i - mus(k)) / sigma2s(k));
  
}

// function that set the statistical unit
void Gauss1::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void Gauss1::save_parameters(const std::string& filename,
                             const unsigned int& chain_id,
                             const unsigned int& temperature){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + "_" + std::to_string(temperature)  + ".csv", std::ios::app);
  
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

// function that return the permutation index of the current state
unsigned int Gauss1::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double Gauss1::log_lik(){
  
  // compute the unnormalized log posterior
  return arma::sum( -0.5*arma::log(lambda0 + ns) + 
                    0.5 * (mus % mus % (lambda0 + ns) / sigma2));
  
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
               const double& m_,
               const double& inv_temp_) : y(y_), 
               hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
               a0(a0_),
               b0(b0_),
               M(M_), V(V_),a_l(a_l_),b_l(b_l_),a_a(a_a_),b_a(b_a_),a_b(a_b_),b_b(b_b_),
               scale(scale_),
               m(m_),
               inv_temp(inv_temp_){
  
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
  log_post = arma::sum(arma::lgamma(alpha + ns) +
    inv_temp * (-0.5 * arma::log(lambdas) -
    alphas % arma::log(betas) + 
    arma::lgamma(alphas))) + 
    inv_temp * (K * (0.5 * std::log(lambda0) + 
    alpha0 * std::log(beta0) - 
    std::lgamma(alpha0)));
  
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
  
  return inv_temp * (std::lgamma(alphas(k) + 0.5) - std::lgamma(alphas(k)) - 
                     0.5 * std::log(2.0 * betas(k) * (1.0 + 1.0 / lambdas(k))) - 
                     (alphas(k) + 0.5) * std::log(1.0 + 0.5 * 
                     (y_i - mus(k)) * (y_i - mus(k)) / (1.0 + 1.0 / lambdas(k)) / betas(k) ));
  
}

// function that set the statistical unit
void Gauss2::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void Gauss2::save_parameters(const std::string& filename,
                             const unsigned int& chain_id,
                             const unsigned int& temperature){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + "_" + std::to_string(temperature)  + ".csv", std::ios::app);
  
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

// function that return the permutation index of the current state
unsigned int Gauss2::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double Gauss2::log_lik(){
  
  // compute the unnormalized log posterior
  return arma::sum(-0.5 * arma::log(lambdas) -
                   alphas % arma::log(betas) + 
                   arma::lgamma(alphas)) + 
                   K * (0.5 * std::log(lambda0) + 
                   alpha0 * std::log(beta0) - 
                   std::lgamma(alpha0));
  
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
               const double& m_,
               const double& inv_temp_) : y(y_), 
               hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
               a0(a0_),
               b0(b0_),
               scale(scale_),
               m(m_),
               inv_temp(inv_temp_){
  
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
  log_post = 0.0;
  for(unsigned int k = 0; k < K; k++){
    
    log_post += std::lgamma(alpha(k) + ns(k)) + 
      inv_temp * (alphas(k) * half_log_det_chol(chol_Bm1s.col(k),D) -
      0.5 * D * std::log(lambdas(k)));
    
    // add the multivariate gamma function term
    for(unsigned int dd = 0; dd < D; dd++){
      log_post += inv_temp * (std::lgamma(0.5 * (alphas(k) - dd)));
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
  return inv_temp * (std::lgamma(a1) + 
                     half_log_det_chol(chol_Bm1s.col(k),D) - 
                     std::lgamma(a2) + 
                     0.5 * D * std::log(b) - 
                     a1 * std::log(1.0 + b * arma::dot(tmp,tmp)));
  
}

// function that set the statistical unit
void MGauss::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MGauss::save_parameters(const std::string& filename,
                             const unsigned int& chain_id,
                             const unsigned int& temperature){
  
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

// function that return the permutation index of the current state
unsigned int MGauss::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double MGauss::log_lik(){
  
  // initialize the log likelihood
  double log_lik = 0.0;
  for(unsigned int k = 0; k < K; k++){
    
    log_lik += alphas(k) * half_log_det_chol(chol_Bm1s.col(k),D) -
      0.5 * D * std::log(lambdas(k));
    
    // add the multivariate gamma function term
    for(unsigned int dd = 0; dd < D; dd++){
      log_lik += std::lgamma(0.5 * (alphas(k) - dd));
    }
    
  }
  
  return log_lik;
  
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
                 const double& m_,
                 const double& inv_temp_) : y(y_), 
                 hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
                 a0(a0_),
                 b0(b0_),
                 scale(scale_),
                 m(m_),
                 inv_temp(inv_temp_){
  
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
    inv_temp * (0.5 * D * arma::log(lambda0 + ns)));
  for(unsigned int k = 0; k < K; k++){
    log_post += inv_temp * (0.5 * arma::sum(mus.col(k) % mus.col(k) * 
      (lambda0 + ns(k)) / sigma2));
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
  
  return inv_temp * (-0.5 * D * std::log(sigma2s(k)) - 
                     0.5 * arma::dot(tmp,tmp) / sigma2s(k));
  
}

// function that set the statistical unit
void MGauss1::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MGauss1::save_parameters(const std::string& filename,
                              const unsigned int& chain_id,
                              const unsigned int& temperature){
  
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

// function that return the permutation index of the current state
unsigned int MGauss1::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double MGauss1::log_lik(){
  
  // initialize the log likelihood
  double log_lik = arma::sum(-0.5 * D * arma::log(lambda0 + ns));
  for(unsigned int k = 0; k < K; k++){
    log_lik += 0.5 * arma::sum(mus.col(k) % mus.col(k) * 
      (lambda0 + ns(k)) / sigma2);
  }
  
  return log_lik;
  
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
                 const double& m_,
                 const double& inv_temp_) : y(y_), 
                 hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
                 a0(a0_),
                 b0(b0_),
                 scale(scale_),
                 m(m_),
                 inv_temp(inv_temp_){
  
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
  log_post = arma::sum(arma::lgamma(alpha + ns) +
    inv_temp * (-0.5 * D * arma::log(lambdas) + D * arma::lgamma(alphas)));
  for(unsigned int k = 0; k < K; k++){
    log_post -= inv_temp * (alphas(k) * arma::sum(arma::log(betas.col(k)))); 
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
  
  return inv_temp * ((std::lgamma(alphas(k) + 0.5) - std::lgamma(alphas(k))) * D -
                     arma::sum( 0.5 * arma::log(2.0 * betas.col(k) * (1.0 + 1.0 / lambdas(k))) +
                     (alphas(k) + 0.5) * arma::log(1.0 + 0.5 * 
                     tmp % tmp / (1.0 + 1.0 / lambdas(k)) / betas.col(k) )));
  
  
}

// function that set the statistical unit
void MGauss2::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MGauss2::save_parameters(const std::string& filename,
                              const unsigned int& chain_id,
                              const unsigned int& temperature){
  
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

// function that return the permutation index of the current state
unsigned int MGauss2::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double MGauss2::log_lik(){
  
  // initialize the log likelihood
  double log_lik = arma::sum(-0.5 * D * arma::log(lambdas) + D * arma::lgamma(alphas));
  for(unsigned int k = 0; k < K; k++){
    log_lik -= alphas(k) * arma::sum(arma::log(betas.col(k))); 
  }
  
  return log_lik;
  
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
             const double& m_,
             const double& inv_temp_) : y(y_), 
             hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
             a0(a0_),
             b0(b0_),
             scale(scale_),
             m(m_),
             inv_temp(inv_temp_){
  
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
    inv_temp * (arma::lgamma(alphas) - 
    alphas % arma::log(betas)));
  
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
  
  return inv_temp * (alphas(k) * std::log(betas(k)) -
                     (alphas(k) + y_i) * std::log(1.0 + betas(k)) +
                     std::lgamma(alphas(k) + y_i) - 
                     std::lgamma(alphas(k)));
  
}

// function that set the statistical unit
void Poiss::set(const unsigned int& i){
  y_i = y(i);
}

// function to generate the collapsed parameters
// and save them to file
void Poiss::save_parameters(const std::string& filename,
                            const unsigned int& chain_id,
                            const unsigned int& temperature){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + "_" + std::to_string(temperature)  + ".csv", std::ios::app);
  
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

// function that return the permutation index of the current state
unsigned int Poiss::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double Poiss::log_lik(){
  
  // initialize the log likelihood
  double log_lik = arma::sum(arma::lgamma(alphas) - 
                             alphas % arma::log(betas));
  
  return log_lik;
  
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
             const double& m_,
             const double& inv_temp_) : y(y_), n_trials(n_trials_), 
             hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
             a0(a0_),
             b0(b0_),
             scale(scale_),
             m(m_),
             inv_temp(inv_temp_){
  
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
    inv_temp * (lbeta(alphas,betas)));
  
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
  
  return inv_temp * (lbeta(alphas(k) + y_i, betas(k) + n_i - y_i) -
                     lbeta(alphas(k),betas(k)));
  
}

// function that set the statistical unit
void Binom::set(const unsigned int& i){
  y_i = y(i);
  n_i = n_trials(i);
}

// function to generate the collapsed parameters
// and save them to file
void Binom::save_parameters(const std::string& filename,
                            const unsigned int& chain_id,
                            const unsigned int& temperature){
  
  // create the connection
  std::ofstream file_pars(filename + "/pars" + std::to_string(chain_id) + "_" + std::to_string(temperature)  +".csv", std::ios::app);
  
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

// function that return the permutation index of the current state
unsigned int Binom::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double Binom::log_lik(){
  
  // initialize the log likelihood
  double log_lik = arma::sum(lbeta(alphas,betas));
  
  return log_lik;
  
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
             const double& m_,
             const double& inv_temp_) : y(y_), 
             hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
             a0(a0_),
             b0(b0_),
             scale(scale_),
             m(m_),
             inv_temp(inv_temp_){
  
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
      inv_temp * (D*std::lgamma(alpha0+beta0+ns(k)));
    for(unsigned int d = 0; d < D; d++){
      log_post += inv_temp * (std::lgamma(alpha0 + n_dk1(d,k)) + 
        std::lgamma(beta0 + ns(k) - n_dk1(d,k)));
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
  return inv_temp * out;
  
}

// function that set the statistical unit
void MBern::set(const unsigned int& i){
  y_i = y.row(i).t();
}

// function to generate the collapsed parameters
// and save them to file
void MBern::save_parameters(const std::string& filename,
                            const unsigned int& chain_id,
                            const unsigned int& temperature){
  
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

// function that return the permutation index of the current state
unsigned int MBern::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double MBern::log_lik(){
  
  // initialize the log likelihood
  double log_lik = 0.0;
  for(unsigned int k = 0; k < K; k++){
    log_lik -= D*std::lgamma(alpha0+beta0+ns(k));
    for(unsigned int d = 0; d < D; d++){
      log_lik += std::lgamma(alpha0 + n_dk1(d,k)) + 
        std::lgamma(beta0 + ns(k) - n_dk1(d,k));
    }
  }
  
  return log_lik;
  
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
                     const double& m_,
                     const double& inv_temp_) : hyperpar_alpha(hyperpar_alpha_), hyperpar_baseline(hyperpar_baseline_), 
                     a0(a0_),
                     b0(b0_),
                     scale(scale_),
                     m(m_),
                     inv_temp(inv_temp_){
  
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
                                const unsigned int& chain_id,
                                const unsigned int& temperature){
  
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

// function that return the permutation index of the current state
unsigned int Partition::permutation(){
  
  // get the ordering of the means's posterior means
  arma::uvec ord = arma::sort_index(ns);
  
  // return the encoded index
  return permutation_rank(ord);
}

// function to compute the unpowered log likelihood
double Partition::log_lik(){
  
  return 0.0;
  
}

// -----------------------------------------------------------------------------
// ------------------------------ SCAN ALGORITHMS ------------------------------
// -----------------------------------------------------------------------------

// marginal Gibbs sampler for finite mixtures
template <typename Model>
void random_scan(Model& model,
                 arma::uvec& c,
                 arma::vec& log_prob,
                 unsigned int& n_lpc,
                 const unsigned int& K,
                 const unsigned int& thin,
                 const unsigned int& n){
  
  // set the index to move and its corresponding atom
  unsigned int i = 0, c_i = 0;
  
  // push the chain ahead thin times
  for(unsigned int tt = 0; tt < thin; tt++){
    
    // sample uniformly from the set of indexes
    i = static_cast<unsigned int>(R::runif(0, n));
    
    // set the current group
    c_i = c(i);
    
    // get the new value of the statistical unit
    model.set(i);
    
    // update its sufficient statistics and parameter
    model.update_par(c_i,-1.0);
    
    // compute the unnormalized full conditional
    for(unsigned int k = 0; k < K; k++){
      log_prob(k) = std::log(model.alpha(k) + model.ns(k)) + 
        model.log_pred(k);
    }
    
    // subtract the current cluster lufc to the log posteriori
    model.log_post -= log_prob(c_i);
    
    // sample the new group for the i-th observation
    c(i) = c_i = gumbel_max(log_prob);
    
    // add the new cluster lufc to the log posteriori
    model.log_post += log_prob(c_i);
    
    // reupdate its sufficient statistics and parameters
    model.update_par(c_i,1.0);
    
  }
  
  // update the number of predictive evaluation
  n_lpc += thin*K;
  
}

// reversible sampler for finite mixtures
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 double& accept_rate,
                 unsigned int& n_lpc,
                 const unsigned int K,
                 const unsigned int& thin){
  
  // set the index to move and its ceding and receiving groups;
  unsigned int i = 0, idx = 0, km = 0, kp = 0;
  
  // initialize the log metropolis ratio and the acceptance probability
  double log_m_ratio = 0.0, accept_prob = 0.0;
  
  // push the chain ahead thin times
  for( unsigned int tt = 0; tt < thin; tt++ ){
    
    // set the direction
    set_directions(km,kp,c,K);
    
    // check that the starting set is not empty
    if(partition[km].size() > 0 ){
      
      // sample one observation from the km group
      idx = static_cast<unsigned int>(R::runif(0,partition[km].size()));
      i = partition[km][idx];
      
      // set the current value for the statistical unit
      model.set(i);
      
      // METROPOLIS-HASTINGS STEP
      
      // subtract the contributes of y_i from km parameters
      // of the predictive distribution and its sufficient statistics
      model.update_par(km,-1.0);
      
      // terminate the computation of the log probability ratio
      
      // define the log metropolis ratio
      log_m_ratio = std::log(model.alpha(kp) + model.ns(kp)) - 
        std::log(model.alpha(km) + model.ns(km)) +
        model.log_pred(kp) - model.log_pred(km);
      
      // update the number of predictive evaluation
      n_lpc += 2;
      
      // report the probability on its scale
      accept_prob = std::min(1.0,std::exp(log_m_ratio + 
        std::log(model.ns(km) + 1.0) - 
        std::log(model.ns(kp) + 1.0)));
      
      // step ahead?
      if(arma::randu() < accept_prob){
        
        // move ahead the original chain
        c(i) = kp;
        
        // update the partition
        
        // set the index to remove equal to the last on the km group
        partition[km][idx] = partition[km].back();
        
        // erase the last piece of the km group
        partition[km].pop_back();
        
        // add to the kp group the new index
        partition[kp].push_back(i);
        
        // leave the km components as they are, update the kps
        model.update_par(kp,1.0);
        
        // update the unnormalized log posterior
        model.log_post += log_m_ratio;
        
      }else{
        
        // leave the kp components as they are, reset the kms
        model.update_par(km,1.0);
        
      }
      
      // update the rate counter
      accept_rate += accept_prob;
      
    }
    
  }
  
}

// non-reversible sampler for finite mixtures
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 std::vector<std::uint8_t>& dir,
                 double& accept_rate,
                 unsigned int& max_ex,
                 unsigned int& ex_count,
                 unsigned int& n_lpc,
                 const double& theta,
                 const unsigned int K,
                 const unsigned int& thin,
                 const unsigned int& n){
  
  // set the index to move and its ceding and receiving groups;
  unsigned int i = 0, idx = 0, idx_dir = 0, km = 0, kp = 0;
  
  // initialize the log metropolis ratio and the acceptance probability
  double log_m_ratio = 0.0, accept_prob = 0.0;
  
  // reset the maximal excursion
  max_ex = 0;
  
  // initialize the velocity
  std::uint8_t vel;
  
  // push the chain ahead thin times
  for( unsigned int tt = 0; tt < thin; tt++ ){
    
    // perturb the direction randomly
    perturb_momentum(dir,theta,K);
    
    // set the direction
    set_directions(km,kp,vel,dir,idx_dir,n,c,K);
    
    // check that the starting set is not empty
    if(partition[km].size() > 0 ){
      
      // sample one observation from the km group
      idx = static_cast<unsigned int>(R::runif(0,partition[km].size()));
      i = partition[km][idx];
      
      // set the current value for the statistical unit
      model.set(i);
      
      // METROPOLIS-HASTINGS STEP
      
      // subtract the contributes of y_i from km parameters
      // of the predictive distribution and its sufficient statistics
      model.update_par(km,-1.0);
      
      // terminate the computation of the log probability ratio
      
      // define the log metropolis ratio
      log_m_ratio = std::log(model.alpha(kp) + model.ns(kp)) - 
        std::log(model.alpha(km) + model.ns(km)) +
        model.log_pred(kp) - model.log_pred(km);
      
      // update the number of predictive evaluation
      n_lpc += 2;
      
      // report the probability on its scale
      accept_prob = std::min(1.0,std::exp(log_m_ratio + 
        std::log(model.ns(km) + 1.0) - 
        std::log(model.ns(kp) + 1.0)));
      
      // step ahead?
      if(arma::randu() < accept_prob){
        
        // move ahead the original chain
        c(i) = kp;
        
        // update the partition
        
        // set the index to remove equal to the last on the km group
        partition[km][idx] = partition[km].back();
        
        // erase the last piece of the km group
        partition[km].pop_back();
        
        // add to the kp group the new index
        partition[kp].push_back(i);
        
        // leave the km components as they are, update the kps
        model.update_par(kp,1.0);
        
        // update the unnormalized log posterior
        model.log_post += log_m_ratio;
        
      }else{
        
        // flip the velocity
        vel ^= 1;
        
        // leave the kp components as they are, reset the kms
        model.update_par(km,1.0);
        
      }
      
      // update the rate counter
      accept_rate += accept_prob;
      
    }else{
      
      // flip the velocity
      vel ^= 1;
      
    }
    
    // perturb the direction randomly
    perturb_momentum(dir,theta,K);
    
    // has the direction overall changed?
    if(vel != dir[idx_dir]){
      
      // check if this exceed the max_excursion counter
      if(ex_count > max_ex){
        max_ex = ex_count;
      }
      
      // reset the excursion counter
      ex_count = 1;
      
      // flip the bit
      dir[idx_dir] ^= 1;
      
    }else{
      
      // increse the excursion counter
      ex_count++;
      
    }
    
  }
  
}



// TEMPLATE FOR SAVING AND PRINTING FUNCTIONS

// finite mixture case
template <typename Model>
void save_and_print(Model& model,
                    unsigned int& conta,
                    arma::uvec& seq_idx,
                    arma::vec& log_p, arma::uvec& n_log_pred_calls,
                    // arma::mat& W,
                    // arma::mat& Theta,
                    arma::uvec& Sigmas,
                    arma::vec& entropy,
                    arma::vec& alphas,
                    const unsigned int& t,
                    const double& refresh,
                    const unsigned int& warm_up,
                    const unsigned int& N,
                    const unsigned int& chain_id,
                    const unsigned int& temperature,
                    const arma::uvec& seq_idx_sampler,
                    const unsigned int& n_lpc,
                    const bool& save_configurations,
                    const bool& save_parameters,
                    const std::string& filename,
                    const arma::uvec& c){
  
  // if the warming up has ended
  if(t >= warm_up){
    
    if(temperature == 1){
      // set the new value for the chain
      
      // log posterior value
      log_p(t-warm_up) = model.log_post;
      
      // number of predictive density call
      n_log_pred_calls(t-warm_up) = n_lpc;
      
      // save the prior check quantities
      // W.row(t-warm_up) = rdirichlet(model.alpha + model.ns).t();
      // for(unsigned int k = 0; k < model.ns.n_elem; k++){
      //   Theta(t-warm_up,k) = R::rnorm(model.mus(k),std::sqrt( model.sigma2s(k) - model.sigma2));
      // }
      
      // return the order statistics permutation
      Sigmas(t-warm_up) = model.permutation();
      
      // save the entropy
      double ntrp = 0.0;
      for(unsigned int k = 0; k < model.ns.n_elem; k++){
        if(model.ns(k) > 0){
          ntrp -= model.ns(k) / model.n * std::log( model.ns(k) / model.n  );
        }
      }
      entropy(t-warm_up) = ntrp;
      
      // save the hyperparameter?
      if(model.hyperpar_alpha){
        alphas(t-warm_up) = model.alpha(0);
      }
      
    }
    
    // save the collapsed parameters?
    if(save_parameters){
      
      // model.save_parameters(filename,chain_id,temperature);
      
      // // save the hyperparameters?
      // if(model.hyperpar_alpha){
      //   // create the connection
      //   std::ofstream file_hyperpar(filename + "/alpha" + std::to_string(chain_id) + ".csv", std::ios::app);
      // 
      //   // save alpha
      //   file_hyperpar << model.alpha << "\n";
      // 
      // }
      
    }
    
    // save the configuration vector?
    if(save_configurations){
      
      // create the connection
      std::ofstream file_conf(filename + "/confs" + std::to_string(chain_id) + "_" + std::to_string(temperature) + ".csv", std::ios::app);
      
      // save the configuration vector
      for(unsigned int i = 0; i < c.n_elem - 1; i++){
        file_conf << c(i) << ",";
      }
      
      // add the last element with an endline
      file_conf << c(c.n_elem-1) << "\n";
      
    }
    
  }
  
  //check if the console update condition is met
  if(temperature == 1){
    
    if(t == seq_idx(conta)){
      
      //update the count
      conta++;
      
      //print to console
      if(t < warm_up){
        Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at " << std::round(refresh * (conta) * 1000) / 10.0 << "%" << std::endl;
      }else if(t == warm_up){
        if(warm_up > 0){
          Rcpp::Rcout << "Chain " << chain_id << ", warm-up currently at 100%" << std::endl;
        }
        conta = 0;
        seq_idx = seq_idx_sampler;
      }else{
        Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at " << std::round(refresh * (conta) * 1000) / 10.0 << "%" << std::endl;
      }
      
    }
    
    // check if we have arrived to the last iteration
    if(t == warm_up + N - 1){
      Rcpp::Rcout << "Chain " << chain_id << ", sampling currently at 100%" << std::endl;
    }
    
  }
  
}

// function that swap the temperature of two chains
template <typename Model>
void swap_chains(std::vector<Model>& models,
                 arma::uvec& idxs,
                 arma::vec& acceptances,
                 bool odd){
  
  // get the number of replicas
  unsigned int T = models.size();
  
  if(T > 1){
    
    // deterministic scan
    unsigned int idx = 0, i = 0, j = 0;
    // for(unsigned int s = T-1; s < 2*T-2; s++){
    for(unsigned int idx = odd; idx < T-1; idx+=2){
      
      // forward and backward
      // idx = (s < T-1) ? s : 2*T - 3 - s;
      i = idxs(idx);
      j = idxs(idx + 1);
      
      // get the two inverse temperature to compare
      double inv_temp_i = models[i].inv_temp;
      double inv_temp_j = models[j].inv_temp;
      
      // compute the two log metropolis ratios
      // of the changing temperature proposal
      double log_m_ratio_i = 0.0, log_m_ratio_j = 0.0;
      
      // check for infinite temperature
      if(inv_temp_i == 0){
        
        // compute the log likelihood from scratch
        log_m_ratio_i = inv_temp_j * models[i].log_lik();
        
      }else{
        
        // use the already computed log posterior
        log_m_ratio_i = (inv_temp_j/inv_temp_i - 1.0) * (
          models[i].log_post - prior_contribute(models[i].ns,
                                                models[i].alpha));
      }
      
      // check for infinite temperature
      if(inv_temp_j == 0){
        
        // compute the log likelihood from scratch
        log_m_ratio_j = inv_temp_i * models[j].log_lik();
        
      }else{
        
        // use the already computed log posterior
        log_m_ratio_j = (inv_temp_i/inv_temp_j - 1.0) * (
          models[j].log_post - prior_contribute(models[j].ns,
                                                models[j].alpha));    
      }
      
      // compute the acceptance probability
      double alpha = std::min(1.0,std::exp( log_m_ratio_i + log_m_ratio_j));
      // acceptances(idx) += 2.0 * 0.5 * alpha;
      
      acceptances(idx) += alpha;
      
      // acceptance step
      if(arma::randu() < alpha){
        
        //acceptances(idx) += 1;
        
        // swap the temperatures
        models[i].inv_temp = inv_temp_j;
        models[j].inv_temp = inv_temp_i;
        
        // adjust the unnormalized log posteriors
        models[i].log_post += log_m_ratio_i; 
        models[j].log_post += log_m_ratio_j;
        
        // swap the position of the two temperatures
        idxs(idx) = j;
        idxs(idx+1) = i;
        
      }
      
    }
    
  }
  
}

// finite mixture case
template <typename Model>
void gibbs_sampler(std::vector<Model>& models,
                   std::vector<arma::uvec>& cs,
                   arma::vec& log_p, arma::uvec& n_log_pred_calls,
                   // arma::mat& W,
                   // arma::mat& Theta,
                   arma::uvec& Sigmas,
                   arma::vec& entropy,
                   arma::vec& alphas,
                   arma::vec& swaps,
                   const unsigned int& N,
                   const unsigned int& warm_up,
                   const unsigned int& thin,
                   const unsigned int& thin_scan,
                   const double& refresh,
                   const unsigned int& chain_id,
                   arma::uvec& seq_idx,
                   const arma::uvec& seq_idx_sampler,
                   const bool& save_configurations,
                   const bool& save_parameters,
                   const std::string& filename,
                   const arma::vec& inv_temps){
  
  // get the sample size
  unsigned int n = cs[0].n_elem;
  
  // get the number of clusters
  unsigned int K = models[0].K;
  
  // get the number of temperatures
  unsigned int T = inv_temps.n_elem;
  
  // initialize the log posterior
  double log_post = models[0].log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the probability vector
  arma::vec log_prob(K);
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // initialize the temperature indexes
  arma::uvec idxs(T);
  for(unsigned int ttt = 0; ttt < T; ttt++){
    idxs(ttt) = ttt;
  }
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    if(t == warm_up) swaps.zeros();
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // propose to swap temperatures
      if(T > 1){
        //swaps += swap_chains(models,idxs) / thin;
        swap_chains(models,idxs, swaps, !( ( (t-1)*thin + tt) % 2) );
      }
      
      // update the hyperparameter?
      //  model.update_hyperpars(log_post);
      
      // random scan at each temperature level
      for(unsigned int ttt = 0; ttt < T; ttt++){
        random_scan(models[idxs(ttt)],cs[idxs(ttt)],log_prob,n_lpc,K,thin_scan,n);
      }
      
    }
    
    // save and print
    for(unsigned int s = 0; s < T; s++){
      save_and_print(models[idxs(s)],conta,seq_idx,log_p,n_log_pred_calls,Sigmas,entropy,alphas,t,refresh,warm_up,N,
                     chain_id,1+s,seq_idx_sampler,n_lpc,save_configurations,
                     save_parameters,filename,cs[idxs(s)]);
    }
    
    // save_and_print(models[idx1],conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
    //                chain_id,seq_idx_sampler,n_lpc,save_configurations,
    //                save_parameters,filename,cs[idx1]);
  }
  
  // normalize the swaps rate
  if(T > 1){
    unsigned int n_even = N * thin / 2;
    unsigned int n_odd = N * thin - n_even;
    for(unsigned int ss = 0; ss < T-1; ss+=2){
      swaps(ss) /= n_odd;
    }
    for(unsigned int ss = 1; ss < T-1; ss+=2){
      swaps(ss) /= n_even;
    }
  }
  
}

// TEMPLATE FOR REVERSIBLE ALGORITHMS

// finite mixture case
template <typename Model>
void reversible_sampler(std::vector<Model>& models,
                        std::vector<arma::uvec>& cs,
                        std::vector<std::vector<std::vector<unsigned int>>>& partitions,
                        arma::vec& log_p, arma::uvec& n_log_pred_calls,
                        // arma::mat& W,
                        // arma::mat& Theta,
                        arma::uvec& Sigmas,
                        arma::vec& entropy,
                        arma::vec& alphas,
                        arma::vec& swaps,
                        arma::mat& acceptance_rates,
                        const unsigned int& N,
                        const unsigned int& warm_up,
                        const unsigned int& thin,
                        const unsigned int& thin_scan,
                        const double& refresh,
                        const unsigned int& chain_id,
                        arma::uvec& seq_idx,
                        const arma::uvec& seq_idx_sampler,
                        const bool& save_configurations,
                        const bool& save_parameters,
                        const std::string& filename,
                        const arma::vec& inv_temps){
  
  // get the number of clusters
  unsigned int K = models[0].K;
  
  // initialize the log posterior
  double log_post = models[0].log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // get the number of temperatures
  unsigned int T = models.size();
  
  // initialize the acceptance rates
  arma::vec accept_rate = arma::zeros<arma::vec>(T);
  
  // initialize the temperature indexes
  arma::uvec idxs(T);
  for(unsigned int ttt = 0; ttt < T; ttt++){
    idxs(ttt) = ttt;
  }
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the acceptance rate
    accept_rate.zeros();
    
    if(t == warm_up) swaps.zeros();
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // propose to swap temperatures
      if(T > 1){
        //swaps += swap_chains(models,idxs) / thin;
        swap_chains(models,idxs, swaps, !( ( (t-1)*thin + tt) % 2) );
      }
      
      // update the hyperparameter?
      //  model.update_hyperpars(log_post);
      
      // random scan for each temperature
      for(unsigned int ttt = 0; ttt < T; ttt++){
        random_scan(models[idxs(ttt)],partitions[idxs(ttt)],cs[idxs(ttt)],accept_rate(ttt),n_lpc,K,thin_scan);
      }
      
    }
    
    // save and print
    for(unsigned int s = 0; s < T; s++){
      save_and_print(models[idxs(s)],conta,seq_idx,log_p,n_log_pred_calls,Sigmas,entropy,alphas,t,refresh,warm_up,N,
                     chain_id,1+s,seq_idx_sampler,n_lpc,save_configurations,
                     save_parameters,filename,cs[idxs(s)]);
    }
    // save_and_print(models[idx1],conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
    //                chain_id,seq_idx_sampler,n_lpc,save_configurations,
    //                save_parameters,filename,cs[idx1]);
    
    if(t >= warm_up){
      acceptance_rates.row(t-warm_up) = accept_rate.t() / thin / thin_scan;
    }
    
  }
  
  // normalize the swaps rate
  if(T > 1){
    unsigned int n_even = N * thin / 2;
    unsigned int n_odd = N * thin - n_even;
    for(unsigned int ss = 0; ss < T-1; ss+=2){
      swaps(ss) /= n_odd;
    }
    for(unsigned int ss = 1; ss < T-1; ss+=2){
      swaps(ss) /= n_even;
    }
  }
  
}

// TEMPLATE FOR NON-REVERSIBLE ALGORITHM

// finite mixture case
template <typename Model>
void non_reversible_sampler(std::vector<Model>& models,
                            std::vector<arma::uvec>& cs,
                            std::vector<std::vector<std::vector<unsigned int>>>& partitions,
                            arma::vec& log_p, arma::uvec& n_log_pred_calls,
                            // arma::mat& W,
                            // arma::mat& Theta,
                            arma::uvec& Sigmas,
                            arma::vec& entropy,
                            arma::vec& alphas,
                            arma::vec& swaps,
                            arma::mat& acceptance_rates,
                            arma::umat& max_excursions,
                            const double& theta,
                            const double& s,
                            const unsigned int& N,
                            const unsigned int& warm_up,
                            const unsigned int& thin,
                            const unsigned int& thin_scan,
                            const double& refresh,
                            const unsigned int& chain_id,
                            arma::uvec& seq_idx,
                            const arma::uvec& seq_idx_sampler,
                            const bool& save_configurations,
                            const bool& save_parameters,
                            const std::string& filename,
                            const arma::vec& inv_temps){
  
  // get the sample size
  unsigned int n = cs[0].n_elem;
  
  // get the number of clusters
  unsigned int K = models[0].K;
  
  // initialize the log posterior
  double log_post = models[0].log_post;
  
  // get the number of temperatures
  unsigned int T = models.size();
  
  // initialize the orientation vectors
  std::vector<std::vector<std::uint8_t>> dirs;
  dirs.resize(T);
  for(unsigned int ttt = 0; ttt < T; ttt++){
    std::vector<std::uint8_t> dir(K*(K-1)/2);
    for(unsigned int i = 0; i < dir.size(); i++){
      dir[i] = (R::runif(0,1) < 0.5) ? 1 : 0;
    }
    dirs[ttt] = dir;
  }
  
  // initialize the acceptance probabilities
  arma::vec accept_rate = arma::zeros<arma::vec>(T);
  
  // initialize the maximal excursions
  arma::uvec max_ex = arma::zeros<arma::uvec>(T);
  
  //  initialize the excursion counters
  arma::uvec ex_count = arma::ones<arma::uvec>(T);
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // initialize the temperature indexes
  arma::uvec idxs(T);
  for(unsigned int ttt = 0; ttt < T; ttt++){
    idxs(ttt) = ttt;
  }
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the acceptance rate
    accept_rate.zeros();
    
    // reset the max excursion counter
    max_ex.zeros();
    
    // reset the excursion counter
    ex_count.ones();
    
    if(t == warm_up) swaps.zeros();
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // propose to swap temperatures
      if(T > 1){
        //swaps += swap_chains(models,idxs) / thin;
        swap_chains(models,idxs, swaps, !( ( (t-1)*thin + tt) % 2) );
      }
      
      // update the hyperparameter?
      //  model.update_hyperpars(log_post);
      
      // classic non-reversible scan at each temperature
      for(unsigned int ttt = 0; ttt < T; ttt++){
        random_scan(models[idxs(ttt)],partitions[idxs(ttt)],cs[idxs(ttt)],dirs[idxs(ttt)],
                    accept_rate(ttt),max_ex(ttt),ex_count(ttt),n_lpc,theta,K,thin_scan,n);
      }
      
    }
    
    // save and print
    for(unsigned int ss = 0; ss < T; ss++){
      save_and_print(models[idxs(ss)],conta,seq_idx,log_p,n_log_pred_calls,Sigmas,entropy,alphas,t,refresh,warm_up,N,
                     chain_id,1+ss,seq_idx_sampler,
                     n_lpc,save_configurations,
                     save_parameters,filename,cs[idxs(ss)]);
    }
    // save_and_print(models[idx1],conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
    //                chain_id,seq_idx_sampler,
    //                n_lpc,save_configurations,
    //                save_parameters,filename,cs[idx1]);
    
    // save the maximum excursion occurred and the acceptance rate
    if(t >= warm_up){
      max_excursions.row(t-warm_up) = max_ex.t();
      acceptance_rates.row(t-warm_up) = accept_rate.t() / thin / thin_scan;
    }
    
  }
  
  // normalize the swaps rate
  if(T > 1){
    unsigned int n_even = N * thin / 2;
    unsigned int n_odd = N * thin - n_even;
    for(unsigned int ss = 0; ss < T-1; ss+=2){
      swaps(ss) /= n_odd;
    }
    for(unsigned int ss = 1; ss < T-1; ss+=2){
      swaps(ss) /= n_even;
    }
  }
  
}

// SAMPLER WRAPPERS
template <typename Model>
void tempered_mixture(std::vector<Model>& models,
                      std::vector<std::vector<std::vector<unsigned int>>>& partitions,
                      std::vector<arma::uvec>& cs,
                      arma::vec& log_p, arma::uvec& n_log_pred_calls,
                      // arma::mat& W,
                      // arma::mat& Theta,
                      arma::uvec& Sigmas,
                      arma::vec& entropy,
                      arma::mat& acceptance_rates,
                      arma::vec& alphas,
                      arma::vec& swaps,
                      arma::umat& max_excursions,
                      const double& theta,
                      const double& s,
                      const unsigned int& N,
                      const unsigned int& warm_up,
                      const unsigned int& thin,
                      const unsigned int& thin_scan,
                      const unsigned int& m,
                      const unsigned int& g,
                      const double& refresh,
                      const unsigned int& chain_id,
                      arma::uvec& seq_idx,
                      const arma::uvec& seq_idx_sampler,
                      const bool& save_configurations,
                      const bool& save_parameters,
                      const std::string& filename,
                      const bool& gibbs,
                      const bool& reversible,
                      const bool& informed,
                      const arma::vec& inv_temps){
  
  if(gibbs){
    
    // gibbs sampler
    gibbs_sampler(models,cs,log_p,n_log_pred_calls,Sigmas,entropy,alphas,swaps,N,warm_up,thin,thin_scan,refresh,
                  chain_id,seq_idx,seq_idx_sampler,
                  save_configurations,save_parameters,filename,inv_temps);
    
  }else{
    if(reversible){
      
      // reversible non informed sampler
      reversible_sampler(models,cs,partitions,log_p,n_log_pred_calls,Sigmas,entropy,alphas,swaps,acceptance_rates,
                         N,warm_up,thin,thin_scan,refresh,
                         chain_id,seq_idx,seq_idx_sampler,
                         save_configurations,save_parameters,filename,inv_temps);
      
    }else{
      
      // non reversible non informed sampler
      non_reversible_sampler(models,cs,partitions,log_p,n_log_pred_calls,Sigmas,entropy,alphas,swaps,acceptance_rates,max_excursions,theta,s,
                             N,warm_up,thin,thin_scan,refresh,
                             chain_id,seq_idx,seq_idx_sampler,
                             save_configurations,save_parameters,filename,inv_temps);
      
    }
  }
}

// [[Rcpp::export]]
Rcpp::List nrMCtempmix(const Rcpp::RObject& y = R_NilValue,
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
                       const Rcpp::Nullable<Rcpp::NumericVector>& inverse_temperatures = R_NilValue,
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
  
  // inverse temperatures
  arma::vec inv_temps;
  if(inverse_temperatures.isNull()){
    inv_temps = arma::zeros<arma::vec>(10);
    for(unsigned int t = 0; t < inv_temps.n_elem; t++){
      inv_temps(t) = std::exp(t * std::log(2.0) * -1.0);
    }
  }else{
    inv_temps = Rcpp::as<arma::vec>(inverse_temperatures);
  }
  
  // get the number of temperatures
  unsigned int T = inv_temps.n_elem;
  
  // get the current configuration
  std::vector<arma::uvec> cs;
  arma::uvec c(n);
  cs.resize(T);
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
  
  for(unsigned int ttt = 0; ttt < T; ttt++){
    cs[ttt] = c;
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
  std::vector<std::vector<std::vector<unsigned int>>> partitions(T);
  
  for(unsigned int ttt = 0; ttt < T; ttt++){
    if(finite){
      partitions[ttt].resize(K);
    }else{
      partitions[ttt].resize(n);
    }
  }
  
  // posterior markov chain
  arma::vec log_p(N);
  arma::uvec n_log_pred_calls(N);
  arma::vec entropy(N);
  arma::vec swaps(T-1);
  
  // prior check quantities
  
  // get the number of permutations
  arma::uvec Sigmas(N);
  // arma::mat W;
  // arma::uvec K_atoms;
  // if(finite){
  //   W = arma::zeros<arma::mat>(N,K);
  // }else{
  //   K_atoms = arma::zeros<arma::uvec>(N);
  //   //entropy = arma::zeros<arma::vec>(N);
  // }
  
  // MH summaries
  arma::mat acceptance_rates;
  arma::umat max_excursions;
  
  // create the filename for the heavier stuff
  std::string filename = get_tempdir_cpp();
  
  if(gibbs && SAM){
    // // reserve n element each to avoid resizing
    // for(unsigned int k = 0; k < K; k++){
    //   
    //   partition[k].reserve(n);
    //   
    // }
    // acceptance_rates = arma::zeros<arma::mat>(N,2);
  }else if(SAM){
    
    // // reserve n element each to avoid resizing
    // for(unsigned int k = 0; k < K; k++){
    //   
    //   partition[k].reserve(n);
    //   
    // }
    // acceptance_rates = arma::zeros<arma::mat>(N,3);
  }else{
    
    for(unsigned int ttt = 0; ttt < T; ttt++){
      // reserve n element each to avoid resizing
      for(unsigned int k = 0; k < K; k++){
        
        partitions[ttt][k].reserve(n);
      } 
    }
    
    // reserve the excursion and acceptance rates vectors
    acceptance_rates = arma::zeros<arma::mat>(N,T);
    
  }
  
  if(!reversible){
    max_excursions = arma::zeros<arma::umat>(N,T);
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
          
          // create a vector of objects
          std::vector<MGauss> models;
          for(unsigned int t = 0; t < inv_temps.n_elem; t++){
            models.emplace_back(partitions[t],y_mat,K,alpha_finite,mu00,lambda0,alpha0,B0,cs[t],gibbs,
                                hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
          }
          
          // fit the model
          tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                           max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,gibbs,reversible,informed,inv_temps);
          
          // save the last configuration at the base temperature
          for(unsigned int s = 0; s < T; s++){
            if(models[s].inv_temp == inv_temps(0)){
              out["c"] = cs[s];
            }
          }
          
        }else{
          // inf_MGauss model(partition,lbl2atm,atm2lbl,labels,y_mat,alpha_infinite,delta,mu00,lambda0,alpha0,B0,c,gibbs && !SAM,
          //                  hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          // 
          // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
          //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
          //                  n_restricted_steps,m,g,refresh,chain_id,
          //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
          //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
          
          // create a vector of objects
          std::vector<MGauss2> models;
          for(unsigned int t = 0; t < inv_temps.n_elem; t++){
            models.emplace_back(partitions[t],y_mat,K,alpha_finite,mu00,lambda0,alpha0,beta00,cs[t],gibbs,
                                hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
          }
          
          // fit the model
          tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                           max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,gibbs,reversible,informed,inv_temps);
          
          // save the last configuration at the base temperature
          for(unsigned int s = 0; s < T; s++){
            if(models[s].inv_temp == inv_temps(0)){
              out["c"] = cs[s];
            }
          }
          
        }else{
          // inf_MGauss2 model(partition,lbl2atm,atm2lbl,labels,y_mat,alpha_infinite,delta,mu00,lambda0,alpha0,beta00,c,gibbs && !SAM,
          //                   hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          // 
          // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
          //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
          //                  n_restricted_steps,m,g,refresh,chain_id,
          //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
          //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
          
          // create a vector of objects
          std::vector<MGauss1> models;
          for(unsigned int t = 0; t < inv_temps.n_elem; t++){
            models.emplace_back(partitions[t],y_mat,K,alpha_finite,sigma20,mu00,lambda0,cs[t],gibbs,
                                hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
          }
          
          // fit the model
          tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                           max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,gibbs,reversible,informed,inv_temps);
          
          // save the last configuration at the base temperature
          for(unsigned int s = 0; s < T; s++){
            if(models[s].inv_temp == inv_temps(0)){
              out["c"] = cs[s];
            }
          }
          
        }else{
          // inf_MGauss1 model(partition,lbl2atm,atm2lbl,labels,y_mat,alpha_infinite,delta,sigma20,mu00,lambda0,c,gibbs && !SAM,
          //                   hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          // 
          // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
          //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
          //                  n_restricted_steps,m,g,refresh,chain_id,
          //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
          //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
          
          // create a vector of objects
          std::vector<Gauss2> models;
          for(unsigned int t = 0; t < inv_temps.n_elem; t++){
            models.emplace_back(partitions[t],y_vec,K,alpha_finite,mu00,lambda0,alpha0,beta00,cs[t],gibbs,
                                hyperpar_alpha,hyperpar_baseline,a0,b0,
                                M,V,a_l,b_l,a_a,b_a,a_b,b_b,
                                scale,slice_max_iter,inv_temps(t));
          }
          
          // fit the model
          tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                           max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,gibbs,reversible,informed,inv_temps);
          
          // save the last configuration at the base temperature
          for(unsigned int s = 0; s < T; s++){
            if(models[s].inv_temp == inv_temps(0)){
              out["c"] = cs[s];
            }
          }
          
        }else{
          // inf_Gauss2 model(partition,lbl2atm,atm2lbl,labels,y_vec,alpha_infinite,delta,mu00,lambda0,alpha0,beta00,c,gibbs && !SAM,
          //                  hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,
          //                  M,V,a_l,b_l,a_a,b_a,a_b,b_b,
          //                  scale,slice_max_iter);
          // 
          // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates, entropy,alphas,deltas,
          //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
          //                  n_restricted_steps,m,g,refresh,chain_id,
          //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
          //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
          
          // create a vector of objects
          std::vector<Gauss1> models;
          for(unsigned int t = 0; t < inv_temps.n_elem; t++){
            models.emplace_back(partitions[t],y_vec,K,alpha_finite,sigma20,mu00,lambda0,cs[t],gibbs,
                                hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
          }
          
          // fit the model
          tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                           max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                           seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                           filename,gibbs,reversible,informed,inv_temps);
          
          // save the last configuration at the base temperature
          for(unsigned int s = 0; s < T; s++){
            if(models[s].inv_temp == inv_temps(0)){
              out["c"] = cs[s];
            }
          }
          
        }else{
          // inf_Gauss1 model(partition,lbl2atm,atm2lbl,labels,y_vec,alpha_infinite,delta,sigma20,mu00,lambda0,c,gibbs && !SAM,
          //                  hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
          // 
          // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
          //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
          //                  n_restricted_steps,m,g,refresh,chain_id,
          //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
          //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
      
      // create a vector of objects
      std::vector<Poiss> models;
      for(unsigned int t = 0; t < inv_temps.n_elem; t++){
        models.emplace_back(partitions[t],y_uvec,K,alpha_finite,alpha0,beta00,cs[t],gibbs,
                            hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
      }
      
      // fit the model
      tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                       max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,gibbs,reversible,informed,inv_temps);
      
      // save the last configuration at the base temperature
      for(unsigned int s = 0; s < T; s++){
        if(models[s].inv_temp == inv_temps(0)){
          out["c"] = cs[s];
        }
      }
      
    }else{
      // inf_Poiss model(partition,lbl2atm,atm2lbl,labels,y_uvec,alpha_infinite,delta,alpha0,beta00,c,gibbs && !SAM,
      //                 hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      // 
      // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
      //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
      //                  n_restricted_steps,m,g,refresh,chain_id,
      //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
      //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
      
      // create a vector of objects
      std::vector<Binom> models;
      for(unsigned int t = 0; t < inv_temps.n_elem; t++){
        models.emplace_back(partitions[t],y_uvec,n_trials_uvec,K,alpha_finite,alpha0,beta00,cs[t],gibbs,
                            hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
      }
      
      // fit the model
      tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                       max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,gibbs,reversible,informed,inv_temps);
      
      // save the last configuration at the base temperature
      for(unsigned int s = 0; s < T; s++){
        if(models[s].inv_temp == inv_temps(0)){
          out["c"] = cs[s];
        }
      }
      
    }else{
      // inf_Binom model(partition,lbl2atm,atm2lbl,labels,y_uvec,n_trials_uvec,alpha_infinite,delta,alpha0,beta00,c,gibbs && !SAM,
      //                 hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      // 
      // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
      //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
      //                  n_restricted_steps,m,g,refresh,chain_id,
      //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
      //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
      
      // create a vector of objects
      std::vector<MBern> models;
      for(unsigned int t = 0; t < inv_temps.n_elem; t++){
        models.emplace_back(partitions[t],y_umat,K,alpha_finite,alpha0,beta00,cs[t],gibbs,
                            hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
      }
      
      // fit the model
      tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                       max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,gibbs,reversible,informed,inv_temps);
      
      // save the last configuration at the base temperature
      for(unsigned int s = 0; s < T; s++){
        if(models[s].inv_temp == inv_temps(0)){
          out["c"] = cs[s];
        }
      }
      
    }else{
      // inf_MBern model(partition,lbl2atm,atm2lbl,labels,y_umat,alpha_infinite,delta,alpha0,beta00,c,gibbs && !SAM,
      //                 hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      // 
      // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
      //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
      //                  n_restricted_steps,m,g,refresh,chain_id,
      //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
      //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
      
      // create a vector of objects
      std::vector<Partition> models;
      for(unsigned int t = 0; t < inv_temps.n_elem; t++){
        models.emplace_back(partitions[t],K,alpha_finite,cs[t],gibbs,
                            hyperpar_alpha,hyperpar_baseline,a0,b0,scale,slice_max_iter,inv_temps(t));
      }
      
      // fit the model
      tempered_mixture(models,partitions,cs,log_p,n_log_pred_calls,Sigmas,entropy,acceptance_rates,alphas,swaps,
                       max_excursions,xi/n,s,N,warm_up,thin,thin_scan,m,g,refresh,chain_id,
                       seq_idx,seq_idx_sampler,save_configurations,save_parameters,
                       filename,gibbs,reversible,informed,inv_temps);
      
      // save the last configuration at the base temperature
      for(unsigned int s = 0; s < T; s++){
        if(models[s].inv_temp == inv_temps(0)){
          out["c"] = cs[s];
        }
      }
      
    }else{
      // inf_Partition model(partition,lbl2atm,atm2lbl,labels,alpha_infinite,delta,c,gibbs && !SAM,
      //                     hyperpar_alpha,hyperpar_delta,hyperpar_baseline,a0,b0,lwr_bound,upr_bound,scale,slice_max_iter);
      // 
      // infinite_mixture(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
      //                  max_excursions,xi/n,N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
      //                  n_restricted_steps,m,g,refresh,chain_id,
      //                  seq_idx,seq_idx_sampler,save_configurations,save_parameters,
      //                  filename,lbl2atm,atm2lbl,labels,gibbs,reversible,informed,SAM);
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
  out["entropy"] = entropy;
  out["swaps"] = swaps;
  out["inverse_temperatures"] = inv_temps;
  // if(finite){
  //   out["W"] = W;
  // }else{
  //   out["K"] = K_atoms;
  // }  
  out["permutations"] = Sigmas; 
  
  
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
