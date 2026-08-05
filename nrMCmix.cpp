#include <iostream>
#include <RcppArmadillo.h>
#include <variant>

// [[Rcpp::depends(RcppArmadillo)]]

#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

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



// marginal Gibbs sampler for finite mixtures
template <typename Model>
void random_scan(Model& model,
                 arma::uvec& c,
                 arma::vec& log_prob,
                 double& log_post,
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
    log_post -= log_prob(c_i);
    
    // sample the new group for the i-th observation
    c(i) = c_i = gumbel_max(log_prob);
    
    // add the new cluster lufc to the log posteriori
    log_post += log_prob(c_i);
    
    // reupdate its sufficient statistics and parameters
    model.update_par(c_i,1.0);
    
  }
  
  // update the number of predictive evaluation
  n_lpc += thin*K;
  
}

// marginal Gibbs sampler for infinite mixtures
template <typename Model>
void random_scan(Model& model,
                 arma::uvec& c,
                 std::vector<double>& log_prob,
                 std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                 std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                 std::vector<unsigned int>& labels,
                 double& log_post,
                 unsigned int& n_lpc,
                 unsigned int& K_atm,
                 const unsigned int& thin,
                 const unsigned int& n){
  
  // set the index to move;
  unsigned int i = 0;
  
  // initialize the atoms and labels
  unsigned int atm = 0, old_atm = 0, old_lbl = 0, new_lbl = 0;
  
  // push the chain ahead thin times
  for(unsigned int tt = 0; tt < thin; tt++){
    
    // sample uniformly from the set of indexes
    i = static_cast<unsigned int>(R::runif(0, n));
    
    // get the current atom
    atm = lbl2atm[c(i)];
    
    // get the new value of the statistical unit
    model.set(i);
    
    // update its sufficient statistics and parameter
    model.update_par(atm,-1.0);
    
    // compute the unnormalized log full conditional
    for(unsigned int k = 0; k < K_atm; k++){
      if(model.ns[k] > 0){
        log_prob[k] = std::log(model.ns[k]-model.delta) + 
          model.log_pred(k);
      }else{
        log_prob[k] = std::log(model.alpha + model.delta*(K_atm-1.0)) + 
          model.log_pred(k);
      }
    }
    
    // update the number of predictive evaluation
    n_lpc += K_atm;
    
    // subtract the current cluster lufc to the log posteriori
    log_post -= log_prob[atm];
    
    // are we deleting a group?
    if(model.ns[atm] > 0){
      // we are not deleting any group
      
      // add the new atom option
      log_prob[K_atm] = std::log(model.alpha + model.delta*K_atm) + 
        model.log_pred_prior();
      
      // update the number of predictive evaluation
      n_lpc++;
      
      // sample the new group for the i-th observation
      atm = gumbel_max(log_prob,K_atm);
      
      // are we creating a new group?
      if(atm == K_atm){
        
        // increase the log probability vector by one entry
        log_prob.push_back(0.0);
        
        // add the new sufficient statistics (only the prior term)
        model.add_new_atom();
        
        // get the new label from the disposable pool
        new_lbl = labels.back();
        
        // remove it from this collection
        labels.pop_back();
        
        // add the new pair in the maps
        lbl2atm[new_lbl] = K_atm;
        atm2lbl[K_atm] = new_lbl;
        
        // increase the current number of atoms
        K_atm++;
        
      }
      
    }else{
      // we are deleting the ceding group
      
      // decrease the number of atoms
      K_atm--;
      
      // sample the new group for the i-th observation
      old_atm = atm;
      atm = gumbel_max(log_prob,K_atm);
      
      // has the statistical unit moved to another atom?
      if(old_atm != atm){
        
        // decrease the log probability vector by one
        log_prob.pop_back();
        log_prob[old_atm] = log_prob.back();
        
        // delete the old sufficient statistic
        model.copy_atom(old_atm,K_atm);
        model.delete_last_atom();
        
        // change the labelings
        
        // get the label of the old atom
        old_lbl = atm2lbl[old_atm];
        
        // get the label of the last atom
        new_lbl = atm2lbl[K_atm];
        
        // replace the label of the old atom with the one of the last one 
        atm2lbl[old_atm] = new_lbl;
        
        // delete the last atom
        atm2lbl.erase(K_atm);
        
        // replace the atom of the new label with the old atom
        lbl2atm[new_lbl] = old_atm;
        
        // delete the label associated with the old atom
        lbl2atm.erase(old_lbl);
        
        // append the old label to the list of possible labels
        labels.push_back(old_lbl);
        
        // if the new atom is the last, set it to be the as the old
        if(atm == K_atm){
          atm = old_atm;
        }
        
      }else{
        // reset the number of atoms to the original one
        K_atm++;
      }
      
    }
    
    // add the new cluster lufc to the log posteriori
    log_post += log_prob[atm];
    
    // update the new sufficient statistics
    // by adding the i-th observation
    model.update_par(atm,1.0);
    
    // update the configuration vector
    c(i) = atm2lbl[atm];
    
  }
  
}

// marginal sampler for infinite mixtures with partition explicitly updated
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 std::vector<double>& log_prob,
                 std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                 std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                 std::vector<unsigned int>& labels,
                 double& log_post,
                 unsigned int& n_lpc,
                 unsigned int& K_atm,
                 const unsigned int& thin,
                 const unsigned int& n){
  
  // set the index to move;
  unsigned int i = 0, idx = 0;
  
  // initialize the atoms and labels
  unsigned int atm = 0, old_atm = 0, old_lbl = 0, new_lbl = 0;
  
  // push the chain ahead thin times
  for(unsigned int tt = 0; tt < thin; tt++){
    
    // sample uniformly from the set of indexes
    i = static_cast<unsigned int>(R::runif(0, n));
    
    // get the current lbl
    old_lbl = c(i);
    
    // get the current atom
    atm = lbl2atm[old_lbl];
    
    // given the current label, sample a unit from its partition
    idx = static_cast<unsigned int>(R::runif(0,partition[old_lbl].size()));
    i = partition[old_lbl][idx];
    
    // get the new value of the statistical unit
    model.set(i);
    
    // update its sufficient statistics and parameter
    model.update_par(atm,-1.0);
    
    // compute the unnormalized log full conditional
    for(unsigned int k = 0; k < K_atm; k++){
      if(model.ns[k] > 0){
        log_prob[k] = std::log(model.ns[k]-model.delta) + 
          model.log_pred(k);
      }else{
        log_prob[k] = std::log(model.alpha + model.delta*(K_atm-1.0)) + 
          model.log_pred(k);
      }
    }
    
    // update the number of predictive evaluation
    n_lpc += K_atm;
    
    // subtract the current cluster lufc to the log posteriori
    log_post -= log_prob[atm];
    
    // are we deleting a group?
    if(model.ns[atm] > 0){
      // we are not deleting any group
      
      // add the new atom option
      log_prob[K_atm] = std::log(model.alpha + model.delta*K_atm) + 
        model.log_pred_prior();
      
      // update the number of predictive evaluation
      n_lpc++;
      
      // sample the new group for the i-th observation
      atm = gumbel_max(log_prob,K_atm);
      
      // are we creating a new group?
      if(atm == K_atm){
        
        // increase the log probability vector by one entry
        log_prob.push_back(0.0);
        
        // add the new sufficient statistics (only the prior term)
        model.add_new_atom();
        
        // get the new label from the disposable pool
        new_lbl = labels.back();
        
        // remove it from this collection
        labels.pop_back();
        
        // add the new pair in the maps
        lbl2atm[new_lbl] = K_atm;
        atm2lbl[K_atm] = new_lbl;
        
        // increase the current number of atoms
        K_atm++;
        
      }
      
    }else{
      // we are deleting the ceding group
      
      // decrease the number of atoms
      K_atm--;
      
      // sample the new group for the i-th observation
      old_atm = atm;
      atm = gumbel_max(log_prob,K_atm);
      
      // has the statistical unit moved to another atom?
      if(old_atm != atm){
        
        // decrease the log probability vector by one
        log_prob.pop_back();
        log_prob[old_atm] = log_prob.back();
        
        // delete the old sufficient statistic
        model.copy_atom(old_atm,K_atm);
        model.delete_last_atom();  
        
        // change the labelings
        
        // get the label of the old atom
        old_lbl = atm2lbl[old_atm];
        
        // get the label of the last atom
        new_lbl = atm2lbl[K_atm];
        
        // replace the label of the old atom with the one of the last one 
        atm2lbl[old_atm] = new_lbl;
        
        // delete the last atom
        atm2lbl.erase(K_atm);
        
        // replace the atom of the new label with the old atom
        lbl2atm[new_lbl] = old_atm;
        
        // delete the label associated with the old atom
        lbl2atm.erase(old_lbl);
        
        // append the old label to the list of possible labels
        labels.push_back(old_lbl);
        
        // if the new atom is the last, set it to be the as the old
        if(atm == K_atm){
          atm = old_atm;
        }
        
      }else{
        // reset the number of atoms to the original one
        K_atm++;
      }
      
    }
    
    // add the new cluster lufc to the log posteriori
    log_post += log_prob[atm];
    
    // update the new sufficient statistics
    // by adding the i-th observation
    model.update_par(atm,1.0);
    
    // update the configuration vector
    new_lbl = atm2lbl[atm];
    c(i) = new_lbl;
    
    // update the partition
    partition[old_lbl][idx] = partition[old_lbl].back();
    partition[old_lbl].pop_back();
    partition[new_lbl].push_back(i);
    
  }
  
}

// reversible sampler for finite mixtures
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 double& accept_rate,
                 double& log_post,
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
        log_post += log_m_ratio;
        
      }else{
        
        // leave the kp components as they are, reset the kms
        model.update_par(km,1.0);
        
      }
      
      // update the rate counter
      accept_rate += accept_prob;
      
    }
    
  }
  
}

// informed version
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 unsigned int* c_km,
                 unsigned int* c_kp,
                 arma::vec& w,
                 double& accept_rate,
                 double& log_post,
                 unsigned int& n_lpc,
                 const unsigned int K,
                 const unsigned int& thin,
                 const unsigned int& m,
                 const unsigned int& g){
  
  // set the index to move and its ceding and receiving groups;
  unsigned int i = 0, idx = 0, km = 0, kp = 0;
  
  // initialize the size of the ceding and receiving groups
  unsigned int n_km = 0, n_kp = 0;
  
  // initialize the log metropolis ratio and the acceptance probability
  double log_alpha = 0.0, accept_prob = 0.0;
  
  // constant part of the weights
  double w_const = 0.0;
  
  // push the chain ahead thin times
  for(unsigned int tt = 0; tt < thin; tt++ ){
    
    // set the direction
    set_directions(km,kp,c,K);
    
    // set the groups of interest
    c_km = partition[km].data();
    c_kp = partition[kp].data();
    
    // create the ceding subset
    reshuffle_partition(c_km,n_km,model.ns(km),m);
    
    // check that the starting set is not empty
    if(n_km > 0 ){
      
      // create the receiving subset
      reshuffle_partition(c_kp,n_kp,model.ns(kp),m-1);
      
      // COMPUTE THE SAMPLING WEIGHTS
      
      // initialize the weights with the constant part
      w_const = std::log(model.alpha(kp) + model.ns(kp)) - 
        std::log(model.alpha(km) + model.ns(km) - 1.0);
      
      // loop over each statistical unit
      for(unsigned int j = 0; j < n_km; j++){
        
        // set the current statistical unit
        model.set(c_km[j]);
        
        // initialize the weights
        w(j) = w_const + model.log_pred(kp);
        
        // remove the statistica unit from the ceding group
        model.update_par(km,-1.0);
        
        // compute the log predictive
        w(j) -= model.log_pred(km);
        
        // put that back inside
        model.update_par(km,1.0);
        
      }
      
      // update the number of predictive evaluation
      n_lpc += n_km;
      
      // sample one from this set given the weights
      idx = gumbel_max(w.subvec(0,n_km-1));
      i = c_km[idx];
      
      // set the current value for the statistical unit
      model.set(i);
      
      // METROPOLIS HASTINGS STEP
      
      // initialize the log acceptance probability as the log sum exp
      // of the proposal weights plus the contribute from the index proposal
      log_alpha = log_sum_exp_g(w.subvec(0,n_km-1),g) +
        std::log(std::max(model.ns(km), static_cast<double>(m))) - 
        std::log(std::max(model.ns(kp) + 1, static_cast<double>(m) ));
      
      // subtract the contributes of y_i from km parameters
      // of the predictive distribution and its sufficient statistics
      model.update_par(km,-1.0);
      
      // give it to the kps
      model.update_par(kp,1.0);
      
      // recompute the new weights
      w(n_kp) = -w(idx);
      
      // loop over each statistical unit
      for(unsigned int j = 0; j < n_kp; j++){
        
        // set the current statistical unit
        model.set(c_kp[j]);
        
        // initialize the weights
        w(j) = -w_const + model.log_pred(km);
        
        // remove the statistica unit from the ceding group
        model.update_par(kp,-1.0);
        
        // compute the log predictive
        w(j) -= model.log_pred(kp);
        
        // put that back inside
        model.update_par(kp,1.0);
        
      }
      
      // update the number of predictive evaluation
      n_lpc += n_kp;
      
      // set the current value for the statistical unit
      model.set(i);
      
      // subtract them in the log probability
      log_alpha -= log_sum_exp_g(w.subvec(0,n_kp),g);
      
      // report the probability on its scale
      accept_prob = std::min(1.0,std::exp(log_alpha));
      
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
        
        // update the unnormalized log posterior
        log_post -= w(n_kp);
        
      }else{
        
        // leave the kp components as they are, reset the kms
        model.update_par(kp,-1.0);
        model.update_par(km,1.0);
        
      }
      
      // update the rate counter
      accept_rate += accept_prob;
      
    }
    
  }
  
}

// reversible sampler for infinite mixtures
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                 std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                 std::vector<unsigned int>& labels,
                 double& accept_rate,
                 double& log_post,
                 unsigned int& n_lpc,
                 unsigned int& K_atm,
                 const unsigned int& thin){
  
  // set the index to move and the corresponding atoms;
  unsigned int i = 0, idx = 0, km = 0, kp = 0;
  
  // initialize labels
  unsigned int old_lbl = 0, new_lbl = 0;
  
  // initilize the acceptance probability and the log metropolis ratio
  double accept_prob = 0.0;
  double log_m_ratio = 0.0;
  
  // lazyness binary variable
  bool lazy = false;
  
  // initialize the non-lazy counter
  unsigned int non_lazy_count = 0;
  
  // push the chain ahead thin times
  for(unsigned int tt = 0; tt < thin; tt++ ){
    
    // set the direction
    lazy = set_directions(old_lbl,km,kp,c,K_atm,lbl2atm,atm2lbl);
    
    // check that the starting set is not empty
    if(!lazy){
      
      // increase the number of non-lazy counts
      non_lazy_count++;
      
      // sample one observation from the km group
      idx = static_cast<unsigned int>(R::runif(0,partition[old_lbl].size()));
      i = partition[old_lbl][idx];
      
      // set the value of the observed variable
      model.set(i);
      
      // initialize the log metropolis ratio and the acceptance probability
      log_m_ratio = 0.0;
      accept_prob = 0.0;
      
      // METROPOLIS HASTINGS STEP
      
      // are we deleting any group?
      if(model.ns[km] > 1){
        // we are not deleting any group
        
        // subtract the contributes of y_i from km parameters
        // of the predictive distribution and its sufficient statistics
        model.update_par(km,-1.0);
        
        // are we creating a new group?
        if(km == kp){
          // yes, then we will use the prior predictive
          log_m_ratio = std::log(model.alpha + K_atm*model.delta) - 
            std::log(model.ns[km] - model.delta) + 
            model.log_pred_prior() - 
            model.log_pred(km);
          
          // compute the acceptance probability
          accept_prob = std::min(1.0, 
                                 std::exp(log_m_ratio + 
                                   std::log(model.ns[km] + 1.0)));
          
          // acceptance step
          if(arma::randu() < accept_prob){
            
            // get a new label
            new_lbl = labels.back();
            
            // erase it from the pool of unused ones
            labels.pop_back();
            
            // assign the new label
            c(i) = new_lbl;
            
            // add the new sufficient statistics (only the prior term)
            model.add_new_atom();
            
            // add the new pair in the maps
            lbl2atm[new_lbl] = K_atm;
            atm2lbl[K_atm] = new_lbl;
            
            // set the index to remove equal to the last on the km group
            partition[old_lbl][idx] = partition[old_lbl].back();
            
            // erase the last piece of the km group
            partition[old_lbl].pop_back();
            
            // create the new atom
            partition[new_lbl].push_back(i);
            
            // leave the km components as they are, update the kps
            model.update_par(K_atm,1.0);  
            
            // increase the current number of atoms
            K_atm++;
            
            // update the unnormalized log posterior
            log_post += log_m_ratio;
          }else{
            
            // leave the kp components as they are, reset the kms
            model.update_par(km,1.0);  
            
          }
          
        }else{
          // no, classic metropolis ratio
          log_m_ratio = std::log(model.ns[kp] - model.delta) - 
            std::log(model.ns[km] - model.delta) + 
            model.log_pred(kp) - 
            model.log_pred(km);
          
          // compute the acceptance probability
          accept_prob = std::min(1.0, 
                                 std::exp(log_m_ratio + 
                                   std::log(model.ns[km] + 1.0) - 
                                   std::log(model.ns[kp] + 1.0)));
          
          // acceptance step
          if(arma::randu() < accept_prob){
            
            // update the partition
            
            // get the new label
            new_lbl = atm2lbl[kp];
            
            // set the index to remove equal to the last on the km group
            partition[old_lbl][idx] = partition[old_lbl].back();
            
            // erase the last piece of the km group
            partition[old_lbl].pop_back();
            
            // add to the kp group the new index
            partition[new_lbl].push_back(i);
            
            // add the new cluster lufc to the log posteriori
            log_post += log_m_ratio;
            
            // update the new sufficient statistics
            // by adding the i-th observation
            model.update_par(kp,1.0);
            
            // update the configuration vector
            c(i) = new_lbl;
            
          }else{
            
            // leave the kp components as they are, reset the kms
            model.update_par(km,1.0);  
            
          }
        }
        
      }else{
        // we are deleting the ceding group
        
        // are we creating a new group?
        if(km != kp){
          // no, otherwise we will immediately accept the proposal,
          // here we use the prior predictive at the denominator
          log_m_ratio = std::log(model.ns[kp] - model.delta) - 
            std::log(model.alpha + (K_atm-1.0)*model.delta) + 
            model.log_pred(kp) - 
            model.log_pred_prior();
          
          // compute the acceptance probability
          accept_prob = std::min(1.0, 
                                 std::exp(log_m_ratio -
                                   std::log(model.ns[kp] + 1.0)));
          
          // acceptance step
          if(arma::randu() < accept_prob){
            
            // decrease the number of atoms
            K_atm--;
            
            // get the new label
            new_lbl = atm2lbl[kp];
            
            // update the configuration vector
            c(i) = new_lbl;
            
            // add the new cluster lufc to the log posteriori
            log_post += log_m_ratio;
            
            // delete the old sufficient statistic
            model.copy_atom(km,K_atm);
            model.delete_last_atom();
            
            // update the partition
            partition[old_lbl].pop_back();
            partition[new_lbl].push_back(i);
            
            // change the labelings
            
            // get the label of the last atom
            new_lbl = atm2lbl[K_atm];
            
            // replace the label of the old atom with the one of the last one 
            atm2lbl[km] = new_lbl;
            
            // delete the last atom
            atm2lbl.erase(K_atm);
            
            // replace the atom of the new label with the old atom
            lbl2atm[new_lbl] = km;
            
            // delete the label associated with the old atom
            lbl2atm.erase(old_lbl);
            
            // append the old label to the list of possible labels
            labels.push_back(old_lbl);
            
            // if the new atom is the last, set it to be the as the old
            if(kp == K_atm){
              kp = km;
            }
            
            // update the new sufficient statistics
            // by adding the i-th observation
            model.update_par(kp,1.0);
            
          }
          
        }else{
          accept_prob = 1.0;
        }
        
      }
      
      // update the number of predictive evaluation
      n_lpc += 2;
      
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
                 double& log_post,
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
        log_post += log_m_ratio;
        
      }else{
        
        // leave the kp components as they are, reset the kms
        model.update_par(km,1.0);
        
        // flip the velocity
        vel ^= 1;
        
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

// informed version
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 unsigned int* c_km,
                 unsigned int* c_kp,
                 arma::vec& w,
                 std::vector<std::uint8_t>& dir,
                 double& accept_rate,
                 unsigned int& max_ex,
                 unsigned int& ex_count,
                 double& log_post,
                 unsigned int& n_lpc,
                 const double& theta,
                 const unsigned int K,
                 const unsigned int& thin,
                 const unsigned int& n,
                 const unsigned int& m,
                 const unsigned int& g){
  
  // set the index to move and its ceding and receiving groups;
  unsigned int i = 0, idx = 0, idx_dir = 0, km = 0, kp = 0;
  
  // initialize the size of the ceding and receiving groups
  unsigned int n_km = 0, n_kp = 0;
  
  // initialize the log metropolis ratio and the acceptance probability
  double log_alpha = 0.0, accept_prob = 0.0;
  
  // constant part of the weights
  double w_const = 0.0;
  
  // initialize the velocity
  std::uint8_t vel;
  
  // push the chain ahead thin times
  for(unsigned int tt = 0; tt < thin; tt++ ){
    
    // perturb the direction randomly
    perturb_momentum(dir,theta,K);
    
    // set the direction
    set_directions(km,kp,vel,dir,idx_dir,n,c,K);
    
    // set the groups of interest
    c_km = partition[km].data();
    c_kp = partition[kp].data();
    
    // create the ceding subset
    reshuffle_partition(c_km,n_km,model.ns(km),m);
    
    // check that the starting set is not empty
    if(n_km > 0 ){
      
      // create the receiving subset
      reshuffle_partition(c_kp,n_kp,model.ns(kp),m-1);
      
      // COMPUTE THE SAMPLING WEIGHTS
      
      // initialize the weights with the constant part
      w_const = std::log(model.alpha(kp) + model.ns(kp)) - 
        std::log(model.alpha(km) + model.ns(km) - 1.0);
      
      // loop over each statistical unit
      for(unsigned int j = 0; j < n_km; j++){
        
        // set the current statistical unit
        model.set(c_km[j]);
        
        // initialize the weights
        w(j) = w_const + model.log_pred(kp);
        
        // remove the statistica unit from the ceding group
        model.update_par(km,-1.0);
        
        // compute the log predictive
        w(j) -= model.log_pred(km);
        
        // put that back inside
        model.update_par(km,1.0);
        
      }
      
      // update the number of predictive evaluation
      n_lpc += n_km;
      
      // sample one from this set given the weights
      idx = gumbel_max(w.subvec(0,n_km-1));
      i = c_km[idx];
      
      // set the current value for the statistical unit
      model.set(i);
      
      // METROPOLIS HASTINGS STEP
      
      // initialize the log acceptance probability as the log sum exp
      // of the proposal weights plus the contribute from the index proposal
      log_alpha = log_sum_exp_g(w.subvec(0,n_km-1),g) +
        std::log(std::max(model.ns(km), static_cast<double>(m))) - 
        std::log(std::max(model.ns(kp) + 1, static_cast<double>(m) ));
      
      // subtract the contributes of y_i from km parameters
      // of the predictive distribution and its sufficient statistics
      model.update_par(km,-1.0);
      
      // give it to the kps
      model.update_par(kp,1.0);
      
      // recompute the new weights
      w(n_kp) = -w(idx);
      
      // loop over each statistical unit
      for(unsigned int j = 0; j < n_kp; j++){
        
        // set the current statistical unit
        model.set(c_kp[j]);
        
        // initialize the weights
        w(j) = -w_const + model.log_pred(km);
        
        // remove the statistica unit from the ceding group
        model.update_par(kp,-1.0);
        
        // compute the log predictive
        w(j) -= model.log_pred(kp);
        
        // put that back inside
        model.update_par(kp,1.0);
        
      }
      
      // update the number of predictive evaluation
      n_lpc += n_kp;
      
      // set the current value for the statistical unit
      model.set(i);
      
      // subtract them in the log probability
      log_alpha -= log_sum_exp_g(w.subvec(0,n_kp),g);
      
      // report the probability on its scale
      accept_prob = std::min(1.0,std::exp(log_alpha));
      
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
        
        // update the unnormalized log posterior
        log_post -= w(n_kp);
        
      }else{
        
        // leave the kp components as they are, reset the kms
        model.update_par(kp,-1.0);
        model.update_par(km,1.0);
        
        // flip the velocity
        vel ^= 1;
        
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

// non-reversible variant for finite mixtures
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 std::vector<std::uint8_t>& dir,
                 double& accept_rate,
                 unsigned int& max_ex,
                 unsigned int& ex_count,
                 double& log_post,
                 unsigned int& n_lpc,
                 const double& theta,
                 const double& s,
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
  
  // initialize the direction pool
  arma::uvec direction_pool(K);
  for(unsigned int k = 0; k < K; k++){
    direction_pool(k) = k;
  }
  
  // initialize the geometric probability and length
  double p = 0.0;
  unsigned int t = 0;
  
  // push the chain ahead thin times
  for( unsigned int tt = 0; tt < thin; tt++ ){
    
    // perturb the direction randomly
    // perturb_momentum(dir,theta,K);
    
    // set the direction
    if(t == 0){
      
      // sample a new direction
      set_directions(km,kp,vel,dir,idx_dir,K,direction_pool,K);
      
      // compute the geometric probability
      p = s / ( model.ns(km) + model.ns(kp) );
      
      // sample the number of iterations to carry on in this direction
      t = static_cast<unsigned int>(std::log(arma::randu()) / std::log1p(-p));
    }else{
      
      // decrease the counter
      t--;
      
    }
    
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
        log_post += log_m_ratio;
        
      }else{
        
        // leave the kp components as they are, reset the kms
        model.update_par(km,1.0);
        
        // flip the velocity
        vel ^= 1;
        
      }
      
      // update the rate counter
      accept_rate += accept_prob;
      
    }else{
      
      // flip the velocity
      vel ^= 1;
      
    }
    
    // perturb the direction randomly
    // perturb_momentum(dir,theta,K);
    
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
      
      // flip the role
      unsigned int k_tmp = km;
      km = kp;
      kp = k_tmp;
      
    }else{
      
      // increse the excursion counter
      ex_count++;
      
    }
    
  }
  
}

// non-reversible sampler for infinite mixtures
template <typename Model>
void random_scan(Model& model,
                 std::vector<std::vector<unsigned int>>& partition,
                 arma::uvec& c,
                 std::vector<std::uint8_t>& dir,
                 std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                 std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                 std::vector<unsigned int>& labels,
                 double& accept_rate,
                 unsigned int& max_ex,
                 unsigned int& ex_count,
                 double& log_post,
                 unsigned int& n_lpc,
                 const double& theta,
                 unsigned int& K_atm,
                 const unsigned int& thin,
                 const unsigned int& n){
  
  // set the index to move and the corresponding atoms;
  unsigned int i = 0, idx = 0, idx_dir = 0, km = 0, kp = 0;
  
  // initialize labels
  unsigned int old_lbl = 0, new_lbl = 0;
  
  // initilize the acceptance probability and the log metropolis ratio
  double accept_prob = 0.0;
  double log_m_ratio = 0.0;
  
  // lazyness binary variable
  bool lazy = false;
  
  // initialize the velocity
  std::uint8_t vel = 0;
  
  // initialize the non-lazy counter
  unsigned int non_lazy_count = 0;
  
  // direction flip indicator
  bool flip = false;
  
  // push the chain ahead thin times
  for(unsigned int tt = 0; tt < thin; tt++ ){
    
    // set flip to false
    flip = false;
    
    // perturb the direction randomly
    perturb_momentum(dir,theta,K_atm);
    
    // set the direction
    lazy = set_directions(old_lbl,km,kp,vel,dir,idx_dir,c,K_atm,lbl2atm,atm2lbl);
    
    // check that the starting set is not empty
    if(!lazy){
      
      // increase the number of effective steps
      non_lazy_count++;
      
      // sample one observation from the km group
      idx = static_cast<unsigned int>(R::runif(0,partition[old_lbl].size()));
      i = partition[old_lbl][idx];
      
      // set the value of the observed variable
      model.set(i);
      
      // METROPOLIS HASTINGS STEP
      
      // set to zero the log metropolis ratio and the acceptance probability
      log_m_ratio = 0.0;
      accept_prob = 0.0;
      
      // are we deleting any group?
      if(model.ns[km] > 1){
        // we are not deleting any group
        
        // subtract the contributes of y_i from km parameters
        // of the predictive distribution and its sufficient statistics
        model.update_par(km,-1.0);
        
        // are we creating a new group?
        if(km == kp){
          // yes, then we will use the prior predictive
          log_m_ratio = std::log(model.alpha + K_atm*model.delta) - 
            std::log(model.ns[km] - model.delta) + 
            model.log_pred_prior() - 
            model.log_pred(km);
          
          // compute the acceptance probability
          accept_prob = std::min(1.0, 
                                 std::exp(log_m_ratio + 
                                   std::log(model.ns[km] + 1.0)));
          
          // acceptance step
          if(arma::randu() < accept_prob){
            
            // get a new label
            new_lbl = labels.back();
            
            // erase it from the pool of unused ones
            labels.pop_back();
            
            // assign the new label
            c(i) = new_lbl;
            
            // set the new velocities
            set_directions(dir,vel,idx_dir,old_lbl,new_lbl,K_atm,lbl2atm,atm2lbl);
            
            // add the new sufficient statistics (only the prior term)
            model.add_new_atom();
            
            // add the new pair in the maps
            lbl2atm[new_lbl] = K_atm;
            atm2lbl[K_atm] = new_lbl;
            
            // set the index to remove equal to the last on the km group
            partition[old_lbl][idx] = partition[old_lbl].back();
            
            // erase the last piece of the km group
            partition[old_lbl].pop_back();
            
            // create the new atom
            partition[new_lbl].push_back(i);
            
            // leave the km components as they are, update the kps
            model.update_par(K_atm,1.0);  
            
            // increase the current number of atoms
            K_atm++;
            
            // update the unnormalized log posterior
            log_post += log_m_ratio;
          }else{
            
            // leave the kp components as they are, reset the kms
            model.update_par(km,1.0); 
            
            // flip
            flip = true;
            
          }
          
        }else{
          // no, classic metropolis ratio
          log_m_ratio = std::log(model.ns[kp] - model.delta) - 
            std::log(model.ns[km] - model.delta) + 
            model.log_pred(kp) - 
            model.log_pred(km);
          
          // compute the acceptance probability
          accept_prob = std::min(1.0, 
                                 std::exp(log_m_ratio + 
                                   std::log(model.ns[km] + 1.0) - 
                                   std::log(model.ns[kp] + 1.0)));
          
          // acceptance step
          if(arma::randu() < accept_prob){
            
            // update the partition
            
            // get the new label
            new_lbl = atm2lbl[kp];
            
            // set the index to remove equal to the last on the km group
            partition[old_lbl][idx] = partition[old_lbl].back();
            
            // erase the last piece of the km group
            partition[old_lbl].pop_back();
            
            // add to the kp group the new index
            partition[new_lbl].push_back(i);
            
            // add the new cluster lufc to the log posteriori
            log_post += log_m_ratio;
            
            // update the new sufficient statistics
            // by adding the i-th observation
            model.update_par(kp,1.0);
            
            // update the configuration vector
            c(i) = new_lbl;
            
          }else{
            
            // leave the kp components as they are, reset the kms
            model.update_par(km,1.0); 
            
            // flip the velocity
            dir[idx_dir] ^= 1;
            
            // flip
            flip = true;
          
          }
          
        }
        
      }else{
        // we are deleting the ceding group
        
        // are we creating a new group?
        if(km != kp){
          // no, otherwise we will immediately accept the proposal,
          // here we use the prior predictive at the denominator
          log_m_ratio = std::log(model.ns[kp] - model.delta) - 
            std::log(model.alpha + (K_atm-1.0)*model.delta) + 
            model.log_pred(kp) - 
            model.log_pred_prior();
          
          // compute the acceptance probability
          accept_prob = std::min(1.0, 
                                 std::exp(log_m_ratio -
                                   std::log(model.ns[kp] + 1.0)));
          
          // acceptance step
          if(arma::randu() < accept_prob){
            
            // decrease the number of atoms
            K_atm--;
            
            // get the new label
            new_lbl = atm2lbl[kp];
            
            // update the configuration vector
            c(i) = new_lbl;
            
            // add the new cluster lufc to the log posteriori
            log_post += log_m_ratio;
            
            // delete the old sufficient statistic
            model.copy_atom(km,K_atm);
            model.delete_last_atom();             
            
            // update the partition
            partition[old_lbl].pop_back();
            partition[new_lbl].push_back(i);
            
            // change the labelings
            
            // get the label of the last atom
            new_lbl = atm2lbl[K_atm];
            
            // replace the label of the old atom with the one of the last one 
            atm2lbl[km] = new_lbl;
            
            // delete the last atom
            atm2lbl.erase(K_atm);
            
            // replace the atom of the new label with the old atom
            lbl2atm[new_lbl] = km;
            
            // delete the label associated with the old atom
            lbl2atm.erase(old_lbl);
            
            // append the old label to the list of possible labels
            labels.push_back(old_lbl);
            
            // if the new atom is the last, set it to be the as the old
            if(kp == K_atm){
              kp = km;
            }
            
            // update the new sufficient statistics
            // by adding the i-th observation
            model.update_par(kp,1.0);
            
          }else{
            
            // flip the velocity
            dir[idx_dir] ^= 1;
            
            // flip
            flip = true;
          }
          
        }else{
          accept_prob = 1.0;
        }
        
      }
      
      // update the number of predictive evaluation
      n_lpc += 2;
      
      // has the direction overall changed?
      if(flip){
        
        // check if this exceed the max_excursion counter
        if(ex_count > max_ex){
          max_ex = ex_count;
        }
        
        // reset the excursion counter
        ex_count = 1;
        
      }else{
        
        // increse the excursion counter
        ex_count++;
        
      }
      
      // update the rate counter
      accept_rate += accept_prob;
    }
    
    // perturb the direction randomly
    perturb_momentum(dir,theta,K_atm);
    
  }
  
}


// // function for both sequential allocation and restricted steps (split kind)
// template<typename Model>
// void restricted_steps_split(Model& model,
//                             arma::vec& log_prob_restricted,
//                             std::vector<std::vector<unsigned int>>& splitted_groups,
//                             double& log_trans_prob,
//                             const unsigned int& atm1,
//                             const unsigned int& K_atm,
//                             const unsigned int* c_k,
//                             const unsigned int& n_restricted_steps){
// 
//   // initialize the unit to move and its new group index
//   unsigned int k = 0, idx = 0;
//   
//   // define the group sizes
//   arma::uvec ns = arma::ones<arma::uvec>(2);
//   unsigned int n_swapped = 0;
// 
//   // initialize the transition probabilities
//   log_trans_prob = 0.0;
// 
//   // should we save them already in the sequential allocation step?
//   bool save_probs = n_restricted_steps == 0;
// 
//   // sequentially allocate the units in the splitted group
//   for(unsigned int h = 2; h < model.ns[atm1]; h++){
// 
//     // get the current observation
//     k = c_k[h];
//     model.set(k);
// 
//     // compute the probabilities
//     log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) +
//       model.log_pred(K_atm);
// 
//     log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) +
//       model.log_pred(K_atm+1);
// 
//     // normalize the two probabilities
//     log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
// 
//     // assign the unit to one of the two groups
//     idx = ( std::log(arma::randu()) < log_prob_restricted(0) ) ? 0 : 1;
// 
//     if(save_probs){
//       // cumulate the log transition probabilities
//       log_trans_prob += log_prob_restricted(idx);
//     }
// 
//     // update the restricted partition
//     splitted_groups[idx].push_back(k);
//     
//     // update the parameters
//     model.update_par(K_atm+idx,1.0);
// 
//   }
//   
//   // for each restricted step
//   for(unsigned int s = 0; s < n_restricted_steps; s++){
// 
//     // check if we have to save the probabilities
//     save_probs = s == n_restricted_steps-1;
// 
//     // get the group sizes
//     ns(0) = splitted_groups[0].size();
//     ns(1) = splitted_groups[1].size();
//     
//     // reset the swaps counter
//     n_swapped = 0;
//     
//     // for each group
//     for(unsigned int g = 0; g < 2; g++){
// 
//       // do a restricted Polya urn update
//       for(unsigned int h = 1; h < ns(g); h++){
// 
//         // get the current observation
//         k = splitted_groups[g][h];
//         model.set(k);
// 
//         // compute the probabilities
//         log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) +
//           model.log_pred(K_atm);
// 
//         log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) +
//           model.log_pred(K_atm+1);
// 
//         // normalize the two probabilities
//         log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
// 
//         // assign the unit to one of the two groups
//         idx = ( std::log(arma::randu()) < log_prob_restricted(0) ) ? 0 : 1;
// 
//         if(idx != g){
//           // update the parameters
//           model.update_par(K_atm + 1 - g,1.0);
//           model.update_par(K_atm + g,-1.0);
// 
//           // update the partition
//           splitted_groups[g][h] = splitted_groups[g].back();
//           splitted_groups[g].pop_back();
//           splitted_groups[1-g].push_back(k);
//           
//           if(g == 0){
//             
//             // don't increase the counter, but...
//             h--;
//             
//             // ...reduce the size
//             ns(0)--;
//             
//             // increase the count of values in the tail of the second group
//             // that have already been scanned
//             n_swapped++;
//             
//           }else{
//             
//             if(n_swapped == 0){
//               // if there are no more already scanned values in the tails
//               
//               // don't increase the counter but reduce the size
//               h--;
//               ns(1)--;
//               
//             }else{
//               
//               // reduce the number of already scanned values in the tails
//               n_swapped--;
//             }
//             
//           }
//           
//         }
// 
//         if(save_probs){
//           // cumulate the log transition probabilities
//           log_trans_prob += log_prob_restricted(idx);
//         }
// 
//       }
// 
//     }
// 
//   }
// 
// }
// 
// // function for both sequential allocation and restricted steps (merge kind)
// template<typename Model>
// void restricted_steps_merge(Model& model,
//                             arma::vec& log_prob_restricted,
//                             std::vector<std::vector<unsigned int>>& splitted_groups,
//                             std::vector<unsigned int>& merged_group,
//                             double& log_trans_prob,
//                             const unsigned int& lbl2,
//                             const arma::uvec& c,
//                             const unsigned int& K_atm,
//                             const unsigned int* c_k,
//                             const unsigned int& n_restricted_steps){
// 
//   // initialize the unit to move and its new group index
//   unsigned int k = 0, idx = 0;
//   
//   // define the group sizes
//   arma::uvec ns = arma::ones<arma::uvec>(2);
//   unsigned int n_swapped = 0;
// 
//   // initialize the transition probabilities
//   log_trans_prob = 0.0;
// 
//   // should we save them already in the sequential allocation step?
//   bool save_probs = n_restricted_steps == 0;
// 
//   // sequentially allocate the units in the splitted group
//   for(unsigned int h = 2; h < merged_group.size(); h++){
// 
//     // get the current observation
//     k = c_k[h];
//     model.set(k);
// 
//     // compute the probabilities
//     log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) +
//       model.log_pred(K_atm);
// 
//     log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) +
//       model.log_pred(K_atm+1);
// 
//     // normalize the two probabilities
//     log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
// 
//     if(save_probs){
//       // last iteration, we don't need to update the split partition
//       // but we need to save the probabilities
// 
//       // assign the unit to the known groups
//       idx = ( c(k) == lbl2 ) ? 0 : 1;
// 
//       // cumulate the log transition probabilities
//       log_trans_prob += log_prob_restricted(idx);
//     }else{
// 
//       // assign the unit to one of the two groups
//       idx = ( std::log(arma::randu()) < log_prob_restricted(0) ) ? 0 : 1;
// 
//       // update the restricted partition
//       splitted_groups[idx].push_back(k);
// 
//     }
// 
//     // update the parameters
//     model.update_par(K_atm+idx,1.0);
// 
//   }
// 
//   // for each restricted step
//   for(unsigned int s = 0; s < n_restricted_steps; s++){
//     
//     // get the group sizes
//     ns(0) = splitted_groups[0].size();
//     ns(1) = splitted_groups[1].size();
//     
//     // reset the swaps counter
//     n_swapped = 0;
// 
//     // check if we have to save the probabilities
//     save_probs = s == n_restricted_steps-1;
// 
//     // for each group
//     for(unsigned int g = 0; g < 2; g++){
// 
//       // do a restricted Polya urn update
//       for(unsigned int h = 1; h < ns(g); h++){
// 
//         // get the current observation
//         k = splitted_groups[g][h];
//         model.set(k);
// 
//         // compute the probabilities
//         log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) +
//           model.log_pred(K_atm);
// 
//         log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) +
//           model.log_pred(K_atm+1);
// 
//         // normalize the two probabilities
//         log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
// 
//         if(save_probs){
//           // last iteration, we don't need to update the split partition
//           // but we need to save the probabilities
// 
//           // assign the unit to the known groups
//           idx = ( c(k) == lbl2 ) ? 0 : 1;
// 
//           // cumulate the log transition probabilities
//           log_trans_prob += log_prob_restricted(idx);
// 
//         }else{
// 
//           // assign the unit to one of the two groups
//           idx = ( std::log(arma::randu()) < log_prob_restricted(0) ) ? 0 : 1;
// 
//         }
// 
//         if(idx != g){
//           // update the parameters
//           model.update_par(K_atm + 1 - g,1.0);
//           model.update_par(K_atm + g,-1.0);
// 
//           if(!save_probs){
//             // update the partition
//             splitted_groups[g][h] = splitted_groups[g].back();
//             splitted_groups[g].pop_back();
//             splitted_groups[1-g].push_back(k);
//             
//             if(g == 0){
//               
//               // don't increase the counter, but...
//               h--;
//               
//               // ...reduce the size
//               ns(0)--;
//               
//               // increase the count of values in the tail of the second group
//               // that have already been scanned
//               n_swapped++;
//               
//             }else{
//               
//               if(n_swapped == 0){
//                 // if there are no more already scanned values in the tails
//                 
//                 // don't increase the counter but reduce the size
//                 h--;
//                 ns(1)--;
//                 
//               }else{
//                 
//                 // reduce the number of already scanned values in the tails
//                 n_swapped--;
//               }
//               
//             }
//           }
// 
//         }
// 
//       }
// 
//     }
// 
//   }
// 
// }

// function for both sequential allocation and restricted steps
template<typename Model>
void restricted_steps(Model& model,
                      arma::vec& log_prob_restricted,
                      std::vector<std::vector<unsigned int>>& splitted_groups,
                      std::vector<unsigned int>& merged_group,
                      double& log_trans_prob,
                      const unsigned int& atm1,
                      const unsigned int& lbl2,
                      const arma::uvec& c,
                      const unsigned int& K_atm,
                      const unsigned int* c_k,
                      const unsigned int& n_restricted_steps,
                      bool split){
  
  // initialize the unit to move and its new group index
  unsigned int k = 0, idx = 0;
  
  // define the group sizes
  arma::uvec ns = arma::ones<arma::uvec>(2);
  unsigned int n_swapped = 0;
  
  // initialize the transition probabilities
  log_trans_prob = 0.0;
  
  // should we save them already in the sequential allocation step?
  bool save_probs = n_restricted_steps == 0;
  
  unsigned int n;
  if(split){
    n = model.ns[atm1];
  }else{
    n = merged_group.size();
  }
  
  // sequentially allocate the units in the splitted group
  for(unsigned int h = 2; h < n; h++){
    
    // get the current observation
    k = c_k[h];
    model.set(k);
    
    // compute the probabilities
    log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) + 
      model.log_pred(K_atm);
    
    log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) + 
      model.log_pred(K_atm+1);
    
    // normalize the two probabilities
    log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
    
    if(!split && save_probs){
      
      // last iteration, we don't need to update the split partition
      // but we need to save the probabilities 
      
      // assign the unit to the known groups
      idx = ( c(k) == lbl2 ) ? 0 : 1;
      
    }else{
      
      // assign the unit to one of the two groups
      idx = ( std::log(arma::randu()) < log_prob_restricted(0) ) ? 0 : 1;
      
      // update the restricted partition
      splitted_groups[idx].push_back(k);
      
    }
    
    if(save_probs){
      // cumulate the log transition probabilities
      log_trans_prob += log_prob_restricted(idx);
    }
    
    // update the parameters
    model.update_par(K_atm+idx,1.0);
    
  }
  
  // for each restricted step
  for(unsigned int s = 0; s < n_restricted_steps; s++){
    
    // get the group sizes
    ns(0) = splitted_groups[0].size();
    ns(1) = splitted_groups[1].size();
    
    // reset the swaps counter
    n_swapped = 0;
    
    // check if we have to save the probabilities
    save_probs = s == n_restricted_steps-1;
    
    // for each group
    for(unsigned int g = 0; g < 2; g++){
      
      // do a restricted Polya urn update
      for(unsigned int h = 1; h < ns(g); h++){
        
        // get the current observation
        k = splitted_groups[g][h];
        model.set(k);
        
        // compute the probabilities
        log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) + 
          model.log_pred(K_atm);
        
        log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) + 
          model.log_pred(K_atm+1);
        
        // normalize the two probabilities
        log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
        
        if(!split && save_probs){
          
          // last iteration, we don't need to update the split partition
          // but we need to save the probabilities 
          
          // assign the unit to the known groups
          idx = ( c(k) == lbl2 ) ? 0 : 1;
          
        }else{
          
          // assign the unit to one of the two groups
          idx = ( std::log(arma::randu()) < log_prob_restricted(0) ) ? 0 : 1;
          
        }
        
        if(idx != g){
          // update the parameters
          model.update_par(K_atm + 1 - g,1.0);
          model.update_par(K_atm + g,-1.0);
          
          if(!save_probs || split){
            // update the partition
            splitted_groups[g][h] = splitted_groups[g].back();
            splitted_groups[g].pop_back();
            splitted_groups[1-g].push_back(k);
            
            if(g == 0){
              
              // don't increase the counter, but...
              h--;
              
              // ...reduce the size
              ns(0)--;
              
              // increase the count of values in the tail of the second group
              // that have already been scanned
              n_swapped++;
              
            }else{
              
              if(n_swapped == 0){
                // if there are no more already scanned values in the tails
                
                // don't increase the counter but reduce the size
                h--;
                ns(1)--;
                
              }else{
                
                // reduce the number of already scanned values in the tails
                n_swapped--;
              }
              
            }
          }
          
        }
        
        if(save_probs){
          // cumulate the log transition probabilities
          log_trans_prob += log_prob_restricted(idx);
        }
        
      }
      
    }
    
  }
  
}

// reversible case
template<typename Model>
void split_and_merge(Model& model,
                     std::vector<std::vector<unsigned int>>& partition,
                     arma::uvec& c,
                     std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                     std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                     std::vector<unsigned int>& labels,
                     const arma::vec& co_oc_tbl,
                     const arma::uvec& pair_alias,
                     bool use_alias,
                     const unsigned int& n_restricted_steps,
                     std::vector<std::vector<unsigned int>>& splitted_groups,
                     std::vector<unsigned int>& merged_group,
                     unsigned int* c_k,
                     unsigned int& K_atm,
                     double& log_post,
                     unsigned int& n_lpc,
                     double& accept_split,
                     double& accept_merge,
                     unsigned int& split_count,
                     const unsigned int& n){
  
  // initialize the log product of transition probabilities
  // and the log metropolis ratio
  double log_trans_prob = 0.0, log_m_ratio = 0.0;
  
  // initialize the restricted gibbs probabilities
  arma::vec log_prob_restricted = arma::zeros<arma::vec>(2);
  
  // initialize the atoms and labels
  unsigned int new_lbl = 0, i = 0, j = 0;
  unsigned int atm1 = 0, atm2 = 0, lbl1 = 0, lbl2 = 0;
  
  // sample two indexes
  if(use_alias){
    
    // sample a pair of indices
    unsigned idx = sample_alias(co_oc_tbl,pair_alias);
    
    // get the corresponding indexes
    i = static_cast<unsigned int>(0.5 * (1.0 + std::sqrt(1.0 + 8*idx)));
    j = idx - i*(i-1)/2;
    
  }else{
    
    // sample two indexes
    i = static_cast<unsigned int>(R::runif(0,n));
    j = static_cast<unsigned int>(R::runif(0,n-1));
    if(j >= i) j++; 
    
  }
  
  // get the corresponding labels
  lbl1 = c(i);
  lbl2 = c(j);
  
  // get the atoms
  atm1 = lbl2atm[lbl1];
  atm2 = lbl2atm[lbl2];
  
  // initialize the acceptance probability
  double accept_alpha = 0.0;
  
  // distinguish between the two cases
  if(atm1 == atm2){
    // split step
    
    // initialize the two new groups
    splitted_groups[1].push_back(i);
    splitted_groups[0].push_back(j);
    
    // reshuffle the group indexes by putting i and j at the first position
    c_k = partition[lbl1].data();
    shuffle_group(c_k,model.ns[atm1],i,j);
    
    // add two more atoms
    model.add_new_atom();
    model.add_new_atom();
    
    // fill them with the first two observations (i and j)
    model.set(j);
    model.update_par(K_atm,1.0);
    
    model.set(i);
    model.update_par(K_atm+1,1.0);
    
    // define the new label
    new_lbl = labels.back();
    
    // allocate the observation into the splitted groups
    // restricted_steps_split(model,log_prob_restricted,splitted_groups,
    //                        log_trans_prob,atm1,K_atm,c_k,n_restricted_steps);
    restricted_steps(model,log_prob_restricted,splitted_groups,merged_group,
                     log_trans_prob,atm1,lbl2,c,K_atm,c_k,n_restricted_steps,true);
    
    // compute the log Metropolis ratio
    log_m_ratio = model.log_marginal(K_atm) + 
      model.log_marginal(K_atm+1) - 
      model.log_marginal(atm1);
    
    // update the number of predictive (and marginal) evaluation
    n_lpc += 2 + (model.ns[atm1]-2)*(1+n_restricted_steps);
    
    // add the EPPF ratio term
    log_m_ratio += std::log(model.alpha + K_atm * model.delta) + 
      std::lgamma(model.ns[K_atm] - model.delta) + 
      std::lgamma(model.ns[K_atm+1] - model.delta) - 
      std::lgamma(1.0 - model.delta) - 
      std::lgamma(model.ns[atm1] - model.delta);
    
    // compute the acceptance probability
    accept_alpha = std::min(1.0, std::exp(log_m_ratio - log_trans_prob));
    // accept_split += accept_alpha;
    split_count++;
    
    // acceptance step
    if(arma::randu() < accept_alpha){
      
      accept_split++;
      
      // update the allocation vector
      c_k = splitted_groups[0].data();
      for(unsigned int h = 0; h < splitted_groups[0].size(); h++){
        c(c_k[h]) = new_lbl;
      }
      
      // rearrange the new atoms
      model.copy_atom(atm1,K_atm+1);
      model.delete_last_atom();
      
      // rearrange the partition
      partition[lbl1] = splitted_groups[1];
      partition[new_lbl] = splitted_groups[0];
      
      // remove the new label from the collection
      labels.pop_back();
      
      // add the new pair in the maps
      lbl2atm[new_lbl] = K_atm;
      atm2lbl[K_atm] = new_lbl;
      
      // increase the number of atoms
      K_atm++;
      
      // update the log posterior
      log_post += log_m_ratio;
      
    }else{
      
      // delete the new atoms
      model.delete_last_atom();
      model.delete_last_atom();
      
    }
    
    // clear the groups memory
    splitted_groups[0].clear();
    splitted_groups[1].clear();
    
  }else{
    // merge step
    
    // add two new atoms
    model.add_new_atom();
    model.add_new_atom();
    
    if(n_restricted_steps > 0){
      // initialize the two new groups
      splitted_groups[1].push_back(i);
      splitted_groups[0].push_back(j);
    }
    
    // add to those atoms the i-th and j-th observations
    model.set(j);
    model.update_par(K_atm,1.0);
    
    model.set(i);
    model.update_par(K_atm+1,1.0);
    
    // copy the two groups into a single one
    merged_group = partition[lbl1];
    merged_group.insert(merged_group.end(),
                        partition[lbl2].begin(),
                        partition[lbl2].end());
    
    // reshuffle the group indexes
    c_k = merged_group.data();
    shuffle_group(c_k,merged_group.size(),i,j);
    
    // create the launching state and cumulate the transition
    // probabilities to reach the original one from it
    // restricted_steps_merge(model,log_prob_restricted,splitted_groups,merged_group,
    //                        log_trans_prob,lbl2,c,K_atm,c_k,n_restricted_steps);
    restricted_steps(model,log_prob_restricted,splitted_groups,merged_group,
                     log_trans_prob,atm1,lbl2,c,K_atm,c_k,n_restricted_steps,false);
    
    // erase the new atoms
    model.delete_last_atom();
    model.delete_last_atom();
    
    // add the new merged atom
    model.add_new_atom();
    model.merge_atoms(atm1,atm2);
    
    // compute log metropolis ratio
    log_m_ratio = model.log_marginal(K_atm) - 
      model.log_marginal(atm1) - 
      model.log_marginal(atm2);
    
    // update the number of predictive (and marginal) evaluation
    n_lpc += 2 + (model.ns[K_atm]-2)*(1+n_restricted_steps);
    
    // add the EPPF ratio term
    log_m_ratio -= std::log(model.alpha + (K_atm-1) * model.delta) + 
      std::lgamma(model.ns[atm1] - model.delta) + 
      std::lgamma(model.ns[atm2] - model.delta) - 
      std::lgamma(1.0 - model.delta) - 
      std::lgamma(model.ns[K_atm] - model.delta);
    
    // compute the acceptance probability
    accept_alpha = std::min(1.0, std::exp( log_m_ratio + log_trans_prob ));
    // accept_merge += accept_alpha;
    
    // acceptance step
    if( arma::randu() < accept_alpha ){
      
      accept_merge++;
      
      // update the allocation vector
      c_k = partition[lbl2].data();
      for(unsigned int h = 0; h < partition[lbl2].size(); h++){
        c(c_k[h]) = lbl1;
      }
      
      // rearrange the new atoms in the first one
      model.copy_atom(atm1,K_atm);
      model.delete_last_atom();
      
      // decrease the number of atoms
      K_atm--;
      
      // erase the second atom
      model.copy_atom(atm2,K_atm);
      model.delete_last_atom();
      
      // rearrange the partition
      partition[lbl1] = merged_group;
      partition[lbl2].clear();
      
      // change the labeling
      
      // get the label of the last atom
      new_lbl = atm2lbl[K_atm];
      
      // set the label of the second atom that have been merged 
      // equal to the label of the last atom
      atm2lbl[atm2] = new_lbl;
      
      // erase the label of the second atom that have been merged
      lbl2atm.erase(lbl2);
      
      // append the old label to the list of possible labels
      labels.push_back(lbl2);
      
      // set the atom corresponding to the previous last atom equal
      // to the one of the erased group
      lbl2atm[new_lbl] = atm2;
      
      // erase the last atom
      atm2lbl.erase(K_atm);
      
      // update the log posterior
      log_post += log_m_ratio;
      
    }else{
      
      // delete the new atom
      model.delete_last_atom();
      
    }
    
    // clear the merged group memory
    merged_group.clear();
    
    if(n_restricted_steps > 0){
      // clear the splitted groups memory
      splitted_groups[0].clear();
      splitted_groups[1].clear();
    }
    
  }
  
}

// with aliasing table
// reversible case
template<typename Model>
void split_and_merge2(Model& model,
                      std::vector<std::vector<unsigned int>>& partition,
                      arma::uvec& c,
                      std::unordered_map<unsigned int, unsigned int>& lbl2atm,
                      std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                      std::vector<unsigned int>& labels,
                      const arma::vec& co_oc_tbl,
                      const arma::uvec& pair_alias,
                      std::vector<std::vector<unsigned int>>& splitted_groups,
                      std::vector<unsigned int>& merged_group,
                      unsigned int* c_k,
                      unsigned int& K_atm,
                      double& log_post,
                      double& accept_split,
                      double& accept_merge,
                      unsigned int& split_count,
                      const unsigned int& n){
  
  // initialize the log product of transition probabilities
  // and the log metropolis ratio
  double log_trans_prob = 0.0, log_m_ratio = 0.0;
  
  // initialize the restricted gibbs probabilities
  arma::vec log_prob_restricted = arma::zeros<arma::vec>(2);
  
  // initialize the atoms and labels
  unsigned int new_lbl = 0;
  unsigned int atm1 = 0, atm2 = 0, lbl1 = 0, lbl2 = 0, idx = 0;
  
  // initialize the index in the restricted Gibbs step
  unsigned int k = 0;
  
  // sample a pair of indices
  idx = sample_alias(co_oc_tbl,pair_alias);
  
  // get the corresponding indexes
  unsigned int i = static_cast<unsigned int>(0.5 * (1.0 + std::sqrt(1.0 + 8*idx)));
  unsigned int j = idx - i*(i-1)/2;
  
  // get the corresponding labels
  lbl1 = c(i);
  lbl2 = c(j);
  
  // get the atoms
  atm1 = lbl2atm[lbl1];
  atm2 = lbl2atm[lbl2];
  
  // initialize the acceptance probability
  double accept_alpha;
  
  // distinguish between the two cases
  if(atm1 == atm2){
    // split step
    
    // initialize the two new groups
    splitted_groups[1].push_back(i);
    splitted_groups[0].push_back(j);
    
    // reshuffle the group indexes by putting i and j at the first position
    c_k = partition[lbl1].data();
    shuffle_group(c_k,model.ns[atm1],i,j);
    
    // add two more atoms
    model.add_new_atom();
    model.add_new_atom();
    
    // fill them with the first two observations (i and j)
    model.set(j);
    model.update_par(K_atm,1.0);
    
    model.set(i);
    model.update_par(K_atm+1,1.0);
    
    // define the new label
    new_lbl = labels.back();
    
    // do a restricted Polya urn update
    for(unsigned int h = 2; h < model.ns[atm1]; h++){
      
      // get the current observation
      k = c_k[h];
      model.set(k);
      
      // compute the probabilities
      log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) + 
        model.log_pred(K_atm);
      
      log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) + 
        model.log_pred(K_atm+1);
      
      // normalize the two probabilities
      log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
      
      // assign the unit to one of the two groups
      idx = ( std::log(arma::randu()) < log_prob_restricted(0) ) ? 0 : 1;
      
      // update the parameters
      model.update_par(K_atm+idx,1.0);
      
      // cumulate the log transition probabilities
      log_trans_prob += log_prob_restricted(idx);
      
      // update the restricted partition
      splitted_groups[idx].push_back(k);
      
    }
    
    // compute the log Metropolis ratio
    log_m_ratio = model.log_marginal(K_atm) + 
      model.log_marginal(K_atm+1) - 
      model.log_marginal(atm1);
    
    // add the EPPF ratio term
    log_m_ratio += std::log(model.alpha + K_atm * model.delta) + 
      std::lgamma(model.ns[K_atm] - model.delta) + 
      std::lgamma(model.ns[K_atm+1] - model.delta) - 
      std::lgamma(1.0 - model.delta) - 
      std::lgamma(model.ns[atm1] - model.delta);
    
    // compute the acceptance probability
    accept_alpha = std::min(1.0, std::exp(log_m_ratio - log_trans_prob));
    accept_split += accept_alpha;
    split_count++;
    
    // acceptance step
    if(arma::randu() < accept_alpha){
      
      // update the allocation vector
      c_k = splitted_groups[0].data();
      for(unsigned int h = 0; h < splitted_groups[0].size(); h++){
        c(c_k[h]) = new_lbl;
      }
      
      // rearrange the new atoms
      model.copy_atom(atm1,K_atm+1);
      model.delete_last_atom();
      
      // rearrange the partition
      partition[lbl1] = splitted_groups[1];
      partition[new_lbl] = splitted_groups[0];
      
      // remove the new label from the collection
      labels.pop_back();
      
      // add the new pair in the maps
      lbl2atm[new_lbl] = K_atm;
      atm2lbl[K_atm] = new_lbl;
      
      // increase the number of atoms
      K_atm++;
      
      // update the log posterior
      log_post += log_m_ratio;
      
    }else{
      
      // delete the new atoms
      model.delete_last_atom();
      model.delete_last_atom();
      
    }
    
    // clear the groups memory
    splitted_groups[0].clear();
    splitted_groups[1].clear();
    
  }else{
    // merge step
    
    // add two new atoms
    model.add_new_atom();
    model.add_new_atom();
    
    // add to those atoms the i-th and j-th observations
    model.set(j);
    model.update_par(K_atm,1.0);
    
    model.set(i);
    model.update_par(K_atm+1,1.0);
    
    // copy the two groups into a single one
    merged_group = partition[lbl1];
    merged_group.insert(merged_group.end(),
                        partition[lbl2].begin(),
                        partition[lbl2].end());
    
    // reshuffle the group indexes
    c_k = merged_group.data();
    shuffle_group(c_k,merged_group.size(),i,j);
    
    // compute the restricted Polya urn updates probabilities
    for(unsigned int h = 2; h < merged_group.size(); h++){
      
      // get the current observation
      k = c_k[h];
      model.set(k);
      
      // compute the probabilities
      log_prob_restricted(0) = std::log(model.ns[K_atm] - model.delta) + 
        model.log_pred(K_atm);
      
      log_prob_restricted(1) = std::log(model.ns[K_atm+1] - model.delta) + 
        model.log_pred(K_atm+1);
      
      // normalize the two probabilities
      log_prob_restricted = log_prob_restricted - log_sum_exp(log_prob_restricted);
      
      // assign the unit to the known groups
      idx = ( c(k) == lbl2 ) ? 0 : 1;
      
      // update the parameters
      model.update_par(K_atm+idx,1.0);
      
      // cumulate the log transition probabilities
      log_trans_prob += log_prob_restricted(idx);
      
    }
    
    // erase the new atoms
    model.delete_last_atom();
    model.delete_last_atom();
    
    // add the new merged atom
    model.add_new_atom();
    model.merge_atoms(atm1,atm2);
    
    // compute log metropolis ratio
    log_m_ratio = model.log_marginal(K_atm) - 
      model.log_marginal(atm1) - 
      model.log_marginal(atm2);
    
    // add the EPPF ratio term
    log_m_ratio -= std::log(model.alpha + (K_atm-1) * model.delta) + 
      std::lgamma(model.ns[atm1] - model.delta) + 
      std::lgamma(model.ns[atm2] - model.delta) - 
      std::lgamma(1.0 - model.delta) - 
      std::lgamma(model.ns[K_atm] - model.delta);
    
    // compute the acceptance probability
    accept_alpha = std::min(1.0, std::exp( log_m_ratio + log_trans_prob ));
    accept_merge += accept_alpha;
    
    // acceptance step
    if( arma::randu() < accept_alpha ){
      
      // update the allocation vector
      c_k = partition[lbl2].data();
      for(unsigned int h = 0; h < partition[lbl2].size(); h++){
        c(c_k[h]) = lbl1;
      }
      
      // rearrange the new atoms in the first one
      model.copy_atom(atm1,K_atm);
      model.delete_last_atom();
      
      // decrease the number of atoms
      K_atm--;
      
      // erase the second atom
      model.copy_atom(atm2,K_atm);
      model.delete_last_atom();
      
      // rearrange the partition
      partition[lbl1] = merged_group;
      partition[lbl2].clear();
      
      // change the labeling
      
      // get the label of the last atom
      new_lbl = atm2lbl[K_atm];
      
      // set the label of the second atom that have been merged 
      // equal to the label of the last atom
      atm2lbl[atm2] = new_lbl;
      
      // erase the label of the second atom that have been merged
      lbl2atm.erase(lbl2);
      
      // append the old label to the list of possible labels
      labels.push_back(lbl2);
      
      // set the atom corresponding to the previous last atom equal
      // to the one of the erased group
      lbl2atm[new_lbl] = atm2;
      
      // erase the last atom
      atm2lbl.erase(K_atm);
      
      // update the log posterior
      log_post += log_m_ratio;
      
    }else{
      
      // delete the new atom
      model.delete_last_atom();
      
    }
    
    // clear the merged group memory
    merged_group.clear();
    
  }
  
}

// non-reversible case
// ...



// TEMPLATE FOR SAVING AND PRINTING FUNCTIONS

// finite mixture case
template <typename Model>
void save_and_print(Model& model,
                    unsigned int& conta,
                    arma::uvec& seq_idx,
                    arma::vec& log_p, arma::uvec& n_log_pred_calls,
                    arma::mat& W,
                    arma::vec& entropy,
                    arma::vec& alphas,
                    const unsigned int& t,
                    const double& refresh,
                    const unsigned int& warm_up,
                    const unsigned int& N,
                    const unsigned int& chain_id,
                    const arma::uvec& seq_idx_sampler,
                    const double& log_post,
                    const unsigned int& n_lpc,
                    const bool& save_configurations,
                    const bool& save_parameters,
                    const std::string& filename,
                    const arma::uvec& c){
  
  // if the warming up has ended
  if(t >= warm_up){
    
    // set the new value for the chain
    
    // log posterior value
    log_p(t-warm_up) = log_post;
    
    // number of predictive density call
    n_log_pred_calls(t-warm_up) = n_lpc;
    
    // save the prior check quantities
    W.row(t-warm_up) = rdirichlet(model.alpha + model.ns).t();
    
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
    
    // save the collapsed parameters?
    if(save_parameters){
      
      model.save_parameters(filename,chain_id);
      
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
      std::ofstream file_conf(filename + "/confs" + std::to_string(chain_id) + ".csv", std::ios::app);
      
      // save the configuration vector
      for(unsigned int i = 0; i < c.n_elem - 1; i++){
        file_conf << c(i) << ",";
      }
      
      // add the last element with an endline
      file_conf << c(c.n_elem-1) << "\n";
      
    }
    
  }
  
  //check if the console update condition is met
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

// infinite mixture case
template <typename Model>
void save_and_print(Model& model,
                    unsigned int& conta,
                    arma::uvec& seq_idx,
                    arma::vec& log_p, arma::uvec& n_log_pred_calls,
                    arma::uvec& K_atoms,
                    arma::vec& entropy,
                    arma::vec& alphas,
                    arma::vec& deltas,
                    const unsigned int& K_atm,
                    const unsigned int& t,
                    const double& refresh,
                    const unsigned int& warm_up,
                    const unsigned int& N,
                    const unsigned int& chain_id,
                    const arma::uvec& seq_idx_sampler,
                    const double& log_post,
                    const unsigned int& n_lpc,
                    const bool& save_configurations,
                    const bool& save_parameters,
                    const std::string& filename,
                    const arma::uvec& c){
  
  // if the warming up has ended
  if(t >= warm_up){
    
    // set the new value for the chain
    
    // log posterior value
    log_p(t-warm_up) = log_post;
    
    // number of predictive density call
    n_log_pred_calls(t-warm_up) = n_lpc;
    
    // save the prior check quantities
    K_atoms(t-warm_up) = K_atm;
    
    // save the entropy
    double ntrp = 0.0;
    for(unsigned int k = 0; k < K_atm; k++){
      ntrp -= model.ns[k] / model.n * std::log( model.ns[k] / model.n  );
    }
    entropy(t-warm_up) = ntrp;
    
    // save the hyperparameters?
    if(model.hyperpar_alpha){
      alphas(t-warm_up) = model.alpha;
    }
    
    if(model.hyperpar_delta){
      deltas(t-warm_up) = model.delta;
    }
    
    // save the collapsed parameters?
    if(save_parameters){
      
      model.save_parameters(filename,chain_id);
      
      // // save the hyperparameters?
      // if(model.hyperpar_alpha){
      //   // create the connection
      //   std::ofstream file_alpha(filename + "/alpha" + std::to_string(chain_id) + ".csv", std::ios::app);
      // 
      //   // save alpha
      //   file_alpha << model.alpha << "\n";
      // 
      // }
      // 
      // if(model.hyperpar_delta){
      //   // create the connection
      //   std::ofstream file_delta(filename + "/delta" + std::to_string(chain_id) + ".csv", std::ios::app);
      // 
      //   // save alpha
      //   file_delta << model.delta << "\n";
      // 
      // }
      
    }
    
    // save the configuration vector?
    if(save_configurations){
      
      // create the connection
      std::ofstream file_conf(filename + "/confs" + std::to_string(chain_id) + ".csv", std::ios::app);
      
      // save the configuration vector
      for(unsigned int i = 0; i < c.n_elem - 1; i++){
        file_conf << c(i) << ",";
      }
      
      // add the last element with an endline
      file_conf << c(c.n_elem-1) << "\n";
      
    }
    
  }
  
  //check if the console update condition is met
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

// TEMPLATE FOR GIBBS SAMPLING

// finite mixture case
template <typename Model>
void gibbs_sampler(Model& model,
                   arma::uvec& c,
                   arma::vec& log_p, arma::uvec& n_log_pred_calls,
                   arma::mat& W,
                   arma::vec& entropy,
                   arma::vec& alphas,
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
                   const std::string& filename){
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of clusters
  unsigned int K = model.K;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the probability vector
  arma::vec log_prob(K);
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      random_scan(model,c,log_prob,log_post,n_lpc,K,thin_scan,n);
      
      // update the hyperparameter
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
  }
  
}

// infinite mixture case
template <typename Model>
void gibbs_sampler(Model& model,
                   arma::uvec& c,
                   arma::vec& log_p, arma::uvec& n_log_pred_calls,
                   arma::uvec& K_atoms,
                   const unsigned int& N,
                   const unsigned int& warm_up,
                   const unsigned int& thin,
                   const unsigned int& thin_scan,
                   const double& refresh,
                   arma::vec& entropy,
                   arma::vec& alphas,
                   arma::vec& deltas,
                   const unsigned int& chain_id,
                   arma::uvec& seq_idx,
                   const arma::uvec& seq_idx_sampler,
                   const bool& save_configurations,
                   const bool& save_parameters,
                   const std::string& filename,
                   std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                   std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                   std::vector<unsigned int>& labels){
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of atm
  unsigned int K_atm = model.K_atm;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the probability vector
  std::vector<double> log_prob;
  
  // reserve n maximum entries
  log_prob.reserve(n);
  
  // initialize with only the current number of atoms plus one
  log_prob.resize(K_atm+1);
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // thinned iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      random_scan(model,c,log_prob,lbl2atm,atm2lbl,labels,log_post,n_lpc,K_atm,thin_scan,n);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
  }
  
}

// TEMPLATE FOR REVERSIBLE ALGORITHMS

// finite mixture case
template <typename Model>
void reversible_sampler(Model& model,
                        arma::uvec& c,
                        std::vector<std::vector<unsigned int>>& partition,
                        arma::vec& log_p, arma::uvec& n_log_pred_calls,
                        arma::mat& W,
                        arma::vec& entropy,
                        arma::vec& alphas,
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
                        const std::string& filename){
  
  // get the number of clusters
  unsigned int K = model.K;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the acceptance rate
  double accept_rate = 0.0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      random_scan(model,partition,c,accept_rate,log_post,n_lpc,K,thin_scan);
      
      // update the hyperparameter?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    if(t >= warm_up){
      acceptance_rates(t-warm_up,0) = accept_rate;
    }
    
  }
  
}

// infinite mixture case
template <typename Model>
void reversible_sampler(Model& model,
                        arma::uvec& c,
                        std::vector<std::vector<unsigned int>>& partition,
                        arma::vec& log_p, arma::uvec& n_log_pred_calls,
                        arma::uvec& K_atoms,
                        arma::mat& acceptance_rates,
                        arma::vec& entropy,
                        arma::vec& alphas,
                        arma::vec& deltas,
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
                        std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                        std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                        std::vector<unsigned int>& labels){
  
  // get the number of atoms
  unsigned int K_atm = model.K_atm;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the acceptance rate
  double accept_rate = 0.0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the acceptance rate
    accept_rate = 0.0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      random_scan(model,partition,c,lbl2atm,atm2lbl,labels,accept_rate,log_post,n_lpc,K_atm,thin_scan);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    if(t >= warm_up){
      acceptance_rates(t-warm_up,0) = accept_rate / thin / thin_scan;
    }
    
  }
  
}

// TEMPLATE FOR NON-REVERSIBLE ALGORITHM

// finite mixture case
template <typename Model>
void non_reversible_sampler(Model& model,
                            arma::uvec& c,
                            std::vector<std::vector<unsigned int>>& partition,
                            arma::vec& log_p, arma::uvec& n_log_pred_calls,
                            arma::mat& W,
                            arma::vec& entropy,
                            arma::vec& alphas,
                            arma::mat& acceptance_rates,
                            arma::uvec& max_excursions,
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
                            const std::string& filename){
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of clusters
  unsigned int K = model.K;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // initialize the orientation vector
  std::vector<std::uint8_t> dir(K*(K-1)/2);
  for(unsigned int i = 0; i < dir.size(); i++){
    dir[i] = (R::runif(0,1) < 0.5) ? 1 : 0;
  }
  
  // initialie the maximal excursion
  unsigned int max_ex = 0;
  
  //  initialize the excursion counter
  unsigned int ex_count = 1;
  
  // initialize the acceptance rate
  double accept_rate = 0.0;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the acceptance rate
    accept_rate = 0.0;
    
    // reset the max excursion counter
    max_ex = 0;
    
    // reset the excursion counter
    ex_count = 1;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      if(s != 0){
        
        // modified version
        random_scan(model,partition,c,dir,accept_rate,max_ex,ex_count,log_post,n_lpc,theta,s,K,thin_scan,n);
      }else{
        
        // classic non-reversible
        random_scan(model,partition,c,dir,accept_rate,max_ex,ex_count,log_post,n_lpc,theta,K,thin_scan,n);
      }
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    // save the maximum excursion occurred and the acceptance rate
    if(t >= warm_up){
      max_excursions(t-warm_up) = max_ex;
      acceptance_rates(t-warm_up,0) = accept_rate / thin / thin_scan;
    }
    
  }
  
}

// infinite mixture case
template <typename Model>
void non_reversible_sampler(Model& model,
                            arma::uvec& c,
                            std::vector<std::vector<unsigned int>>& partition,
                            arma::vec& log_p, arma::uvec& n_log_pred_calls,
                            arma::uvec& K_atoms,
                            arma::mat& acceptance_rates,
                            arma::vec& entropy,
                            arma::vec& alphas,
                            arma::vec& deltas,
                            arma::uvec& max_excursions,
                            const double& theta,
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
                            std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                            std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                            std::vector<unsigned int>& labels){
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of atoms
  unsigned int K_atm = model.K_atm;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // initialize the orientation vector
  std::vector<std::uint8_t> dir(n*(n-1)/2);
  for(unsigned int i = 0; i < dir.size(); i++){
    dir[i] = (R::runif(0,1) < 0.5) ? 1 : 0;
  }
  
  // initialie the maximal excursion
  unsigned int max_ex = 0;
  
  //  initialize the excursion counter
  unsigned int ex_count = 1;
  
  // initialize the acceptance rate
  double accept_rate = 0.0;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the acceptance rate
    accept_rate = 0.0;
    
    // reset the max excursion counter
    max_ex = 0;
    
    // reset the excursion counter
    ex_count = 1;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      random_scan(model,partition,c,dir,lbl2atm,atm2lbl,labels,
                  accept_rate,max_ex,ex_count,log_post,n_lpc,theta,K_atm,thin_scan,n);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    // save the maximum excursion occurred and the acceptance rate
    if(t >= warm_up){
      max_excursions(t-warm_up) = max_ex;
      acceptance_rates(t-warm_up,0) = accept_rate / thin / thin_scan;
    }
    
  }
  
}

// TEMPLATE FOR INFORMED REVERSIBLE SAMPLER

// finite mixture case
template <typename Model>
void reversible_informed_sampler(Model& model,
                                 arma::uvec& c,
                                 std::vector<std::vector<unsigned int>>& partition,
                                 arma::vec& log_p, arma::uvec& n_log_pred_calls,
                                 arma::mat& W,
                                 arma::vec& entropy,
                                 arma::vec& alphas,
                                 arma::mat& acceptance_rates,
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
                                 const std::string& filename){
  
  // get the number of clusters
  unsigned int K = model.K;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // initialize the vector of indexes for the ceding group
  auto* c_km = partition[0].data();
  
  // initialize the vector of indexes for the receiving group
  auto* c_kp = partition[0].data();
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // define the weights vectors
  arma::vec w(m);
  
  // initialize the acceptance rate
  double accept_rate = 0.0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the acceptance rate
    accept_rate = 0.0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      random_scan(model,partition,c,c_km,c_kp,w,accept_rate,log_post,n_lpc,K,thin_scan,m,g);
      
      // update the hyperparameter?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    // save the acceptance rate
    if(t >= warm_up){
      acceptance_rates(t-warm_up,0) = accept_rate / thin / thin_scan;
    }
    
  }
  
}

// TEMPLATE FOR INFORMED NON-REVERSIBLE SAMPLER

// finite mixture case
template <typename Model>
void non_reversible_informed_sampler(Model& model,
                                     arma::uvec& c,
                                     std::vector<std::vector<unsigned int>>& partition,
                                     arma::vec& log_p, arma::uvec& n_log_pred_calls,
                                     arma::mat& W,
                                     arma::vec& entropy,
                                     arma::vec& alphas,
                                     arma::mat& acceptance_rates,
                                     arma::uvec& max_excursions,
                                     const double& theta,
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
                                     const std::string& filename){
  
  // initialie the maximal excursion
  unsigned int max_ex = 0;
  
  // initialize the excursion counter
  unsigned int ex_count = 1;
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of clusters
  unsigned int K = model.K;
  
  // initialize the orientation vector
  std::vector<std::uint8_t> dir(K*(K-1)/2);
  for(unsigned int i = 0; i < dir.size(); i++){
    dir[i] = (R::runif(0,1) < 0.5) ? 1 : 0;
  }
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // initialize the vector of indexes for the ceding group
  auto* c_km = partition[0].data();
  
  // initialize the vector of indexes for the receiving group
  auto* c_kp = partition[0].data();
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // define the weights vectors
  arma::vec w(m);
  
  // initialize the acceptance rate
  double accept_rate = 0.0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // for loop
  for (unsigned int t = 0; t < (N + warm_up); t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the acceptance rate
    accept_rate = 0.0;
    
    // reset the max excursion counter
    max_ex = 0;
    
    // reset the excursion counter
    ex_count = 1;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // random scan
      random_scan(model,partition,c,c_km,c_kp,w,dir,
                  accept_rate,max_ex,ex_count,log_post,n_lpc,theta,K,thin_scan,n,m,g);
      
      // update the hyperparameter
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,W,entropy,alphas,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    // save the maximum excursion occurred and the acceptance rate
    if(t >= warm_up){
      max_excursions(t-warm_up) = max_ex;
      acceptance_rates(t-warm_up,0) = accept_rate / thin / thin_scan;
    }
    
  }
  
}

// SPLIT AND MERGE SAMPLER FOR INFINITE MIXTURE (COLLAPSED GIBBS)

// marginal sampler
template <typename Model>
void split_and_merge_gibbs(Model& model,
                           std::vector<std::vector<unsigned int>>& partition,
                           arma::uvec& c,
                           arma::vec& log_p, arma::uvec& n_log_pred_calls,
                           arma::uvec& K_atoms,
                           arma::mat& acceptance_rates,
                           arma::vec& entropy,
                           arma::vec& alphas,
                           arma::vec& deltas,
                           const unsigned int& N,
                           const unsigned int& warm_up,
                           const unsigned int& thin,
                           const unsigned int& thin_scan,
                           const unsigned int& thin_SAM,
                           const bool& NUSAMS,
                           const Rcpp::Nullable<Rcpp::NumericMatrix>& NU_weights,
                           arma::mat& NUW,
                           const bool& return_weights,
                           const unsigned int& n_restricted_steps,
                           const double& refresh,
                           const unsigned int& chain_id,
                           arma::uvec& seq_idx,
                           const arma::uvec& seq_idx_sampler,
                           const bool& save_configurations,
                           const bool& save_parameters,
                           const std::string& filename,
                           std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                           std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                           std::vector<unsigned int>& labels){
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of atm
  unsigned int K_atm = model.K_atm;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the probability vector
  std::vector<double> log_prob;
  
  // reserve n maximum entries
  log_prob.reserve(n);
  
  // initialize with only the current number of atoms plus one
  log_prob.resize(K_atm+1);
  
  // initialize the two splitted groups and the merged group
  std::vector<std::vector<unsigned int>> splitted_groups;
  splitted_groups.resize(2);
  splitted_groups[0].reserve(n);
  splitted_groups[1].reserve(n);
  
  std::vector<unsigned int> merged_group;
  merged_group.reserve(n);
  
  // initalize the pointer to the group to split
  auto* c_k = merged_group.data();
  
  // initialize the acceptance probabilities
  double accept_split = 0.0, accept_merge = 0.0;
  
  // initialize the unrolled co occurrence probability matrix
  // and the aliasing table
  arma::vec co_oc_tbl;
  arma::uvec pair_alias;
  
  // check the matrix is already present
  bool estimate_NU_weights = NU_weights.isNull();
  
  if(NUSAMS){
    co_oc_tbl = arma::ones<arma::vec>(n*(n-1)/2) * 1e-5;
    pair_alias = arma::zeros<arma::uvec>(n*(n-1)/2);
    
    if(!estimate_NU_weights){
      
      // copy it
      Rcpp::NumericMatrix NUW(NU_weights);
      
      // unrolled it into a vector
      unsigned int itr = 0;
      for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < i; j++){
          if(c(i) == c(j)){
            co_oc_tbl(itr) = NUW(i,j);
          }
          itr++;
        }
      }
    }
  }
  
  // initialize the split counter
  unsigned int split_count = 0;
  
  // initialize the number of log predictive density calls
  unsigned int n_lpc = 0;
  
  // warm up phase
  for(unsigned int t = 0; t < warm_up; t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // split and merge steps
      for(unsigned int ttt = 0; ttt < thin_SAM; ttt++){
        
        split_and_merge(model,partition,c,lbl2atm,atm2lbl,labels,
                        co_oc_tbl,pair_alias,false,n_restricted_steps,
                        splitted_groups,merged_group,c_k,
                        K_atm,log_post,n_lpc,accept_split,accept_merge,split_count,n);
        
        // update the length of the probability vector
        if(log_prob.size() < K_atm + 1){
          log_prob.push_back(0.0);
        }else if(log_prob.size() > K_atm + 1){
          log_prob.pop_back();
        }
        
      }
      
      // random scan step
      random_scan(model,partition,c,log_prob,lbl2atm,atm2lbl,labels,log_post,n_lpc,K_atm,thin_scan,n);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    if(NUSAMS && estimate_NU_weights){
      
      // update the co-occurrence probabilities
      unsigned int itr = 0;
      for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < i; j++){
          if(c(i) == c(j)){
            co_oc_tbl(itr)++;
          }
          itr++;
        }
      }
      
    }
    
  }
  
  if(warm_up > 0 && NUSAMS){
    
    if(estimate_NU_weights){
      
      // normalize the co occurrence probabilities
      co_oc_tbl /= warm_up;
      
      // save the weights?
      if(return_weights){
        unsigned int itr = 0;
        for(unsigned int i = 0; i < n; i++){
          for(unsigned int j = 0; j < i; j++){
            NUW(i,j) = co_oc_tbl(itr);
            itr++;
          }
        }
      }
      
      // compute the unnormalized log probability
      co_oc_tbl = 0.5 * ( arma::log(co_oc_tbl) + arma::log1p(-co_oc_tbl) );
      
    }
    
    // compute the aliasing table
    create_aliasing_table(co_oc_tbl,pair_alias);
    
  }
  
  // sampling phase
  for (unsigned int t = 0; t < N; t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the split and merge rates and the split count
    accept_split = 0.0;
    accept_merge = 0.0;
    split_count = 0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // split and merge step using the aliasing table
      for(unsigned int ttt = 0; ttt < thin_SAM; ttt++){
        
        // iteration
        split_and_merge(model,partition,c,lbl2atm,atm2lbl,labels,
                        co_oc_tbl,pair_alias,NUSAMS,n_restricted_steps,
                        splitted_groups,merged_group,c_k,
                        K_atm,log_post,n_lpc,accept_split,accept_merge,split_count,n);
        
        // update the length of the probability vector
        if(log_prob.size() < K_atm + 1){
          log_prob.push_back(0.0);
        }else if(log_prob.size() > K_atm + 1){
          log_prob.pop_back();
        }
        
      }
      
      // random scan step
      random_scan(model,partition,c,log_prob,lbl2atm,atm2lbl,labels,log_post,n_lpc,K_atm,thin_scan,n);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t+warm_up,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    acceptance_rates(t,0) = accept_split / thin_SAM / thin; // split_count;
    acceptance_rates(t,1) = accept_merge / thin_SAM / thin; // (thin_SAM - split_count);
    
  }
  
}

// reversible
template <typename Model>
void split_and_merge_reversible(Model& model,
                                std::vector<std::vector<unsigned int>>& partition,
                                arma::uvec& c,
                                arma::vec& log_p, arma::uvec& n_log_pred_calls,
                                arma::uvec& K_atoms,
                                arma::mat& acceptance_rates,
                                arma::vec& entropy,
                                arma::vec& alphas,
                                arma::vec& deltas,
                                const unsigned int& N,
                                const unsigned int& warm_up,
                                const unsigned int& thin,
                                const unsigned int& thin_scan,
                                const unsigned int& thin_SAM,
                                const bool& NUSAMS,
                                const Rcpp::Nullable<Rcpp::NumericMatrix>& NU_weights,
                                arma::mat& NUW,
                                const bool& return_weights,
                                const unsigned int& n_restricted_steps,
                                const double& refresh,
                                const unsigned int& chain_id,
                                arma::uvec& seq_idx,
                                const arma::uvec& seq_idx_sampler,
                                const bool& save_configurations,
                                const bool& save_parameters,
                                const std::string& filename,
                                std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                                std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                                std::vector<unsigned int>& labels){
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of atm
  unsigned int K_atm = model.K_atm;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the two splitted groups and the merged group
  std::vector<std::vector<unsigned int>> splitted_groups;
  splitted_groups.resize(2);
  splitted_groups[0].reserve(n);
  splitted_groups[1].reserve(n);
  
  std::vector<unsigned int> merged_group;
  merged_group.reserve(n);
  
  // initalize the pointer to the group to split
  auto* c_k = merged_group.data();
  
  // initialize the acceptance probabilities
  double accept_split = 0.0, accept_merge = 0.0, accept_rate = 0.0;
  
  // initialize the unrolled co occurrence probability matrix
  // and the aliasing table
  arma::vec co_oc_tbl;
  arma::uvec pair_alias;
  
  // check the matrix is already present
  bool estimate_NU_weights = NU_weights.isNull();
  
  if(NUSAMS){
    co_oc_tbl = arma::zeros<arma::vec>(n*(n-1)/2);
    pair_alias = arma::zeros<arma::uvec>(n*(n-1)/2);
    
    if(!estimate_NU_weights){
      
      // copy it
      Rcpp::NumericMatrix NUW(NU_weights);
      
      // unrolled it into a vector
      unsigned int itr = 0;
      for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < i; j++){
          if(c(i) == c(j)){
            co_oc_tbl(itr) = NUW(i,j);
          }
          itr++;
        }
      }
    }
  }
  
  // initialize the split counter
  unsigned int split_count = 0;
  
  // initialize the number of call to the predictive distribution
  unsigned int n_lpc = 0;
  
  // warm up phase
  for(unsigned int t = 0; t < warm_up; t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // split and merge steps
      for(unsigned int ttt = 0; ttt < thin_SAM; ttt++){
        
        split_and_merge(model,partition,c,lbl2atm,atm2lbl,labels,
                        co_oc_tbl,pair_alias,false,n_restricted_steps,
                        splitted_groups,merged_group,c_k,
                        K_atm,log_post,n_lpc,accept_split,accept_merge,split_count,n);
        
      }
      
      // random scan step
      random_scan(model,partition,c,lbl2atm,atm2lbl,labels,accept_rate,log_post,n_lpc,K_atm,thin_scan);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    if(NUSAMS && estimate_NU_weights){
      
      // update the co-occurrence probabilities
      unsigned int itr = 0;
      for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < i; j++){
          if(c(i) == c(j)){
            co_oc_tbl(itr)++;
          }
          itr++;
        }
      }
      
    }
    
  }
  
  if(warm_up > 0 && NUSAMS){
    
    if(estimate_NU_weights){
      
      // normalize the co occurrence probabilities
      co_oc_tbl /= warm_up;
      
      // save the weights?
      if(return_weights){
        unsigned int itr = 0;
        for(unsigned int i = 0; i < n; i++){
          for(unsigned int j = 0; j < i; j++){
            NUW(i,j) = co_oc_tbl(itr);
            itr++;
          }
        }
      }
      
      // compute the unnormalized log probability
      co_oc_tbl = 0.5 * ( arma::log(co_oc_tbl) + arma::log1p(-co_oc_tbl) );
      
    }
    
    // compute the aliasing table
    create_aliasing_table(co_oc_tbl,pair_alias);
    
  }
  
  // sampling phase
  for (unsigned int t = 0; t < N; t++){
    
    // reset the number of calls
    n_lpc = 0;
    
    // reset the split and merge rates and the split count
    accept_split = 0.0;
    accept_merge = 0.0;
    accept_rate = 0.0;
    split_count = 0;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // split and merge steps
      for(unsigned int ttt = 0; ttt < thin_SAM; ttt++){
        
        // iteration
        split_and_merge(model,partition,c,lbl2atm,atm2lbl,labels,
                        co_oc_tbl,pair_alias,NUSAMS,n_restricted_steps,
                        splitted_groups,merged_group,c_k,
                        K_atm,log_post,n_lpc,accept_split,accept_merge,split_count,n);
        
      }
      
      // random scan step
      random_scan(model,partition,c,lbl2atm,atm2lbl,labels,accept_rate,log_post,n_lpc,K_atm,thin_scan);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t+warm_up,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    acceptance_rates(t,0) = accept_split / thin_SAM / thin; // split_count;
    acceptance_rates(t,1) = accept_merge / thin_SAM / thin; // (thin_SAM - split_count);
    acceptance_rates(t,2) = accept_rate / thin_scan / thin;
    
  }
  
}

// non reversible
template <typename Model>
void split_and_merge_non_reversible(Model& model,
                                    std::vector<std::vector<unsigned int>>& partition,
                                    arma::uvec& c,
                                    arma::vec& log_p, arma::uvec& n_log_pred_calls,
                                    arma::uvec& K_atoms,
                                    arma::mat& acceptance_rates,
                                    arma::vec& entropy,
                                    arma::vec& alphas,
                                    arma::vec& deltas,
                                    arma::uvec& max_excursions,
                                    const double& theta,
                                    const unsigned int& N,
                                    const unsigned int& warm_up,
                                    const unsigned int& thin,
                                    const unsigned int& thin_scan,
                                    const unsigned int& thin_SAM,
                                    const bool& NUSAMS,
                                    const Rcpp::Nullable<Rcpp::NumericMatrix>& NU_weights,
                                    arma::mat& NUW,
                                    const bool& return_weights,
                                    const unsigned int& n_restricted_steps,
                                    const double& refresh,
                                    const unsigned int& chain_id,
                                    arma::uvec& seq_idx,
                                    const arma::uvec& seq_idx_sampler,
                                    const bool& save_configurations,
                                    const bool& save_parameters,
                                    const std::string& filename,
                                    std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                                    std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                                    std::vector<unsigned int>& labels){
  
  // get the sample size
  unsigned int n = c.n_elem;
  
  // get the number of atm
  unsigned int K_atm = model.K_atm;
  
  // initialize the log posterior
  double log_post = model.log_post;
  
  // initialize the orientation vector
  std::vector<std::uint8_t> dir(n*(n-1)/2);
  for(unsigned int i = 0; i < dir.size(); i++){
    dir[i] = (R::runif(0,1) < 0.5) ? 1 : 0;
  }
  
  // initialie the maximal excursion
  unsigned int max_ex = 0;
  
  // initialize the excursion counter
  unsigned int ex_count = 1;
  
  // and a count indicator to cycle between these
  unsigned int conta = 0;
  
  // initialize the two splitted groups and the merged group
  std::vector<std::vector<unsigned int>> splitted_groups;
  splitted_groups.resize(2);
  splitted_groups[0].reserve(n);
  splitted_groups[1].reserve(n);
  
  std::vector<unsigned int> merged_group;
  merged_group.reserve(n);
  
  // initalize the pointer to the group to split
  auto* c_k = merged_group.data();
  
  // initialize the acceptance probabilities
  double accept_rate = 0.0, accept_split = 0.0, accept_merge = 0.0;
  
  // initialize the unrolled co occurrence probability matrix
  // and the aliasing table
  arma::vec co_oc_tbl;
  arma::uvec pair_alias;
  
  // check the matrix is already present
  bool estimate_NU_weights = NU_weights.isNull();
  
  if(NUSAMS){
    co_oc_tbl = arma::zeros<arma::vec>(n*(n-1)/2);
    pair_alias = arma::zeros<arma::uvec>(n*(n-1)/2);
    
    if(!estimate_NU_weights){
      
      // copy it
      Rcpp::NumericMatrix NUW(NU_weights);
      
      // unrolled it into a vector
      unsigned int itr = 0;
      for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < i; j++){
          if(c(i) == c(j)){
            co_oc_tbl(itr) = NUW(i,j);
          }
          itr++;
        }
      }
    }
  }
  
  // initialize the split counter
  unsigned int split_count = 0;
  
  // initialize the number of call to the predictive distribution
  unsigned int n_lpc = 0;
  
  // warm up phase
  for(unsigned int t = 0; t < warm_up; t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the split and merge rates and the split count
    accept_split = 0.0;
    accept_merge = 0.0;
    accept_rate = 0.0;
    split_count = 0;
    
    // reset the max excursion counter
    max_ex = 0;
    
    // reset the excursion counter
    ex_count = 1;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // split and merge steps
      for(unsigned int ttt = 0; ttt < thin_SAM; ttt++){
        
        split_and_merge(model,partition,c,lbl2atm,atm2lbl,labels,
                        co_oc_tbl,pair_alias,false,n_restricted_steps,
                        splitted_groups,merged_group,c_k,
                        K_atm,log_post,n_lpc,accept_split,accept_merge,split_count,n);
        
      }
      
      // random scan step
      random_scan(model,partition,c,dir,lbl2atm,atm2lbl,labels,
                  accept_rate,max_ex,ex_count,log_post,n_lpc,theta,K_atm,thin_scan,n);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    if(NUSAMS && estimate_NU_weights){
      
      // update the co-occurrence probabilities
      unsigned int itr = 0;
      for(unsigned int i = 0; i < n; i++){
        for(unsigned int j = 0; j < i; j++){
          if(c(i) == c(j)){
            co_oc_tbl(itr)++;
          }
          itr++;
        }
      }
      
    }
    
  }
  
  if(warm_up > 0 && NUSAMS){
    
    if(estimate_NU_weights){
      
      // normalize the co occurrence probabilities
      co_oc_tbl /= warm_up;
      
      // save the weights?
      if(return_weights){
        unsigned int itr = 0;
        for(unsigned int i = 0; i < n; i++){
          for(unsigned int j = 0; j < i; j++){
            NUW(i,j) = co_oc_tbl(itr);
            itr++;
          }
        }
      }
      
      // compute the unnormalized log probability
      co_oc_tbl = 0.5 * ( arma::log(co_oc_tbl) + arma::log1p(-co_oc_tbl) );
      
    }
    
    // compute the aliasing table
    create_aliasing_table(co_oc_tbl,pair_alias);
    
  }
  
  for(unsigned int t = 0; t < N; t++){
    
    // set to zero the number of predictive density calls
    n_lpc = 0;
    
    // reset the split and merge rates and the split count
    accept_split = 0.0;
    accept_merge = 0.0;
    accept_rate = 0.0;
    split_count = 0;
    
    // reset the max excursion counter
    max_ex = 0;
    
    // reset the excursion counter
    ex_count = 1;
    
    // thined iterations
    for(unsigned int tt = 0; tt < thin; tt++){
      
      // split and merge steps
      for(unsigned int ttt = 0; ttt < thin_SAM; ttt++){
        
        // reset the split and merge rates and the split count
        accept_split = 0.0;
        accept_merge = 0.0;
        split_count = 0;
        
        // do an iteration
        split_and_merge(model,partition,c,lbl2atm,atm2lbl,labels,
                        co_oc_tbl,pair_alias,NUSAMS,n_restricted_steps,
                        splitted_groups,merged_group,c_k,
                        K_atm,log_post,n_lpc,accept_split,accept_merge,split_count,n);
        
      }
      
      // random scan step
      random_scan(model,partition,c,dir,lbl2atm,atm2lbl,labels,
                  accept_rate,max_ex,ex_count,log_post,n_lpc,theta,K_atm,thin_scan,n);
      
      // update the hyperparameters?
      model.update_hyperpars(log_post);
      
    }
    
    // save and print
    save_and_print(model,conta,seq_idx,log_p,n_log_pred_calls,K_atoms,entropy,alphas,deltas,K_atm,t+warm_up,refresh,warm_up,N,
                   chain_id,seq_idx_sampler,
                   log_post,n_lpc,save_configurations,
                   save_parameters,filename,c);
    
    // save the maximum excursion occurred and the acceptance rates
    acceptance_rates(t,0) = accept_split / thin_SAM / thin; // split_count;
    acceptance_rates(t,1) = accept_merge / thin_SAM / thin; // (thin_SAM - split_count);
    acceptance_rates(t,2) = accept_rate / thin_scan / thin;
    max_excursions(t) = max_ex;
    
  }
  
}

// SAMPLER WRAPPERS
template <typename Model>
void finite_mixture(Model& model,
                    std::vector<std::vector<unsigned int>>& partition,
                    arma::uvec& c,
                    arma::vec& log_p, arma::uvec& n_log_pred_calls,
                    arma::mat& W,
                    arma::vec& entropy,
                    arma::mat& acceptance_rates,
                    arma::vec& alphas,
                    arma::uvec& max_excursions,
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
                    const bool& informed){
  
  if(gibbs){
    
    // gibbs sampler
    gibbs_sampler(model,c,log_p,n_log_pred_calls,W,entropy,alphas,N,warm_up,thin,thin_scan,refresh,
                  chain_id,seq_idx,seq_idx_sampler,
                  save_configurations,save_parameters,filename);
    
    
    
  }else{
    if(reversible){
      
      if(informed){
        
        // reversible informed sampler
        reversible_informed_sampler(model,c,partition,log_p,n_log_pred_calls,W,entropy,alphas,acceptance_rates,
                                    N,warm_up,thin,thin_scan,m,g,refresh,
                                    chain_id,seq_idx,seq_idx_sampler,
                                    save_configurations,save_parameters,filename);
        
        
      }else{
        
        // reversible non informed sampler
        reversible_sampler(model,c,partition,log_p,n_log_pred_calls,W,entropy,alphas,acceptance_rates,
                           N,warm_up,thin,thin_scan,refresh,
                           chain_id,seq_idx,seq_idx_sampler,
                           save_configurations,save_parameters,filename);
      }
      
    }else{
      if(informed){
        
        // non reversible informed
        non_reversible_informed_sampler(model,c,partition,log_p,n_log_pred_calls,W,entropy,alphas,acceptance_rates,max_excursions,theta,
                                        N,warm_up,thin,thin_scan,m,g,refresh,
                                        chain_id,seq_idx,seq_idx_sampler,
                                        save_configurations,save_parameters,filename);
        
      }else{
        
        // non reversible non informed sampler
        non_reversible_sampler(model,c,partition,log_p,n_log_pred_calls,W,entropy,alphas,acceptance_rates,max_excursions,theta,s,
                               N,warm_up,thin,thin_scan,refresh,
                               chain_id,seq_idx,seq_idx_sampler,
                               save_configurations,save_parameters,filename);
        
        
      }
    }
  }
}

// infinite sampler wrapper
template <typename Model>
void infinite_mixture(Model& model,
                      std::vector<std::vector<unsigned int>>& partition,
                      arma::uvec& c,
                      arma::vec& log_p, arma::uvec& n_log_pred_calls,
                      arma::uvec& K_atoms,
                      arma::mat& acceptance_rates,
                      arma::vec& entropy,
                      arma::vec& alphas,
                      arma::vec& deltas,
                      arma::uvec& max_excursions,
                      const double& theta,
                      const unsigned int& N,
                      const unsigned int& warm_up,
                      const unsigned int& thin,
                      const unsigned int& thin_scan,
                      const unsigned int& thin_SAM,
                      const bool& NUSAMS,
                      const Rcpp::Nullable<Rcpp::NumericMatrix>& NU_weights,
                      arma::mat& NUW,
                      const bool& return_weights,
                      const unsigned int& n_restricted_steps,
                      const unsigned int& m,
                      const unsigned int& g,
                      const double& refresh,
                      const unsigned int& chain_id,
                      arma::uvec& seq_idx,
                      const arma::uvec& seq_idx_sampler,
                      const bool& save_configurations,
                      const bool& save_parameters,
                      const std::string& filename,
                      std::unordered_map<unsigned int,unsigned int>& lbl2atm,
                      std::unordered_map<unsigned int, unsigned int>& atm2lbl,
                      std::vector<unsigned int>& labels,
                      const bool& gibbs,
                      const bool& reversible,
                      const bool& informed,
                      const bool& SAM){
  
  // gibbs?
  if(gibbs){
    
    // gibbs sampler
    if(SAM){
      split_and_merge_gibbs(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                            N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                            n_restricted_steps,refresh,
                            chain_id,seq_idx,seq_idx_sampler,
                            save_configurations,save_parameters,filename,
                            lbl2atm,atm2lbl,labels);
    }else{
      
      gibbs_sampler(model,c,log_p,n_log_pred_calls,K_atoms,N,warm_up,thin,thin_scan,refresh,entropy,alphas,deltas,
                    chain_id,seq_idx,seq_idx_sampler,
                    save_configurations,save_parameters,filename,
                    lbl2atm,atm2lbl,labels);
    }
    
  }else{
    // reversible?
    if(reversible){
      
      // informed
      if(informed){
        
        
      }else{
        
        if(SAM){
          split_and_merge_reversible(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                                     N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                                     n_restricted_steps,refresh,
                                     chain_id,seq_idx,seq_idx_sampler,
                                     save_configurations,save_parameters,filename,
                                     lbl2atm,atm2lbl,labels);
        }else{
          // reversible sampler
          reversible_sampler(model,c,partition,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                             N,warm_up,thin,thin_scan,refresh,
                             chain_id,seq_idx,seq_idx_sampler,
                             save_configurations,save_parameters,filename,
                             lbl2atm,atm2lbl,labels);
        }
      }
    }else{
      if(informed){
        
        // informed
        
      }else{
        
        if(SAM){
          
          split_and_merge_non_reversible(model,partition,c,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                                         max_excursions,theta,
                                         N,warm_up,thin,thin_scan,thin_SAM,NUSAMS,NU_weights,NUW,return_weights,
                                         n_restricted_steps,refresh,
                                         chain_id,seq_idx,seq_idx_sampler,
                                         save_configurations,save_parameters,filename,
                                         lbl2atm,atm2lbl,labels);
          
        }else{
          // non reversible sampler
          non_reversible_sampler(model,c,partition,log_p,n_log_pred_calls,K_atoms,acceptance_rates,entropy,alphas,deltas,
                                 max_excursions,theta,
                                 N,warm_up,thin,thin_scan,refresh,
                                 chain_id,seq_idx,seq_idx_sampler,
                                 save_configurations,save_parameters,filename,
                                 lbl2atm,atm2lbl,labels);
        }
      }
    }
  }
  
}



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
    
  }else if(which == 1){
    
    // lambda0
    
    for(unsigned int k = 0; k < ns.size(); k++){
      
      lambdas[k] += (new_value - lambda0);
      mus[k] = ( (lambda0 + ns[k]) * mus[k] + mu0 * (new_value - lambda0) ) / lambdas[k];
      betas[k] = beta0 + 0.5 * ( mu0*mu0*new_value + sums_y2[k] - mus[k]*mus[k]*lambdas[k] );
      
    }
    
    lambda0 = new_value;
    
  }else if(which == 2){
    
    // alpha0
    
    for(unsigned int k = 0; k < ns.size(); k++){
      
      alphas[k] += (new_value - alpha0);
      
    }
    
    alpha0 = new_value;
    
  }else if(which == 3){
    
    // beta0
    
    for(unsigned int k = 0; k < ns.size(); k++){
      
      betas[k] += (new_value - beta0);
      
    }
    
    beta0 = new_value;
    
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

/*** R

get_labels_draws <- function(sampler, chain_id = 1){
  
  #get the allocation vectors
  filename <- paste0(sampler$filename,"/confs",chain_id,".csv")
  confs <- as.matrix(read.csv(filename, header = FALSE))
  colnames(confs) <- rownames(confs) <- NULL
  
  #delete the file
  file.remove(filename)
  
  #return the labels
  confs
  
}

*/
