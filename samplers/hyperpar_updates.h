#ifndef HYPERPAR_UPDATES_H
#define HYPERPAR_UPDATES_H

#include <iostream>
#include <RcppArmadillo.h>
#include "supportive_functions.h"
#include "models.h"

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

#endif // HYPERPAR_UPDATES_H