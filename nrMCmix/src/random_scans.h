#ifndef RANDOM_SCANS_H
#define RANDOM_SCANS_H

#include <iostream>
#include <RcppArmadillo.h>
#include "supportive_functions.h"
#include "models.h"

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

#endif // RANDOM_SCANS_H
