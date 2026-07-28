#ifndef SPLIT_AND_MERGE_H
#define SPLIT_AND_MERGE_H

#include <iostream>
#include <RcppArmadillo.h>
#include "supportive_functions.h"
#include "models.h"

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

#endif // SPLIT_AND_MERGE_H