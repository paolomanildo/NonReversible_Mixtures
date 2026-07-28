#ifndef SAMPLERS_H
#define SAMPLERS_H

#include <iostream>
#include <RcppArmadillo.h>
#include "supportive_functions.h"
#include "models.h"
#include "random_scans.h"
#include "split_and_merge.h"

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

#endif // SAMPLERS_H