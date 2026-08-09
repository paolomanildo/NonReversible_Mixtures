#load the temperature schedules
load("schedules.RData")

#function to compute the evolution of the TV
tv <- function(sigmas, K = 3){
  
  #get the maximum number of elements
  Kfact <- factorial(K)
  
  #initialize the Monte Carlo estimate of the distribution
  p_hat <- numeric(Kfact)
  
  #initialize the TVs
  TVs <- numeric(NROW(sigmas))
  
  #compute sequentially the TV distance
  for(i in seq_along(sigmas)){
    
    #get the current sigma
    sigma <- sigmas[i]
    
    #update the estimate
    p_hat[sigma] <- p_hat[sigma] + 1
    
    #compute the distance
    TVs[i] <- 0.5 * sum( abs(p_hat / i - 1/Kfact) )
    
  }
  
  #return the TVs
  TVs
}

#simulate the TV distance
iteration1 <- function(y = y, K = 3, alpha = 1, sigma2 = 1,
                       inverse_temperatures = 1,
                       warm_up = 0, N = 2000, thin = 100,
                       thin_scan = NROW(y)){
  
  #initialize the chain at random
  init <- sample(K,length(y),TRUE)
  
  #get the different thinnings
  thin <- NROW(y) * 100 / thin_scan
  
  #fit MG without tempering
  tmp <- capture.output(MG <- nrMCtempmix(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = 1,
                                         warm_up = warm_up, N = N,
                                         thin_scan = thin_scan, thin = thin,
                                         init = init, gibbs = TRUE))
  MG_tv <- tv(MG$permutation[,1]+ 1)
  MG_pred_calls <- MG$n_pred_calls
  MG_ESS <- coda::effectiveSize(MG$permutation[-seq_len(N/2),1] + 1)
  MG_sESS <- MG_ESS / sum(MG_pred_calls[-seq_len(N/2)])
  
  #fit NR without tempering
  tmp <- capture.output(NR <- nrMCtempmix(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = 1,
                                         warm_up = warm_up, N = N, 
                                         thin_scan = thin_scan, thin = thin,
                                         init = init))
  NR_tv <- tv(NR$permutation[,1]+ 1)
  NR_pred_calls <- NR$n_pred_calls
  NR_ESS <- coda::effectiveSize(NR$permutation[-seq_len(N/2),1] + 1)
  NR_sESS <- NR_ESS / sum(NR_pred_calls[-seq_len(N/2)])
  
  #fit MG with tempering
  tmp <- capture.output(MG <- nrMCtempmix(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = inverse_temperatures,
                                         warm_up = warm_up, N = N, 
                                         thin_scan = thin_scan, thin = thin,
                                         init = init, gibbs = TRUE))
  MG_wt_tv <- tv(MG$permutation[,1]+ 1)
  MG_wt_pred_calls <- MG$n_pred_calls
  MG_wt_ESS <- coda::effectiveSize(MG$permutation[-(seq_len(N/2)),1] + 1)
  MG_wt_sESS <- MG_wt_ESS / sum(MG_wt_pred_calls[-seq_len(N/2)]) * length(inverse_temperatures)
  
  #fit NR with tempering
  tmp <- capture.output(NR <- nrMCtempmix(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = inverse_temperatures,
                                         warm_up = warm_up, N = N,                                          
                                         thin_scan = thin_scan, thin = thin,
                                         init = init))
  NR_wt_tv <- tv(NR$permutation[,1]+ 1)
  NR_wt_pred_calls <- NR$n_pred_calls
  NR_wt_ESS <- coda::effectiveSize(NR$permutation[-(seq_len(N/2)),1] + 1)
  NR_wt_sESS <- NR_wt_ESS / sum(NR_wt_pred_calls[-seq_len(N/2)]) * length(inverse_temperatures)
  
  #return the quantities of interest
  c(MG_tv,NR_tv,MG_wt_tv,NR_wt_tv,
    MG_pred_calls,NR_pred_calls,MG_wt_pred_calls,NR_wt_pred_calls,
    MG_ESS,NR_ESS,MG_wt_ESS,NR_wt_ESS)
  
}

#function for a single iteration
iteration2 <- function(thin_scan = 1000, B = 100){
  
  #create the clusters
  n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
  if(is.na(n_cores)) n_cores <- 1 
  
  #parallelize the for loop
  library(doParallel)
  library(foreach)
  
  #initialize the clusters
  cl <- parallel::makeCluster(n_cores)
  registerDoParallel(cl)
  
  clusterExport(
    cl,
    c(
      "thin_scan",
      "inv_temps1",
      "y1",
      "tv",
      "iteration1",
    ),
    envir = environment()
  )
  
  #replicas
  res1 <- foreach(
    j = seq_len(B),
    .packages = c("nrMCtempmix","coda"),   
    .errorhandling = "pass",
    .combine = "cbind"
  ) %dopar% {
    
    #set the seed
    set.seed(j)
    
    #iteration
    iteration1(y = y1, thin_scan = thin_scan,
               inverse_temperatures = inv_temps1$inverse_temperatures)
    
  }
  
  #save the results
  save(res1, file = 
         paste0("results1.RData"))

  #replicas
  res4 <- foreach(
    j = seq_len(B),
    .packages = c("nrMCtempmix","coda"),   
    .errorhandling = "pass",
    .combine = "cbind"
  ) %dopar% {
    
    #set the seed
    set.seed(j)
    
    #iteration
    iteration1(y = y4, thin_scan = thin_scan,
               inverse_temperatures = inv_temps4$inverse_temperatures)
    
  }
  
  #save the results
  save(res4, file = 
         paste0("results4.RData"))

  #stop the clusters
  stopCluster(cl)
}

iteration2(100)
