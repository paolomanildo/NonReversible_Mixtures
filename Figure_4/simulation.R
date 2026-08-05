#function for prior checking the spherical Gaussian DP
rprior <- function(n = 1000L, d = 1L, alpha = 1, delta = 0,
                   sigma2 = 1, mu0 = rep(0,d), lambda0 = 0.1,
                   warm_up = 0, N = 1000, N_NUW_est = 1000,
                   thin_scan = 0, thin = 1, thin_SAM = 5, SAM = FALSE,
                   NUSAMS = FALSE, NUW = NULL, n_restricted_steps = 0,
                   gibbs = FALSE, reversible = FALSE,
                   prior_only = FALSE, seed = NULL){
  
  #load the package
  require(nrMCmix)
  
  #set the seed and the common starting value
  set.seed(seed)
  init <- sample(10,n,TRUE)
  
  #simulate from the prior?
  if(prior_only){
    #yes
    
    t(replicate(N, {
      
      #simulate the partition from the prior
      ns <- generate_data(n = n, alpha = alpha, delta = delta,
                          kernel = "partition")$ns[,1]
      
      #return the entropy and number of groups size
      -sum(ns/n * log(ns/n))
      
    }))
    
  }else{
    #no
    
    #simulate some data from the prior
    y <- generate_data(n = n, d = d, alpha = alpha, delta = delta,
                       sigma2 = sigma2, mu0 = mu0, lambda0 = lambda0)$y
    
    if(d == 1) y <- y[,1]
    
    #if NUSAMS estimate first the weights
    if(NUSAMS){
      sinka <- capture.output(NUW <- nrMCmix(y = y, alpha = alpha, delta = delta, sigma2 = sigma2,
                                             mu0 = mu0, lambda0 = lambda0, warm_up = N_NUW_est,
                                             return_weights = TRUE,
                                             N = 0, thin_scan = thin_scan, thin = 1,
                                             thin_SAM = thin_SAM, n_restricted_steps = n_restricted_steps,
                                             SAM = TRUE, NUSAMS = TRUE, NU_weights = NULL,
                                             gibbs = gibbs, reversible = reversible)$NUW)
    }
    
    #fit the data with the generative model
    sinka <- capture.output(
      
      fit <- nrMCmix(y = y, alpha = alpha, delta = delta, sigma2 = sigma2,
                     mu0 = mu0, lambda0 = lambda0, warm_up = warm_up,
                     N = N, thin_scan = thin_scan, thin = thin,
                     thin_SAM = thin_SAM, n_restricted_steps = n_restricted_steps,
                     SAM = SAM, NUSAMS = NUSAMS, NU_weights = NUW,
                     gibbs = gibbs, reversible = reversible, init = init)
      
    )
    
    #do another 1000 iterations at convergence and return the effective sample size
    #and number of predictive calls
    sinka <- capture.output(
      
      fit2 <- nrMCmix(y = y, alpha = alpha, delta = delta, sigma2 = sigma2,
                      mu0 = mu0, lambda0 = lambda0, warm_up = warm_up,
                      N = N, thin_scan = thin_scan, thin = 1,
                      thin_SAM = thin_SAM, n_restricted_steps = n_restricted_steps,
                      SAM = SAM, NUSAMS = NUSAMS, NU_weights = NUW,
                      gibbs = gibbs, reversible = reversible, init = fit$c[,1] + 1)
      
    )
    
    ESS <- coda::effectiveSize(fit2$entropy)
    n_pred_calls <- sum(fit2$n_pred_calls)
    
    #return the entropy trace and the number of predictive calls
    c(fit$entropy,fit$n_pred_calls,ESS,n_pred_calls)
    
  }
  
}

#function that compute one iteration
iteration <- function(n = 1000L, d = 1L, alpha = 1, delta = 0,
                      sigma2 = 1, mu0 = rep(0,d), lambda0 = 0.1,
                      warm_up = 0, N = 1000, N_NUW_est = 1000,
                      thin_scan = 0, thin = 1, thin_SAM = 5,
                      n_restricted_steps = 0, seed = NULL){
  
  MG <- rprior(seed = seed, n = n, d = d,
               alpha = alpha, delta = delta, sigma2 = sigma2,
               mu0 = mu0, lambda0 = lambda0, warm_up = warm_up,
               N = N, thin_scan = thin_scan, thin = thin,
               gibbs = TRUE)
  
  NR <- rprior(seed = seed, n = n, d = d,
               alpha = alpha, delta = delta, sigma2 = sigma2,
               mu0 = mu0, lambda0 = lambda0, warm_up = warm_up,
               N = N, thin_scan = thin_scan, thin = thin)
  
  SAM <- rprior(seed = seed, n = n, d = d,
                alpha = alpha, delta = delta, sigma2 = sigma2,
                mu0 = mu0, lambda0 = lambda0, warm_up = warm_up,
                N = N, thin_scan = thin_scan, thin = thin,
                thin_SAM = thin_SAM, n_restricted_steps = n_restricted_steps,
                SAM = TRUE, NUSAMS = TRUE, NUW = NULL,
                gibbs = TRUE)
  
  SAM_NR <- rprior(seed = seed, n = n, d = d,
                   alpha = alpha, delta = delta, sigma2 = sigma2,
                   mu0 = mu0, lambda0 = lambda0, warm_up = warm_up,
                   N = N, thin_scan = thin_scan, thin = thin,
                   thin_SAM = thin_SAM, n_restricted_steps = n_restricted_steps,
                   SAM = TRUE, NUSAMS = TRUE, NUW = NULL)
  
  c(MG[seq_len(2*N)],
    NR[seq_len(2*N)],
    SAM[seq_len(2*N)],
    SAM_NR[seq_len(2*N)],
    MG[2*N+1],NR[2*N+1],SAM[2*N+1],SAM_NR[2*N+1],
    MG[2*N+2],NR[2*N+2],SAM[2*N+2],SAM_NR[2*N+2])
}

#function that return one result for all scenarios
iteration2 <- function(n_grid = round(10^seq(3, 4, length.out = 5)),
                       d_grid = c(1,3,5), 
                       alpha = .1, delta = 0,
                       sigma2 = 1, lambda0 = 0.1,
                       warm_up = 0, N = 1000, N_NUW_est = 1000,
                       thin_scan = 0, thin = 10, thin_SAM = 5,
                       n_restricted_steps = 0, seed = NULL){
  
  #create the grid
  configs <- as.matrix(expand.grid(n_grid,d_grid))
  out <- matrix(NA,8*N+8,NROW(configs))
  
  #fill it
  for(i in seq_len(NROW(configs))){
    
    out[,i] <- iteration(n = configs[i,1],
                         d = configs[i,2],
                         alpha = alpha,
                         delta = delta,
                         sigma2 = sigma2,
                         mu0 = rep(0,configs[i,2]),
                         lambda0 = lambda0,
                         warm_up = warm_up,
                         N = N,
                         N_NUW_est = N_NUW_est,
                         thin_scan = thin_scan,
                         thin = thin,
                         thin_SAM = thin_SAM,
                         n_restricted_steps = n_restricted_steps,
                         seed = seed)
  }
  
  #return the matrix
  out
}

#initialize the results
B <- 100

library(doParallel)
library(foreach)

#create the clusters
n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
if(is.na(n_cores)) n_cores <- 1 

#initialize the clusters
cl <- parallel::makeCluster(n_cores)
registerDoParallel(cl)

#update the functions and objects
clusterExport(
  cl,
  c(
    "rprior",
    "iteration",
    "iteration2"
  ),
  envir = environment()
)

#replicas
res <- foreach(
  b = seq_len(B),
  .packages = c("nrMCmix","coda"),   
  .errorhandling = "pass"
) %dopar% {
  
  iteration2(seed = b)
  
}

#stop the clusters
stopCluster(cl)
