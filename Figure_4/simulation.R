library(nrMCmix)

#function for prior checking the spherical Gaussian DP
rprior <- function(n = 1000L, d = 1L, alpha = 0.1, delta = 0,
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
      c(-sum(ns/n * log(ns/n)),length(ns))
      
    }))
    
  }else{
    #no
    
    #simulate some data from the prior
    y <- generate_data(n = n, d = d, alpha = alpha, delta = delta,
                       sigma2 = sigma2, mu0 = mu0, lambda0 = lambda0)$y
    
    if(d == 1) y <- y[,1]
    
    #if NUSAMS estimate first the weights
    sinka <- capture.output(NUW <- nrMCmix(y = y, alpha = alpha, delta = delta, sigma2 = sigma2,
                                           mu0 = mu0, lambda0 = lambda0, warm_up = N_NUW_est,
                                           return_weights = TRUE,
                                           N = 0, thin_scan = thin_scan, thin = thin,
                                           thin_SAM = thin_SAM, n_restricted_steps = n_restricted_steps,
                                           SAM = TRUE, NUSAMS = TRUE, NU_weights = NULL,
                                           gibbs = gibbs, reversible = reversible)$NUW)
    
    #fit the data with the generative model
    sinka <- capture.output(
      
      fit <- nrMCmix(y = y, alpha = alpha, delta = delta, sigma2 = sigma2,
                     mu0 = mu0, lambda0 = lambda0, warm_up = warm_up,
                     N = N, thin_scan = thin_scan, thin = thin,
                     thin_SAM = thin_SAM, n_restricted_steps = n_restricted_steps,
                     SAM = SAM, NUSAMS = NUSAMS, NU_weights = NUW,
                     gibbs = gibbs, reversible = reversible, init = init)
      
    )
    
    #return the entropy and K_star trace
    c(fit$entropy,fit$n_pred_calls)
    
  }
  
}

#define the empirical TV distance
TV <- function(x,y,discrete = FALSE, n_grid = 512){
  
  if(discrete){
    
    #use the relative frequencies
    K <- max(x,y, na.rm = TRUE)
    X_val <- tabulate(x,K) / length(x)
    Y_val <- tabulate(y,K) / length(y)
    
    #return the TV
    0.5 * sum( abs(X_val - Y_val) )
    
  }else{
    
    #use numerical integration
    
    #get the common support of x and y
    x_lim <- range(x,y, na.rm = TRUE)
    x_lim <- x_lim + c(-1,1) * 0.001 * diff(x_lim)
    
    #compute the two densities
    x_grid <- seq(x_lim[1],x_lim[2],l = n_grid)
    X_val <- density(x,from = x_lim[1], to = x_lim[2], n = n_grid)$y
    Y_val <- density(y,from = x_lim[1], to = x_lim[2], n = n_grid)$y
    
    #compute the TV distance with the trapezoidal rule:
    
    #save the absolute differences
    abs_diff <- abs(X_val - Y_val)
    
    #return the integral using the trapezoidal rule
    0.5 * sum( diff(x_grid) * (head(abs_diff, -1) + tail(abs_diff, -1)) / 2 )
    
  }
  
}

#function that compute one iteration
iteration <- function(n = 1000L, d = 1L, alpha = 0.1, delta = 0,
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
  
  c(MG,NR,SAM,SAM_NR)
}

#function that return one result for all scenarios
iteration2 <- function(n_grid = round(10^seq(3, 4, length.out = 5)),
                       d_grid = c(1,3,5), 
                       alpha = 0.1, delta = 0,
                       sigma2 = 1, lambda0 = 0.1,
                       warm_up = 0, N = 10000, N_NUW_est = 1000,
                       thin_scan = 0, thin = 1, thin_SAM = 5,
                       n_restricted_steps = 0, seed = NULL){
  
  #create the grid
  configs <- as.matrix(expand.grid(n_grid,d_grid))
  out <- matrix(NA,8*N,NROW(configs))
  
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
B <- 1000

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
  .packages = c("nrMCmix"),   
  .errorhandling = "pass"
) %dopar% {
  
  iteration2(seed = b)
  
}

#stop the clusters
stopCluster(cl)

#save the results
save(res, file = "results_long01.RData")

n_grid = round(10^seq(3, 4, length.out = 5))
d_grid = c(1,3,5)
grea <- as.matrix(expand.grid(n_grid,d_grid))
N <- dim(res[[1]])[1] / 8

#get the quantities of interest
entropies <- sapply(res, function(x) {
  sapply(seq_len(NCOL(x)),function(i){
    cbind(x[seq_len(N),i],
          x[2*N + seq_len(N),i],
          x[4*N + seq_len(N),i],
          x[6*N + seq_len(N),i])
  }, simplify = "array")
}, simplify = "array")

n_pred_calls <- sapply(res, function(x) {
  sapply(seq_len(NCOL(x)),function(i){
    cbind(x[N + seq_len(N),i],
          x[3*N + seq_len(N),i],
          x[5*N + seq_len(N),i],
          x[7*N + seq_len(N),i])
  }, simplify = "array")
}, simplify = "array")

#simulate from the real prior
prior_draws <- sapply(n_grid,rprior,prior_only = TRUE, N = 1e4,
                      simplify = "array")

#compute the TV distances
TVs <- vector("list",NROW(grea))
for(cc in seq_along(TVs)){
  
  #entropy
  TVs[[cc]]$entropy <- sapply(seq_len(dim(entropies)[2]), function(s) {
    ccc <- which(n_grid == grea[cc,1])[1]
    sapply(seq_len(dim(entropies)[1]), function(i){
      TV(prior_draws[,1,ccc],entropies[i,s,cc,])
    })
  })
  
  #n_pred_calls
  TVs[[cc]]$n_pred_calls <- sapply(seq_len(dim(n_pred_calls)[2]), function(s) {
    ccc <- which(n_grid == grea[cc,1])[1]
    sapply(seq_len(dim(n_pred_calls)[1]), function(i){
      mean(n_pred_calls[i,s,cc,])
    })
  })
  
}

#determine the number of predictive call to reach convergence
#for each scenario, then at convergence, take 1000 iteration and compute the ESS
T2C <- function(X){
  
  #get the TV upper quantile of SAM_NR after burn in
  q99 <- quantile(X$entropy[-(1:1000),4],0.99)
  
  #get the number of predictive calls
  n_calls <- sapply(seq_along(1:4), function(i){
    n_iter <- which(X$entropy[,i] <= q99)[1]
    if(is.na(n_iter)) n_iter <- 3000
    sum(X$n_pred_calls[seq_len(n_iter),i])
  })
  
  #return it
  n_calls
  
}

#compute the average effective sample size of the entropy
#ESS <- apply(entropies,2:4,coda::effectiveSize)
#ESSs <- ESS / apply(n_pred_calls,2:4, function(x) sum(x[-(1:1000)]))

save(n_grid,d_grid,grea,N,TVs,#ESS,ESSs,
     entropies,n_pred_calls,file = "processed_output01.RData")

