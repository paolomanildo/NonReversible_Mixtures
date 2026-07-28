#function for a generic simulation that returns the cluster size trace
cluster_sizes <- function(B = 300, #simulation replicas
                          n = 2000, #sample size
                          d = 1, #dimension
                          N = 150, #chains iteration
                          K = 2, #number of clusters
                          init = NULL, #initialization vector
                          K_init = K, #number of clusters at the first iteration
                          kernel = "gaussian", #mixture kernel
                          mu0 = 0, #mean hyperparameter for the mean parameters (Gaussaian case)
                          lambda0 = 1, #precision hyperparameter for the mean parameters (Gaussian case)
                          sigma2 = 1, #known variance (Gaussian case)
                          alpha0 = 1, #shape parameter gamma prior (Poisson case)
                          beta0 = 1, #rate parameter gamma prior (Poisson case)
                          alpha = 1, #Dirichlet concentration parameter
                          sample_from_model = TRUE, #generative sampling?
                          sample_data = function(n) NULL#function to sample the data
){
  
  #load the package
  require(nrMCmix)
  
  #initialize the output
  MG_cluster_sizes <- R_cluster_sizes <- NR_cluster_sizes <- array(NA, dim = c(N,K,B))
  
  #fill them
  for(b in seq_len(B)){
    
    #generate the data
    set.seed(b)
    if(sample_from_model){
      
      y <- generate_data(n = n,
                         d = d,
                         K = K,
                         alpha = alpha,
                         mu0 = mu0,
                         lambda0 = lambda0,
                         sigma2 = sigma2,
                         alpha0 = alpha0,
                         beta0 = beta0,
                         kernel = kernel)$y[,seq_len(d)]
    }else{
      
      y <- sample_data(n)
      
    }

    if(is.null(init)){
      #sample the initial value for the chain
      init <- sample(K_init,n,TRUE)
    }
    
    #get the configuration Markov chain for the MG
    tmp <- capture.output(MG <- nrMCmix(y,K = K, alpha = alpha,
                                       mu0 = mu0, lambda0 = lambda0, sigma2 = sigma2,
                                       alpha0 = alpha0, beta0 = beta0,
                                       N = N-1,warm_up = 0,
                                       gibbs = TRUE,
                                       kernel = kernel, init = init,
                                       save_configurations = TRUE))
    MG_lbls <- get_labels_draws(MG) + 1
    
    #store the cluster size trace
    MG_cluster_sizes[,,b] <- t(apply(rbind(c(init),MG_lbls),1,function(x) {
      ns <- sapply(seq_len(K),function(k) sum(x == k))
      ns / sum(ns)
    }))
    
    #do the same for the reversible one
    tmp <- capture.output(R <- nrMCmix(y,K = K, alpha = alpha,
                                       mu0 = mu0, lambda0 = lambda0, sigma2 = sigma2,
                                       alpha0 = alpha0, beta0 = beta0,
                                       N = N-1,warm_up = 0,
                                       reversible = TRUE,
                                       kernel = kernel, init = init,
                                       save_configurations = TRUE))
    R_lbls <- get_labels_draws(R) + 1
    
    #store the cluster size trace
    R_cluster_sizes[,,b] <- t(apply(rbind(c(init),R_lbls),1,function(x) {
      ns <- sapply(seq_len(K),function(k) sum(x == k))
      ns / sum(ns)
    }))
    
    #do the same for the non-reversible one
    tmp <- capture.output(NR <- nrMCmix(y,K = K, alpha = alpha,
                                       mu0 = mu0, lambda0 = lambda0, sigma2 = sigma2,
                                       alpha0 = alpha0, beta0 = beta0,
                                       N = N-1,warm_up = 0,
                                       kernel = kernel, init = init,
                                       save_configurations = TRUE))
    NR_lbls <- get_labels_draws(NR) + 1
    
    #store the cluster size trace
    NR_cluster_sizes[,,b] <- t(apply(rbind(c(init),NR_lbls),1,function(x) {
      ns <- sapply(seq_len(K),function(k) sum(x == k))
      ns / sum(ns)
    }))
    
    #update the user
    cat("\n",b,"over",B)
    
  }
  
  #return the two arrays
  list(MG = MG_cluster_sizes,
       R = R_cluster_sizes,
       NR = NR_cluster_sizes)
  
}
