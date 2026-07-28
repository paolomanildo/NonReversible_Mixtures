#load the data
y <- read.csv("COMBO17.csv")$Mcz

#function to fit the model via MCMC
fit <- function(gibbs = FALSE, SAM = FALSE, NUSAMS = FALSE,
                NUW = NULL, n_restricted_steps = 0, thin_SAM = 5,
                N = 1000, warm_up = 1000, thin = 1, thin_scan = 0,
                draws = FALSE, init = NULL, compute_similarity = FALSE,
                save_parameters = FALSE,
                alpha = 2,
                delta = 0.1,
                mu0 = 0.15,
                lambda0 = 0.006,
                alpha0 = 0.33,
                beta0 = 0.001){
  
  #load the package
  tryCatch(require(nrMCmix), error = function(x) message'Cannot found package "nrMCmix", try load it by Rcpp::sourceCpp("nrMCmix.cpp")')
  
  #function to extract the hyperparameters values
  get_hyperpar <- function(sampler, chain_id = 1){
    filename <- paste0(sampler$filename, "/pars", chain_id, 
                       ".csv")
    confs <- as.matrix(read.csv(filename, header = FALSE))
    colnames(confs) <- rownames(confs) <- NULL
    file.remove(filename)
    confs
  }
  
  #initialize the output
  out <- list()
  
  #fit the model
  tmp <- nrMCmix(y,
                hyperpar_alpha = TRUE,
                hyperpar_delta = TRUE,
                hyperpar_baseline = TRUE,
                M = 0,
                V = 10,
                a_l = 1,
                b_l = 1,
                a_a = 1,
                b_a = 1,
                a_b = 5,
                b_b = 2,
                warm_up = warm_up,
                N = N,
                thin = thin,
                thin_scan = thin_scan,
                save_configurations = draws,
                save_parameters = save_parameters,
                thin_SAM = thin_SAM,
                gibbs = gibbs,
                SAM = SAM,
                NUSAMS = NUSAMS,
                NU_weights = NUW,
                n_restricted_steps = n_restricted_steps,
                init = init,
                alpha = alpha,
                delta = delta,
                mu0 = mu0,
                lambda0 = lambda0,
                alpha0 = alpha0,
                beta0 = beta0)
  
  out$sampler <- tmp
  
  #get the optimal partition and hyperpars draws
  if(draws){
    lbls <- get_labels_draws(tmp)
    idx <- salso::salso(lbls)
    out$partition <- idx
    out$lbls <- lbls
  }
  if(save_parameters){
    hyperpars <- cbind(tmp$alpha,tmp$delta,get_hyperpar(tmp))
    out$hyperpars <- hyperpars
  }
  
  if(draws){
    #reconstruct the density
    x_grid <- seq(min(y) - diff(range(y))*0.05,
                  max(y) + diff(range(y))*0.05,
                  l = 512)
    y_vals <- sapply(seq_len(NROW(lbls)),function(i){
      get_latent_process(lbls[i,],y = y, alpha = hyperpars[i,1],
                         delta = hyperpars[i,2],x_grid = x_grid,
                         mu0 = hyperpars[i,3],
                         lambda0 = hyperpars[i,4],
                         alpha0 = hyperpars[i,5],
                         beta0 = hyperpars[i,6])$y
    })
    out$x_grid <- x_grid
    out$y_vals <- y_vals
    
  }
  
  #compute the similarity matrix
  if(compute_similarity){
    
    S <- compute_similarity_matrix(lbls)
    
    #add a nugget term
    S[S == 1] <- 1-1e-5
    S[S == 0] <- 1e-5
    out$similarity <- S
  }
  
  return(out)
}

#get a long run of the sampler
long_run <- fit(save_parameters = TRUE, draws = TRUE,
                thin = 1000, SAM = TRUE, compute_similarity = TRUE)

#save the results
save(long_run, file = "long_run.RData")
