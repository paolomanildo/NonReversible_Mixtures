y <- read.csv("COMBO17.csv")$Mcz
 
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
  require(nrMCmix)
  
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

# #density plot
# {
#   par(mfrow = c(1,2))
#   {
#     par(mar = c(4.3,4.1,.6,2.1))
#     
#     # fit plot
#     colors <- c(
#       "#222222",
#       "#DDDDDD",
#       "#444444",
#       "#CCCCCC",
#       "#666666",
#       "#BBBBBB",
#       "#888888",
#       "#AAAAAA",
#       "#999999"
#     )
#     
#     # posterior mean density
#     y_mean <- apply(long_run$y_vals, 1, mean, na.rm = TRUE)
#     
#     # histogram
#     hist(
#       y,
#       prob = TRUE,
#       breaks = 80,
#       border = "white",
#       col = "grey92",
#       main = "",
#       xlab = "y",
#       ylab = expression(f(y)),
#       xaxs = "i",
#       yaxs = "i",
#       ylim = c(0,4)
#     )
#     
#     # posterior sampled densities
#     set.seed(1)
#     for(i in sample(ncol(long_run$y_vals), 100)) {
#       lines(
#         long_run$x_grid,
#         long_run$y_vals[, i],
#         #col = adjustcolor("#E69F00", alpha.f = 0.15),
#         col = adjustcolor("gray", alpha.f = 0.15),
#         lwd = 1
#       )
#     }
#     
#     # posterior mean density
#     lines(
#       long_run$x_grid,
#       y_mean,
#       #col = "#8B0000",
#       col = "black",
#       lwd = 1
#     )
#     
#     # cluster summaries
#     ranges <- do.call(
#       "rbind",
#       tapply(y, long_run$partition, function(x)
#         c(min(x), mean(x), max(x)))
#     )
#     par(xpd = TRUE)
#     for(i in seq_len(nrow(ranges))) {
#       
#       segments(
#         x0 = ranges[i,1],
#         y0 = 0,
#         x1 = ranges[i,3],
#         y1 = 0,
#         col = colors[i],
#         lwd = 4
#       )
#       
#       points(
#         ranges[i,2],
#         0,
#         pch = 18,
#         col = colors[i],
#         cex = 1.5
#       )
#     }
#     par(xpd = FALSE)
#     box()
#     grid()
#   }
#   ax <- axTicks(1)
#   ax_lbl <- as.character(ax)
#   ax_lbl[1] <- "0.0"
#   ax_lbl[6] <- "1.0"
#   
#   #similarity matrix plot
#   {
#     S_ord <- long_run$similarity[order(y,decreasing = FALSE), order(y,decreasing = FALSE)]
#     image(S_ord, axes = FALSE, col = gray.colors(100, start = 1, end = 0))
#     axis(1, at = ax/max(y), labels = ax_lbl)
#     axis(2, at = ax/max(y), labels = ax_lbl)
#     box()
#     abline(h = ax[-1]/max(y), v = ax[-1]/max(y),
#            col = "lightgray", lty = "dotted", lwd = 0.8)
#   }
# }

#function that compute one run of all the algorithms
iteration <- function(thin_SAM = 5,
                      n_restricted_steps = 0,
                      NUW = NULL,
                      N = 1000,
                      warm_up = 0,
                      thin = 100,
                      thin_scan = 0,
                      init = NULL,
                      alpha = 2,
                      delta = 0.1,
                      mu0 = 0.15,
                      lambda0 = 0.006,
                      alpha0 = 0.33,
                      beta0 = 0.001,
                      seed = NULL){
  
  set.seed(seed)
  
  if(is.null(init)){
    init <- sample(10,length(y),TRUE)
  }
  
  sinka <- capture.output(MG <- fit(gibbs = TRUE,
                                    N = N,
                                    warm_up = warm_up,
                                    thin = thin,
                                    thin_scan = thin_scan,
                                    init = init,
                                    alpha = alpha,
                                    delta = delta,
                                    mu0 = mu0,
                                    lambda0 = lambda0,
                                    alpha0 = alpha0,
                                    beta0 = beta0))
  
  sinka <- capture.output(NR <- fit(N = N, warm_up = warm_up,
                                    thin = thin, thin_scan = thin_scan,
                                    init = init,
                                    alpha = alpha,
                                    delta = delta,
                                    mu0 = mu0,
                                    lambda0 = lambda0,
                                    alpha0 = alpha0,
                                    beta0 = beta0))
  
  sinka <- capture.output(SAM <- fit(gibbs = TRUE,
                                     N = N, warm_up = warm_up,
                                     thin = thin, thin_scan = thin_scan,
                                     init = init,
                                     alpha = alpha,
                                     delta = delta,
                                     mu0 = mu0,
                                     lambda0 = lambda0,
                                     alpha0 = alpha0,
                                     beta0 = beta0,
                                     NUW = NUW, n_restricted_steps = n_restricted_steps,
                                     SAM = TRUE, NUSAMS = TRUE))
  
  sinka <- capture.output(SAM_NR <- fit(N = N, warm_up = warm_up,
                                        thin = thin, thin_scan = thin_scan,
                                        init = init,
                                        alpha = alpha,
                                        delta = delta,
                                        mu0 = mu0,
                                        lambda0 = lambda0,
                                        alpha0 = alpha0,
                                        beta0 = beta0,
                                        NUW = NUW, n_restricted_steps = n_restricted_steps,
                                        SAM = TRUE, NUSAMS = TRUE))
  
  c(MG$sampler$entropy,
    NR$sampler$entropy,
    SAM$sampler$entropy,
    SAM_NR$sampler$entropy,
    MG$sampler$K,
    NR$sampler$K,
    SAM$sampler$K,
    SAM_NR$sampler$K,
    MG$sampler$n_pred_calls,
    NR$sampler$n_pred_calls,
    SAM$sampler$n_pred_calls,
    SAM_NR$sampler$n_pred_calls)
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
    "y",
    "long_run",
    "fit",
    "iteration"
  ),
  envir = environment()
)

#replicas
res <- foreach(
  b = seq_len(B),
  .packages = c("nrMCmix"),   
  .errorhandling = "pass"
) %dopar% {
  
  N <- NROW(long_run$hyperpars)
  
  #transient phase
  transient <- iteration(seed = b)
  
  #stationary phase
  stationary <- iteration(seed = b,
                          init = long_run$sampler$c[,1] + 1,
                          NUW = long_run$similarity,
                          alpha = long_run$hyperpars[N,1],
                          delta = long_run$hyperpars[N,2],
                          mu0 = long_run$hyperpars[N,3],
                          lambda0 = long_run$hyperpars[N,4],
                          alpha0 = long_run$hyperpars[N,5],
                          beta0 = long_run$hyperpars[N,6])
  
  #get the quantity of interest
  stationary <- c(as.numeric(coda::effectiveSize(stationary[seq_len(N)])),
                  as.numeric(coda::effectiveSize(stationary[N + seq_len(N)])),
                  as.numeric(coda::effectiveSize(stationary[2*N + seq_len(N)])),
                  as.numeric(coda::effectiveSize(stationary[3*N + seq_len(N)])),
                  sum(stationary[8*N + seq_len(N)]),
                  sum(stationary[9*N + seq_len(N)]),
                  sum(stationary[10*N + seq_len(N)]),
                  sum(stationary[11*N + seq_len(N)]))
  
  c(transient,stationary)
}

#stop the clusters
stopCluster(cl)

N <- NROW(long_run$hyperpars)

#get the quantities of interest
entropies <- sapply(res, function(x) {
  cbind(x[seq_len(N)],
        x[N + seq_len(N)],
        x[2*N + seq_len(N)],
        x[3*N + seq_len(N)])
}, simplify = "array")

Ks <- sapply(res, function(x) {
  cbind(x[4*N + seq_len(N)],
        x[5*N + seq_len(N)],
        x[6*N + seq_len(N)],
        x[7*N + seq_len(N)])
}, simplify = "array")

n_pred_calls <- sapply(res, function(x) {
  cbind(x[8*N + seq_len(N)],
        x[9*N + seq_len(N)],
        x[10*N + seq_len(N)],
        x[11*N + seq_len(N)])
}, simplify = "array")

#get the reference distribution
target_entropy <- long_run$sampler$entropy[,1]
target_K <- long_run$sampler$K[,1]

#compute the TV distances
TV_ent <- sapply(1:4, function(i){
  sapply(seq_len(NROW(entropies)), function(j){
    TV(target_entropy,entropies[j,i,])
  })
})

TV_K <- sapply(1:4, function(i){
  sapply(seq_len(NROW(Ks)), function(j){
    TV(target_K,Ks[j,i,], discrete = TRUE)
  })
})

#compute the ESS
ESS <- t(sapply(res,function(x) {
  x[12*N + 1:4]
}))

#compute the scaled ESS
sESS <- t(sapply(res,function(x) {
  x[12*N + 1:4] / x[12*N + 5:8]
}))

save(TV_ent,TV_K,ESS,sESS, file = "processed_output.RData")

