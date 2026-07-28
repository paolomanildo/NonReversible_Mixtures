### TEMPERING
library(nrTemperdMixtures)

#y <- read.csv("galaxy.csv", header = TRUE)$x

#function to simulate some dataset
sim_data <- function(n = 1000,
                     w = c(0.1,0.7,0.2),
                     mu = c(-2,1,2),
                     sigma2 = 1,
                     seed = 123){
  set.seed(seed)
  z <- sample(length(w),n,TRUE,prob = w)
  rnorm(n,mu[z],sqrt(sigma2))
}

y1 <- sim_data(mu = c(-2,1,2))
y2 <- sim_data(mu = c(-4,1,4))
y3 <- sim_data(mu = c(-5,1,5))
y4 <- sim_data(mu = c(-8,1,8))

#function that return the acceptance rate at a given temperature
acc_rate <- function(x,
                     y,
                     K = 3,
                     alpha = 1,
                     sigma2 = 1,
                     lambda0 = 0.01,
                     mu0 = 0,
                     gibbs = FALSE,
                     reversible = FALSE,
                     warm_up = 1000,
                     N = 1000,
                     thin = 1,
                     thin_scan = 0){
  
  sinka <- capture.output(rate <- mean(nrMCMCtemp(y, K = K, alpha = alpha, sigma2 = 1,
                                                       mu0 = mu0, lambda0 = lambda0,
                                                       warm_up = warm_up, N = N,
                                                       thin = thin, thin_scan = thin_scan,
                                                       gibbs = gibbs, reversible = reversible,
                                                       inverse_temperatures = x)$acceptance_rates))
  
  rate
  
}

#function that found the optimal highest temperature
search_max_temp <- function(y,
                            max_degree = 30,
                            tol = 0.01,
                            K = 3,
                            alpha = 1,
                            sigma2 = 1,
                            lambda0 = 0.01,
                            mu0 = 0,
                            gibbs = FALSE,
                            reversible = FALSE,
                            warm_up = 1000,
                            N = 1000,
                            thin = 1,
                            thin_scan = 0){
  
  #define the temperature grid
  x_grid <- 2^-(seq(0,20,l = 30))
  
  #get the acceptance rates
  x_val <- sapply(x_grid,acc_rate,y = y, K = K,
                  alpha = alpha, sigma2 = sigma2,
                  lambda0 = lambda0, mu0 = mu0,
                  gibbs = gibbs, reversible = reversible,
                  warm_up = warm_up, N = N,
                  thin = thin, thin_scan = thin_scan)
  
  #get the first closest value to the maximum
  val <- x_grid[which(abs(x_val - max(x_val)) < 0.01)[1]]
  
  #return the quantities of interest
  list(value = val,
       grid = x_grid,
       objective = x_val)
}

#function to create a temperature ladder
#via adaptive tuning
create_ladder <- function(y,
                          K = 3,
                          alpha = 1,
                          sigma2 = 1,
                          lambda0 = 0.01,
                          mu0 = 0,
                          gibbs = FALSE,
                          reversible = FALSE,
                          warm_up = 1000,
                          N = 1000,
                          thin = 1,
                          thin_scan = 0,
                          target_rate = 0.234,
                          n_temp_max = 10,
                          inv_temps = 1,
                          min_val = NULL,
                          min_search_val = NULL,
                          max_degree = 30,
                          tol = 0.01,
                          max_it = 100,
                          eps = 1e-5){
  
  require(nrTemperdMixtures)
  
  #initialize the output
  out <- list()
  
  #if the minimum value is null, search it
  if(is.null(min_val)){
    tmp <- search_max_temp(y = y,
                           max_degree = max_degree,
                           tol = tol, 
                           K = K,
                           alpha = alpha, sigma2 = sigma2,
                           lambda0 = lambda0, mu0 = mu0,
                           gibbs = gibbs, reversible = reversible,
                           warm_up = warm_up, N = N,
                           thin = thin, thin_scan = thin_scan)
    
    out$temperature_search <- tmp
    min_val <- tmp$value
  }
  
  #define the lowest temperature considered
  if(is.null(min_search_val)){
    min_search_val <- min_val / 2
  }
  
  #define the loss
  loss <- function(x,inv_temps,y){
    
    sinka <- capture.output(rate <- tail(drop(nrMCMCtemp(y = y, K = K, alpha = alpha, sigma2 = 1,
                                                         mu0 = mu0, lambda0 = lambda0,
                                                         warm_up = warm_up, N = N,
                                                         thin = thin, thin_scan = thin_scan,
                                                         gibbs = gibbs, reversible = reversible,
                                                         inverse_temperatures = c(inv_temps,x))$swaps),1))
    
    rate - target_rate
  }
  
  #bisection algorithm
  find_optimum <- function(y,
                           inv_temps = 1,
                           min_val = 0.001,
                           max_it = 100,
                           eps = 1e-5){
    
    #bisection algoritjhm extremes
    curr_high <- tail(inv_temps,1)
    curr_low <- min_val
    
    exit <- FALSE
    iters <- 1
    old_val <- curr_high
    while(!exit && iters <= max_it){
      
      #take the mid value in the interval
      val <- mean(c(curr_high,curr_low))
      
      #get the function whose root is of interest
      err <- loss(val,inv_temps,y)
      
      #if it is lower than zero reduce the interval from the left
      if(err < 0){
        curr_low <- val
      }else{
        #otherwise from the right
        curr_high <- val
      }
      
      if( abs(val - old_val)/old_val < eps){
        exit <- TRUE
      }else{
        old_val <- val
        iters <- iters + 1
      }
      
    }
    
    #return the value
    list(value = val,
         objective = err,
         iters = iters)
    
  }
  
  exit <- FALSE
  iters <- NULL
  objective <- NULL
  while(!exit && length(inv_temps) < n_temp_max){
    opt <- find_optimum(y = y,
                        inv_temps = inv_temps,
                        min_val = min_search_val,
                        max_it = max_it,
                        eps = eps)
    inv_temps <- c(inv_temps,opt$value)
    objective <- c(objective,opt$objective)
    iters <- c(iters,opt$iters)
    
    if(opt$value < min_val){
      exit <- TRUE
    }
  }
  
  out$iters <- iters
  out$objective <- objective
  out$inverse_temperatures <- inv_temps
  
  return(out)
}

#function to compute the evolution of the TV
tv <- function(theta){
  
  # get the permutation
  perms <- matrix(c(1,2,3,
                    1,3,2,
                    2,1,3,
                    2,3,1,
                    3,1,2,
                    3,2,1),6,3,byrow = TRUE)
  
  #get the lbls
  lbls <- apply(theta,1,function(x) {
    tmp <- order(x)
    which(apply(perms,1,function(xx) all(tmp == xx)))[1]
  })
  
  #compute the frequencies sequentially
  ns <- matrix(0,NROW(lbls),6)
  ns[1,lbls[1]] <- 1 
  for(i in seq_along(lbls)[-1]){
    lbl <- lbls[i]
    ns[i,] <- ns[i-1,]
    ns[i,lbl] <- ns[i,lbl] + 1
  }
  ns <- ns / seq_len(NROW(lbls))
  
  #compute the TV distance
  apply(ns,1,function(x){
    0.5 * sum(abs(x-1/6))
  })
}

#get the temperature ladder
inv_temps1 <- create_ladder(y1, n_temp_max = 30)
inv_temps2 <- create_ladder(y2, n_temp_max = 30)
inv_temps3 <- create_ladder(y3, n_temp_max = 30)
inv_temps4 <- create_ladder(y4, n_temp_max = 30)

#do the same for alpha = 0.1
#inv_temps01 <- create_ladder(y, n_temp_max = 30)

#simulate the TV distance
iteration <- function(y = y, K = 3, alpha = 1, sigma2 = 1,
                      inverse_temperatures = 1,
                      warm_up = 0, N = 1000, thin = 100,
                      thin_scan = NROW(y)){
  
  #initialize the chain at random
  init <- sample(K,length(y),TRUE)
  
  #get the different thinnings
  thin <- NROW(y) * 100 / thin_scan
  
  #fit MG without tempering
  tmp <- capture.output(MG <- nrMCMCtemp(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = 1,
                                         warm_up = warm_up, N = N,
                                         thin_scan = thin_scan, thin = thin,
                                         init = init, gibbs = TRUE))
  MG_wt_tv <- tv(MG$Theta)
  
  #fit NR without tempering
  tmp <- capture.output(NR <- nrMCMCtemp(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = 1,
                                         warm_up = warm_up, N = N, 
                                         thin_scan = thin_scan, thin = thin,
                                         init = init))
  NR_wt_tv <- tv(NR$Theta)
  
  #fit MG with tempering
  tmp <- capture.output(MG <- nrMCMCtemp(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = inverse_temperatures,
                                         warm_up = warm_up, N = N, 
                                         thin_scan = thin_scan, thin = thin,
                                         init = init, gibbs = TRUE, thin_scan = ))
  MG_tv <- tv(MG$Theta)
  
  #fit NR with tempering
  tmp <- capture.output(NR <- nrMCMCtemp(y, K = K, 
                                         alpha = alpha, sigma2 = sigma2,
                                         inverse_temperatures = inverse_temperatures,
                                         warm_up = warm_up, N = N,                                          
                                         thin_scan = thin_scan, thin = thin,
                                         init = init))
  NR_tv <- tv(NR$Theta)
  
  #compute the ESS of quantities that are invariant to the labels
  tmp <- capture.output(MG_wt <- coda::effectiveSize(nrMCMCtemp(y, K = K, 
                                                             alpha = alpha, sigma2 = sigma2,
                                                             inverse_temperatures = 1,
                                                             warm_up = N, N = N, thin = 1,
                                                             init = init,
                                                             thin_scan = thin_scan, thin = thin/100,
                                                             gibbs = TRUE)$entropy))
  
  tmp <- capture.output(NR_wt <- coda::effectiveSize(nrMCMCtemp(y, K = K, 
                                                             alpha = alpha, sigma2 = sigma2,
                                                             inverse_temperatures = 1,
                                                             warm_up = N, N = N, thin = 1,
                                                             init = init,
                                                             thin_scan = thin_scan, thin = thin/100,
                                                             gibbs = FALSE)$entropy))
  
  #compute the ESS of quantities that are invariant to the labels
  tmp <- capture.output(MG <- coda::effectiveSize(nrMCMCtemp(y, K = K, 
                                                             alpha = alpha, sigma2 = sigma2,
                                                             inverse_temperatures = inverse_temperatures,
                                                             warm_up = N, N = N, thin = 1,
                                                             init = init,
                                                             thin_scan = thin_scan, thin = thin/100,
                                                             gibbs = TRUE)$entropy))
  
  tmp <- capture.output(NR <- coda::effectiveSize(nrMCMCtemp(y, K = K, 
                                                             alpha = alpha, sigma2 = sigma2,
                                                             inverse_temperatures = inverse_temperatures,
                                                             warm_up = N, N = N, thin = 1,
                                                             init = init,
                                                             thin_scan = thin_scan, thin = thin/100,
                                                             gibbs = FALSE)$entropy))
  ESSs <- as.numeric(c(MG_wt,NR_wt,MG,NR))
  
  #return the quantities of interest
  c(ESSs,MG_wt_tv,NR_wt_tv,MG_tv,NR_tv)
  
}

#simulation
B <- 100

#create the clusters
n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
if(is.na(n_cores)) n_cores <- 1 

#parallelize the for loop
library(doParallel)
library(foreach)

#initialize the clusters
cl <- parallel::makeCluster(n_cores)
registerDoParallel(cl)

#update the functions and objects
clusterExport(
  cl,
  c(
    "y1",
    "y2",
    "y3",
    "y4",
    "tv",
    "iteration",
    "inv_temps1",
    "inv_temps2",
    "inv_temps3",
    "inv_temps4"
  ),
  envir = environment()
)

iter <- function(thin_scan = 1000){
  
  clusterExport(
    cl,
    c(
      "thin_scan"
    ),
    envir = environment()
  )
  
  #replicas
  res1 <- foreach(
    j = seq_len(B),
    .packages = c("nrTemperdMixtures","coda"),   
    .errorhandling = "pass",
    .combine = "cbind"
  ) %dopar% {
    
    #set the seed
    set.seed(j)
    
    #iteration
    iteration(y = y1, thin_scan = thin_scan,
              inverse_temperatures = inv_temps1$inverse_temperatures)
    
  }
  
  #save the results
  #save(inv_temps1,res1, file = "results1.RData")
  
  #replicas
  res2 <- foreach(
    j = seq_len(B),
    .packages = c("nrTemperdMixtures","coda"),   
    .errorhandling = "pass",
    .combine = "cbind"
  ) %dopar% {
    
    #set the seed
    set.seed(j)
    
    #iteration
    iteration(y = y2, thin_scan = thin_scan,
              inverse_temperatures = inv_temps2$inverse_temperatures)
    
  }
  
  #save the results
  #save(inv_temps2,res2, file = "results2.RData")
  
  #replicas
  res3 <- foreach(
    j = seq_len(B),
    .packages = c("nrTemperdMixtures","coda"),   
    .errorhandling = "pass",
    .combine = "cbind"
  ) %dopar% {
    
    #set the seed
    set.seed(j)
    
    #iteration
    iteration(y = y3, thin_scan = thin_scan,
              inverse_temperatures = inv_temps3$inverse_temperatures)
    
  }
  
  #save the results
  #save(inv_temps3,res3, file = "results3.RData")
  
  #replicas
  res4 <- foreach(
    j = seq_len(B),
    .packages = c("nrTemperdMixtures","coda"),   
    .errorhandling = "pass",
    .combine = "cbind"
  ) %dopar% {
    
    #set the seed
    set.seed(j)
    
    #iteration
    iteration(y = y4, thin_scan = thin_scan,
              inverse_temperatures = inv_temps4$inverse_temperatures)
    
  }
  
  #save the results
  #save(inv_temps4,res4, file = "results4.RData")
  
  list(res1,res2,res3,res4)
}

thin_scans <- c(100,250,500,800,1000)
results <- vector("list",length(thin_scans))
for(i in seq_along(thin_scans)){
  results[[i]] <- iter(thin_scans[i])
  save(results, file = "resultss.RData")
}

#stop the clusters
stopCluster(cl)

load("results(1).RData")

# 
# load("results1.RData")
# load("results2.RData")
# load("results3.RData")
# load("results4.RData")
# plot(inv_temps1$temperature_search$grid,inv_temps1$temperature_search$objective, log = "x", type = "l")
# plot(inv_temps2$temperature_search$grid,inv_temps2$temperature_search$objective, log = "x", type = "l")
# plot(inv_temps3$temperature_search$grid,inv_temps3$temperature_search$objective, log = "x", type = "l")
# plot(inv_temps4$temperature_search$grid,inv_temps4$temperature_search$objective, log = "x", type = "l")
# inv_temps1$inverse_temperatures
# inv_temps2$inverse_temperatures
# inv_temps3$inverse_temperatures
# inv_temps4$inverse_temperatures
# 
#ESS
boxplot(t(res[1:4,]))
boxplot(t(res01[1:4,]))
boxplot(t(res01wt[1:4,]))
boxplot(t(reswt[1:4,]))
#ok
# 
# #convergence time
# {
#   res <- res1
#   x_grid <- seq_len(1000)
#   vals <- cbind(rowMeans(res[5:1004,]),
#                 rowMeans(res[1005:2004,]),
#                 rowMeans(res[2005:3004,]),
#                 rowMeans(res[3005:4004,]))
#   
#   matplot(vals, x = x_grid, xlab = "Iterations", ylab = "TV",
#           col = c("black","darkred","darkgreen","darkblue"),
#           lwd = 2, lty = 1, type = "l")
#   grid()
#   legend("topright", legend = c("MG","NR","Tempered","Tempered + NR"),
#          col = c("black","darkred","darkgreen","darkblue"),
#          lwd = 2, lty = 1, bg = "transparent", bty = "n")
#   
# }
# 
# 
# #ggplots
# 
# library(ggplot2)
# library(tidyr)
# library(gridExtra)
# 
# {
#   
#   #p1
#   p1 <- plot(NRMCMC::nrmix(y1, K = 3, alpha = c(1,1,1),
#                            sigma2 = 1, n_chains = 1)) + 
#     ggplot2::theme(legend.position = "none")
#   
#   #p2
#   x_grid <- seq_len(1000)
#   
#   vals1 <- cbind(
#     MG            = rowMeans(res1[5:1004, ]),
#     NR            = rowMeans(res1[1005:2004, ]),
#     Tempered      = rowMeans(res1[2005:3004, ]),
#     Tempered_NR = rowMeans(res1[3005:4004, ])
#   )
#   
#   vals4 <- cbind(
#     MG            = rowMeans(res4[5:1004, ]),
#     NR            = rowMeans(res4[1005:2004, ]),
#     Tempered      = rowMeans(res4[2005:3004, ]),
#     Tempered_NR = rowMeans(res4[3005:4004, ])
#   )
#   
#   
#   df_long1 <- pivot_longer(
#     data.frame(
#       Iterations = x_grid,
#       vals1
#     ),
#     cols = -Iterations,
#     names_to = "Method",
#     values_to = "TV"
#   )
#   
#   
#   df_long4 <- pivot_longer(
#     data.frame(
#       Iterations = x_grid,
#       vals4
#     ),
#     cols = -Iterations,
#     names_to = "Method",
#     values_to = "TV"
#   )
#   
#   p2 <- ggplot(df_long1,
#                aes(x = Iterations, y = TV, colour = Method, linetype = Method)) +
#     geom_line(linewidth = 1) +
#     scale_colour_manual(
#       name = NULL,
#       values = c(
#         "MG" = "black",
#         "NR" = "grey30",
#         "Tempered" = "grey55",
#         "Tempered_NR" = "grey75"
#       ),
#       labels = c("MG", "NR", "Tempered", "Tempered + NR")
#     ) +
#     scale_linetype_manual(
#       name = NULL,
#       values = c(
#         "MG" = "dotted",
#         "NR" = "dotdash",
#         "Tempered" = "dashed",
#         "Tempered_NR" = "solid"
#       ),
#       labels = c("MG", "NR", "Tempered", "Tempered + NR")
#     ) +
#     labs(x = "Iterations", y = "TV distance") +
#     theme_bw() +
#     theme(
#       panel.grid.major = element_line(colour = "grey85"),
#       panel.grid.minor = element_line(colour = "grey92"),
#       legend.position = c(0.75, 0.7),
#       legend.background = element_blank(),
#       legend.key = element_blank()
#     )
#   
#   #p3
#   
#   df1 <- as.data.frame(t(res1[1:4, ]))
#   df4 <- as.data.frame(t(res4[1:4, ]))
#   
#   colnames(df1) <- colnames(df4) <- c("MG", "NR", "Tempered", "Tempered_NR")
#   
#   df_long1_2 <- pivot_longer(
#     df1,
#     cols = everything(),
#     names_to = "Method",
#     values_to = "Value"
#   )
#   
#   df_long1_2$Method <- factor(
#     df_long1_2$Method,
#     levels = c("MG", "NR", "Tempered", "Tempered_NR"),
#     labels = c("MG", "NR", "Tempered", "Tempered + NR")
#   )
#   
#   df_long4_2 <- pivot_longer(
#     df4,
#     cols = everything(),
#     names_to = "Method",
#     values_to = "Value"
#   )
#   
#   df_long4_2$Method <- factor(
#     df_long4_2$Method,
#     levels = c("MG", "NR", "Tempered", "Tempered_NR"),
#     labels = c("MG", "NR", "Tempered", "Tempered + NR")
#   )
#   
#   p3 <- ggplot(df_long1_2, aes(x = Value, y = Method, fill = Method)) +
#     geom_boxplot(
#       width = 0.6,
#       outlier.shape = 16,
#       outlier.size = 1.5,
#       alpha = 0.8
#     ) +
#     scale_fill_manual(
#       values = c(
#         "MG" = "black",
#         "NR" = "grey30",
#         "Tempered" = "grey55",
#         "Tempered_NR" = "grey75"
#       )
#     ) +
#     labs(
#       x = "Entropy ESS",
#       y = NULL,
#       fill = NULL
#     ) +
#     theme_bw(base_size = 14) +
#     theme(
#       legend.position = "none",
#       panel.grid.major.y = element_blank(),
#       panel.grid.minor = element_blank()
#     )
#   
#   
#   #p4
#   p4 <- plot(NRMCMC::nrmix(y4, K = 3, alpha = c(1,1,1),
#                            sigma2 = 1, n_chains = 1)) + 
#     ggplot2::theme(legend.position = "none")
#   
#   #p5
#   p5 <- ggplot(df_long4,
#                aes(x = Iterations, y = TV, colour = Method, linetype = Method)) +
#     geom_line(linewidth = 1) +
#     scale_colour_manual(
#       name = NULL,
#       values = c(
#         "MG" = "black",
#         "NR" = "grey30",
#         "Tempered" = "grey55",
#         "Tempered_NR" = "grey75"
#       ),
#       labels = c("MG", "NR", "Tempered", "Tempered + NR")
#     ) +
#     scale_linetype_manual(
#       name = NULL,
#       values = c(
#         "MG" = "dotted",
#         "NR" = "dotdash",
#         "Tempered" = "dashed",
#         "Tempered_NR" = "solid"
#       ),
#       labels = c("MG", "NR", "Tempered", "Tempered + NR")
#     ) +
#     labs(x = "Iterations", y = "TV distance") +
#     theme_bw() +
#     theme(
#       panel.grid.major = element_line(colour = "grey85"),
#       panel.grid.minor = element_line(colour = "grey92"),
#       legend.position = c(0.75, 0.7),
#       legend.background = element_blank(),
#       legend.key = element_blank()
#     )
#   
#   #p6
#   p6 <- ggplot(df_long4_2, aes(x = Value, y = Method, fill = Method)) +
#     geom_boxplot(
#       width = 0.6,
#       outlier.shape = 16,
#       outlier.size = 1.5,
#       alpha = 0.8
#     ) +
#     scale_fill_manual(
#       values = c(
#         "MG" = "black",
#         "NR" = "grey30",
#         "Tempered" = "grey55",
#         "Tempered_NR" = "grey75"
#       )
#     ) +
#     labs(
#       x = "Entropy ESS",
#       y = NULL,
#       fill = NULL
#     ) +
#     theme_bw(base_size = 14) +
#     theme(
#       legend.position = "none",
#       panel.grid.major.y = element_blank(),
#       panel.grid.minor = element_blank()
#     )
#   
#   # p1: move ylab to title, remove xlab
#   p1 <- p1 +
#     labs(title = "Density", x = NULL, y = NULL) +
#     theme(
#       plot.title = element_text(hjust = 0.5),
#       axis.title.x = element_blank(),
#       #axis.text.x = element_blank(),
#       axis.ticks.x = element_blank()
#     )
#   
#   
#   # p2: move ylab to title, remove xlab
#   p2 <- p2 +
#     labs(
#       title = "TV distance",
#       x = NULL,
#       y = NULL
#     ) +
#     theme(
#       plot.title = element_text(hjust = 0.5),
#       #axis.text.x = element_blank(),
#       axis.ticks.x = element_blank()
#     )
#   
#   
#   # p3: move ylab to title, remove xlab
#   p3 <- p3 +
#     labs(
#       title = "Entropy ESS",
#       x = NULL,
#       y = NULL
#     ) +
#     theme(
#       plot.title = element_text(hjust = 0.5),
#       #axis.text.x = element_blank(),
#       axis.ticks.x = element_blank()
#     ) + xlim(c(0,100))
#   
#   p4 <- p4 +
#     labs(title = NULL, x = NULL, y = NULL) +
#     theme(
#       plot.title = element_text(hjust = 0.5),
#       axis.title.x = element_blank(),
#       #axis.text.x = element_blank(),
#       axis.ticks.x = element_blank()
#     )
#   
#   p5 <- p5 +
#     labs(
#       title = NULL,
#       x = "Iterations",
#       y = NULL
#     ) +
#     theme(
#       plot.title = element_text(hjust = 0.5),
#       #axis.text.x = element_blank(),
#       axis.ticks.x = element_blank()
#     )
#   
#   p6 <- p6 + 
#     labs(
#       title = NULL,
#       x = NULL,
#       y = NULL
#     ) +
#     theme(
#       plot.title = element_text(hjust = 0.5),
#       #axis.text.x = element_blank(),
#       axis.ticks.x = element_blank()
#     ) + xlim(c(0,1300))
#   
#   grid.arrange(p1, p2, p3,
#                p4, p5, p6,
#                ncol = 3, nrow = 2)
# }
# 
# 
# ###
# #function that estimate the comunication barrier
# estimate_barrier <- function(y,
#                              K = 3,
#                              alpha = rep(1,K),
#                              sigma2 = 1,
#                              mu0 = 0,
#                              lambda0 = 0.1,
#                              warm_up = 0,
#                              N = 1000,
#                              thin = 1,
#                              thin_scan = 0,
#                              gibbs = FALSE,
#                              reversible = FALSE,
#                              inverse_temperatures = NULL,
#                              TT = 30,
#                              n_iters = 10){
#   
#   require(nrTemperdMixtures)
#   
#   #start with a geometric grid
#   if(is.null(inverse_temperatures)){
#     inverse_temperatures <- seq(1,0,l = TT + 1)
#     #inverse_temperatures <- c(2^-seq(0,20, l = TT),0)
#   }
#   
#   #estimate the pairwise swap rejection probabilities
#   rs <- numeric(TT)
#   for(i in seq_len(n_iters)){
#     sinka <- capture.output(fit <- nrMCMCtemp(y, K = K, alpha = alpha, sigma2 = sigma2,
#                                               mu0 = mu0, lambda0 = lambda0,
#                                               inverse_temperatures = inverse_temperatures,
#                                               warm_up = warm_up, N = N, thin = thin,
#                                               thin_scan = thin_scan,
#                                               gibbs = gibbs, reversible = reversible))
#     rs <- 1 - as.numeric(fit$swaps)
#     
#     #cumulate them
#     Lambda <- cumsum(rs)
#     
#     #fit a monotone spline
#     sp <- suppressWarnings(splinefun(Lambda/sum(rs),
#                                      log(rev(inverse_temperatures)[-1]),
#                                      method = "monoH.FC"))
#     
#     plot(rev(inverse_temperatures)[-1],Lambda,
#          type = "o")
#      
#     #update the interval
#     inverse_temperatures <- c(1,exp(sp(seq(1,0, l = TT)[-1])),0)
#     
#   }
#   
#   #return the inverse temperature
#   inverse_temperatures
#   
# }
# 
# K = 3
# alpha = rep(1,K)
# sigma2 = 1
# mu0 = 0
# lambda0 = 0.1
# warm_up = 0
# N = 1000
# thin = 1
# thin_scan = 0
# gibbs = FALSE
# reversible = FALSE
# inverse_temperatures = NULL
# TT = 30
# 
# 
# acc <- function(x,inv_temp = 1){
#   sinka <- capture.output(
#     fit <- nrMCMCtemp(y, K = 3, sigma2 = 1,
#                       inverse_temperatures = c(inv_temp,x))
#   )
#   fit$swaps[NROW(fit$swaps)]
# }
# 
