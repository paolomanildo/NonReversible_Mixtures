### FIGURE 1: LARGEST COMPONENT MARKOV CHAIN SIMULATION

#function for a generic simulation that returns the cluster size trace
cluster_sizes <- function(B = 300, #simulation replicas
                          n = 2000, #sample size
                          d = 1, #dimension
                          N = 150, #chains iteration
                          K = 2, #number of clusters
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
  require(nrNUSAMS)
  
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
    
    #sample the initial value for the chain
    init <- sample(K_init,n,TRUE)
    
    #get the configuration Markov chain for the MG
    tmp <- capture.output(MG <- nrMCMC(y,K = K, alpha = alpha,
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
    tmp <- capture.output(R <- nrMCMC(y,K = K, alpha = alpha,
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
    tmp <- capture.output(NR <- nrMCMC(y,K = K, alpha = alpha,
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

### VISUALIZATION FUNCTIONS

### spaghetti plot of the largest component markov chain
spaghetti <- function(X,
                      title = "",
                      gg = FALSE,
                      ylim = c(0.5,1),
                      col = adjustcolor("darkgray",0.3)){
  
  #ggplots?
  if(gg){
    
    #create the data frame for ggplot
    df <- data.frame(prop = c(X),
                     chain_iter = rep(seq_len(NROW(X)),NCOL(X)),
                     sim_iter = rep(seq_len(NCOL(X)),each = NROW(X)))
    
    #create also the data frame with the mean line
    df_mean <- data.frame(prop = apply(X,1,mean),
                          chain_iter = seq_len(NROW(X)),
                          sim_iter = 1)
    
    ggplot2::ggplot(df, ggplot2::aes(x = chain_iter,
                                     y = prop,
                                     group = sim_iter)) +
      ggplot2::geom_line(color = col) +
      ggplot2::geom_line(data = df_mean, 
                         ggplot2::aes(x = chain_iter, y = prop),
                         color = "black", linewidth = 1) +  
      ggplot2::labs(x = "Iterations",
                    y = "First Component Proportion",
                    title = title) +
      ggplot2::theme_minimal() +
      ggplot2::theme(legend.position = "none") 
    
  }else{
    
    #get the dimensions
    N <- NROW(X)
    B <- NCOL(X)
    
    #empty plot
    plot(c(1,N),ylim,xlab = "Iterations",
         ylab = "First component proportion",
         main = title,
         type = "n")
    
    #add the lines
    for(b in seq_len(B)){
      lines(seq_len(N),X[,b], col = col)
    }
    #add the mean line
    lines(seq_len(N),apply(X,1,mean),col = "black")
  }
  
}

### violin plots of the marginal distribution evolution
violin <- function(X,
                   Y,
                   title = "Chains Marginal Distribution",
                   ylim = c(0,1),
                   col1 = "black",
                   col2 = "darkgray",
                   gg = FALSE, by = 10){
  
  
  if(gg){
    
    #get the range for the densities
    x_lim <- range(X, Y)
    
    #create the grid
    x_grid <- seq(x_lim[1], x_lim[2], length.out = 512)
    
    #create the time stamps
    tt <- c(1, seq(0, NROW(X), by = 10)[-1])
    
    #initialize the list with the densities
    polygon_data_list <- list()
    
    #loop over each time stamp
    for (i in seq_along(tt)) {
      
      #get the densities
      dens_X <- density(X[1:tt[i],], from = x_lim[1], to = x_lim[2])$y
      dens_Y <- density(Y[1:tt[i],], from = x_lim[1], to = x_lim[2])$y
      
      #normalize them
      dens_X <- dens_X / max(dens_X)
      dens_Y <- dens_Y / max(dens_Y)
      
      #create a data frame with all the information
      df_Y <- data.frame(
        x = c(tt[i] - dens_Y * 4.5, rep(tt[i], length(x_grid))),
        y = c(x_grid, rev(x_grid)),
        type = "Y",
        iteration = tt[i]  
      )
      
      df_X <- data.frame(
        x = c(tt[i] + dens_X * 4.5, rep(tt[i], length(x_grid))),
        y = c(x_grid, rev(x_grid)),
        type = "X",
        iteration = tt[i]  
      )
      
      #put them into the list      
      polygon_data_list[[i]] <- rbind(df_Y, df_X)
    }
    
    #concatentate the list
    polygon_data <- do.call("rbind",polygon_data_list)
    
    #create the plot
    ggplot2::ggplot(polygon_data, ggplot2::aes(x = x, y = y, fill = type)) +
      ggplot2::geom_polygon(ggplot2::aes(group = interaction(type, iteration)), color = NA, alpha = 0.7) +
      ggplot2::scale_fill_manual(values = c("Y" = col1, "X" = col2)) +
      ggplot2::scale_x_continuous(name = "Iterations") +
      ggplot2::scale_y_continuous(name = "First component proportion", breaks = tt) +
      ggplot2::theme_minimal() +
      ggplot2::ggtitle(title) +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
        legend.position = "none"  
      )
    
  }else{
    
    #every 10 iterations compute the empirical density
    x_lim <- range(X,Y)
    x_grid <- seq(x_lim[1],x_lim[2], l = 512)
    tt <- c(1,seq(0,NROW(X), by = by)[-1])
    X_densities <- sapply(tt,function(ttt) {
      tmp <- density(X[seq_len(ttt),], from = x_lim[1], to = x_lim[2])$y
      tmp/max(tmp)
    })
    Y_densities <- sapply(tt,function(ttt) {
      tmp <- density(Y[seq_len(ttt),], from = x_lim[1], to = x_lim[2])$y
      tmp/max(tmp)
    })
    
    #empty plot
    plot(c(1,NROW(X)),x_lim, xlab = "Iterations",
         ylab = "First component proportion", xaxt = "n",
         type = "n", main = title)
    axis(1,tt,tt)
    
    #add the densities
    for(i in seq_along(tt)){
      
      #Y_densities on the left
      polygon(c(tt[i] - Y_densities[,i]*by*0.45,rep(tt[i],length(x_grid))),
              c(x_grid,rev(x_grid)), col = col1, border = NA)
      
      #X_densities on the right
      polygon(c(tt[i] + X_densities[,i]*by*0.45,rep(tt[i],length(x_grid))),
              c(x_grid,rev(x_grid)), col = col2, border = NA)
      
    }
    
  }
  
}

#scatter plot
cloud_plot <- function(X, which = 1:2){
  
  #take the tail
  plot(t(X[NROW(X),which,]), xlim = c(0,1), ylim = c(0,1),
       xlab = "", ylab = "", pch = 16, cex = .5)
  
}

#histogram
histo <- function(X, title = "Marginal"){
  
  hist(X[NROW(X),1,], bty = "n", col = "black", prob = TRUE,
       xlab = "", ylab = "", main = title )

}

### FIGURE 1

figure1 <- cluster_sizes(n = 2000, N = 150, B = 100,
                         alpha = 0.5, K = 2,
                         mu0 = 0, lambda0 = 1, sigma2 = 1,
                         sample_from_model = FALSE,
                         sample_data = function(n){
                           z <- sample(1:2,n,TRUE,c(0.9,0.1))
                           rnorm(n,c(0.9,-0.9)[z],1)
                         })

spaghetti(apply(figure1$MG,c(1,3), max), title = "Marginal")
spaghetti(apply(figure1$R,c(1,3), max), title = "Reversible")
spaghetti(apply(figure1$NR,c(1,3), max), title = "Non Reversible")

violin(apply(figure1$MG,c(1,3), max),apply(figure1$NR,c(1,3), max))

### FIGURE 2

figure2_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 3,
                             kernel = "partition",
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               numeric(n)
                             })

violin(figure2_top$MG[,1,],figure2_top$NR[,1,])
cloud_plot(figure2_top$MG)
cloud_plot(figure2_top$R)
cloud_plot(figure2_top$NR)

figure2_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                alpha = 0.1, K = 3,
                                kernel = "partition",
                                sample_from_model = FALSE,
                                sample_data = function(n){
                                  numeric(n)
                                })

violin(figure2_bottom$MG[,1,],figure2_bottom$NR[,1,])
cloud_plot(figure2_bottom$MG)
cloud_plot(figure2_bottom$R)
cloud_plot(figure2_bottom$NR)

### FIGURE 3

figure3_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 3,
                             kernel = "gaussian",
                             mu0 = 0, lambda0 = 1, sigma2 = 1,
                             sample_from_model = TRUE)

violin(figure3_top$MG[,1,],figure3_top$NR[,1,])
cloud_plot(figure3_top$MG)
cloud_plot(figure3_top$NR)

figure3_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                alpha = 0.1, K = 3,
                                kernel = "gaussian",
                                mu0 = 0, lambda0 = 1, sigma2 = 1,
                                sample_from_model = TRUE)

violin(figure3_bottom$MG[,1,],figure3_bottom$NR[,1,])
cloud_plot(figure3_bottom$MG)
cloud_plot(figure3_bottom$NR)

### FIGURE 4

figure4 <- cluster_sizes(n = 1000, d = 18, N = 100, B = 500, 
                         alpha = c(4,1,1,1,1), K = 5,
                         mu0 = rep(0,18),
                         sigma2 = 18 * 2,
                         lambda0 = 18 * 4,
                         sample_from_model = TRUE)


histo(figure4$MG); curve(dbeta(x,4,4), add = TRUE, col = "gray")
histo(figure4$NR,"Non reversible");  curve(dbeta(x,4,4), add = TRUE, col = "gray")
violin(figure4$MG[,1,],figure4$NR[,1,])

### FIGURE 5

figure5_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 2, 
                             mu0 = 0, lambda0 = 1, sigma2 = 1,
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               rnorm(n,2,1)
                             })

histo(figure5_top$MG)
histo(figure5_top$NR)
violin(figure5_top$MG[,1,],figure5_top$NR[,1,])

figure5_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 0.1, K = 2, K_init = 1, 
                             mu0 = 0, lambda0 = 1, sigma2 = 1,
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               rnorm(n,2,1)
                             })

histo(figure5_bottom$MG)
histo(figure5_bottom$NR)
violin(figure5_bottom$MG[,1,],figure5_bottom$NR[,1,])

### FIGURE 6
n <- 1000
B <- 300
N <- 50
figure6 <- sapply(c(3,10,20,50), function(K) {
  
  #initialize the output
  PNR_cluster_sizes <- QNR_cluster_sizes <- matrix(NA,N,B)
  
  #set alpha
  alpha <- c(1,rep(1/(K-1),K-1))
  
  #fill them
  for(b in seq_len(B)){
    
    #sample the initial value for the chain
    init <- sample(K,n,TRUE)
    
    #get the configuration Markov chain for the classic non-reversible algorithm
    tmp <- capture.output(PNR <- nrMCMC(numeric(n),K = K, alpha = alpha,
                                       N = N-1,warm_up = 0,
                                       kernel = "partition", init = init,
                                       save_configurations = TRUE))
    PNR_lbls <- get_labels_draws(PNR) + 1
    
    #store the cluster size trace
    PNR_cluster_sizes[,b] <- apply(rbind(c(init),PNR_lbls),1,function(x) {
      ns <- sapply(seq_len(K),function(k) sum(x == k))
      ns[1] / sum(ns)
    })
    
    #do the same for its variant
    tmp <- capture.output(QNR <- nrMCMC(numeric(n),K = K, alpha = alpha,
                                        N = N-1,warm_up = 0,
                                        kernel = "partition", init = init,
                                        save_configurations = TRUE))
    QNR_lbls <- get_labels_draws(QNR) + 1
    
    #store the cluster size trace
    QNR_cluster_sizes[,b] <- apply(rbind(c(init),QNR_lbls),1,function(x) {
      ns <- sapply(seq_len(K),function(k) sum(x == k))
      ns[1] / sum(ns)
    })
    
    #update the user
    cat("\n",b,"over",B)
    
  }
  
  #return the two arrays
  list(PNR = PNR_cluster_sizes, QNR = QNR_cluster_sizes)
})

violin(figure6[1,1][[1]],figure6[2,1][[1]], by = 5)
violin(figure6[1,2][[1]],figure6[2,2][[1]], by = 5)
violin(figure6[1,3][[1]],figure6[2,3][[1]], by = 5)
violin(figure6[1,4][[1]],figure6[2,4][[1]], by = 5)

### FIGURE 7

figure7_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 3,
                             kernel = "poisson",
                             alpha0 = 1, beta0 = 1,
                             sample_from_model = TRUE)

violin(figure7_top$MG[,1,],figure7_top$NR[,1,])
cloud_plot(figure7_top$MG)
cloud_plot(figure7_top$NR)

figure7_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                alpha = 0.1, K = 3,
                                kernel = "poisson",
                                alpha0 = 1, beta0 = 1,
                                sample_from_model = TRUE)

violin(figure7_bottom$MG[,1,],figure7_bottom$NR[,1,])
cloud_plot(figure7_bottom$MG)
cloud_plot(figure7_bottom$NR)


### REFEREE

n <- 2000
N <- 150
B <- 100

#load the package
require(nrNUSAMS)

xi_grid <- c(0.1,0.3,0.5,
             1,3,5,
             10,30,50,
             100,300,500,
             1000)

#initialize the output
largest_clusters <- array(NA, dim = c(N,length(xi_grid),B))

#fill them
for(b in seq_len(B)){
  
  #generate the data
  set.seed(b)
  z <- sample(2,n,TRUE,c(0.9,0.1))
  y <- rnorm(n,c(0.9,-0.9)[z],1)
  
  #sample the initial value for the chain
  init <- sample(2,n,TRUE)
  
  for(i in seq_along(xi_grid)){
    #get the configuration Markov chain
    tmp <- capture.output(NR <- nrMCMC(y,K = 2, alpha = 0.5,
                                       mu0 = 0, lambda0 = 1, sigma2 = 1,
                                       N = N-1,warm_up = 0,
                                       xi = xi_grid[i],
                                       init = init,
                                       save_configurations = TRUE))
    lbls <- get_labels_draws(NR) + 1
    largest_clusters[,i,b] <- apply(rbind(c(init),lbls),1,function(x) {
      ns <- sapply(seq_len(2),function(k) sum(x == k))
      max(ns) / sum(ns)
    })
    
  }
  
  #update the user
  cat("\n",b,"over",B)
  
}

matplot(apply(largest_clusters,1:2,mean), type = "l", lty = c(1:3,1:3,1:3,1:3,1), col = c(1,1,1,
                                                                            2,2,2,
                                                                            3,3,3,
                                                                            4,4,4,5))
legend("bottomright",
       legend = c("0.1–0.5",
                  "1–5",
                  "10–50",
                  "100–500",
                  "1000"),
       col = c(1:4,5),
       lty = c(1,1,1,1,1),
       cex = 0.9,
       bty = "n",
       title = expression(xi))

{
  y_vals <- apply(largest_clusters,1:2,mean)
  
  xlim <- c(1,150)
  ylim <- range(y_vals,1)
  
  plot(xlim,ylim, type = "n", xlab = "", ylab = "")
  polygon(c(seq_len(150),
            rev(seq_len(150))),
          c(apply(y_vals[,1:3],1,min),rev(apply(y_vals[,1:3],1,max))),
          col = adjustcolor("black",0.8), border = NA)
  
  polygon(c(seq_len(150),
            rev(seq_len(150))),
          c(apply(y_vals[,4:6],1,min),rev(apply(y_vals[,4:6],1,max))),
          col = adjustcolor("darkgray",0.8),  border = NA)
  
  polygon(c(seq_len(150),
            rev(seq_len(150))),
          c(apply(y_vals[,7:9],1,min),rev(apply(y_vals[,7:9],1,max))), 
          col = adjustcolor("gray",0.8), border = NA)
  
  polygon(c(seq_len(150),
            rev(seq_len(150))),
          c(apply(y_vals[,10:12],1,min),rev(apply(y_vals[,10:12],1,max))), 
          col = adjustcolor("lightgray",0.8), border = NA)
  
  lines(seq_len(150),y_vals[,13], col = "darkred")
  
  legend("bottomright",
         legend = c("[0.10.5]",
                    "[1,5]",
                    "[10,50]",
                    "[100,500]",
                    "1000"),
         col = c("black","lightgray","gray","darkgray","darkred"),
         bty = "n",
         lwd = 3,
         title = expression("      " ~ xi))
}

#ok
colors <- c(rev(sapply(colorRampPalette(c("gray","black"))(10),adjustcolor,1)),
            rev(sapply(colorRampPalette(c("yellow","darkred"))(14),adjustcolor,1)))


colors <- rev(sapply(colorRampPalette(c("lightgray","gray","black"))(length(xi_grid)),adjustcolor,1))

pal <- colorRampPalette(c("white","gray","black"))
colors <- rev(pal(200)[as.numeric(cut(xi_grid, breaks = 200))])

matplot(apply(largest_clusters,1:2,mean),
        type = "l", col = colors,
        lty = 1, xlab = "", ylab = "", ylim = c(0.5,1))

#add the reversible one
lines(apply(apply(figure1$R,c(1,3), max),1,mean), col = "red")
