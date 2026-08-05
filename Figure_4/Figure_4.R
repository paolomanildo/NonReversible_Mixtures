#ANALYSIS
load("results01.RData")

n_grid = round(10^seq(3, 4, length.out = 5))
d_grid = c(1,3,5)
grea <- as.matrix(expand.grid(n_grid,d_grid))
N <- (dim(res[[1]])[1] - 8) / 8

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

#ESS
ESS <- sapply(res, function(x) {
  t(sapply(seq_len(NCOL(x)),function(i){
    x[8*N + 1:4,i]
  }, simplify = "array"))
}, simplify = "array")

sESS <- sapply(res, function(x) {
  t(sapply(seq_len(NCOL(x)),function(i){
    x[8*N + 1:4,i] / x[8*N  + 5:8,i] 
  }, simplify = "array"))
}, simplify = "array")

#simulate from the real prior
prior_draws <- sapply(n_grid,rprior,prior_only = TRUE,
                      N = 1e4, alpha = .1,
                      simplify = "array")

#compute the TV distances

#get the common support
x_lim <- range(prior_draws,entropies, na.rm = TRUE)
x_lim <- x_lim + c(-1,1) * 0.001 * diff(x_lim)

Y_vals <- apply(prior_draws,3,function(y){
  p0 <- mean(y == 0)
  c(p0,(1-p0) * density(y[y != 0],from = x_lim[1], to = x_lim[2], n = 512)$y)
})

TV <- function(x,y){
  
  #compute the two densities
  x_grid <- seq(x_lim[1],x_lim[2],l = 512)
  p0 <- mean(x == 0)
  X_val <- (1-p0) * density(x[x != 0],from = x_lim[1], to = x_lim[2], n = 512)$y

  #compute the TV distance with the trapezoidal rule:
  
  #save the absolute differences for the continouos part
  abs_diff <- abs(X_val - y[-1])
  
  #return the integral using the trapezoidal rule
  0.5 * ( abs(p0 - y[1]) + 
            sum( diff(x_grid) * (head(abs_diff, -1) + tail(abs_diff, -1)) / 2 ))
  
}

TVs <- vector("list",NROW(grea))
for(cc in seq_along(TVs)){
  
  #entropy
  TVs[[cc]]$entropy <- sapply(seq_len(dim(entropies)[2]), function(s) {
    ccc <- which(n_grid == grea[cc,1])[1]
    sapply(seq_len(dim(entropies)[1]), function(i){
      TV(entropies[i,s,cc,],Y_vals[,ccc])
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

{
  X11()
  par(mfrow = c(5,3), mar = c(.1,2.1,.1,.1))
  for(i in 1:15){
    matplot(TVs[[i]]$entropy, type = "l")
  }
}

#determine the number of predictive call to reach convergence
#for each scenario, then at convergence, take 1000 iteration and compute the ESS
T2C <- function(X){
  
  #get the TV upper quantile of SAM_NR after burn in
  q05 <- quantile(X$entropy[-(1:500),4],0.5)
  
  #get the number of predictive calls
  n_calls <- sapply(seq_along(1:4), function(i){
    n_iter <- which(X$entropy[,i] <= q05)[1]
    if(is.na(n_iter)) n_iter <- 1000
    sum(X$n_pred_calls[seq_len(n_iter),i])
  })
  
  #return it
  n_calls
  
}

#compute the number of predictive calls to convergence
n_preds_to_conv <- t(sapply(TVs,T2C))

#plot
library(ggplot2)
library(patchwork)

#create the data frame
df <- expand.grid(n = n_grid,d = d_grid,
                  sampler = c("MG","NR","NUSAMS(0,5,1)","NUSAMS(0,5,1) + NR"))
df$T2C <- rep(NA,NROW(df))
for(i in seq_len(NROW(grea))){
  idx <- sapply(1:4,function(j) NROW(grea)*(j-1) + i)
  
  df$T2C[idx] <- T2C(TVs[[i]])
}
df$ESS <- c(apply(ESS,1:2,mean))
df$sESS <- c(apply(sESS,1:2,mean))

#time to convergence plots
p1 <- ggplot(
  df,
  aes(
    x = n,
    y = T2C,
    color = sampler,
    shape = sampler,
    group = sampler
  )
) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.8) +
  facet_wrap(
    ~d,
    nrow = 1,
    labeller = label_bquote(d == .(d))
  ) +
  labs(
    y = "N° predictives to converge\n(log scale)",
    color = NULL,
    shape = NULL
  ) +
  scale_x_log10() +
  scale_y_log10() +
  scale_shape_manual(
    values = c(16, 17, 15, 18)
  ) +
  scale_color_manual(
    values = c("black", "grey35", "grey55", "grey75")
  ) +
  theme_bw() +
  theme(
    legend.position = "none",
    legend.key.width = unit(1.7, "cm"),
    legend.text.position = "left",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_text(size = 8)
  )

#ESS plot
p2 <- ggplot(
  df,
  aes(
    x = n,
    y = ESS,
    color = sampler,
    shape = sampler,
    group = sampler
  )
) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.8) +
  facet_wrap(
    ~d,
    nrow = 1,
    labeller = label_bquote(d == .(d))
  ) +
  labs(
    x = "Sample size (log scale)",
    y = "ESS (log scale)",
    color = NULL,
    shape = NULL
  ) +
  scale_x_log10() +
  scale_y_log10() +
  scale_shape_manual(
    values = c(16, 17, 15, 18)
  ) +
  scale_color_manual(
    values = c("black", "grey35", "grey55", "grey75")
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    strip.text = element_blank(),
    legend.key.width = unit(1.7, "cm"),
    legend.text.position = "left",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank(),
    axis.title.y = element_text(size = 8)
  )

#scaled ESS
p3 <- ggplot(
  df,
  aes(
    x = n,
    y = sESS,
    color = sampler,
    shape = sampler,
    group = sampler
  )
) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.8) +
  facet_wrap(
    ~d,
    nrow = 1,
    labeller = label_bquote(d == .(d))
  ) +
  labs(
    x = "Sample size (log scale)",
    y = "ESS per predictive\n(log scale)",
    color = NULL,
    shape = NULL
  ) +
  scale_x_log10() +
  scale_y_log10() +
  scale_shape_manual(
    values = c(16, 17, 15, 18)
  ) +
  scale_color_manual(
    values = c("black", "grey35", "grey55", "grey75")
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    strip.text = element_blank(),
    legend.key.width = unit(1.7, "cm"),
    legend.text.position = "left",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank(),
    axis.title.y = element_text(size = 8)
  )

#merge the plots
p1 / p3 +
  plot_layout(
    heights = c(1, 1.15)
  )
