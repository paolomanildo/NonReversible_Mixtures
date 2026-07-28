library(ggplot2)
library(patchwork)

#load the data
load("processed_output01.RData")

#set the grid values
n_grid = round(10^seq(3, 4, length.out = 5))
d_grid = c(1,3,5)
grea <- as.matrix(expand.grid(n_grid,d_grid))
N <- dim(res[[1]])[1] / 8

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

#create the data frame
df <- expand.grid(n = n_grid,d = d_grid,
                    sampler = c("MG","NR","NUSAMS(0,5,1)","NUSAMS(0,5,1) + NR"))
df$T2C <- rep(NA,NROW(df))
for(i in seq_len(NROW(grea))){
  idx <- sapply(1:4,function(j) NROW(grea)*(j-1) + i)
  
  df$T2C[idx] <- T2C(TVs[[i]])
}
df$ESS <- c(t(apply(ESS,1:2,median)))
df$ESSs <- c(t(apply(ESSs,1:2,median)))

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
    y = "N° predictives to converge (log scale)",
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
    y = "ESS per predictive (log scale)",
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
    y = ESSs,
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
    y = "ESS per predictive (log scale)",
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
