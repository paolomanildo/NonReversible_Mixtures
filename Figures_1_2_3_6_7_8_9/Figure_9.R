source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 9 - simulation for the Poisson kernel
set.seed(123)
figure9_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                              alpha = 1, K = 3,
                              kernel = "poisson",
                              alpha0 = 1, beta0 = 1,
                              sample_from_model = TRUE)

cloud_plot(figure9_top$MG, gg = FALSE)
#cloud_plot(figure9_top$R, gg = FALSE)
cloud_plot(figure9_top$NR, gg = FALSE, title = "Non-reversible")
violin(figure9_top$MG[,1,],figure9_top$NR[,1,])

figure9_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                 alpha = 0.1, K = 3,
                                 kernel = "poisson",
                                 alpha0 = 1, beta0 = 1,
                                 sample_from_model = TRUE)

cloud_plot(figure9_bottom$MG, gg = FALSE)
#cloud_plot(figure9_top$R, gg = TRUE)
cloud_plot(figure9_bottom$NR, gg = FALSE, title = "Non-reversible")
violin(figure9_bottom$MG[,1,],figure9_bottom$NR[,1,])
