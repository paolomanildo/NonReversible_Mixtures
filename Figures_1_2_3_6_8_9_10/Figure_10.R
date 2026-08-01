source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 10 - simulation for the Poisson kernel
set.seed(123)
figure10_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                              alpha = 1, K = 3,
                              kernel = "poisson",
                              alpha0 = 1, beta0 = 1,
                              sample_from_model = TRUE)

cloud_plot(figure10_top$MG, gg = FALSE)
#cloud_plot(figure10_top$R, gg = FALSE)
cloud_plot(figure10_top$NR, gg = FALSE, title = "Non-reversible")
violin(figure10_top$MG[,1,],figure10_top$NR[,1,])

figure10_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                 alpha = 0.1, K = 3,
                                 kernel = "poisson",
                                 alpha0 = 1, beta0 = 1,
                                 sample_from_model = TRUE)

cloud_plot(figure10_bottom$MG, gg = FALSE)
#cloud_plot(figure10_top$R, gg = TRUE)
cloud_plot(figure10_bottom$NR, gg = FALSE, title = "Non-reversible")
violin(figure10_bottom$MG[,1,],figure10_bottom$NR[,1,])


