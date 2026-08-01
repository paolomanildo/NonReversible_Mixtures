source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 2 - simulation from the Gaussian case
set.seed(123)
figure2_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 3,
                             kernel = "gaussian",
                             mu0 = 0, lambda0 = 1, sigma2 = 1,
                             sample_from_model = TRUE)

cloud_plot(figure2_top$MG, gg = FALSE)
#cloud_plot(figure2_top$R, gg = TRUE)
cloud_plot(figure2_top$NR, gg = FALSE)
violin(figure2_top$MG[,1,],figure2_top$NR[,1,])

figure2_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                alpha = 0.1, K = 3,
                                kernel = "gaussian",
                                mu0 = 0, lambda0 = 1, sigma2 = 1,
                                sample_from_model = TRUE)

cloud_plot(figure2_bottom$MG, gg = FALSE)
#cloud_plot(figure2_bottom$R, gg = FALSE)
cloud_plot(figure2_bottom$NR, gg = FALSE)
violin(figure2_bottom$MG[,1,],figure2_bottom$NR[,1,])

