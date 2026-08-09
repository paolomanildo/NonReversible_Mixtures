source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 8 - high dimensional example
set.seed(123)
figure8 <- cluster_sizes(n = 1000, d = 18, N = 100, B = 500, 
                         alpha = c(4,1,1,1,1), K = 5,
                         mu0 = rep(0,18),
                         sigma2 = 18 * 2,
                         lambda0 = 18 * 4,
                         sample_from_model = TRUE)


histo(figure8$MG); curve(dbeta(x,4,4), add = TRUE, col = "gray")
histo(figure8$NR,"Non-reversible");  curve(dbeta(x,4,4), add = TRUE, col = "gray")
violin(figure8$MG[,1,],figure8$NR[,1,])
