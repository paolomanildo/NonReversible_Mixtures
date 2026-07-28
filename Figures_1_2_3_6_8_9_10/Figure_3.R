source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 3 - overfitted mixture simulation
figure3 <- cluster_sizes(n = 1000, d = 18, N = 100, B = 500, 
                         alpha = c(4,1,1,1,1), K = 5,
                         mu0 = rep(0,18),
                         sigma2 = 18 * 2,
                         lambda0 = 18 * 4,
                         sample_from_model = TRUE)


histo(figure3$MG); curve(dbeta(x,4,4), add = TRUE, col = "gray", gg = TRUE)
histo(figure3$NR,"Non-reversible");  curve(dbeta(x,4,4), add = TRUE, col = "gray", gg = TRUE)
violin(figure3$MG[,1,],figure3$NR[,1,], gg = TRUE)
