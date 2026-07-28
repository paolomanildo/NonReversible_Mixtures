source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 3 - overfitted mixture simulation
figure3_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 2, K_init = 1,
                             mu0 = 0, lambda0 = 1, sigma2 = 1,
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               rnorm(n,2,1)
                             })

histo(figure3_top$MG)
histo(figure3_top$NR)
violin(figure3_top$MG[,1,],figure3_top$NR[,1,], gg = TRUE)

figure3_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 0.1, K = 2, K_init = 1, 
                             mu0 = 0, lambda0 = 1, sigma2 = 1,
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               rnorm(n,2,1)
                             })

histo(figure3_bottom$MG)
histo(figure3_bottom$NR)
violin(figure3_bottom$MG[,1,],figure3_bottom$NR[,1,], gg = TRUE)
