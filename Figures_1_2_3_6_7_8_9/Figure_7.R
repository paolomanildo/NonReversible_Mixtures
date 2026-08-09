source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 7 - simulation from the prior distribution
set.seed(123)
figure7_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 3,
                             kernel = "partition",
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               numeric(n)
                             })

cloud_plot(figure7_top$MG, gg = FALSE)
#cloud_plot(figure7_top$R, gg = FALSE)
cloud_plot(figure7_top$NR, gg = FALSE, title = "Non-reversible")
violin(figure7_top$MG[,1,],figure7_top$NR[,1,], gg = TRUE)

figure7_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                alpha = 0.1, K = 3,
                                kernel = "partition",
                                sample_from_model = FALSE,
                                sample_data = function(n){
                                  numeric(n)
                                })

cloud_plot(figure7_bottom$MG, gg = FALSE)
#cloud_plot(figure7_bottom$R, gg = FALSE)
cloud_plot(figure7_bottom$NR, gg = FALSE, title = "Non-reversible")
violin(figure7_bottom$MG[,1,],figure7_bottom$NR[,1,], gg = TRUE)
