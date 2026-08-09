source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 7 - simulation from the prior distribution
set.seed(123)
figure8_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 3,
                             kernel = "partition",
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               numeric(n)
                             })

cloud_plot(figure8_top$MG, gg = FALSE)
#cloud_plot(figure8_top$R, gg = FALSE)
cloud_plot(figure8_top$NR, gg = FALSE, title = "Non-reversible")
violin(figure8_top$MG[,1,],figure8_top$NR[,1,], gg = TRUE)

figure8_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                alpha = 0.1, K = 3,
                                kernel = "partition",
                                sample_from_model = FALSE,
                                sample_data = function(n){
                                  numeric(n)
                                })

cloud_plot(figure8_bottom$MG, gg = FALSE)
#cloud_plot(figure8_bottom$R, gg = FALSE)
cloud_plot(figure8_bottom$NR, gg = FALSE, title = "Non-reversible")
violin(figure8_bottom$MG[,1,],figure8_bottom$NR[,1,], gg = TRUE)
