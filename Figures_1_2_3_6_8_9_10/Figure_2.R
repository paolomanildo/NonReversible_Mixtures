source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 2
figure2_top <- cluster_sizes(n = 1000, N = 100, B = 300,
                             alpha = 1, K = 3,
                             kernel = "partition",
                             sample_from_model = FALSE,
                             sample_data = function(n){
                               numeric(n)
                             })

violin(figure2_top$MG[,1,],figure2_top$NR[,1,], gg = TRUE)
cloud_plot(figure2_top$MG, gg = TRUE)
cloud_plot(figure2_top$R, gg = TRUE)
cloud_plot(figure2_top$NR, gg = TRUE)

figure2_bottom <- cluster_sizes(n = 1000, N = 100, B = 300,
                                alpha = 0.1, K = 3,
                                kernel = "partition",
                                sample_from_model = FALSE,
                                sample_data = function(n){
                                  numeric(n)
                                })

violin(figure2_bottom$MG[,1,],figure2_bottom$NR[,1,], gg = TRUE)
cloud_plot(figure2_bottom$MG, gg = TRUE)
cloud_plot(figure2_bottom$R, gg = TRUE)
cloud_plot(figure2_bottom$NR, gg = TRUE)
