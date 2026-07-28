source("simulate_cluster_sizes.R")
source("plot_functions.R")

### FIGURE 1
n <- 2000
set.seed(1920)
z <- sample(1:2,n,TRUE,c(0.9,0.1))
y <- rnorm(n,c(0.9,-0.9)[z],1)
figure1 <- cluster_sizes(n = 2000, N = 150, B = 100,
                         alpha = 0.5, K = 2,
                         mu0 = 0, lambda0 = 1, sigma2 = 1,
                         sample_from_model = FALSE,
                         sample_data = function(n){
                           y
                         })

spaghetti(apply(figure1$MG,c(1,3), max), title = "Marginal Gibbs", which_show = 18, gg = FALSE)
spaghetti(apply(figure1$R,c(1,3), max), title = "Reversible", which_show = 18, gg = FALSE)
spaghetti(apply(figure1$NR,c(1,3), max), title = "Non-Reversible", which_show = 18, gg = FALSE)


violin(apply(figure1$MG,c(1,3), max),apply(figure1$NR,c(1,3), max), gg = TRUE)

