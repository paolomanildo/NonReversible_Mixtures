#load the results
load("results.RData")

#define the empirical TV distance
TV <- function(x,y,discrete = FALSE, n_grid = 512){
  
  if(discrete){
    
    #use the relative frequencies
    K <- max(x,y, na.rm = TRUE)
    X_val <- tabulate(x,K) / length(x)
    Y_val <- tabulate(y,K) / length(y)
    
    #return the TV
    0.5 * sum( abs(X_val - Y_val) )
    
  }else{
    
    #use numerical integration
    
    #get the common support of x and y
    x_lim <- range(x,y, na.rm = TRUE)
    x_lim <- x_lim + c(-1,1) * 0.001 * diff(x_lim)
    
    #compute the two densities
    x_grid <- seq(x_lim[1],x_lim[2],l = n_grid)
    X_val <- density(x,from = x_lim[1], to = x_lim[2], n = n_grid)$y
    Y_val <- density(y,from = x_lim[1], to = x_lim[2], n = n_grid)$y
    
    #compute the TV distance with the trapezoidal rule:
    
    #save the absolute differences
    abs_diff <- abs(X_val - Y_val)
    
    #return the integral using the trapezoidal rule
    0.5 * sum( diff(x_grid) * (head(abs_diff, -1) + tail(abs_diff, -1)) / 2 )
    
  }
  
}

#compute the ESS
ESS <- t(sapply(res, function(x){
  x[8000 + 1:4]
}))

#compute the scaled ESS
sESS <- t(sapply(res, function(x){
  x[8000 + 1:4] / x[8000 + 5:8]
}))

#compute the TV distances
TVs <- t(sapply(1:1000, function(i){
  c(TV(long_run$sampler$entropy,sapply(res,function(x) x[1000*0 + i])),
    TV(long_run$sampler$entropy,sapply(res,function(x) x[1000*1 + i])),
    TV(long_run$sampler$entropy,sapply(res,function(x) x[1000*2 + i])),
    TV(long_run$sampler$entropy,sapply(res,function(x) x[1000*3 + i])))
}))

#set the threshold
hist(TVs, n = 100)
abline(v = 0.05, col = 2)

#get the number of iteration to conververgence
T2C <- apply(TVs,2,function(x) {
  which(x <= 0.05)[1]
})

#plot
matplot(TVs, type = "l", lty = 1)
abline(v = T2C, col = 1:4)
legend("topright",lty = 1, col = 1:4, legend = c("MG","NR","NUSAMS(0,5,1)","NUSAMS(0,5,1) + NR"),
       bg = "transparent", bty = "n", cex = .8)
                                       
#get the number of predictive calls to convergence
n_preds_to_conv <- t(sapply(1:500, function(i){
  c(sum(res[[i]][4*1000 + seq_len(T2C[1])]),
    sum(res[[i]][5*1000 + seq_len(T2C[2])]),
    sum(res[[i]][6*1000 + seq_len(T2C[3])]),
    sum(res[[i]][7*1000 + seq_len(T2C[4])]))
}))

#estimate and standard error using the delta method assuming samplers' independence
summaries <- function(x,y, digits = 3){
  x_bar <- mean(x)
  y_bar <- mean(y)
  se_x <- sqrt(var(x) / length(x))
  se_y <- sqrt(var(y) / length(y))
  est <- x_bar / y_bar
  se <- est * sqrt( (se_x / x_bar)^2 + (se_y / y_bar)^2 )
  paste0(sprintf(paste0("%.",digits,"f"),est),
         " (",sprintf(paste0("%.",digits,"f"),se),")")
}

#table summary
df <- as.data.frame(rbind(
  c("NR" = summaries(n_preds_to_conv[,2], n_preds_to_conv[,1]),
    "NUSAMS(0,5,1)" = summaries(n_preds_to_conv[,3],n_preds_to_conv[,1]),
    "NUSAMS(0,5,1)+NR" = summaries(n_preds_to_conv[,4],n_preds_to_conv[,1])),
  
  c("NR" = summaries(sESS[,2],sESS[,1]),
    "NUSAMS(0,5,1)" = summaries(sESS[,3],sESS[,1]),
    "NUSAMS(0,5,1)+NR" = summaries(sESS[,4],sESS[,1]))))

rownames(df) <- c("Relative  n° predictive to converge",
                  "Relative ESS per predictive")


df <- t(df)

knitr::kable(df, format = "latex")
