#get the results
load("results.RData")

#define the grid for the refreshment rate
xi_grid <- 10^seq(log10(0.01), log10(1000), length.out = 100)

#smoothed plot
{
  m1 <- smooth.spline(xi_grid,results[1,1,], cv = TRUE)
  m2 <- smooth.spline(xi_grid,results[3,1,], cv = TRUE)
  m2 <- smooth.spline(xi_grid,results[3,1,], spar = m2$spar + .8 )
  
  op <- par(no.readonly = TRUE)
  par(mfrow = c(1,2), mar = c(4.1,4.1,1.1,2.1))
  plot(m1$x,m1$y, type = "l", log = "x", 
       xlab = expression(xi ~ "(log scale)"),
       ylab = "N° itetations to convergence",
       xaxt = "n", cex.lab = 1.3)
  axis(1,at = c(0.01,0.1,1,10,100,1000),
       labels = c("0.01","0.1","1","10","100","1000"))
  grid(lwd = 2)
  plot(m2$x,m2$y, type = "l", log = "x", 
       xlab = expression(xi ~ "(log scale)"), ylab = "ESS",
       xaxt = "n", cex.lab = 1.3)
  axis(1,at = c(0.01,0.1,1,10,100,1000),
       labels = c("0.01","0.1","1","10","100","1000"))
  grid(lwd = 2)
  par(op)
}
