### VISUALIZATION FUNCTIONS

### spaghetti plot of the largest component markov chain
spaghetti <- function(X,
                      title = "",
                      gg = FALSE,
                      ylim = c(0.5,1),
                      ylab = "",
                      xlab = "",
                      col = gray(0.6),
                      which_show = NULL){
  
  #ggplots?
  if(gg){
    
    #create the data frame for ggplot
    df <- data.frame(prop = c(X),
                     chain_iter = rep(seq_len(NROW(X)),NCOL(X)),
                     sim_iter = rep(seq_len(NCOL(X)),each = NROW(X)))
    
    #create also the data frame with the mean line
    if(is.null(which_show)){
      df_mean <- data.frame(prop = apply(X,1,mean),
                            chain_iter = seq_len(NROW(X)),
                            sim_iter = 1)
    }else{
      df_mean <- data.frame(prop = X[,which_show],
                            chain_iter = seq_len(NROW(X)),
                            sim_iter = 1)
    }
    
    
    ggplot2::ggplot(df, ggplot2::aes(x = chain_iter,
                                     y = prop,
                                     group = sim_iter)) +
      ggplot2::geom_line(color = col) +
      ggplot2::geom_line(data = df_mean, 
                         ggplot2::aes(x = chain_iter, y = prop),
                         color = "black", linewidth = 1) +  
      ggplot2::labs(x = xlab,
                    y = ylab,
                    title = title) +
      ggplot2::theme_minimal() +
      ggplot2::theme(legend.position = "none") 
    
  }else{
    
    #get the dimensions
    N <- NROW(X)
    B <- NCOL(X)
    
    #empty plot
    plot(c(1,N),ylim,xlab = xlab,
         ylab = ylab,
         main = title,
         type = "n",
         cex.lab = 1.5,
         cex.axis = 1.5,
         cex.main = 1.5)
    
    #add the lines
    for(b in seq_len(B)){
      lines(seq_len(N),X[,b], col = col, lwd = 3)
    }
    #add the mean line
    if(is.null(which_show)){
      lines(seq_len(N),apply(X,1,mean),col = "black", lwd = 3)
      lines(seq_len(N),apply(X,1,mean),col = "black", lwd = 3)
    }else{
      lines(seq_len(N),X[,which_show],col = "black", lwd = 3)
      lines(seq_len(N),X[,which_show],col = "black", lwd = 3)
    }
  }
  
}

### violin plots of the marginal distribution evolution
violin <- function(X,
                   Y,
                   title = "Marginal distribution of the chains",
                   ylim = c(0,1),
                   col1 = "black",
                   col2 = "darkgray",
                   gg = FALSE, by = 10){
  
  #load the package
  require(vioplot)
  
  #define the indexes to plot the results
  ind <- c(1, seq(from = by,to = NROW(X), by = by))
  
  #get the reversible
  R <- t(X[ind,])
  colnames(R) <- ind
  R <- as.data.frame(R)
  
  #get the non-reversible
  NR <- t(Y[ind,])
  colnames(NR) <- ind
  NR <- as.data.frame(NR)
  
  #plot the non-reversible
  vioplot::vioplot(NR, side = "left", col = col1,
                   border = "NA",  drawRect = F, main = title,
                   cex.axis = 1.5, cex.main = 1.5)
  
  #plot the reversible
  vioplot::vioplot(R, side = "right", col = col2, border = "NA",
                   drawRect = F, add = T)
  
}

#scatter plot
cloud_plot <- function(X, which = 1:2, gg = FALSE, title = "Marginal Gibbs"){
  
  if(gg){
    
    #take the tail
    df <- data.frame(
      x = X[NROW(X), which[1], ],
      y = X[NROW(X), which[2], ]
    )
    
    ggplot2::ggplot(df, ggplot2::aes(x = x, y = y)) +
      ggplot2::geom_point(size = 0.5) +
      ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
      ggplot2::labs(x = NULL, y = NULL) +
      ggplot2::theme_classic() + 
      ggplot2::labs(title = title)
    
  }else{
    
    #take the tail
    plot(t(X[NROW(X),which,]), xlim = c(0,1), ylim = c(0,1),
         xlab = "", ylab = "", pch = 19, cex.axis = 1.5,
         main = title, cex.main = 1.5)
    
  }
  
}

#histogram
histo <- function(X, title = "Marginal Gibbs", gg = FALSE){
  
  if(gg){
    
    df <- data.frame(
      x = X[NROW(X), 1, ]
    )
    
    ggplot2::ggplot(df, ggplot2::aes(x = x)) +
      ggplot2::geom_histogram(
        bins = 30,              
        fill = "black",
        color = "black",
        ggplot2::aes(y = ggplot2::after_stat(density))
      ) +
      ggplot2::labs(
        x = NULL,
        y = NULL,
        title = title
      ) +
      ggplot2::theme_classic() +
      ggplot2::theme(
        panel.border = ggplot2::element_blank()
      )
    
  }else{
    
    hist(X[NROW(X),1,], bty = "n", col = "black", prob = TRUE,
         xlab = "", ylab = "", main = title , cex.main = 1.5, cex.axis = 1.5)
  }
  
}
