### VISUALIZATION FUNCTIONS

### spaghetti plot of the largest component markov chain
spaghetti <- function(X,
                      title = "",
                      gg = FALSE,
                      ylim = c(0.5,1),
                      col = adjustcolor("darkgray",0.3),
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
      ggplot2::labs(x = "Iterations",
                    y = "First Component Proportion",
                    title = title) +
      ggplot2::theme_minimal() +
      ggplot2::theme(legend.position = "none") 
    
  }else{
    
    #get the dimensions
    N <- NROW(X)
    B <- NCOL(X)
    
    #empty plot
    plot(c(1,N),ylim,xlab = "Iterations",
         ylab = "First component proportion",
         main = title,
         type = "n")
    
    #add the lines
    for(b in seq_len(B)){
      lines(seq_len(N),X[,b], col = col)
    }
    #add the mean line
    if(is.null(which_show)){
      lines(seq_len(N),apply(X,1,mean),col = "black")
    }else{
      lines(seq_len(N),X[,which_show],col = "black")
    }
  }
  
}

### violin plots of the marginal distribution evolution
violin <- function(X,
                   Y,
                   title = "Chains Marginal Distribution",
                   ylim = c(0,1),
                   col1 = "black",
                   col2 = "darkgray",
                   gg = FALSE, by = 10){
  
  
  if(gg){
    
    #get the range for the densities
    x_lim <- range(X, Y)
    
    #create the grid
    x_grid <- seq(x_lim[1], x_lim[2], length.out = 512)
    
    #create the time stamps
    tt <- c(1, seq(0, NROW(X), by = 10)[-1])
    
    #initialize the list with the densities
    polygon_data_list <- list()
    
    #loop over each time stamp
    for (i in seq_along(tt)) {
      
      #get the densities
      dens_X <- density(X[1:tt[i],], from = x_lim[1], to = x_lim[2])$y
      dens_Y <- density(Y[1:tt[i],], from = x_lim[1], to = x_lim[2])$y
      
      #normalize them
      dens_X <- dens_X / max(dens_X)
      dens_Y <- dens_Y / max(dens_Y)
      
      #create a data frame with all the information
      df_Y <- data.frame(
        x = c(tt[i] - dens_Y * 4.5, rep(tt[i], length(x_grid))),
        y = c(x_grid, rev(x_grid)),
        type = "Y",
        iteration = tt[i]  
      )
      
      df_X <- data.frame(
        x = c(tt[i] + dens_X * 4.5, rep(tt[i], length(x_grid))),
        y = c(x_grid, rev(x_grid)),
        type = "X",
        iteration = tt[i]  
      )
      
      #put them into the list      
      polygon_data_list[[i]] <- rbind(df_Y, df_X)
    }
    
    #concatentate the list
    polygon_data <- do.call("rbind",polygon_data_list)
    
    #create the plot
    ggplot2::ggplot(polygon_data, ggplot2::aes(x = x, y = y, fill = type)) +
      ggplot2::geom_polygon(ggplot2::aes(group = interaction(type, iteration)), color = NA, alpha = 0.7) +
      ggplot2::scale_fill_manual(values = c("Y" = col1, "X" = col2)) +
      ggplot2::scale_x_continuous(name = "Iterations") +
      ggplot2::scale_y_continuous(name = "First component proportion", breaks = tt) +
      ggplot2::theme_minimal() +
      ggplot2::ggtitle(title) +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
        legend.position = "none"  
      )
    
  }else{
    
    #every 10 iterations compute the empirical density
    x_lim <- range(X,Y)
    x_grid <- seq(x_lim[1],x_lim[2], l = 512)
    tt <- c(1,seq(0,NROW(X), by = by)[-1])
    X_densities <- sapply(tt,function(ttt) {
      tmp <- density(X[seq_len(ttt),], from = x_lim[1], to = x_lim[2])$y
      tmp/max(tmp)
    })
    Y_densities <- sapply(tt,function(ttt) {
      tmp <- density(Y[seq_len(ttt),], from = x_lim[1], to = x_lim[2])$y
      tmp/max(tmp)
    })
    
    #empty plot
    plot(c(1,NROW(X)),x_lim, xlab = "Iterations",
         ylab = "First component proportion", xaxt = "n",
         type = "n", main = title)
    axis(1,tt,tt)
    
    #add the densities
    for(i in seq_along(tt)){
      
      #Y_densities on the left
      polygon(c(tt[i] - Y_densities[,i]*by*0.45,rep(tt[i],length(x_grid))),
              c(x_grid,rev(x_grid)), col = col1, border = NA)
      
      #X_densities on the right
      polygon(c(tt[i] + X_densities[,i]*by*0.45,rep(tt[i],length(x_grid))),
              c(x_grid,rev(x_grid)), col = col2, border = NA)
      
    }
    
  }
  
}

#scatter plot
cloud_plot <- function(X, which = 1:2, gg = FALSE){
  
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
      ggplot2::theme_classic()
    
  }else{
    
    #take the tail
    plot(t(X[NROW(X),which,]), xlim = c(0,1), ylim = c(0,1),
         xlab = "", ylab = "", pch = 16, cex = .5)
    
  }
  
}

#histogram
histo <- function(X, title = "Marginal", gg = FALSE){
  
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
         xlab = "", ylab = "", main = title )
  }
  
}
