#load the data
load("long_run.RData")
y <- read.csv("COMBO17.csv")$Mcz

{
  par(mfrow = c(1,2))

  #density plot
  {
    par(mar = c(4.3,4.1,.6,2.1))

    # fit plot
    colors <- c(
      "#222222",
      "#DDDDDD",
      "#444444",
      "#CCCCCC",
      "#666666",
      "#BBBBBB",
      "#888888",
      "#AAAAAA",
      "#999999"
    )

    # posterior mean density
    y_mean <- apply(long_run$y_vals, 1, mean, na.rm = TRUE)

    # histogram
    hist(
      y,
      prob = TRUE,
      breaks = 80,
      border = "white",
      col = "grey92",
      main = "",
      xlab = "y",
      ylab = expression(f(y)),
      xaxs = "i",
      yaxs = "i",
      ylim = c(0,4)
    )

    # posterior sampled densities
    set.seed(1)
    for(i in sample(ncol(long_run$y_vals), 100)) {
      lines(
        long_run$x_grid,
        long_run$y_vals[, i],
        #col = adjustcolor("#E69F00", alpha.f = 0.15),
        col = adjustcolor("gray", alpha.f = 0.15),
        lwd = 1
      )
    }

    # posterior mean density
    lines(
      long_run$x_grid,
      y_mean,
      #col = "#8B0000",
      col = "black",
      lwd = 1
    )

    # cluster summaries
    ranges <- do.call(
      "rbind",
      tapply(y, long_run$partition, function(x)
        c(min(x), mean(x), max(x)))
    )
    par(xpd = TRUE)
    for(i in seq_len(nrow(ranges))) {

      segments(
        x0 = ranges[i,1],
        y0 = 0,
        x1 = ranges[i,3],
        y1 = 0,
        col = colors[i],
        lwd = 4
      )

      points(
        ranges[i,2],
        0,
        pch = 18,
        col = colors[i],
        cex = 1.5
      )
    }
    par(xpd = FALSE)
    box()
    grid()
  }
  ax <- axTicks(1)
  ax_lbl <- as.character(ax)
  ax_lbl[1] <- "0.0"
  ax_lbl[6] <- "1.0"

  #similarity matrix plot
  {
    S_ord <- long_run$similarity[order(y,decreasing = FALSE), order(y,decreasing = FALSE)]
    image(S_ord, axes = FALSE, col = gray.colors(100, start = 1, end = 0))
    axis(1, at = ax/max(y), labels = ax_lbl)
    axis(2, at = ax/max(y), labels = ax_lbl)
    box()
    abline(h = ax[-1]/max(y), v = ax[-1]/max(y),
           col = "lightgray", lty = "dotted", lwd = 0.8)
  }
}
