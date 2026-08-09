library(ggplot2)
library(tidyr)
library(gridExtra)

load("results1final.RData")
load("results4final.RData")
load("schedules.RData")

###
{
  # p1
  p1 <- plot(
    NRMCMC::nrmix(
      y1, K = 3, alpha = c(1, 1, 1),
      sigma2 = 1, n_chains = 1
    )
  ) +
    ggplot2::theme(legend.position = "none")
  
  
  # p2
  x_grid <- seq_len(2000)
  
  vals1 <- cbind(
    MG            = rowMeans(res1[1:2000,]),
    NR            = rowMeans(res1[2000 + round(seq(1,2000, l = 2000)),]),
    Tempered      = rowMeans(res1[4001:6000, ]),
    Tempered_NR = rowMeans(res1[6000 + round(seq(1,2000, l = 2000)), ])
  )
  
  vals4 <- cbind(
    MG            = rowMeans(res4[1:2000, ]),
    NR            = rowMeans(res4[2000 + round(seq(1,2000, l = 2000)), ]),
    Tempered      = rowMeans(res4[4001:6000, ]),
    Tempered_NR = rowMeans(res4[6000 + round(seq(1,2000, l = 2000)), ])
  )
  
  df_long1 <- pivot_longer(
    data.frame(
      Iterations = x_grid,
      vals1
    ),
    cols = -Iterations,
    names_to = "Method",
    values_to = "TV"
  )
  
  df_long1$Iterations[df_long1$Method == "MG"] <- rowMeans(apply(res1[8000 + 1:2000,],2,cumsum))
  df_long1$Iterations[df_long1$Method == "NR"] <- rowMeans(apply(res1[10000 + 1:2000,],2,cumsum))
  df_long1$Iterations[df_long1$Method == "Tempered"] <- rowMeans(apply(res1[12000 + 1:2000,]/length(inv_temps1$inverse_temperatures),2,cumsum))
  df_long1$Iterations[df_long1$Method == "Tempered_NR"] <- rowMeans(apply(res1[14000 + 1:2000,]/length(inv_temps1$inverse_temperatures),2,cumsum))
  
  
  df_long4 <- pivot_longer(
    data.frame(
      Iterations = x_grid,
      vals4
    ),
    cols = -Iterations,
    names_to = "Method",
    values_to = "TV"
  )
  
  df_long4$Iterations[df_long4$Method == "MG"] <- rowMeans(apply(res4[8000 + 1:2000,],2,cumsum))
  df_long4$Iterations[df_long4$Method == "NR"] <- rowMeans(apply(res4[10000 + 1:2000,],2,cumsum))
  df_long4$Iterations[df_long4$Method == "Tempered"] <- rowMeans(apply(res4[12000 + 1:2000,]/length(inv_temps4$inverse_temperatures),2,cumsum))
  df_long4$Iterations[df_long4$Method == "Tempered_NR"] <- rowMeans(apply(res4[14000 + 1:2000,]/length(inv_temps4$inverse_temperatures),2,cumsum))
  
  p2 <- ggplot(
    df_long1,
    aes(
      x = Iterations,
      y = TV,
      colour = Method,
      linetype = Method
    )
  ) +
    geom_line(linewidth = 1) +
    scale_colour_manual(
      name = NULL,
      values = c(
        "MG" = "black",
        "NR" = "grey30",
        "Tempered" = "grey55",
        "Tempered_NR" = "grey75"
      ),
      labels = c(
        "MG",
        "NR",
        "Tempered",
        "Tempered + NR"
      )
    ) +
    scale_linetype_manual(
      name = NULL,
      values = c(
        "MG" = "dotted",
        "NR" = "dotdash",
        "Tempered" = "dashed",
        "Tempered_NR" = "solid"
      ),
      labels = c(
        "MG",
        "NR",
        "Tempered",
        "Tempered + NR"
      )
    ) +
    labs(
      x = "Iterations",
      y = "TV distance"
    ) +
    theme_bw() +
    theme(
      panel.grid.major = element_line(colour = "grey85"),
      panel.grid.minor = element_line(colour = "grey92"),
      legend.position = c(0.75, 0.7),
      legend.background = element_blank(),
      legend.key = element_blank()
    )
  
  
  # p3
  df1 <- data.frame(
    MG = log(res1[16000 + 1,] / colSums(res1[9000 + 1:1000,])),
    NR = log(res1[16000 + 2,] / colSums(res1[11000 + 1:1000,])),
    Tempered = log(res1[16000 + 3,] / colSums(res1[13000 + 1:1000,] / length(inv_temps1$inverse_temperatures))),
    Tempered_NR = log(res1[16000 + 4,] / colSums(res1[15000 + 1:1000,] / length(inv_temps1$inverse_temperatures)))
  )
  
  df4 <- data.frame(
    MG = log(res4[16000 + 1,] / colSums(res1[9000 + 1:1000,])),
    NR = log(res4[16000 + 2,] / colSums(res1[11000 + 1:1000,])),
    Tempered = log(res4[16000 + 3,] / colSums(res4[13000 + 1:1000,] / length(inv_temps4$inverse_temperatures))),
    Tempered_NR = log(res4[16000 + 4,] / colSums(res4[15000 + 1:1000,] / length(inv_temps4$inverse_temperatures) ))
  )
  
  df_long1_2 <- pivot_longer(
    df1,
    cols = everything(),
    names_to = "Method",
    values_to = "Value"
  )
  
  df_long1_2$Method <- factor(
    df_long1_2$Method,
    levels = c(
      "MG",
      "NR",
      "Tempered",
      "Tempered_NR"
    ),
    labels = c(
      "MG",
      "NR",
      "Tempered",
      "Tempered + NR"
    )
  )
  
  df_long4_2 <- pivot_longer(
    df4,
    cols = everything(),
    names_to = "Method",
    values_to = "Value"
  )
  
  df_long4_2$Method <- factor(
    df_long4_2$Method,
    levels = c(
      "MG",
      "NR",
      "Tempered",
      "Tempered_NR"
    ),
    labels = c(
      "MG",
      "NR",
      "Tempered",
      "Tempered + NR"
    )
  )
  
  p3 <- ggplot(
    df_long1_2,
    aes(
      x = Value,
      y = Method,
      fill = Method
    )
  ) +
    geom_boxplot(
      width = 0.6,
      outlier.shape = 16,
      outlier.size = 1.5,
      alpha = 0.8
    ) +
    scale_fill_manual(
      values = c(
        "MG" = "black",
        "NR" = "grey30",
        "Tempered" = "grey55",
        "Tempered_NR" = "grey75"
      )
    ) +
    labs(
      x = "Entropy ESS",
      y = NULL,
      fill = NULL
    ) +
    theme_bw(base_size = 14) +
    theme(
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  
  # p4
  p4 <- plot(
    NRMCMC::nrmix(
      y4, K = 3, alpha = c(1, 1, 1),
      sigma2 = 1, n_chains = 1
    )
  ) +
    ggplot2::theme(legend.position = "none")
  
  
  # p5
  p5 <- ggplot(
    df_long4,
    aes(
      x = Iterations,
      y = TV,
      colour = Method,
      linetype = Method
    )
  ) +
    geom_line(linewidth = 1) +
    scale_colour_manual(
      name = NULL,
      values = c(
        "MG" = "black",
        "NR" = "grey30",
        "Tempered" = "grey55",
        "Tempered_NR" = "grey75"
      ),
      labels = c(
        "MG",
        "NR",
        "Tempered",
        "Tempered + NR"
      )
    ) +
    scale_linetype_manual(
      name = NULL,
      values = c(
        "MG" = "dotted",
        "NR" = "dotdash",
        "Tempered" = "dashed",
        "Tempered_NR" = "solid"
      ),
      labels = c(
        "MG",
        "NR",
        "Tempered",
        "Tempered + NR"
      )
    ) +
    labs(
      x = "N° predictive evaluations",
      y = "TV distance"
    ) +
    theme_bw() +
    theme(
      panel.grid.major = element_line(colour = "grey85"),
      panel.grid.minor = element_line(colour = "grey92"),
      legend.position = c(0.75, 0.7),
      legend.background = element_blank(),
      legend.key = element_blank()
    )
  
  
  # p6
  p6 <- ggplot(
    df_long4_2,
    aes(
      x = Value,
      y = Method,
      fill = Method
    )
  ) +
    geom_boxplot(
      width = 0.6,
      outlier.shape = 16,
      outlier.size = 1.5,
      alpha = 0.8
    ) +
    scale_fill_manual(
      values = c(
        "MG" = "black",
        "NR" = "grey30",
        "Tempered" = "grey55",
        "Tempered_NR" = "grey75"
      )
    ) +
    labs(
      x = "Entropy ESS",
      y = NULL
    ) +
    theme_bw(base_size = 14) +
    theme(
      legend.position = "none",
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  
  # ------------------------------------------------------------
  # FINAL FORMATTING
  # ------------------------------------------------------------
  
  # Common formatting for the plots
  common_theme <- theme(
    plot.title = element_text(
      size = 18,
      hjust = 0.5,
      face = "plain"
    ),
    axis.title.x = element_text(
      size = 17
    ),
    axis.text.x = element_text(
      size = 16
    ),
    axis.ticks.x = element_blank()
  )
  
  
  # p1: title instead of y-axis label, remove x-axis label
  p1 <- p1 +
    labs(
      title = "Density",
      x = NULL,
      y = NULL
    ) +
    theme(
      plot.title = element_text(
        size = 18,
        hjust = 0.5
      ),
      axis.title.x = element_blank(),
      axis.ticks.x = element_blank()
    )
  
  
  # p2
  p2 <- p2 +
    labs(
      title = "Permutations distance from stationarity",
      x = NULL,
      y = NULL
    ) +
    theme(
      plot.title = element_text(
        size = 16,
        hjust = 0.5
      ),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size = 16),
      axis.ticks.x = element_blank()
    ) +
    xlim(0, 4e8)
  
  
  # p3
  p3 <- p3 +
    labs(
      title = "Permutations ESS per predictive\n(log scale)",
      x = NULL,
      y = NULL
    ) +
    theme(
      plot.title = element_text(
        size = 16,
        hjust = 0.5
      ),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size = 16),
      axis.ticks.x = element_blank()
    ) +
    xlim(-17, -12)
  
  
  # p4
  p4 <- p4 +
    labs(
      title = NULL,
      x = NULL,
      y = NULL
    ) +
    theme(
      plot.title = element_text(
        size = 18,
        hjust = 0.5
      ),
      axis.title.x = element_blank(),
      axis.ticks.x = element_blank()
    )
  
  
  # p5
  p5 <- p5 +
    labs(
      title = NULL,
      x = "N° predictive evaluations",
      y = NULL
    ) +
    theme(
      plot.title = element_text(
        size = 18,
        hjust = 0.5
      ),
      axis.title.x = element_text(size = 17),
      axis.text.x = element_text(size = 16),
      axis.ticks.x = element_blank()
    ) +
    xlim(0, 4e8)
  
  
  # p6
  p6 <- p6 +
    labs(
      title = NULL,
      x = "",
      y = NULL
    ) +
    theme(
      plot.title = element_text(
        size = 18,
        hjust = 0.5
      ),
      axis.ticks.x = element_blank()
    ) +
    xlim(-17, -12)
  
  
  # ------------------------------------------------------------
  # COMBINE
  # ------------------------------------------------------------
  
  grid.arrange(
    p1, p2, p3,
    p4, p5, p6,
    ncol = 3,
    nrow = 2
  )
}
