#
# Pairwise Diebold-Mariano forecasting tests
#

library(tidyverse)
library(reshape2)
library(forecast)
library(xtable)

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling")

get_model_losses = function(filename) {
  # Load data
  python__Losses = read_csv(paste0(getwd(), "/R/", filename), na = "")
  
  # Remove the very first columns, which is just redundant indexes
  python__Losses = python__Losses[,c(-1)]
  
  #M0_names = colnames(python__Losses)
  #M0 = length(M0_names)
  #t = dim(python__Losses)[1]
  
  # Extract individual models
  loss_mat = as.matrix(python__Losses)
  
  return(loss_mat)
}

# Filenames
losses_filenames = c(
  "mcs_small_fix_2007",
  "mcs_small_fix_2011",
  "mcs_small_ew_2007",
  "mcs_small_ew_2011",
  "mcs_small_rw_2007",
  "mcs_small_rw_2011"
)

# Diebold-Mariano --------------------------


DM_results = list()

i = 1
for (f in losses_filenames) {
  # Print --------------------------------
  cat("\n\n+ ---------------------- \n")
  cat(paste("Loss file:", toupper(f)), "\n\n")
  
  # Get losses
  loss_mat = get_model_losses(paste0(f, ".csv"))
  M0_names = colnames(loss_mat)
  # Clean up names 
  M0_names = str_remove_all(str_remove_all(M0_names, fixed(" [EW]")), fixed(" [RW]"))
  L = length(M0_names)
  
  p_val_mat = matrix(nrow = L, ncol = L)
  
  for (j in 1:L) {
    for (k in 1:L) {
      # Run test only against different model
      if (k != j) {
        test_jk = dm.test(
          loss_mat[,j],
          loss_mat[,k],
          alternative = "g",
          h = 1,
          power = 2
        )
        p_val_mat[j,k] = test_jk$p.value
      } 
    }
  }
  
  rownames(p_val_mat) = M0_names
  colnames(p_val_mat) = M0_names
  
  # Print --------------------------------
  cat("+ ---------------------- \n")
  cat(paste("Loss file:", toupper(f)), "\n\n")
  cat("Diebold-Mariano pairwise test p-values:\n")
  print(p_val_mat)
  
  # Plot ---------------------------------
  
  #p_val_mat_plot = melt(t(p_val_mat[1:(L-1),2:L]))
  p_val_mat_plot = melt(t(p_val_mat))
  
  p_val_mat_plot$Var1 <- factor(p_val_mat_plot$Var1)
  p_val_mat_plot$Var2 <- factor(p_val_mat_plot$Var2)
  
  ggplot(data =  p_val_mat_plot, aes(x = Var1, y = Var2)) +
    geom_tile(aes(fill = value)) +
    geom_text(aes(label = ifelse(is.na(value), " ", substring(sprintf("%.3f", value),2))), vjust = 0.5) +
    #scale_fill_gradientn(colours=c("salmon","salmon","white","white"),
    #                     values=c(0,0.09998,0.101,1),
    #                     #breaks=c(0.1),
    #                     #guide = "none", 
    #                     na.value = "gray85") +
    scale_fill_stepsn(breaks=c(0, 0.0999999, 0.1),
                      colours=c("salmon","white","white"),
                      guide = "none",
                      na.value = "gray85") +
    coord_fixed() +
    scale_y_discrete(limits=rev) +
    scale_x_discrete(position="top") + 
    xlab("") +
    ylab("") +
    ggtitle("") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust=0, colour="black"), 
          axis.text.y = element_text(color="black"),
          plot.margin=grid::unit(c(0,0,0,0), "mm"))
  
  # Save
  ggsave(paste0(getwd(), "/R/Pairwise Tests/", f, "_DMtest_pvals.png"), 
         width = 15.5, height = 12, units = "cm")
  
}


# -----