#
# Perform the MCS over the 1-step-ahead forecasts of models
#
# Ref: "The Model Confidence Set" (2011), Hansen et al.
# 
# NOTE: run in R version 4.2.3 (2023-03-15 ucrt)

library(tidyverse)
library(lubridate)
#library(MCS)
library(modelconf) # see: https://github.com/nielsaka/modelconf/
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

# MCS -------------------------

MCS_results = list()

i = 1
for (f in losses_filenames) {
  # Get losses
  loss_mat = get_model_losses(paste0(f, ".csv"))
  M0_names = colnames(loss_mat)
  
  # alpha = 0.1
  #MCS_010 = MCSprocedure(
  #  Loss = loss_mat,
  #  alpha = 0.1,
  #  #statistic = "TR",
  #  B = 3000,
  #  k = 3,
  #)
  
  MCS_010 = estMCS(
    loss = loss_mat^2,  # squared losses
    B = 10000,
    l = 2
  )
  
  p_values_table_010 = data.frame(
    p_value = MCS_010[, "MCS p-val"],
    row.names = M0_names
  ) %>% arrange(p_value)
  
  # alpha = 0.25
  #MCS_025 = MCSprocedure(
  #  Loss = loss_mat,
  #  alpha = 0.25,
  #  #statistic = "TR",
  #  B = 3000,
  #  k = 3,
  #)
  
  MCS_025 = estMCS(
    loss = loss_mat^2,  # squared losses
    B = 10000,
    l = 2
  )
  
  p_values_table_025 = data.frame(
    p_value = MCS_025[, "MCS p-val"],
    row.names = M0_names
  ) %>% arrange(p_value)
  
  
  # Save results 
  mcs_results_i = list(
    M0_names,
    list(
      p_values_table_010,
      M0_names[MCS_010[, "MCS p-val"] > 0.1]
    ),
    list(
      p_values_table_025,
      M0_names[MCS_025[, "MCS p-val"] > 0.25]
    )
  )
  names(mcs_results_i) = c("models_names", "alpha=0_10", "alpha=0_25")
  MCS_results[[i]] = mcs_results_i
  
  i = i + 1
  
  rm(loss_mat, M0_names) 
  rm(MCS_010, MCS_025, p_values_table_010, p_values_table_025)
  rm(mcs_results_i)
}
names(MCS_results) = losses_filenames

# Print results -------------------------

# NOTE: the dataframe assumes that the methods (and method names)
#       remain unchanged across different files
results_DF = data.frame(Method = MCS_results[[1]][[1]])

MCS_include_stars = function(m, mcs25, mcs10) {
  s = ""
  if (m %in% mcs25) {
    s = paste0(s, "*")
  }
  if (m %in% mcs10) {
    s = paste0(s, "*")
  }
  return(s)
}

i = 1
for (r in names(MCS_results)) {
  cat("+ ---------------------------\n")
  cat(paste0("+ Setup: ",  toupper(r), "\n\n"))
  
  # Results
  cat(": Included models (MCS_alpha = 0.10): \n", 
      paste0("-  ", MCS_results[[i]][[2]][[2]], "\n"), 
      "\n")
  cat(": Included models (MCS_alpha = 0.25): \n", 
      paste0("-  ", MCS_results[[i]][[3]][[2]], "\n"), 
      "\n")
  
  # Fill dataframe column
  results_DF = cbind(
    results_DF,
    newcol = mapply(function(m) MCS_include_stars(
      m,
      MCS_results[[i]][[3]][[2]],
      MCS_results[[i]][[2]][[2]]
    ),
    MCS_results[[i]][[1]]
    )
  )
  names(results_DF)[i+1] = gsub("_", " ", toupper(r))
  
  i = i + 1
}

xtable(results_DF[,-1])

# Export results
write.csv(results_DF, paste0(getwd(), "/R/", "MCS_small_stars.csv"), row.names = FALSE)

# Save workspace
save.image(paste0(getwd(), "/R/", "MCS_small_results_", today(),".RData"))

# -----