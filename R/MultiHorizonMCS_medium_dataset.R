#
# Perform the Multi-Horizon MCS over the multistep forecasts of models
#
# Ref: "Multi-Horizon Forecast Comparison" (2021), Quaedvlieg, Rogier
# 
# NOTE: run in R version 4.2.3 (2023-03-15 ucrt)

rm(list=ls()) # clear workspace

library(tidyverse)
library(lubridate)
library(tictoc)
library(xtable)
#library(MultiHorizonSPA) # see: https://github.com/lucabarbaglia/MultiHorizonSPA

source("./R/lib_uMCS.R") # load uMCS functions

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling")

get_model_losses = function(filename, max_horizon) {
  # Load data
  python__multihLosses = read_csv(paste0(getwd(), "/R/", filename), na = "")
  
  M0_names = as.list(unique(python__multihLosses["M0"]))[[1]]
  M0 = length(M0_names)
  H = dim(unique(python__multihLosses["H"]))[1]
  t = dim(unique(python__multihLosses["T"]))[1]
  
  # Extract individual models
  loss_list = vector(mode = "list", length = M0)
  
  for (m in 1:M0) {
    # Filter
    loss_mat_m = python__multihLosses %>%
      filter(M0 == M0_names[m]) %>%
      pivot_wider(
        names_from = H,
        values_from = Loss
      ) %>%
      select(
        -c("M0", "T")
      ) %>%
      modify(  # squared losses
        ~ .^2
      ) %>%
      as.matrix()
    
    # Slice to max horizon
    loss_list[[m]] = loss_mat_m[,1:max_horizon]
    
    # Info
    cat("Listed model: ", M0_names[m], "\n")
  }
  names(loss_list) = M0_names
  
  return(loss_list)
}

# Filenames
losses_filenames = c(
  "mhmcs_medium_multistep_fix_2007",
  "mhmcs_medium_multistep_fix_2011",
  "mhmcs_medium_multistep_ew_2007",
  "mhmcs_medium_multistep_ew_2011",
  "mhmcs_medium_multistep_rw_2007",
  "mhmcs_medium_multistep_rw_2011"
)

# Set maximal horizon to consider in uMCS test
losses_max_horizon = 8

# Multi-Horizon MCS -------------------------

MHMCS_results = list()

i = 1
for (f in losses_filenames) {
  # Get losses
  loss_list = get_model_losses(paste0(f, ".csv"), losses_max_horizon)
  M0_names = names(loss_list)
  
  # alpha = 0.1
  # MHMCS_010 = FastMultiHorizonMCS(
  #   loss_list, 
  #   alpha_mcs = 0.1,
  #   L = 3, 
  #   B = 1000, 
  #   unif_or_average = 'u', 
  #   num_cores = 4,
  #   seed = 202209
  # )
  
  # p_values_table_010 = data.frame(
  #   p_value = MHMCS_010$p_values,
  #   row.names = M0_names
  # ) %>% arrange(p_value)
  
  # # alpha = 0.25
  # MHMCS_025 = FastMultiHorizonMCS(
  #   loss_list, 
  #   alpha_mcs = 0.25,
  #   L = 3, 
  #   B = 1000, 
  #   unif_or_average = 'u', 
  #   num_cores = 4,
  #   seed = 202209
  # )
  
  # p_values_table_025 = data.frame(
  #   p_value = MHMCS_025$p_values,
  #   row.names = M0_names
  # ) %>% arrange(p_value)
  
  tic()
  MHMCS = uMCS(
    loss_list, 
    alpha_ij=0.1, 
    B=1000, 
    l=2, 
    seed=12345, 
    sorted=FALSE
  )
  toc()
  
  # Save results 
  p_values_table = data.frame(
    p_value = MHMCS$MCS_p_values,
    row.names = M0_names
  ) %>% 
    arrange(p_value) 
  
  p_values_table_010 = p_values_table %>% filter(p_value > 0.1)
  p_values_table_025 = p_values_table %>% filter(p_value > 0.25)
  
  # Save results 
  mhmcs_results_i = list(
    M0_names,
    list(
      p_values_table_010,
      rownames(p_values_table_010)
    ),
    list(
      p_values_table_025,
      rownames(p_values_table_025)
    )
  )
  names(mhmcs_results_i) = c("models_names", "alpha=0_10", "alpha=0_25")
  MHMCS_results[[i]] = mhmcs_results_i
  
  i = i + 1
  
  rm(loss_list, M0_names, MHMCS) 
  rm(p_values_table, p_values_table_010, p_values_table_025)
  rm(mhmcs_results_i)
}
names(MHMCS_results) = losses_filenames


# Print results -------------------------

# NOTE: the dataframe assumes that the methods (and method names)
#       remain unchanged across different files
results_DF = data.frame(Method = MHMCS_results[[1]][[1]])

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
for (r in names(MHMCS_results)) {
  cat("+ ---------------------------\n")
  cat(paste0("+ Setup: ",  toupper(r), "\n\n"))
  
  # Results
  cat(": Included models (MCS_alpha = 0.10): \n", 
      paste0("-  ", MHMCS_results[[i]][[2]][[2]], "\n"), 
      "\n")
  cat(": Included models (MCS_alpha = 0.25): \n", 
      paste0("-  ", MHMCS_results[[i]][[3]][[2]], "\n"), 
      "\n")
  
  # Fill dataframe column
  results_DF = cbind(
    results_DF,
    newcol = mapply(function(m) MCS_include_stars(
      m,
      MHMCS_results[[i]][[3]][[2]],
      MHMCS_results[[i]][[2]][[2]]
    ),
    MHMCS_results[[i]][[1]]
    )
  )
  names(results_DF)[i+1] = gsub("_", " ", toupper(r))
  
  i = i + 1
}

# xtable(results_DF)
# print(xtable(results_DF, digits=3), include.rownames=FALSE)

# Export results
write.csv(results_DF, paste0(getwd(), "/R/", "MHMCS_medium_stars.csv"), row.names = FALSE)

# Save workspace
save.image(paste0(getwd(), "/R/", "MHMCS_medium_results_", today(),".RData"))

# -----