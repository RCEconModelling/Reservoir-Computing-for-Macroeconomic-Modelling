#
#

library(tidyverse)
library(lubridate)
library(sandwich)

handlers(global = TRUE)
handlers("progress")

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling")

## LOAD DATA ------------------------------

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
    # cat("Listed model: ", M0_names[m], "\n")
  }
  names(loss_list) = M0_names
  
  return(loss_list)
}

# Filenames
losses_filenames = c(
  "mhmcs_small_multistep_fix_2007",
  "mhmcs_small_multistep_fix_2011",
  "mhmcs_small_multistep_ew_2007",
  "mhmcs_small_multistep_ew_2011",
  "mhmcs_small_multistep_rw_2007",
  "mhmcs_small_multistep_rw_2011"
)

# Set maximal horizon to consider in uMCS test
losses_max_horizon = 8

# loss_list = get_model_losses(paste0(losses_filenames[1], ".csv"), losses_max_horizon)
loss_list = get_model_losses(paste0(losses_filenames[5], ".csv"), losses_max_horizon)

# Cut loss list
# loss_list = loss_list[c(1, 2, 3)]
# loss_list = loss_list[7:9]

source("./R/lib_uMCS.R")

result = uMCS(loss_list, alpha_mcs=0.1, alpha_ij=0.1, B=50, l=2, seed=12345, sorted=TRUE)

## RESULT ------------------------------

cat("\n| Models: ", names(loss_list))
cat("\n| uMCS results")
cat("\n|   p-val  Name")
for (m in 1:length(loss_list)) {
  cat(sprintf("\n| - %.2f", result$MCS_p_values[m]), " ", names(loss_list)[result$models[m]])
}

# # MCS
# MCS_mat_t_uSPA = result$sample_t_uSPA - result$sample_c_uSPA
# MCS_mat_boot_t_uSPA = result$boot_t_uSPA - result$boot_c_uSPA
# 
# M = length(loss_list)
# 
# MCS_models_order = c()
# MCS_pvals_order = c()
# for (m in 1:(M-1)) {
#   candidate = arrayInd(which.max(MCS_mat_t_uSPA), c(M,M))
#   idx_cand = candidate[1]
#   # append
#   MCS_models_order = append(MCS_models_order, idx_cand)
#   # p-value
#   max_t = MCS_mat_t_uSPA[candidate] 
#   boot_max_t = apply(MCS_mat_boot_t_uSPA, 3, \(x) max(x, na.rm=TRUE))
#   pval_cand = mean(abs(boot_max_t) > abs(max_t))
#   MCS_pvals_order = append(MCS_pvals_order, pval_cand)
#   # remove candidate from model set
#   MCS_mat_t_uSPA[idx_cand,] = NA
#   MCS_mat_t_uSPA[,idx_cand] = NA
#   MCS_mat_boot_t_uSPA[idx_cand,,] = NA
#   MCS_mat_boot_t_uSPA[,idx_cand,] = NA
# }
# # final model
# MCS_models_order = append(MCS_models_order, (1:M)[! 1:M %in% MCS_models_order])
# MCS_pvals_order = append(MCS_pvals_order, 1)
# 
# # strictly increasing p-values
# MCS_adjusted_pvals_order = cummax(MCS_pvals_order)

## RESULT ------------------------------

# cat("\n| Models: ", names(loss_list))
# cat("\n| uMCS results")
# cat("\n|   p-val  Name")
# for (m in 1:M) { 
#   cat(sprintf("\n| - %.2f", MCS_pvals_order[m]), " ", names(loss_list)[MCS_models_order[m]])
# }

# %%%%%