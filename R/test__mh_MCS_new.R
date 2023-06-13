#
#

library(tidyverse)
library(lubridate)
library(tictoc)
library(sandwich)
library(doFuture)
registerDoFuture()
plan(multisession)

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
    loss_list[[m]] = loss_mat_m[,1:max_horizon] #* 10000
    
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

loss_list = get_model_losses(paste0(losses_filenames[1], ".csv"), losses_max_horizon)

# Cut loss list
loss_list = loss_list[c(1, 4, 7)]

# library(progressr)
# handlers(global = TRUE)
# handlers("progress", "beepr")

# uniform_MHMCS = function(losses,  alpha_ij, alpha_mcs, B, block_size) {
#   return(0)
# }

M = length(loss_list)
N = nrow(loss_list[[1]])
H = ncol(loss_list[[1]])

alpha_ij = 0.1
alpha_mcs = 0.1
B = 300
block_size = 4

## --------------------------------------------------------
## PRELIMINARIES ------------------------------------------

# Comparison indices
ij_idx = list()
k = 1
for (i in 1:(M-1)) {
  for (j in (i+1):M) {
    ij_idx[[k]] = c(i, j)
    k = k + 1
  }
}
K = length(ij_idx)

# ij_idx_vech = function(idx) {
#   # Wrap-around indices for computing MCS
#   if (idx[1] > idx[2]) {
#     return(c(idx[2], idx[1]))
#   }
#   return(idx)
# }

math_M = function(vec) {
  mat = matrix(NA, nrow=M, ncol=M)
  for (i in 1:M) {
    for (j in 1:M) {
      mat[i,j] = vec[k_from_ij[i,j]]
    }
  }
  return(mat)
}


## --------------------------------------------------------
## SAMPLE STATISTIC ---------------------------------------

# loss_diff = list()
# loss_diff_var = list
# for (h in 1:H) {
#   loss_diff_h = matrix(NA, nrow=N, ncol=K)
#   for (k in 1:K) {
#     i = ij_idx[[k]][1]
#     j = ij_idx[[k]][2]
#     loss_diff_h[,k] = loss_list[[i]][,h] - loss_list[[j]][,h]
#   }
#   loss_diff[[h]] = loss_diff_h
#   loss_diff_var[[h]] = NeweyWest(loss_diff_h, lag=3)
#   rm(loss_diff_h)
# }

# Get uSPA t-stat from sample
d_ij_h = matrix(NA, nrow=K, ncol=H)
sample_tstat_uSPA = matrix(NA, nrow=M, ncol=M)
for (k in 1:K) {
  i = ij_idx[[k]][1]
  j = ij_idx[[k]][2]
  diff_loss_k = loss_list[[i]] - loss_list[[j]]
  # Mean
  d_ij_h = colMeans(diff_loss_k)
  # Variance
  V_ij_h = colMeans((diff_loss_k - d_ij_h)^2)
  # uSPA t-stat
  sample_tstat_uSPA[i,j] = min(+d_ij_h / sqrt(V_ij_h))
  sample_tstat_uSPA[j,i] = min(-d_ij_h / sqrt(V_ij_h)) # flip sign
}

## --------------------------------------------------------
## BOOTSTRAP ----------------------------------------------

# Block boostrap indices
L = floor(N / block_size)
ub = L * block_size
# block_idx = seq(1, ub, by=block_size)
block_idx = 1:(ub - block_size)

# pbar = progressor(along = B)

set.seed(2022222)

tic()
# bootstrap_uSPA = foreach(b = 1:B, .options.future = list(seed = TRUE)) %dofuture% {
for (b in 1:B) {
  # (1) First bootstrap round
  boot_sample_idx = sample(block_idx, L, replace=TRUE)
  boot_blocks = lapply(boot_sample_idx, function(i) seq(i, i+block_size-1))
  b_idx = unlist(boot_blocks)
  
  d_ij_hb = matrix(NA, nrow=K, ncol=H)
  # V_ij_hb = matrix(NA, nrow=K, ncol=H)
  # tstat_uSPA_ij_b_vec = rep(NA, K)
  tstat_uSPA_ij_b = matrix(NA, nrow=M, ncol=M)
  for (k in 1:K) {
    i = ij_idx[[k]][1]
    j = ij_idx[[k]][2]
    diff_loss_k = loss_list[[i]][b_idx,] - loss_list[[j]][b_idx,]
    # Mean
    d_ij_hb[k,] = colMeans(diff_loss_k)
    # Variance
    dev_loss_k = diff_loss_k - d_ij_hb[k,]
    vv_k = rep(0, H)
    for (l in 1:L) {
      l_idx = (1+(l-1)*block_size):(l*block_size)
      vv_k = vv_k + colSums(dev_loss_k[l_idx,])^2
    }
    V_ij_hb = vv_k / (block_size * L)
    # uSPA t-stat
    tstat_uSPA_ij_b[i,j] = min(+(d_ij_hb[k,] - d_ij_h) / sqrt(V_ij_hb))
    tstat_uSPA_ij_b[j,i] = min(-(d_ij_hb[k,] - d_ij_h) / sqrt(V_ij_hb)) # flip sign
  }
  
  # (2) Second bootstrap round
  # tstat_uSPA_ij_b_2 = matrix(NA, nrow=K, ncol=B)
  tstat_uSPA_ij_doubleboot = list()
  for (b_2 in 1:B){
    boot_2_sample_idx = sample(block_idx, L, replace=TRUE)
    boot_2_blocks = lapply(boot_2_sample_idx, function(i) seq(i, i+block_size-1))
    # Re-sample from current block
    b_2_idx = b_idx[unlist(boot_2_blocks)]
    
    tstat_uSPA_ij_b_2 = matrix(NA, nrow=M, ncol=M)
    for (k in 1:K) {
      i = ij_idx[[k]][1]
      j = ij_idx[[k]][2]
      diff_loss_k_2 = loss_list[[i]][b_2_idx,] - loss_list[[j]][b_2_idx,]
      # Mean
      d_ij_hb_2 = colMeans(diff_loss_k_2)
      # Variance
      dev_loss_k_2 = diff_loss_k_2 - d_ij_hb_2
      vv_k_2 = rep(0, H)
      for (l in 1:L) {
        l_idx = (1+(l-1)*block_size):(l*block_size)
        vv_k_2 = vv_k_2 + colSums(dev_loss_k_2[l_idx,])^2
      }
      V_ij_hb_2 = vv_k_2 / (block_size * L)
      # uSPA t-stat
      tstat_uSPA_ij_b_2[i,j] = min(+(d_ij_hb_2 - d_ij_hb) / sqrt(V_ij_hb_2))
      tstat_uSPA_ij_b_2[j,i] = min(-(d_ij_hb_2 - d_ij_hb) / sqrt(V_ij_hb_2)) # flip sign
    }
    tstat_uSPA_ij_doubleboot[[b_2]] = tstat_uSPA_ij_b_2
  }
  
  # (3) Collect critical value
  crit_uSPA_ij_b = matrix(NA, nrow=M, ncol=M)
  for (i in 1:M) {
    for (j in 1:M) {
      # Recall: ij_idx[[k]] = c(i, j)
      # cat('[', i, ',', j ,']', sep='')
      if (i != j) {
        # sign_ij = -1 + 2*(i < j)
        # k_ij = k_from_ij[i,j]
        # tstat_uSPA_ij_b[i,j] = sign_ij * tstat_uSPA_ij_b_vec[k_ij]
        crit_uSPA_ij_b[i,j] = quantile(
          sapply(tstat_uSPA_ij_doubleboot, function(x) x[i,j]), alpha_ij
        )
      }
    }
  }
  # for (k in 1:K) {
  #   crit_uSPA_ij_b[k] = quantile(tstat_uSPA_ij_b_2[k,], alpha_ij)
  # }
  
  # (4) Centered stat
  centered_uSPA_ij = tstat_uSPA_ij_b - crit_uSPA_ij_b
  
  # Status
  # pbar(sprintf("b = %g", b))
  
  list(tstat_uSPA_ij_b, centered_uSPA_ij)
}
toc()

## --------------------------------------------------------
## HM-MCS -------------------------------------------------

# Compute outer centered statistic
# boot_quant_tstat_uSPA = matrix(NA, nrow=M, ncol=M)
boot_centered_uSPA = matrix(NA, nrow=M, ncol=M)
for (i in 1:M) {
  for (j in 1:M) {
    if (i != j) {
      boot_quant_tstat_uSPA_ij = quantile(
        sapply(bootstrap_uSPA, function(x) x[[1]][i,j]), alpha_mcs
      )
      boot_centered_uSPA[i,j] = (
        sample_tstat_uSPA[i,j] - boot_quant_tstat_uSPA_ij
      )
    }
  }
}

# Compute final quantiles
boot_pvals_uSPA = matrix(0, nrow=M, ncol=M)
for (b in 1:B) {
  boot_pvals_uSPA = boot_pvals_uSPA + (
    boot_centered_uSPA < bootstrap_uSPA[[b]][[2]]
  ) / B
}

# MCS
boot_mh_mcs_pvals = rep(0, M)
for (m in 1:M) {
  boot_mh_mcs_pvals[m] = max(boot_pvals_uSPA[m,], na.rm=TRUE)
}

cat(boot_mh_mcs_pvals)
