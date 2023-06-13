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
loss_list = loss_list[c(1, 2, 4)]

M = length(loss_list)
N = dim(loss_list$Mean)[1]
H = dim(loss_list$Mean)[2]

## HELPERS

make_idx_ij = function(M) {
  K = floor(M*(M-1)/2)
  ij_idx = array(NA, c(K,2))
  k = 1
  for (i in 1:(M-1)) {
    for (j in (i+1):M) {
      ij_idx[k,] = c(i, j)
      k = k + 1
    }
  }
  # K = length(ij_idx)
  return(ij_idx)
}

make_blocks = function(N, block_size) {
  R = N - block_size + 1
  blocks = array(NA, c(block_size, R))
  for (o in 1:block_size) {
    blocks[o,] = o:(N-block_size+o)
  }
  return(blocks)
}

make_boot_idx = function(N, block_size, B) {
  R = N - block_size + 1
  L = floor(N / block_size)
  uN = L * block_size
  blocks = make_blocks(N, block_size)
  boot_idx = array(NA, c(uN, B))
  for (b in 1:B) {
    idx_b = sample(R, L, replace=TRUE)
    boot_idx[,b] = c(blocks[,idx_b])
  }
  return(boot_idx)
}

make_blocksum_mat = function(N, block_size) {
  L = floor(N / block_size)
  uN = L * block_size
  J = array(0, c(L,uN))
  for (l in 1:L) {
    J[l,(1+(l-1)*block_size):(l*block_size)] = 1
  }
  return(J)
}


# Prepare difference data
idx_ij = make_idx_ij(M)
K = dim(idx_ij)[1]

# data = array(0, c(K,N,H))
# for (k in 1:K) {
#   i_k = idx_ij[k,][1]
#   j_k = idx_ij[k,][2]
#   data[k,,] = loss_list[[i_k]] - loss_list[[j_k]]
# }
# 
# d_ij_h = array(NA, c(K,H))
# for (k in 1:K) {
#   d_ij_h[k,] = colMeans(data[k,,])
# }

data = array(0, c(M,N,H))
for (m in 1:M) {
  data[m,,] = loss_list[[m]]
}

make_stat = function(data_i, data_j, block_size, blocksum_mat) {
  d_ij_h_t = data_i - data_j
  d_ij_h = colMeans(d_ij_h_t)
  omega2_ij_h = colMeans(
    (blocksum_mat %*% (d_ij_h_t - d_ij_h))^2 / block_size
  )
  stat_ij_h = (d_ij_h / sqrt(omega2_ij_h))
  return(stat_ij_h)
}

## BOOTSTRAP
block_size = 2
B = 500

blocksum_mat = make_blocksum_mat(N, block_size)

data_mean = array(NA, c(M,H))
for (m in 1:M) {
  data_mean[m,] = colMeans(data[m,,])
}

data_stat_ij = array(NA, c(M,M))
for (k in 1:K) {
  i_k = idx_ij[k,][1]
  data_i = data[i_k,,]
  j_k = idx_ij[k,][2]
  data_j = data[j_k,,]
  
  d_ij_h = colMeans(data_i - data_j)
  
  omega2_ij_h = colMeans(
    (blocksum_mat %*% ((data_i - data_j) - d_ij_h))^2 / block_size
  )
  
  # data_stat_ij[k,] = (d_ij_h / sqrt(omega2_ij_h))
  data_stat_ij[i_k,j_k] = min(+ d_ij_h / sqrt(omega2_ij_h))
  data_stat_ij[j_k,i_k] = min(- d_ij_h / sqrt(omega2_ij_h))
}

set.seed(123456)
# boot_d_ij_hb = array(NA, c(K,B,H))
# boot_omega2_ij_hb = array(NA, c(K,B,H))
boot_stat_ij_hb = array(NA, c(M,M,B))
boot_cstat_ij_hb = array(NA, c(M,M,B))
# boot_inner_stat_ij_hb = array(NA, c(M,M,B))
for (k in 1:K) {
  cat("[+] k = ", k, "/", K, "\n", sep='')
  
  boot_idx = make_boot_idx(N, block_size, B)
  for (b in 1:B) {
    i_k = idx_ij[k,][1]
    data_i = data[i_k,boot_idx[,b],] - data_mean[i_k,]
    j_k = idx_ij[k,][2]
    data_j = data[j_k,boot_idx[,b],] - data_mean[j_k,]
    
    # boot_d_ij_hb_t = data_i - data_j
    # 
    # boot_d_ij_hb[k,b,] = colMeans(boot_d_ij_hb_t)
    # boot_omega2_ij_hb[k,b,] = colMeans(
    #   (blocksum_mat %*% (boot_d_ij_hb_t - boot_d_ij_hb[k,b,]))^2 / block_size
    # )
    # 
    # tmp_stat = boot_d_ij_hb[k,b,] / sqrt(boot_omega2_ij_hb[k,b,])
    
    tmp_stat = make_stat(data_i, data_j, block_size, blocksum_mat)
    
    boot_stat_ij_hb[i_k,j_k,b] = min(+ tmp_stat)
    boot_stat_ij_hb[j_k,i_k,b] = min(- tmp_stat)
    
    ## Inner bootstrap
    bb_boot_stat_ij_hb = array(NA, c(2,B))
    
    mu_i_b = colMeans(data_i)
    mu_j_b = colMeans(data_j)
    
    bb_boot_idx = make_boot_idx(dim(boot_idx)[1], block_size, B)
    for (bb in 1:B) {
      bb_b_idx = boot_idx[bb_boot_idx[,bb],b]
      bb_data_i = data[i_k,bb_b_idx,] - mu_i_b   # data_mean[i_k,]
      bb_data_j = data[j_k,bb_b_idx,] - mu_j_b   # data_mean[j_k,]
      
      bb_tmp_stat = make_stat(bb_data_i, bb_data_j, block_size, blocksum_mat)
      
      bb_boot_stat_ij_hb[1,bb] = min(+ bb_tmp_stat)
      bb_boot_stat_ij_hb[2,bb] = min(- bb_tmp_stat)
      
      # if (bb == 1) {
      #   boot_inner_stat_ij_hb[i_k,j_k,b] = min(+ bb_tmp_stat)
      #   boot_inner_stat_ij_hb[j_k,i_k,b] = min(- bb_tmp_stat)
      # }
    }
    
    # centered statistic from inner bootstrap
    boot_cstat_ij_hb[i_k,j_k,b] = (
      min(+ tmp_stat) - quantile(bb_boot_stat_ij_hb[1,], 0.1)
    )
    boot_cstat_ij_hb[j_k,i_k,b] = (
      min(- tmp_stat) - quantile(bb_boot_stat_ij_hb[2,], 0.1)
    )
    
    # rm(bb_boot_stat_ij_hb,bb_boot_idx)
  }
}

## Critical-value-centered statistic from outer bootstrap
# boot_crit_ij = array(NA, c(M,M))
data_cstat_ij = array(NA, c(M,M))
for (k in 1:K) {
  i_k = idx_ij[k,][1]
  j_k = idx_ij[k,][2]
  
  data_cstat_ij[i_k,j_k] = (
    data_stat_ij[i_k,j_k] - quantile(boot_stat_ij_hb[i_k,j_k,], 0.1) # +stat
  )
  data_cstat_ij[j_k,i_k] = (
    data_stat_ij[j_k,i_k] - quantile(boot_stat_ij_hb[j_k,i_k,], 0.1)  # -stat
  )
}

## MCS procedure
MCS_models = c()
MCS_pvals = c()
for (m in 1:(M-1)) {
  candidate_idx = arrayInd(which.max(abs(data_cstat_ij)), c(M,M))
  candidate_model = max(candidate_idx)
  MCS_models = append(MCS_models, candidate_model)
  MCS_pvals = append(MCS_pvals, (
    mean(data_cstat_ij[candidate_idx] < boot_cstat_ij_hb[candidate_idx[1],candidate_idx[2],])
  ))
}



# cat("\n\n uSPA p-value (one sided): ", mean(min(data_stat_ij) < boot_stat_ij_hb), "\n")

# hist(boot_stat_ij_hb)

# h = 1; plot(data_i[,h], type="l"); lines(data_j[,h], col="red")


# %%%%%