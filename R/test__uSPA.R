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
loss_list = loss_list[c(1, 2)]

M = length(loss_list)
N = dim(loss_list[[1]])[1]
H = dim(loss_list[[1]])[2]

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

# make_stat = function(data_i, data_j, block_size, blocksum_mat) {
#   d_ij_h_t = data_i - data_j
#   d_ij_h = colMeans(d_ij_h_t)
#   omega2_ij_h = colMeans(
#     (blocksum_mat %*% (d_ij_h_t - d_ij_h))^2 / block_size
#   )
#   # stat_ij_h = min(d_ij_h / sqrt(omega2_ij_h))
#   stat_ij_h = (d_ij_h / sqrt(omega2_ij_h))
#   return(stat_ij_h)
# }

## BOOTSTRAP
block_size = 4
B = 1000

blocksum_mat = make_blocksum_mat(N, block_size)

data_mean = array(NA, c(M,H))
for (m in 1:M) {
  data_mean[m,] = colMeans(data[m,,])
}

data_stat_ij = array(NA, c(K,H))
for (k in 1:K) {
  i_k = idx_ij[k,][1]
  data_i = data[i_k,,]
  j_k = idx_ij[k,][2]
  data_j = data[j_k,,]
  
  d_ij_h = colMeans(data_i - data_j)
  
  omega2_ij_h = colMeans(
    (blocksum_mat %*% ((data_i - data_j) - d_ij_h))^2 / block_size
  )
  
  data_stat_ij[k,] = (d_ij_h / sqrt(omega2_ij_h))
}

# set.seed(12345)
# boot_data_mean_c = array(NA, c(M,B,H))
# for (m in 1:M) {
#   boot_idx = make_boot_idx(N, block_size, B)
#   for (b in 1:B) {
#     boot_data_mean_c[m,b,] = colMeans(data[m,boot_idx[,b],]) - data_mean[m,]
#   }
# }

boot_d_ij_hb = array(NA, c(K,B,H))
boot_omega2_ij_hb = array(NA, c(K,B,H))
boot_stat_ij_hb = array(NA, c(K,B))
set.seed(12345)
for (k in 1:K) {
  boot_idx = make_boot_idx(N, block_size, B)
  for (b in 1:B) {
    i_k = idx_ij[k,][1]
    data_i = data[i_k,boot_idx[,b],] - data_mean[i_k,]
    j_k = idx_ij[k,][2]
    data_j = data[j_k,boot_idx[,b],] - data_mean[j_k,]
    
    boot_d_ij_hb_t = data_i - data_j
    
    boot_d_ij_hb[k,b,] = colMeans(boot_d_ij_hb_t)
    boot_omega2_ij_hb[k,b,] = colMeans(
      (blocksum_mat %*% (boot_d_ij_hb_t - boot_d_ij_hb[k,b,]))^2 / block_size
    )
    
    boot_stat_ij_hb[k,b] = min(boot_d_ij_hb[k,b,] / sqrt(boot_omega2_ij_hb[k,b,]))
  }
}

pval_uSPA = mean(min(data_stat_ij) < boot_stat_ij_hb)

cat("\n\n uSPA p-value (one sided): ", pval_uSPA, "\n")


hist(boot_stat_ij_hb)

# h = 1; plot(data_i[,h], type="l"); lines(data_j[,h], col="red")


# %%%%%