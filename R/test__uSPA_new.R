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

loss_list = get_model_losses(paste0(losses_filenames[1], ".csv"), losses_max_horizon)

# Cut loss list
loss_list = loss_list[c(8, 6)]

M = length(loss_list)
N = dim(loss_list[[1]])[1]
H = dim(loss_list[[1]])[1]


## HELPERS ------------------------------

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

# Kernels and HAC variance [see Ox code by Quaedvlieg (2021)]
quadspectral_kernel = function(x) {
  a = 6/5 * x
  return( 3 * (sin(pi*a) / (pi*a) -  cos(pi*a) / (pi*a))^2 )
}

bartlett_kenel = function(x) {
  return( (1 - abs(x)) * (abs(x) <= 1) )
}

epanechnikov_kernel = function(x) {
  return( 3/4*(1 - x^2) * (abs(x) <= 1) ) 
}

HAC_variance = function(X, kernel="quadspectral") {
  N = dim(X)[1]
  H = dim(X)[2]
  switch (kernel,
    bartlett = {
      bw = floor(1.2 * N^(1/3))
      weights = bartlett_kenel(seq(1,N-1)/bw)
    },
    epanechnikov = {
      bw = floor(1.2 * N^(1/3))
      weights = epanechnikov_kernel(seq(1,N-1)/bw)
    },
    quadspectral = {
      bw = floor(1.3 * N^(1/5))
      weights = quadspectral_kernel(seq(1,N-1)/bw)
    }
  )
  X_acf = apply(diff_ij, 2, \(x) acf(x, lag.max=(N-1), plot=FALSE)$acf)
  Gamma = t(replicate(N, diag(var(X)))) * X_acf
  V = colSums(Gamma * c(1, 2*weights))
  return(V)
}


# Prepare difference data and statistic
diff_ij = loss_list[[1]] - loss_list[[2]]

diff_ij_mean = colMeans(diff_ij)

data_stat_ij = min(
  sqrt(N) * diff_ij_mean / sqrt(HAC_variance(diff_ij))
)


## BOOTSTRAP ------------------------------
block_size = 2
B = 1000

blocksum_mat = make_blocksum_mat(N, block_size)

boot_diff_ij_mean = array(NA, c(B,H))
boot_omega2_ij = array(NA, c(B,H))
boot_stat_ij = array(NA, c(B))

set.seed(12345)
boot_idx = make_boot_idx(N, block_size, B)
for (b in 1:B) {
  boot_diff_ij_b = diff_ij[boot_idx[,b],] - diff_ij_mean
  boot_diff_ij_b_mean = colMeans(boot_diff_ij_b)
  boot_diff_ij_mean[b,] = boot_diff_ij_b_mean
  
  boot_omega2_ij[b,] = colMeans(
    (blocksum_mat %*% (boot_diff_ij_b - boot_diff_ij_b_mean))^2 / block_size
  )
  
  boot_stat_ij[b] = min(
    sqrt(N) * boot_diff_ij_b_mean / sqrt(boot_omega2_ij[b,])
  )
}

# Quantile
quant_uSPA = quantile(boot_stat_ij, 0.1)

## RESULT ------------------------------

pval_uSPA = mean(data_stat_ij < boot_stat_ij)

cat("\n| Models:  i =", names(loss_list)[1], "\n|          j =", names(loss_list)[2])
cat("\n|\n| uSPA p-value (one sided): ", pval_uSPA, "\n\n")


hist(boot_stat_ij)

# h = 1; plot(data_i[,h], type="l"); lines(data_j[,h], col="red")


# %%%%%