#
# Implement the uMC test of Quaedvlieg (2021) following the original Ox code
# 

library(tictoc)
library(doFuture)
registerDoFuture()
plan(multisession)

# library(progressr)

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

HAC_variance = function(X, kernel="bartlett") {
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
  X_acf = apply(X, 2, \(v) acf(v, lag.max=(N-1), plot=FALSE)$acf)
  Gamma = t(replicate(N, diag(var(X)))) * X_acf
  V = colSums(Gamma * c(1, 2*weights))
  return(V)
}

## uSPA ------------------------------

bootstrap_uSPA = function(diff_ij, alpha=0.1, B=1000, l=2, seed=12345) {
  N = dim(diff_ij)[1]
  H = dim(diff_ij)[2]
  
  block_size = l
  blocksum_mat = make_blocksum_mat(N, block_size)
  
  diff_ij_mean = colMeans(diff_ij)
  
  # boot_diff_ij_mean = array(NA, c(B,H))
  # boot_omega2_ij = array(NA, c(B,H))
  boot_stat_ij = array(NA, c(B))
  
  if (!is.na(seed)) {
    set.seed(seed) 
  }
  boot_idx = make_boot_idx(N, block_size, B)
  for (b in 1:B) {
    boot_diff_ij_b = diff_ij[boot_idx[,b],] - diff_ij_mean
    boot_diff_ij_b_mean = colMeans(boot_diff_ij_b)
    # boot_diff_ij_mean[b,] = boot_diff_ij_b_mean
    
    boot_omega2_ij_b = colMeans(
      (blocksum_mat %*% (boot_diff_ij_b - boot_diff_ij_b_mean))^2 / block_size
    )
    
    boot_stat_ij[b] = min(
      sqrt(N) * boot_diff_ij_b_mean / sqrt(boot_omega2_ij_b)
    )
  }
  q_uSPA = quantile(boot_stat_ij, alpha)
  
  return(q_uSPA)
}

## uMCS ------------------------------

uMCS = function(
    loss_list, 
    alpha_ij, 
    B=1000, 
    l=2, 
    seed=12345, 
    sorted=FALSE
) {
  M = length(loss_list)
  N = nrow(loss_list[[1]])
  H = ncol(loss_list[[1]])
  
  block_size = l
  blocksum_mat = make_blocksum_mat(N, block_size)
  
  ij_idx = list() # differences indices
  k = 1
  for (i in 1:M) {
    for (j in 1:M) {
      if (i != j) {
        ij_idx[[k]] = c(i, j)
        k = k + 1
      }
    }
  }
  K = length(ij_idx)
  
  # sample statistics
  sample_t_uSPA = matrix(NA, nrow=M, ncol=M)
  sample_c_uSPA = matrix(NA, nrow=M, ncol=M)
  for (k in 1:K) {
    i = ij_idx[[k]][1]
    j = ij_idx[[k]][2]
    diff_ij = loss_list[[i]] - loss_list[[j]]
    sample_t_uSPA[i,j] = min(
      sqrt(N) * colMeans(diff_ij) / sqrt(HAC_variance(diff_ij))
    )
    sample_c_uSPA[i,j] = bootstrap_uSPA(diff_ij, alpha=alpha_ij, l=l, seed=seed)
  }
  
  # pre-compute loss differences
  pre_diff_ij = array(NA, c(N,H,K))
  for (k in 1:K) {
    i = ij_idx[[k]][1]
    j = ij_idx[[k]][2]
    pre_diff_ij[,,k] = loss_list[[i]] - loss_list[[j]]
  }
  
  # bootstrap
  boot_t_uSPA = array(NA, c(M,M,B))
  boot_c_uSPA = array(NA, c(M,M,B))
  
  if (!is.na(seed)) {
    set.seed(seed) 
  }
  boot_idx = make_boot_idx(N, block_size, B)
  
  # # sequential
  # for (b in 1:B) {
  #   for (k in 1:K) {
  #     i = ij_idx[[k]][1]
  #     j = ij_idx[[k]][2]
  #     # diff_ij = loss_list[[i]] - loss_list[[j]]
  #     diff_ij = pre_diff_ij[,,k] # pre-computed differences
  # 
  #     boot_diff_ij_b = (diff_ij - colMeans(diff_ij))[boot_idx[,b],]
  #     boot_diff_ij_b_mean = colMeans(boot_diff_ij_b)
  #     boot_omega2_ij_b = colMeans(
  #       (blocksum_mat %*% (boot_diff_ij_b - boot_diff_ij_b_mean))^2 / block_size
  #     )
  # 
  #     boot_t_uSPA[i,j,b] = min(
  #       sqrt(N) * boot_diff_ij_b_mean / sqrt(boot_omega2_ij_b)
  #     )
  #     boot_c_uSPA[i,j,b] = bootstrap_uSPA(boot_diff_ij_b, alpha=alpha_ij, l=l, seed=NA)
  #   }
  # }
  
  # parallel
  boot = foreach(b = 1:B, .options.future = list(seed = TRUE)) %dofuture% {
    boot_t_uSPA_b = array(NA, c(M,M))
    boot_c_uSPA_b = array(NA, c(M,M))
    for (k in 1:K) {
      i = ij_idx[[k]][1]
      j = ij_idx[[k]][2]
      # diff_ij = loss_list[[i]] - loss_list[[j]]
      diff_ij = pre_diff_ij[,,k] # pre-computed differences

      boot_diff_ij_b = (diff_ij - colMeans(diff_ij))[boot_idx[,b],]
      boot_diff_ij_b_mean = colMeans(boot_diff_ij_b)
      boot_omega2_ij_b = colMeans(
        (blocksum_mat %*% (boot_diff_ij_b - boot_diff_ij_b_mean))^2 / block_size
      )

      boot_t_uSPA_b[i,j] = min(
        sqrt(N) * boot_diff_ij_b_mean / sqrt(boot_omega2_ij_b)
      )
      boot_c_uSPA_b[i,j] = bootstrap_uSPA(boot_diff_ij_b, alpha=alpha_ij, l=l, seed=NA)
    }

    list(boot_t_uSPA_b=boot_t_uSPA_b, boot_c_uSPA_b=boot_c_uSPA_b)
  }
  # unload bootstrap
  for (b in 1:B) {
    boot_t_uSPA[,,b] = boot[[b]]$boot_t_uSPA_b
    boot_c_uSPA[,,b] = boot[[b]]$boot_c_uSPA_b
  }
  
  # return(list(
  #   sample_t_uSPA=sample_t_uSPA,
  #   sample_c_uSPA=sample_c_uSPA,
  #   boot_t_uSPA=boot_t_uSPA,
  #   boot_c_uSPA=boot_c_uSPA
  # ))
  
  # MCS
  MCS_mat_t_uSPA = sample_t_uSPA - sample_c_uSPA
  MCS_mat_boot_t_uSPA = boot_t_uSPA - boot_c_uSPA
  
  MCS_models_order = c()
  MCS_pvals_order = c()
  for (m in 1:(M-1)) {
    candidate = arrayInd(which.max(MCS_mat_t_uSPA), c(M,M))
    idx_cand = candidate[1]
    # append
    MCS_models_order = append(MCS_models_order, idx_cand)
    # p-value
    max_t = MCS_mat_t_uSPA[candidate] 
    boot_max_t = apply(MCS_mat_boot_t_uSPA, 3, \(x) max(x, na.rm=TRUE))
    pval_cand = mean(abs(boot_max_t) > abs(max_t))
    MCS_pvals_order = append(MCS_pvals_order, pval_cand)
    # remove candidate from model set
    MCS_mat_t_uSPA[idx_cand,] = NA
    MCS_mat_t_uSPA[,idx_cand] = NA
    MCS_mat_boot_t_uSPA[idx_cand,,] = NA
    MCS_mat_boot_t_uSPA[,idx_cand,] = NA
  }
  # final model
  MCS_models_order = append(MCS_models_order, (1:M)[! 1:M %in% MCS_models_order])
  MCS_pvals_order = append(MCS_pvals_order, 1)
  
  # strictly increasing p-values
  MCS_adjusted_pvals_order = cummax(MCS_pvals_order)
  
  if (!sorted) {
    sort = match(1:M, MCS_models_order)
    MCS_models_order = 1:M
    MCS_pvals_order = MCS_pvals_order[sort]
    MCS_adjusted_pvals_order = MCS_adjusted_pvals_order[sort]
  }

  return(list(
    models = MCS_models_order, 
    MCS_p_values = MCS_adjusted_pvals_order,
    boot_p_values = MCS_pvals_order
  ))
}


# %%%%%