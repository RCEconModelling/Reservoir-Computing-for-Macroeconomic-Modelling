MCSprocedureDestudentized=function (Loss, alpha = 0.15, B = 5000, cl = NULL, ram.allocation = TRUE, 
          statistic = "Tmax", k = NULL, min.k = 3, verbose = TRUE) 
{
  time.start = Sys.time()
  Loss = as.matrix(Loss)
  M_start = ncol(Loss)
  colnames(Loss) = gsub(".", "_", colnames(Loss), fixed = T)
  if (is.null(colnames(Loss))) {
    colnames(Loss) = paste("model", 1:ncol(Loss), sep = "_")
  }
  if (is.null(cl)) {
    cl_number = 0
  }
  if (!is.null(cl)) {
    max.cores = length(cl)
  }
  B = round(B)
  #if (B < 1000) {
  #  cat(paste("Warning: B is small"))
  #}
  if (any(is.na(Loss))) {
    stop("NAs in Loss are not allowed")
  }
  if (any(abs(Loss) == Inf)) {
    stop("Inf in Loss are not allowed")
  }
  N = nrow(Loss)
  repeat {
    M = ncol(Loss)
    model.names = colnames(Loss)
    col.names.d = do.call(c, lapply(1:M, function(x) {
      paste(model.names[x], model.names[-x], sep = ".")
    }))
    d = do.call(cbind, lapply(1:M, function(x) {
      Loss[, x] - Loss[, -x]
    }))
    colnames(d) = col.names.d
    d_ij_mean = colMeans(d)
    d_i_mean = sapply(1:M, function(x) {
      sum(d_ij_mean[paste(model.names[x], model.names[-x], 
                          sep = ".")])/(M - 1)
    })
    names(d_i_mean) = model.names
    block.length = NULL
    foo = expand.grid(1:M, 1:M)[, 2:1]
    foo = foo[foo[, 1] != foo[, 2], ]
    index = col.names.d[foo[, 1] < foo[, 2]]
    if (ncol(d) > 2) {
      d_cut = as.list(as.data.frame(d[, index]))
      if (!is.null(cl)) {
        k = max(na.omit(as.numeric(parSapply(cl, d_cut, 
                                             function(x) {
                                               try(ar(x)$order, silent = TRUE)
                                             }))))
      }
      if (is.null(cl)) {
        k = max(na.omit(as.numeric(sapply(d_cut, function(x) {
          try(ar(x)$order, silent = TRUE)
        }))))
      }
    }
    else {
      k = ar(d[, 1])$order
    }
    if (k < min.k) {
      k = min.k
    }
    v = ceiling(nrow(d)/k)
    n = nrow(d)
    if (!is.null(cl)) {
      indexes_b = parLapply(cl, 1:B, boot.block, v = v, 
                            n = n, k = k)
    }
    if (is.null(cl)) {
      indexes_b = lapply(1:B, boot.block, v = v, n = n, 
                         k = k)
    }
    if (!is.null(cl)) {
      if (ram.allocation) {
        weigth.in.cl = (as.numeric(object.size(d))/1e+06) * 
          3.2
        used.memory = sum(sapply(ls(), function(x) object.size(get(x))))/1e+06
        cl_number = min(as.numeric(ceiling((memory.limit() * 
                                              0.8 - used.memory)/weigth.in.cl) - 1), max.cores)
      }
      else {
        cl_number = max.cores
      }
    }
    else {
      cl_number = 1
    }
    d = as.data.frame(d)
    if (cl_number > 1) {
      d_ij_avg_resampled = parLapply(cl[1:cl_number], indexes_b, 
                                     function(x, d, N) {
                                       colSums((d[x, , drop = FALSE]))/N
                                     }, d = d, N = N)
    }
    else {
      d_ij_avg_resampled = lapply(indexes_b, function(x, 
                                                      d, N) {
        colSums((d[x, , drop = FALSE]))/N
      }, d = d, N = N)
    }
    if (!is.null(cl)) {
      d_b_i_mean = do.call(rbind, parLapply(cl, d_ij_avg_resampled, 
                                            d_b_i_mean.fun, M = M, model.names = model.names))
    }
    if (is.null(cl)) {
      d_b_i_mean = do.call(rbind, lapply(d_ij_avg_resampled, 
                                         d_b_i_mean.fun, M = M, model.names = model.names))
    }
  
    d_ij_avg_resampled = do.call(rbind, d_ij_avg_resampled)
    
    TR = max(abs(d_ij_mean))
    TM = max(d_i_mean)
    Tb_R = sapply(1:B, function(i) {
      max(abs(d_ij_avg_resampled[i, ] - d_ij_mean))
    })
    Tb_M = sapply(1:B, function(i) {
      max((d_b_i_mean[i, ] - d_i_mean))
    })
    Pr = length(which(TR < Tb_R))/B
    Pm = length(which(TM < Tb_M))/B
    v_i_M = d_i_mean
    v_i_M_order = order(v_i_M, decreasing = TRUE)
    TR_all = t(as.matrix(d_ij_mean))
    v_i_R = sapply(1:M, function(x) {
      max(TR_all[1, paste(model.names[x], model.names[-x], 
                          sep = ".")])
    })
    names(v_i_R) = model.names
    v_i_R_order = order(v_i_R, decreasing = TRUE)
    Pm_h0_mk = NULL
    for (i in 1:M) {
      if (i == 1) {
        model_temp = model.names
      }
      else {
        model_temp = model.names[-v_i_M_order[1:(i - 
                                                   1)]]
      }
      TM_temp = max(d_i_mean[model_temp])
      Pm_h0_mk = c(Pm_h0_mk, length(which(TM_temp < Tb_M))/B)
    }
    mcs_Pm = sapply(1:M, function(x) max(Pm_h0_mk[1:x]))
    combine.names = names(d_ij_mean)
    Pr_h0_mk = NULL
    for (i in 1:M) {
      remove = NULL
      if (i == 1) {
        model_temp = combine.names
      }
      else {
        remove = do.call(c, lapply(1:(i - 1), function(x) {
          which(gsub(model.names[v_i_R_order[x]], "", 
                     combine.names) != combine.names)
        }))
        model_temp = combine.names[-remove]
      }
      if (i < M) {
        TR_temp = max(abs(d_ij_mean[model_temp]))
        Pr_h0_mk = c(Pr_h0_mk, length(which(TR_temp < 
                                              Tb_R))/B)
      }
      else {
        Pr_h0_mk = c(Pr_h0_mk, 1)
      }
    }
    mcs_Pr = sapply(1:M, function(x) max(Pr_h0_mk[1:x]))
    matrix_show = matrix(NA, M, 7, dimnames = list(model.names, 
                                                   c("Rank_M", "v_M", "MCS_M", "Rank_R", "v_R", "MCS_R", 
                                                     "Loss")))
    matrix_show[, "v_M"] = v_i_M
    matrix_show[names(sort(v_i_M)), "Rank_M"] = 1:M
    matrix_show[model.names[v_i_M_order], "MCS_M"] = mcs_Pm
    matrix_show[, "v_R"] = v_i_R
    matrix_show[names(sort(v_i_R)), "Rank_R"] = 1:M
    matrix_show[model.names[v_i_R_order], "MCS_R"] = mcs_Pr
    matrix_show[, "Loss"] = colMeans(Loss)
    rm(list = "indexes_b")
    if (!is.null(cl)) 
      clusterEvalQ(cl, {
        rm(list = ls())
        gc()
      })
    gc()
    if (statistic == "Tmax") {
      p2test = Pm
    }
    if (statistic == "TR") {
      p2test = Pr
    }
    if (p2test > alpha ) {
      if (verbose) {
        cat(paste("\n###########################################################################################################################\n"))
        cat(paste("Superior Set Model created\t:\n"))
        show(matrix_show)
        cat(paste("p-value\t:\n"))
        print(p2test)
        cat(paste("\n###########################################################################################################################"))
      }
      break
    }
    else {
      if (statistic == "Tmax") 
        eliminate = which(v_i_M == max(v_i_M))
      if (statistic == "TR") 
        eliminate = which(v_i_R == max(v_i_R))
      if (verbose) 
        cat(paste("\nModel", model.names[eliminate], 
                  "eliminated", Sys.time()))
      Loss = as.matrix(Loss[, -eliminate])
      colnames(Loss) = model.names[-eliminate]
      if (ncol(Loss) == 1) {
        if (verbose) {
          cat(paste("\n###########################################################################################################################\n"))
          cat(paste("Superior Set Model created\t:\n"))
          matrix_show = matrix(matrix_show[-eliminate, 
                                           ], nrow = 1, dimnames = list(colnames(Loss), 
                                                                        colnames(matrix_show)))
          show(matrix_show)
          cat(paste("p-value\t:\n"))
          show(p2test)
          cat(paste("\n###########################################################################################################################"))
        }
        break
      }
    }
  }
  elapsed.time = Sys.time() - time.start
  n_elim = M_start - nrow(matrix_show)
  out = new("SSM", show = matrix_show, Info = list(model.names = rownames(matrix_show), 
                                                   elapsed.time = elapsed.time, statistic = statistic, n_elim = n_elim, 
                                                   mcs_pvalue = p2test, alpha = alpha, B = B, k = k), Bootstrap = list(TR = list(Stat = TR, 
                                                                                                                                 BootDist = Tb_R), Tmax = list(Stat = TM, BootDist = Tb_M)))
  return(out)
}