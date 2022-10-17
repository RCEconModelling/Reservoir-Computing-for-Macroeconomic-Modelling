library(MASS)
source("MCSprocedureDestudentized.R")
source("MCS.R")

# Set-up parameters
Svalues=c(2,4)
M0=2
P=200


# DGP parameters
thetavalues=c(0,0.1,0.2,0.5)
delta=0.2

for (rho in c(0,0.5)){
  
  
  # Simulations and boostrap
  Sim=1000
  B=400
  set.seed(1000)
  min.k=3
  alpha=0.1
  
  # Define boot.block function used in code
  boot.block <- function(x, v, n, k) {
    startIndexes = sample(1:(n - k), v + 1)
    blocks = do.call(c, lapply(startIndexes, function(p, k) p +
                               seq_len(k), k = k))
    return(blocks[1:n])
  }
  
  
  results=matrix(NA,2*length(thetavalues),length(Svalues))
  
  for (Scount in 1:length(Svalues)){
    
    for (thetacount in 1:length(thetavalues)){
      
      
      # Set-up parameters inside loop
      S=Svalues[Scount]
      theta=thetavalues[thetacount]
      
      
      theta1=theta*kronecker(diag(M0),matrix(1,1,S/2))
      
      Mstar=t(apply(theta1,1,function(x){x==apply(theta1,2,min)}))+0
      Sigma=toeplitz(rho^seq(0,(S-1),1))
      
      
      # Initialise rejection rate results
      MCSrej=0
      
      # Initialise rejection rate results for Hansen et al (2011) procedure independent across s
      MCSHLNrej=0
      
      # Simulation reps
      for (i in 1:Sim){
        
        cat(paste("\nSimulation", i,"of",Sim,Sys.time()))
        
        # Generate blocks of PxM matrices for epsilon, stacked horizontally S times, with correlation across S and not M
        eps=matrix(mvrnorm(P*M0,matrix(0,S),Sigma),nrow=P)
        
        LossMat=matrix(NA,(P+1),S*M0)
        LossMat[1,]=rnorm(S*M0)
        for (t in 1:P){
          LossMat[(1+t),]=delta*LossMat[t,]+eps[t,]
        }
        
        # Add on mean to LossMat and remove initial condition
        LossMat=LossMat+matrix(1,P+1,1)%*%t(matrix(theta1))
        LossMat=LossMat[-1,]
        
        # Put Loss into a list of S matrices, one PxM matrix for each k=1,...,S
        LossMat=lapply(1:S, function(x) {LossMat[,(M0*(x-1)+1):(M0*x)]})
        
        # Get colnames by each matrix of list
        LossMat=lapply(LossMat, function(x) {colnames(x)<-paste("model", 1:ncol(x), sep = "_");x})
        
        # Set minimum bootstrap length (same as MCSprocedure)
        min.k=3
        N = lapply(LossMat,nrow)
        
        Loss=LossMat
        
        MCS=matrix(1,M0,S)
        
        repeat {
          # Generate number of models per horizon, which changes at different steps of the algorithm
          M = lapply(Loss,ncol)
          model.names = lapply(Loss,colnames)
          
          # Generate colnames for loss differentials
          # Ensure that if a given horizon has only one model remaining, then it will have null columns for d
          col.names.d = lapply(model.names, function(y) {do.call(c, lapply(1:length(y), function(x) {
            if(length(y)>1){
              paste(y[x], y[-x], sep = ".")} else{}
          }))})
          
          # Generate loss differentials by horizon, there will be a different number for each as we loop
          d = lapply(1:S, function(y) do.call(cbind, lapply(1:M[[y]], function(x) {
            Loss[[y]][, x] - Loss[[y]][, -x]
          })))
          
          # Apply colnames to loss differentials
          d=lapply(1:S, function(x) {colnames(d[[x]])=col.names.d[[x]];d[[x]]})
          
          # Calculate means
          d_ij_mean = lapply(d,colMeans)
          
          # Generate indices for model comparisons 
          foo = lapply(1:S, function(x) expand.grid(1:M[[x]], 1:M[[x]])[, 2:1])
          foo = lapply(1:S, function (x) foo[[x]][foo[[x]][, 1] != foo[[x]][, 2], ])
          # Get model comparison names only for unique comparisons (1 vs 2 to 7, 2 versus 3 to 7 etc.)
          index = lapply(1:S, function(x) col.names.d[[x]][foo[[x]][, 1] < foo[[x]][, 2]])
          
          # Choose block length k to be the max of the AR(k) orders for each loss differential series
          # We want a common block length across all comparisons
          
          # N.B do.call(c,.) makes a list of lists into a list
          if (do.call(min,lapply(d,ncol)) > 2) {
            d_cut = lapply(1:S,function(x) as.list(as.data.frame(d[[x]][, index[[x]]])))
            
            k = max(na.omit(as.numeric(sapply(do.call(c,d_cut), function(x) {
              try(ar(x)$order, silent = TRUE)
            }))))
            
          } else {
            k = min.k
          }
          
          
          # Set min lag to be 3 (as per the MCSprocedure R script)
          if (k < min.k) {
            k = min.k
          }
          
          # Number of blocks
          v = ceiling(nrow(d[[1]])/k)
          # Number of obs 
          n = nrow(d[[1]])
          
          # Get block bootstrap indices
          indexes_b = lapply(1:B, boot.block, v = v, n = n, 
                             k = k)
          
          
          d = lapply(d,as.data.frame)
          
          
          # Resample B times for each model comparison and horizon and take mean 
          d_ij_avg_resampled = lapply(1:S, function(z) lapply(indexes_b, function(x, 
                                                                                  d, N) {
            colSums((d[x, , drop = FALSE]))/N
          }, d = d[[z]], N = N[[z]]))
          
          
          
          # Stack up bootstrap means 
          d_ij_avg_resampled = lapply(1:S, function (x) do.call(rbind, d_ij_avg_resampled[[x]]))
          
          # N.B. The bootstrap variance calculation is commented out as we use the de-studentised stat
          # This can be added back in to studentize the test statistic and bootstrap draws
          
          # Get bootstrap variance for each comparison, each horizon
          #d_var_ij = lapply(1:S, function(x) matrix(colSums(t(apply(d_ij_avg_resampled[[x]], 
          #                                                          1, function(z) (z - d_ij_mean[[x]])^2)))/B, nrow = 1))
          # Get colnames for bootvar
          #d_var_ij=lapply(1:S, function(x) {colnames(d_var_ij[[x]])=names(d_ij_mean[[x]]);d_var_ij[[x]]})
          
          # Get test statistic
          # N.B. Apply max within horizon and then across horizon which gives the final TR
          TR = do.call(max,lapply(1:S,function(x) max((abs(d_ij_mean[[x]])))))
          # Can also use bootstrap variance used to normalize test stat - see HLN11 Online Appendix
          
          
          # Get B times the max over M at each horizon, and then take max over horizons
          # N.B. B x 1 bootstrap vector per horizon, then take the max row-wise aross horizons
          Tb_R = apply(do.call(cbind,lapply(1:S,function(x) sapply(1:B, function(i) {
            max(abs(d_ij_avg_resampled[[x]][i, ] - d_ij_mean[[x]]))
          }))),1,max)
          
          # Get p-value
          Pr = length(which(TR < Tb_R))/B
          
          # Calculate all t-stats to identify the eliminated model
          TR_all = lapply(1:S,function(x) ((d_ij_mean[[x]])))
          
          # Get model names with appended horizon, to identify eliminated model AND horizon name
          model.names.horizon=do.call(c,lapply(1:S, function(x) paste(model.names[[x]],"_horizon_",x,sep="")))
          
          rm(list = "indexes_b")
          
          p2test = Pr
          
          if (p2test > alpha) {
            
            cat(paste("\n###########################################################################################################################\n"))
            cat(paste("Superior Set Model created\t:\n"))
            show(model.names.horizon)
            cat(paste("p-value\t:\n"))
            print(p2test)
            cat(paste("\n###########################################################################################################################"))
            break
          }
          else {
            eliminatehorizon=which(sapply(TR_all,function(x) TR %in% x))
            elimcompname=lapply(eliminatehorizon,function(x) names(TR_all[[x]][which(TR==TR_all[[x]])]))
            eliminatemodel=as.numeric(substr(elimcompname,7,gregexpr("m",elimcompname)[[1]][2]-2))
            
            # Save off eliminated model in MCS matrix
            MCS[eliminatemodel,eliminatehorizon]=0
            
            cat(paste("\nModel", eliminatemodel,"horizon",eliminatehorizon, 
                      "eliminated", Sys.time()))
            # After at least one model is eliminated from horizon k, eliminatemodel does not correspond to the column of model k
            eliminatemodelpos=lapply(1:length(eliminatehorizon),function(x) which(colnames(Loss[[eliminatehorizon[x]]]) %in% paste("model_",eliminatemodel[x],sep="")))
            Loss[eliminatehorizon]= lapply(1:length(eliminatehorizon),function(x) as.matrix(as.data.frame(Loss[[eliminatehorizon[x]]])[-c(eliminatemodelpos[[x]])]))
            if (all(sapply(Loss,ncol)==1)) {
              
              cat(paste("\n###########################################################################################################################\n"))
              cat(paste("Superior Set Model created\t:\n"))
              model.names.horizon=model.names.horizon[-c(which(model.names.horizon %in% paste("model",eliminatemodel,"horizon",eliminatehorizon,sep="_")))]
              
              show(model.names.horizon)
              cat(paste("p-value\t:\n"))
              show(p2test)
              cat(paste("\n###########################################################################################################################"))
              break
            }
          }
          
          # repeat loop end
        }
        
        # Store results
        MCSrej=MCSrej+sum(MCS==0)
        
        # Now do the HLN procedure for independent s
        Loss=LossMat
        
        MCS=matrix(0,M0,S)
        
        for (i in 1:S){
          
          MCSdata=as.matrix(as.numeric(sapply(strsplit(c(row.names((MCSprocedureDestudentized(Loss[[i]],alpha=alpha,statistic = "TR",B=B))@show)),split="_",fixed=TRUE),function(x) (x[2]))))
          MCS[MCSdata,i]=1
        }
        
        
        # Store results
        MCSHLNrej=MCSHLNrej+sum(MCS==0)
        
        # i loop end
      }
      
      # Average over simulations
      MCSrej=MCSrej/(Sim*S)
      MCSHLNrej=MCSHLNrej/(Sim*S)
      
      # Store Results
      results[thetacount,Scount]=MCSrej
      results[length(thetavalues)+thetacount,Scount]=MCSHLNrej
      
      path=paste("Sim Results/Temp/Update-M",M0,"-P",P,"-theta",theta,"-S",S,"-delta-",delta,"-rho-",rho,"-alpha-",alpha,".csv",sep="")
      write.table(results,path, row.names=FALSE, col.names=FALSE, sep=",")
      
      
      # thetacount loop end
    }
    
    # Scount loop end
  }
  
  
  path=paste("Sim Results/Results-M",M0,"-P",P,"-delta-",delta,"-rho-",rho,"-alpha-",alpha,".csv",sep="")
  write.table(results,path, row.names=FALSE, col.names=FALSE, sep=",")
  
  
}