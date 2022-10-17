### Pairwise model forecasting tests multi-step ###

#Diebold, F.X. and Mariano, R.S. (1995) Comparing predictive accuracy. Journal of Business and Economic Statistics, 13, 253-263.

#Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of forecasting, 13(2), 281-291.

### ------------------------------------------- ###

# Syntax 

#dm.test(
#  e1,                                                  #error M1
#  e2,                                                  #error M2
#  alternative = c("two.sided", "less", "greater"),     #hypothesis
#  h = 1,                                               #forecast horizon 
#  power = 1                                            #power used in the loss function (power 1 as we already have SFE as inputs)
#)

### -------------------------------- ###


# LIBRARIES
library(forecast) 

# DIRECTORY
primary_directory = 'C:/Users/sophi/OneDrive/Desktop/RC Data'
setwd(primary_directory)

# LOAD DATA
dd = data.frame() #input errors
dd = read.csv(file = 'multi-step.csv')                  #change file name accordingly
colnames(dd)[1] <- gsub('^...','',colnames(dd)[1])
dm = matrix(nrow=40,ncol=8)                             #rearranged data, change to nrow=40 for 2007 and nrow=24 for 2011
out = matrix(nrow=8,ncol=8)                             #output test results

# VARIABLE NAMES
names = c("Mean","MIDAS","DFM [A]","DFM [B]","singleESN [A]","singleESN [B]","multiESN [A]","multiESN [B]")
colnames(out) = names
rownames(out) = names


for(i in 0:7) {
  
  # ARRANGE DATA
  j = 1
  while (j <= 8){
    gg = dd[which(dd$H==i & dd$M0==names[j]),]
    dm[,j] = gg$Loss
    j = j+1
  }
  dm_out = data.frame(dm)
  colnames(dm_out) = names
  
  # DM TEST
  for(k in 1:ncol(dm_out)) {       
    e1 = dm_out[,k]
    l = k+1
    while (l <= ncol(dm_out)) {
      e2 = dm_out[,l]
      test = dm.test(e1,e2,alternative = c("g"),h = i+1,power = 2)
      #(test$statistic)      # test statistic
      #print(test$p.value)        # p-value
      out[k,l] = test$p.value
      l = l+1
    }
  }
  
  for(k in 1:ncol(dm_out)) {       
    e1 = dm_out[,k]
    l = k+1
    while (l <= ncol(dm_out)) {
      e2 = dm_out[,l]
      test = dm.test(e1,e2,alternative = c("l"),h = i+1,power = 2)
      #print(test$statistic)      # test statistic
      #print(test$p.value)        # p-value
      out[l,k] = test$p.value
      l = l+1
    }
  }
  
  output_filename = paste0(i+1,'_multi-step.csv')
  write.csv(out, output_filename) #change file name accordingly
  
}


 



