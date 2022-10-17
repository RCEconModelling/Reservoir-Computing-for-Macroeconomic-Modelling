### Pairwise model forecasting tests ###

#Diebold, F.X. and Mariano, R.S. (1995) Comparing predictive accuracy. Journal of Business and Economic Statistics, 13, 253-263.

#Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of forecasting, 13(2), 281-291.

### -------------------------------- ###

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
dd = read.csv(file = 'errors.csv')                  #change file name accordingly
colnames(dd)[1] <- gsub('^...','',colnames(dd)[1])
dm = matrix(nrow=ncol(dd),ncol=ncol(dd))            #output test results
dm[1,]=colnames(dd)
dm[,1]=colnames(dd)

# DM TEST
for(i in 2:ncol(dd)) {       
  e1 = dd[,i]
  j = i+1
  while (j <= ncol(dd)) {
    e2 = dd[,j]
    test = dm.test(e1,e2,alternative = c("g"),h = 1,power = 2)
    #(test$statistic)      # test statistic
    #print(test$p.value)        # p-value
    dm[i,j] = test$p.value
    j = j+1
  }
}

for(i in 2:ncol(dd)) {       
  e1 = dd[,i]
  j = i+1
  while (j <= ncol(dd)) {
    e2 = dd[,j]
    test = dm.test(e1,e2,alternative = c("l"),h = 1,power = 2)
    #print(test$statistic)      # test statistic
    #print(test$p.value)        # p-value
    dm[j,i] = test$p.value
    j = j+1
  }
}
#dm

# OUTPUT    
write.csv(dm, 'output_pairwise.csv', row.names=FALSE) #change file name accordingly
