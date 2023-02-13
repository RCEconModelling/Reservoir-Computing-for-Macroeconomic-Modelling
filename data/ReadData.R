# NOTE

# Data files need to be set with release dates dd/mm/yyyy for each observation.
# Data file with highest frequency is reference file and must span over the full sample period. 
# File must be sourced, otherwise the code won't obtain curr_dir.
# It requires the "rugarch" package to estimate the model. If not installed, run the commmand "install.packages("rugarch")

# DIRECTORY

curr_dir = dirname(parent.frame(2)$ofile) #get current path when sourcing the file
setwd(curr_dir)


# LIBRARIES

library(rugarch) #required to estimate GARCH model


# SET SAMPLE 

startDate = "1959-01-01"
endDate = "2022-12-31"
date_format = '%d/%m/%Y'
model_counter = 1         #to save model coefficients
m = list()                #to save model coefficients
# DATA FRAMES

ref.data = data.frame()
data.d = data.frame()
data.w = data.frame()
data.m = data.frame()
dara.q = data.frame()
data.y = data.frame()


# LOAD DATA (hide if non applicable)

data.d = read.csv(file = 'day.csv')
#data.w = read.csv(file = 'week_sa.csv')
data.m = read.csv(file = 'month.csv')
data.q = read.csv(file = 'quarter.csv')
#data.y = read.csv(file = 'year.csv')


# ONLY USE IF LOADED DATA HAS DATE INCORRECTLY SPECIFIED

#colnames(data.d)[1] <- gsub('^...','',colnames(data.d)[1])
#colnames(data.w)[1] <- gsub('^...','',colnames(data.w)[1])
#colnames(data.m)[1] <- gsub('^...','',colnames(data.m)[1])
#colnames(data.q)[1] <- gsub('^...','',colnames(data.q)[1])
#colnames(data.y)[1] <- gsub('^...','',colnames(data.y)[1])

#data.d$Date = as.Date(data.d$Date,format="%d/%m/%Y")
#data.w$Date = as.Date(data.w$Date,format="%d/%m/%Y")
#data.m$Date = as.Date(data.m$Date,format="%d/%m/%Y")
#data.q$Date = as.Date(data.q$Date,format="%d/%m/%Y")
#data.y$Date = as.Date(data.y$Date,format="%d/%m/%Y")


# SPECIFY FUNCTIONS FOR DATA TRANSFORMATION

D = function(x){
  c(NA,diff(x))
}
DD = function(x){
  c(NA,NA,diff(diff(x)))
}
L = function(x){
  c(log(x))
}
DL = function(x){
  c(NA,diff(log(x)))
}
DDL = function(x){
  c(NA,NA,diff(diff(log(x))))
}
DPCT = function(x){
  x1 = x[-(1)]
  for (i in 1:length(x1)){
    tt = (x[i+1]/x[i] - 1)
    x1[i] = tt
  }
  c(NA,x1)
}
fitGARCH = function(x, return_coefs = FALSE,model_counter){
  # ---- fitGARCH fits a GARCH(1,1) model into the data ----
  # inputs:
  #   x: numeric array, containing log-diferences of prices
  #   return_coefs = boolean, If TRUE, returns a 2 element named list where "coefs" contains
  #                           the coefficients of the estimation and "volatility" contains the
  #                           estimated volatility using a GARCH(1,1). Default value = FALSE
  # outputs: 
  #   output: list containing a numeric array with the same length as x that contains the volatility
  #           estimations produced by the GARCH(1,1) model
  # 
  # notes:
  #   -NA's are ignored in the estimation. The output will contain the same number and position of NA's
  # - The function asumes that the data will be pre-filtered before applying the transformation. E.g. If we
  #   want to estimate the model with data from 1990-01-01 onwards, x should be already filtered to ensure
  #   the correct estimation
  
  x = DL(x) #Compute log-returns from price series to estimate volatility
  na_mask = is.na(x)
  spec = ugarchspec(variance.model = list(model = 'sGARCH',
                                          garchOrder = c(1,1)),
                    mean.model = list(armaOrder = c(0,0)))
  fit = ugarchfit(data = x[!na_mask], 
                  spec = spec,
                  fit.control = list(scale = 0, rec.init = 'all'),
                  rseed = 2) 
  #note: scale = 1 scales values  by std dev to ensure convergence (use when including full sample)
  vol = fit@fit$sigma #returns model-implied volatility
  x[!na_mask] = vol
  
  if (return_coefs){ #check if we want coefficients or not
    #output = list(coefs = coef(fit), volatility = x)
    output = x
    model_coefficients = coef(fit)
    m[[model_counter]]<<- model_coefficients #assign fit coefs to global env.
    model_counter <<- model_counter + 1
  }else{
    output = x
  }
  return(output)
} 

transform = function(x){
  c2 = which(x[1,]==2)
  c3 = which(x[1,]==3)
  c4 = which(x[1,]==4)
  c5 = which(x[1,]==5)
  c6 = which(x[1,]==6)
  c7 = which(x[1,]==7)
  c8 = which(x[1,]==8)
  x1 = x[-(1),]
  for (i in c2){
    tt = D(x1[,i]) 
    x1[,i] = tt
  }
  for (i in c3){
    tt = DD(x1[,i]) 
    x1[,i] = tt
  }
  for (i in c4){
    tt = L(x1[,i]) 
    x1[,i] = tt
  }
  for (i in c5){
    tt = DL(x1[,i]) 
    x1[,i] = tt
  }
  for (i in c6){
    tt = DDL(x1[,i]) 
    x1[,i] = tt
  }
  for (i in c7){
    tt = DPCT(x1[,i]) 
    x1[,i] = tt
  }
  for (i in c8){
    tt = fitGARCH(x1[,i], return_coefs = TRUE, model_counter = model_counter)
    x1[,i] = tt
  }
  x = x1
}

date_filtering = function(x,start_date, end_date, date_format,get_transform_header = FALSE){
  transform_header = x[1,]
  if(get_transform_header){assign("x_header",transform_header,env = globalenv())}
  x$Date = as.Date(x$Date,format = date_format)
  x = x[x$Date >= start_date & x$Date <= end_date,]
  x[1,] = transform_header
  return(x)
}


# TRANSFORM DATA

x = date_filtering(data.d,
                   start_date =  startDate,
                   end_date = endDate, 
                   date_format = date_format,
                   get_transform_header = TRUE)
data.d = transform(x)
daily_garch_coefs = data.frame(Reduce(rbind, m,init = NULL),
                               row.names = colnames(x_header)[which(x_header == 8)])
#write output for GARCH(1,1) parameters for daily data
write.csv(daily_garch_coefs,'daily_garch_coefs.csv')
rm(list = c('m',"x_header"))


#x = date_filtering(data.w,
#                   start_date =  startDate,
#                   end_date = endDate, 
#                   date_format = date_format)
#data.w = transform(x)

x = date_filtering(data.m,
                   start_date =  startDate,
                   end_date = endDate, 
                   date_format = date_format)
data.m = transform(x)

x = date_filtering(data.q,
                   start_date =  startDate,
                   end_date = endDate, 
                   date_format = date_format)
data.q = transform(x)

#x = date_filtering(data.y,
#                   start_date =  startDate,
#                   end_date = endDate, 
#                   date_format = date_format)
#data.y = transform(x)


# MERGE DATA

ref.data = data.d
# ref.data = merge(ref.data,data.w,by="Date",all.x=TRUE)
ref.data = merge(ref.data,data.m,by="Date",all.x=TRUE)
ref.data = merge(ref.data,data.q,by="Date",all.x=TRUE)
#ref.data = merge(ref.data,data.y,by="Date",all.x=TRUE)


# REPEAT OBSERVATIONS TO FILL 'NA' FOR LOW FREQUENCY SERIES

replace_na = function(x,a=!is.na(x)){
  x[which(a)[c(NA,1:sum(a))][cumsum(a)+1]]
}

new.data = data.frame(ref.data$Date)

for (i in 2:ncol(ref.data)){
  new.data = data.frame(new.data,replace_na(ref.data[,i]))
}

colnames(new.data) = names(ref.data)


# OUTPUT   

d0 = gsub(pattern = '[-]','',startDate) #remove '-' from dates
d1 = gsub(pattern = '[-]','',endDate)
# output_nofill_filename = paste0(d0,'_', d1,'_output_data.csv')
# output_fill_filename = paste0(d0,'_', d1,'_output_filled_data.csv')
output_nofill_filename = paste0('fullsample','_output_data.csv')
output_fill_filename = paste0('fullsample','_output_filled_data.csv')

write.csv(ref.data, output_nofill_filename, row.names=FALSE)
write.csv(new.data, output_fill_filename, row.names=FALSE)

