#
#
# 

library(tidyverse)
library(lubridate)
library(MultiHorizonSPA)

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling")

# Load data
python__multihLosses = read_csv(paste0(getwd(), "/R/Rdata__mhmcs_multisa_2007.csv"), na = "")

M0_names = as.list(unique(python__multihLosses["M0"]))[[1]]
M0 = length(M0_names)
H = dim(unique(python__multihLosses["H"]))[1]
t = dim(unique(python__multihLosses["T"]))[1]

# Extract individual models
loss_list = vector(mode = "list", length = M0)

for (m in 1:M0) {
  # Filter
  loss_list[[m]] = python__multihLosses %>%
    filter(M0 == M0_names[m]) %>%
    pivot_wider(
      names_from = H,
      values_from = Loss
    ) %>%
    select(
      -c("M0", "T")
    ) %>%
    as.matrix()
  
  # Info
  cat("Listed model: ", M0_names[m], "\n")
}

# Multi-Horizon MCS -------------------------

# alpha = 0.1
MHMCS_010 = FastMultiHorizonMCS(
  loss_list, 
  alpha_mcs = 0.1,
  L = 3, 
  B = 300, 
  unif_or_average = 'u', 
  num_cores = 2,
  seed = 202209
)

p_values_table_010 = data.frame(
  p_value = MHMCS_010$p_values,
  row.names = M0_names
) %>% arrange(p_value)

MHMCS_025 = FastMultiHorizonMCS(
  loss_list, 
  alpha_mcs = 0.25,
  L = 3, 
  B = 300, 
  unif_or_average = 'u', 
  num_cores = 2,
  seed = 202209
)

p_values_table_025 = data.frame(
  p_value = MHMCS_025$p_values,
  row.names = M0_names
) %>% arrange(p_value)
  

# Results
cat("Included models (MCS_alpha = 0.1): \n", 
    paste0(M0_names[MHMCS_010$MCS_set], "\n"), 
    "\n")
cat("Included models (MCS_alpha = 0.25): \n", 
    paste0(M0_names[MHMCS_025$MCS_set], "\n"), 
    "\n")

print(p_values_table_010)
print(p_values_table_025)


#####