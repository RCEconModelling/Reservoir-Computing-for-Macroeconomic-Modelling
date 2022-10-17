#
# Weave LaTeX table of results (Relative RMSFE + Multi-horizon MCS) for MEDIUM dataset
# 

library(tidyverse)
library(stringr)
library(xtable)

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling")

RelRMSFE_MH_medium = read_csv(paste0(getwd(), "/R/", "RelRMSFE_MH_medium.csv"), na = "")[,c(-1)]
MHMCS_medium = read_csv(paste0(getwd(), "/R/", "MHMCS_medium_stars.csv"), na = " ")

# Adjust column names in different dataframes
colnames(MHMCS_medium) = str_replace_all(colnames(MHMCS_medium), "MHMCS MEDIUM MULTISTEP ", "")
colnames(RelRMSFE_MH_medium) = str_replace_all(colnames(RelRMSFE_MH_medium), "_", " ")

# Pivot MCS
MHMCS_medium_pivot = MHMCS_medium %>% 
  pivot_longer(
    cols=`FIX 2007`:`RW 2011`,
    names_to = "Setup",
    values_to = "uMCS"
  ) %>%
  rename(
    Model = Method
  )

# Generate new "Setup" column for RelRMSFE
RelRMSFE_MH_medium_mod = RelRMSFE_MH_medium %>%
  mutate(
    across(`FIX 2007`:`RW 2011`, ~ ifelse(!is.na(.), cur_column(), NA))
  ) %>% 
  rowwise() %>%
  mutate(
    Setup = max(c_across(`FIX 2007`:`RW 2011`), na.rm = TRUE)
  ) %>%
  select(
    -c(`FIX 2007`:`RW 2011`)
  )

# Merge
results_table_DF = RelRMSFE_MH_medium_mod %>%
  left_join(
    MHMCS_medium_pivot,
    by = c("Setup", "Model")
  ) %>%
  relocate(
    Setup, Model, `1`:`8`, uMCS
  )

print(xtable(results_table_DF, digits=3))

# -----