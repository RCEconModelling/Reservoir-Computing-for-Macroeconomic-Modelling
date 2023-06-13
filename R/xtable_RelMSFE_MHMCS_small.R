#
# Weave LaTeX table of results (Relative MSFE + Multi-horizon MCS) for SMALL dataset
#
# NOTE: run in R version 4.2.3 (2023-03-15 ucrt)

library(tidyverse)
library(stringr)
library(xtable)

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling")

RelRMSFE_MH_small = read_csv(paste0(getwd(), "/R/", "RelRMSFE_MH_small.csv"), na = "")[,c(-1)]
MHMCS_small = read_csv(paste0(getwd(), "/R/", "MHMCS_small_stars.csv"), na = " ")

# Adjust column names in different dataframes
colnames(MHMCS_small) = str_replace_all(colnames(MHMCS_small), "MHMCS SMALL MULTISTEP ", "")
colnames(RelRMSFE_MH_small) = str_replace_all(colnames(RelRMSFE_MH_small), "_", " ")

# Pivot MCS
MHMCS_small_pivot = MHMCS_small %>% 
  pivot_longer(
    cols=`FIX 2007`:`RW 2011`,
    names_to = "Setup",
    values_to = "uMCS"
  ) %>%
  rename(
    Model = Method
  )

# Make RealMSFE by squaring RelRMSFE
RelMSFE_MH_small = RelRMSFE_MH_small %>%
  mutate(
    across(`1`:`8`, ~ .^2)
  )

# Generate new "Setup" column for RelRMSFE
RelMSFE_MH_small_mod = RelMSFE_MH_small %>%
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
results_table_DF = RelMSFE_MH_small_mod %>%
  left_join(
    MHMCS_small_pivot,
    by = c("Setup", "Model")
  ) %>%
  relocate(
    Setup, Model, `1`:`8`, uMCS
  )

print(xtable(results_table_DF, digits=3), include.rownames=FALSE)

# -----