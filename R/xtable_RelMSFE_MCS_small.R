#
# Weave LaTeX table of results (Relative MSFE + MCS) for SMALL dataset
# 
# NOTE: run in R version 4.2.3 (2023-03-15 ucrt)

library(tidyverse)
library(xtable)

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling")

RelMSFE_small = read_csv(paste0(getwd(), "/R/", "RelMSFE_small.csv"), na = "")[,c(-1)]
MCS_small = read_csv(paste0(getwd(), "/R/", "MCS_small_stars.csv"), na = " ")

# Join dataframes
results_table_DF = data.frame(
  row.names = rownames(MCS_small)
)

results_table_DF["Method"] = MCS_small$Method
for (c in colnames(RelMSFE_small)) {
  results_table_DF[paste(c, "RelMSFE")] = RelMSFE_small[c]
  results_table_DF[paste(c, "MCS")] = MCS_small[c]
}

print(xtable(results_table_DF, digits=3), include.rownames=FALSE)

# -----