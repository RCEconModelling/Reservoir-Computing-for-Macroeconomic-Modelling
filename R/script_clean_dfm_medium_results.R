library(tidyverse)
library(lubridate)

# Set working directory
setwd("~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling/dfm_medium_raw")

# Target folder
target_dir = "~/GitHub/Reservoir-Computing-for-Macroeconomic-Modelling/dfm_medium"

# File name specs
inputs_prefix = "test_predictions_"
#inputs_prefix = "train_predictions_"

inputs_filenames = mapply(paste0, inputs_prefix, seq(0, 96, 12), ".csv")

# List all results folders
folders = dir()

for (f in folders) {
  for (d in inputs_filenames) {
    raw = read.csv(paste(getwd(), f, d, sep = "/"))
    
    # Filter
    clean = raw %>% select(Pred.Date, Target.Date, X0)
    
    # Save
    dir.create(file.path(target_dir, f), showWarnings = FALSE)
    write.csv(clean, paste(target_dir, f, d, sep = "/"), row.names = FALSE)
  }
  
  cat("[#] Folder:  ", f, " -  Done!\n")
}