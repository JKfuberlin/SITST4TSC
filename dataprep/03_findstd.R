library(data.table)
library(dplyr)
library(tidyverse)

labels =  "/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl_buffered_labels_balanced_species.csv"
filepath = "/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl04_balanced_buffered_reshaped_nonstandardized_species/"

labels = data.table::fread(labels)
csv = list.files(path = filepath,full.names = T, pattern = "\\.csv$") 
cat('csv loaded')

for(species in 0:9) {
    thisspecies = dplyr::filter(labels, encoded == species)
    targetfiles = as.character(thisspecies$ID)
    vals = data.frame()
    for(id in targetfiles){
        thisfile = data.table::fread(paste0(filepath,id,".csv"))
        # print(head(thisfile))
        vals = rbind(thisfile, vals)   
    }
    print(species)
    # Calculate standard deviation per column
    sd_per_column <- apply(vals, 2, sd)
    # Print the standard deviation per column
    print(sd_per_column)
}


#### same for BI
labels =  "/home/j/data/BI/BI_labels_unbalanced.csv"
filepath = "/home/j/data/BI/csv_BI_reshaped_nonstan/"
