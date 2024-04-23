library(data.table)
library(dplyr)
library(tidyverse)

labels =  "/home/j/data/balanced_labels_pxl.csv"
filepath = "/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl03_balanced_buffered_unshaped_species/"

labels = data.table::fread(labels)
csv = list.files(path = filepath,full.names = T, pattern = "\\.csv$") 
cat('csv loaded')

for(species in 0:9) {
    thisspecies = dplyr::filter(labels, encoded == species)
    targetfiles = thisspecies$ID
    vals = data.frame()
    for(id in targetfiles){
        thisfile = data.table::fread(paste0(filepath,id,".csv"))
        rbind(vals, thisfile)
    }
}
