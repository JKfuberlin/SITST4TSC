# this script uses parallel processing to reshape the csv files consisting of reflectance values of the SITS for pixels from Forsteinrichtung
# The tables are formated from wide to long format
# it then calculates the day of year (DOY) and standardizes the reflectance values
# the reshaped data is then written to a new csv file
# each csv should now consist of 10 rows for the reflectance values of each band and the DOY in two different formats

######
# install and/or load packages
######


list.of.packages <- c(
  "tictoc", # benchmarking
  "tidyr",
  "dplyr",
  "sjmisc", # for data reshaping
  "foreach", # parallelization
  "doParallel", # parallelization
  "lubridate", # for date operations
  "data.table" # reading in multicolumn csv faster
)

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages) > 0){
  install.packages(new.packages, dep=TRUE)
}

#loading packages
for(package.i in list.of.packages){
  suppressPackageStartupMessages(
    library(
      package.i, 
      character.only = TRUE
    )
  )
}

STANDARDIZE = FALSE
################
### define function
################

process_file <- function(file_path) {
  # read csv file
  a <- data.table::fread(file_path)
  
  # remove unneeded columns
  a <- dplyr::select(a, -c(1,3,4,5,6,7,8,9)) # hardcoding, keeping only OBJECTID
  
  # change colnames
  # i end up with incredibly messed up column names after extraction, such as
  # "mean.mean..force.FORCE.C1.L2.ard.X0059_Y0057.20150806_LEVEL2_SEN2A_BOA.tif1"
  # i need to clean up the column name because it is the base for rearranging the csv files to be turnt into tensors later on
  colnames = names(a)
  relevant_colnames = colnames[2:length(colnames)] # not changing ID
  new_colnames = character()
  new_colnames = c(new_colnames, "OBJECTID") # adding back ID
  for (i in 1:length(relevant_colnames)) { # manipulating column names and adding them back
    this.colname = relevant_colnames[i]
    split_string <- strsplit(this.colname, "\\.")
    first.word = unlist(split_string[[1]][1]) # i need to find out the statistic metric, which can be found at the beginning of the string
    new.colname = paste(this.colname,first.word, sep = ".") # then i put the metric at the end
    new.colname  <- sub(".*?([0-9]{8}.*)", "\\1", new.colname) # i now remove everything before the date
    new.colname <- sub("_LEVEL2_SEN2[AB]_BOA\\.", "", new.colname) # and clean up some more
    new_colnames = c(new_colnames, new.colname)
  }
  
  a2 = a
  names(a2) = new_colnames
  
  # pivot longer
  b <- tidyr::pivot_longer(a2, cols = -c(OBJECTID),
                           names_sep = "tif", # looks for tif in the column name and uses the info after (bandnumber)...
                           names_to = c(".value", "band")) # ...to write each band value into the corresponding row containing the date 
  
  # rotate df
  c <- sjmisc::rotate_df(b, rn = T, cn = F)
  
  # remove first 2 rows (should only contain ID and band by now)
  d <- c %>% dplyr::slice(3:nrow(c))

  # Exclude rows with NA values
  d <- d[complete.cases(d), ]

  # Make sure, everything is numeric beacuse of weirdness
  d[] <- lapply(d, as.numeric)

  # Filter out rows with any negative value
  d <- d[!apply(d < 0, 1, any), ]
  
  # rename first colname to 'date'
  d <- dplyr::rename(d, 'date' = 'TRUE') # somehow the first colname is assigned "TRUE"
  
  # turn date into lubridate format
  d$date <- lubridate::ymd(d$date)
  
  # calc number of days passed since 1-1-2015
  TS.origin <- lubridate::ymd("20150101")
  
  # calculate DOY
  int <- lubridate::interval(TS.origin, d$date)
  per <- lubridate::as.period(int, unit = 'day')
  d <- d %>% dplyr::mutate(DOY = lubridate::as.period(lubridate::interval(TS.origin, d$date)))
  d$DOY <- lubridate::time_length(d$DOY, unit='days')

  # calculate DOY
  d <- d %>% dplyr::mutate(DOY = yday(date), year = year(date), DOY2 = DOY + 365 * (year - 2015) + dplyr::if_else(leap_year(date) & DOY > 59, 1, 0))
  
  # round numeric columns
  e <- d %>% dplyr::mutate_if(is.numeric, ~round(., 0))

  # drop date column and year column
  e = e[, -1]
  e = e[, -12]

  # # Select all columns except the last two
  columns_to_standardize <- 1:(ncol(e) - 2)  

  df = e
  # Standardize selected columns
  # df[, columns_to_standardize] <- lapply(df[, columns_to_standardize], as.numeric)
  # if (STANDARDIZE == TRUE) {
  #   df[, columns_to_standardize] <- scale(df[, columns_to_standardize])
  # } else {
  #    # divide by 10000
  # df[, columns_to_standardize] <- df[, columns_to_standardize] / 10000
  # }
  
  # write to csv file
#   if (STANDARDIZE == TRUE) {
#     data.table::fwrite(df, file = paste0("/home/j/data/BI/csv_BI_reshaped/", basename(file_path)))
#   } else {
#   data.table::fwrite(df, file = paste0("/home/j/data/BI/csv_BI_reshaped_nonstan/", basename(file_path)))
# }
data.table::fwrite(df, file = paste0("/home/j/data/BI/csv_BI_reshaped/", basename(file_path)))
}

######
# the thing about standardization is whether to normalize each band separately or all the values at once. 
# scale() does it per column (except DOY) in the following way:
# Mean Centering: It subtracts the mean of each variable from all the values in that variable. This centers the distribution of each variable around zero.
# Scaling: It divides each value by the standard deviation of the variable. This scales the distribution of each variable, ensuring that the variance of each variable is equal to 1.
# read all csv 

tic()
csv = list.files(path = "/home/j/data/BI/csv_bi/",full.names = T, pattern = "\\.csv$") 
cat('csv loaded')
toc()

tic()
for (file in csv) {
  process_file(file)
}

toc()
gc()

labels = read.csv("/home/j/data/BI/BI_labels_unbalanced.csv")
ids = format(labels$ID, scientific=FALSE)
dir = "/home/j/data/BI/csv_BI_reshaped/"
files = list.files(dir)


ids <- trimws(ids)

# Extract numeric portion from file names
file_ids <- as.numeric(sub("\\.csv$", "", files))

# IDs without corresponding files
ids_without_files <- ids[!ids %in% file_ids]

# Files without corresponding IDs
files_without_ids <- files[!file_ids %in% ids]

# Print results
print("IDs without corresponding files:")
print(ids_without_files)

print("Files without corresponding IDs:")
print(files_without_ids)

# remove files without corresponding IDs
file.remove(files_without_ids)