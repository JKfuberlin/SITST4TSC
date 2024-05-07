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

################


######
# Set up parallelization
######


# detect cores and set them to n - 2
parallel::detectCores()
n.cores <- parallel::detectCores() - 2
#create the cluster
my.cluster <- parallel::makeCluster(
  n.cores,
  type = "FORK",
  outfile = "/home/j/outfile2"
)

#check cluster definition (optional)
print(my.cluster)

#register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

#check if it is registered (optional)
foreach::getDoParRegistered()

standardize = 1


################
### define function
################

process_file <- function(file_path) {
  # read csv file
  a <- data.table::fread(file_path)
  
  # remove unneeded columns
  a <- dplyr::select(a, -c(1, 3,4,5)) # hardcoding, keeping only OBJECTID
  # a2 <- dplyr::select(a, -c(X,KATEGORIE:SHAPE_STLe))
  
  # change colnames
  # i end up with incredibly messed up column names after extraction, such as99803077186.csv
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
                           names_to = c(".value", "band")) # ...to write each band value into the corresponding row while maintaining the date information inside the column names 
  
  # rotate df to merge the information from each band with the corresponding date
  c <- sjmisc::rotate_df(b, rn = T, cn = F) # rn = T means rownames become columnnames cn means
  
  # remove first 2 rows (should only contain ID and band by now)
  d <- c %>% dplyr::slice(3:nrow(c))

  # Exclude rows with NA values
  d <- d[complete.cases(d), ]

  # Make sure, everything is numeric beacuse of weirdness
  d[] <- lapply(d, as.numeric)

  # Filter out rows with any negative value
  e <- d[!apply(d < 0, 1, any), ]
  
  # rename first colname to 'date'
  e <- dplyr::rename(e, 'date' = 'TRUE') # somehow the first colname is assigned "TRUE"
  
  # turn date into lubridate format
  e$date <- lubridate::ymd(e$date)
  
  # calc number of days passed since 1-1-2015
  TS.origin <- lubridate::ymd("20150101")
  
  # calculate DOY
  int <- lubridate::interval(TS.origin, e$date)
  per <- lubridate::as.period(int, unit = 'day')
  e <- e %>% dplyr::mutate(DOY = lubridate::as.period(lubridate::interval(TS.origin, e$date)))
  e$DOY <- lubridate::time_length(e$DOY, unit='days')

  # calculate DOY
  e <- e %>% dplyr::mutate(DOY = yday(date), year = year(date), DOY2 = DOY + 365 * (year - 2015) + dplyr::if_else(leap_year(date) & DOY > 59, 1, 0))
  
  # round numeric columns
  e <- e %>% dplyr::mutate_if(is.numeric, ~round(., 0))

  # drop date column
  e = e[, -1]
  e = e[, -12]

  # # Select all columns except the last two
  columns_to_standardize <- 1:(ncol(e) - 2)  

  df = e
  # Standardize selected columns
  df[, columns_to_standardize] <- lapply(df[, columns_to_standardize], as.numeric)
  # if (standardize == 1) {
  #   df[, columns_to_standardize] <- scale(df[, columns_to_standardize])
  # } else {
  #    # divide by 10000
  # df[, columns_to_standardize] <- df[, columns_to_standardize] / 10000
  # }
  
  # write to csv file
  # if (standardize == 1) {
  #   data.table::fwrite(df, file = paste0("/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl04_balanced_buffered_reshaped_standardized_species/", basename(file_path)))

  # } else {
  # data.table::fwrite(df, file = paste0("/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl04_balanced_buffered_reshaped_nonstandardized_species/", basename(file_path)))
  data.table::fwrite(df, file = paste0("/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl04_balanced_buffered_reshaped_species/", basename(file_path)))

}


# Plaintext explanation of what i (try to) do:
# original file has on col per band per timestep,so first each timestep is written into a separate row containing only the bands of said timestep
# all NA observations are deleted at this point. After extraction the Value should be -9999 or NA, i do not replace it by 0 but delete the entire timestep
# two separate DOYs are calculated, one that represents the absolute position within the multiyear timeseries, the other to represent the timestep within the seasonal pattern
# standardization is essentially dropped because the reflectance values already represent percentages of reflectance at the sensor level and only need to be divided by 
# 10.000(theoretical max value) to get floating point values which are necessary for the DL

######

# the thing about standardization is whether to normalize each band separately or all the values at once. 
# scale() does it per column (except DOY) in the following way:
# Mean Centering: It subtracts the mean of each variable from all the values in that variable. This centers the distribution of each variable around zero.
# Scaling: It divides each value by the standard deviation of the variable. This scales the distribution of each variable, ensuring that the variance of each variable is equal to 1.



# read all csv 
tic()
csv = list.files(path = "/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl03_balanced_buffered_unshaped_species/",full.names = T, pattern = "\\.csv$") 
cat('csv loaded')
toc()

tic()
# for (file in csv) {
#   process_file(file)
# }

foreach::registerDoSEQ() # initialize sequential parallel backend
clusterExport(my.cluster, c("csv", "process_file"))
result <- parLapply(my.cluster, csv, process_file)
parallel::stopCluster(cl = my.cluster)

toc()
gc()