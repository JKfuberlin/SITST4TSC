# This script is used to inspect the buffered samples. It reads the buffered samples, moves the corresponding csv files to a different folder, 
# encodes the species and evergreen variables to create pxl_buffered_labels_unbalanced.csv, balances the samples, and saves the balanced samples to a csv file.

library("plyr")
library("tidyverse")
library("sf")
library("data.table")
library("dplyr")
library("doParallel")
library("foreach")

# Set Flow variables
BUFFERANDMOVEFILES = FALSE # Careful, this MOVES, not COPIES!
CREATEUNBALANCEDLABELS = FALSE
BALANCELABELSANDMOVEDATA = FALSE # Careful, this MOVES, not COPIES!
SPECIES = TRUE
EVERGREEN = FALSE
CREATEGPKG = TRUE

# settings
a = sf::read_sf("/home/j/data/FE_train_3035_buffered_10m.gpkg")
HDD = "/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904"

# Define function to move files
move_files <- function(file) {
  file.copy(file.path(folder_path, file), file.path(destination_dir, file))
  file.remove(file.path(folder_path, file))
}
# Define function to balance a dataset
balance_dataset <- function(data, var, seed) {  # data: input dataframe, var: target variable to balance (string), filepath: output file path, seed: random seed (integer) # nolint
  set.seed(seed)
  class_counts <- table(data[[var]]) # Count the number of observations in each class
  min_count <- min(class_counts) # Find the minimum number of observations among classes
  balanced_df <- data.frame() # Initialize an empty dataframe to store the balanced samples
  for (class_name in names(class_counts)) { # Loop through each class
    class_df <- data[data[[var]] == class_name,]   # Subset the dataframe for the current class
    sampled_indices <- sample(1:nrow(class_df), min_count, replace = FALSE)   # Sample the minimum number of observations for the current class
    balanced_df <- rbind(balanced_df, class_df[sampled_indices, ]) # Append the sampled observations to the balanced dataframe
  }
  return(balanced_df) # Return the balanced dataframe
}

# Careful, this MOVES, not COPIES!
if (BUFFERANDMOVEFILES) { # use the already buffered gpkg file to move the corresponding csv files to a new folder
  cl <- makeCluster((detectCores()-2))  # Detect available cores
  ID_vector = a$ID
  folder_path <- file.path(HDD, "data/pxl01_unbalanced_unbuffered/")  # Directory where files are stored
  destination_dir <- file.path(HDD, "data/pxl02_unbalanced_buffered/")  # Directory where you want to move the files
  files <- list.files(folder_path)  # Get list of files in the folder
  file_ids <- as.integer(sub("\\.csv$", "", files))  # Extract IDs from filenames
  files_to_move <- files[file_ids %in% ID_vector]  # Find files to move
  if (length(files_to_move) > 0) {  # Move files in parallel
    foreach(file = files_to_move) %dopar% {
      move_files(file)
    }
    print(paste("Files moved:", files_to_move))
  } else {
    print("No files to move.")
  }
  stopCluster(cl) # Stop parallel backend
}
if (CREATEUNBALANCEDLABELS) { # convert the BST1_BA_1 to encoded values
  a = dplyr::select(a, c("ID","BST1_BA_1"))
  # create a vector with the encoded values based on the dictionary
  code <- c(210, 310, 410, 4, 630, 710, 7, 8, 9)  # i exclude 110 to trick the positional encoding because it cannot start at 0
  # create a vector with the corresponding ranges for the "oak" and other categories
  pine_range <- c(410:490)
  oak_range <- c(600:620)
  sycamore_range <- c(820:829)
  other_evergreen_range <- c(1, 120:199, 220:299, 320:399, 420:499)
  other_dec_range <- c(6, 690, 810, 830:980, 500:590) # Larix 500-599 put into deciduous
  evergreen =  c(1, 100:499)
  deciduous = c(6, 500:980)

  a$encoded <- ifelse(a$BST1_BA_1 %in% oak_range, 4,
                      ifelse(a$BST1_BA_1 == 110, 0, 
                            ifelse(a$BST1_BA_1 %in% sycamore_range, 7, 
                                    ifelse(a$BST1_BA_1 %in% other_evergreen_range, 8, 
                                          ifelse(a$BST1_BA_1 %in% other_dec_range, 9,
                                                  match(a$BST1_BA_1, code))))))

  a$evergreen = ifelse(a$BST1_BA_1 %in% evergreen, 1,
                      ifelse(a$BST1_BA_1 %in% deciduous, 0, NA))

  hist(a$evergreen)

  # check result
  b = a[sample(nrow(a), 100), ]
  unique(a$encoded)

  hist(a$encoded)
  count_0spruce = nrow(a[a$encoded == 0,])
  count_1fir = nrow(a[a$encoded == 1,])
  count_2dgl = nrow(a[a$encoded == 2,])
  count_3pine = nrow(a[a$encoded == 3,])
  count_4oak = nrow(a[a$encoded == 4,])
  count_5redoak = nrow(a[a$encoded == 5,])
  count_6beech = nrow(a[a$encoded == 6,])
  count_7sycamore = nrow(a[a$encoded == 7,])
  count_8otherevergreen = nrow(a[a$encoded == 8,])
  count_9otherdec = nrow(a[a$encoded == 9,])

  data.table::fwrite(a, '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl_buffered_labels_unbalanced.csv')
}
if (BALANCELABELSANDMOVEDATA) { # balances the dataset while writing corresponding labels and moves the balanced csv files to a new folder
  a = data.table::fread(file.path(HDD, "data/pxl_buffered_labels_unbalanced.csv"))
  if (SPECIES) {
    balanced_df = balance_dataset(a, 'encoded', 420)
    target_ID = balanced_df$ID
    balanced_labels = a[a$ID %in% target_ID,]
    data.table::fwrite(balanced_labels, file.path(HDD, "data/pxl_buffered_labels_balanced_species.csv"))
    folder_path <- file.path(HDD, "data/pxl02_unbalanced_buffered/")  # Directory where your files are stored
    destination_dir <- file.path(HDD, "data/pxl03_balanced_buffered_unshaped_species/")  # Directory where you want to move the files
    files <- list.files(folder_path)  # Get list of files in the folder
    file_ids <- as.integer(sub("\\.csv$", "", files))  # Extract IDs from filenames
    files_to_move <- files[file_ids %in% target_ID]  # Find files to move
    cl <- makeCluster((detectCores()-2))  # Detect available cores
    if (length(files_to_move) > 0) {  # Move files in parallel
      foreach(file = files_to_move) %dopar% {
        move_files(file)
      }
      print(paste("Files moved:", files_to_move))
    } else {
      print("No files to move.")
    }
    stopCluster(cl) # Stop parallel backend

    if (CREATEGPKG) { # create gpkg from balanced dataset of the species
          balanced_labels = data.table::fread(file.path(HDD, "data/pxl_buffered_labels_balanced_species.csv"))
          a = sf::read_sf("/home/j/data/FE_train_3035_buffered_10m.gpkg")
          b = dplyr::filter(a, ID %in% balanced_labels$ID)
          b = dplyr::select(b, c("ID","Tile_ID","BST1_BA_1"))
          sf::st_write(b, file.path(HDD, "data/pxl_buffered_balanced_species.gpkg"))
        }
      }

    if (EVERGREEN) {
    balanced_df = balance_dataset(a, 'evergreen', 420)
    target_ID = balanced_df$ID
    balanced_labels = a[a$ID %in% target_ID,]
    data.table::fwrite(balanced_labels, file.path(HDD, "data/pxl_buffered_labels_balanced_evergreen.csv"))
    folder_path <- file.path(HDD, "pxl02_unbalanced_buffered/")  # Directory where your files are stored
    destination_dir <- file.path(HDD, "data/pxl03_balanced_buffered_unshaped_evergreen/")  # Directory where you want to move the files
    files <- list.files(folder_path)  # Get list of files in the folder
    file_ids <- as.integer(sub("\\.csv$", "", files))  # Extract IDs from filenames
    files_to_move <- files[file_ids %in% target_ID]  # Find files to move
    if (length(files_to_move) > 0) {  # Move files in parallel
      foreach(file = files_to_move) %dopar% {
        move_files(file)
      }
      print(paste("Files moved:", files_to_move))
    } else {
      print("No files to move.")
    }
    stopCluster(cl) # Stop parallel backend
        if (CREATEGPKG) { # create gpkg from balanced dataset of the species
          b = dplyr::filter(a, ID %in% balanced_df$ID)
          b = dplyr::select(b, c("ID","encoded"))
          sf::st_write(b, file.path(HDD, "data/pxl_buffered_balanced_evergreen.gpkg"))
        }
      }
}