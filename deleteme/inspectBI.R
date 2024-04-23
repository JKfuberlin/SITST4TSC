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