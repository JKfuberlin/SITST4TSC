import numpy as np
import torch
from torch import nn, Tensor
# import sits_classifier.utils.csv_utils as csv_utils
import csv_utils as csv_utils
import os
import pandas as pd

'create a numpy dataset from csv files'


SPECIES = True

# # settings
PATH = '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data'
DATA_DIR = os.path.join(PATH, 'pxl04_balanced_buffered_reshaped_standardized_species')
LABEL_PATH = '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl_buffered_labels_balanced_species.csv'


if __name__ == "__main__":
    labels = pd.read_csv(LABEL_PATH, sep=',', header=0) 
    len(labels)
    labels = labels.drop(columns=['geom']) # remove the column 'geom' from the labels
    labels.dropna(inplace = True) # somehow a line is interpreted as NA, most probably the header
    len(labels)
    labels = labels.astype(int)
    x_data, y_data = csv_utils.to_numpy_subset(DATA_DIR, labels, SPECIES)  # turn csv files into numpy dataset 
    x_data_season = x_data[:, :, 0:11] # 1 - 11 subsets all bands + DOY indicating the season
    x_data_years = np.concatenate((x_data[:, :, 0:10], x_data[:, :, 11:12]), axis=2) # 1 - 10 + 12 subsets all bands + DOY indicating the year
    np.save('/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/x_set_pxl_buffered_balanced_species_season.npy', x_data_season)
    np.save('/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/x_set_pxl_buffered_balanced_species_years.npy', x_data_years)
    np.save('/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/y_set_pxl_buffered_balanced_species.npy', y_data)
