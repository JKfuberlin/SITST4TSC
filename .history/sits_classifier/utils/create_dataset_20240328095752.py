import numpy as np
import torch
from torch import nn, Tensor
# import sits_classifier.utils.csv_utils as csv_utils
import csv_utils as csv_utils
import os
import pandas as pd

SPECIES = True

# # settings
PATH = '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data'
DATA_DIR = os.path.join(PATH, 'pxl04_balanced_buffered_reshaped_standardized_species')
LABEL_PATH = '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl_buffered_labels_balanced_species.csv'

def numpy_to_tensor(x_data: np.ndarray, y_data: np.ndarray) -> tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    y_data = y_data.reshape(-1)  # This reshapes the y_data numpy array from a 2-dimensional array with shape (n, 1) to a 1-dimensional array with shape (n, ).
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    sz, seq, num_bands = x_set.size(0), x_set.size(1), x_set.size(2) # retrieve size, sequence length and number of bands
    x_set = x_set.view(sz, seq,num_bands) # sz is the amount of samples, seq is the sequence length, and num_bands is the number of features
    return x_set, y_set

if __name__ == "__main__":
    labels = pd.read_csv(LABEL_PATH, sep=',', header=0) 
    len(labels)
    labels = labels.drop(columns=['geom']) # remove the column 'geom' from the labels
    labels.dropna(inplace = True) # somehow a line is interpreted as NA, most probably the header
    len(labels)
    labels = labels.astype(int)
    x_data, y_data = csv_utils.to_numpy_subset(DATA_DIR, labels, SPECIES)  # turn csv file into numpy dataset while balancing the data based on minority class in dataset
    x_data_season = x_data[:, :, 0:11] # 1 - 11 subsets all bands + DOY indicating the season
    x_data_years = np.concatenate((x_data[:, :, 0:10], x_data[:, :, 11:12]), axis=2) # 1 - 10 + 12 subsets all bands + DOY indicating the year
    x_data_season, y_set = numpy_to_tensor(x_data_season, y_data)  # turn dataset into tensor format
    x_data_years, y_set = numpy_to_tensor(x_data_years, y_data)  # dirty solution but avoids reprogramming the function
    torch.save(x_data_season, '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/x_set_pxl_buffered_balanced_species_season.pt')
    torch.save(x_data_years, '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/x_set_pxl_buffered_balanced_species_years.pt')
    torch.save(y_set, '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/y_set_pxl_buffered_balanced_species.pt')

# Annex 1 tensor.view() vs tensor.reshape()
#     view method:
#         The view method returns a new tensor that shares the same data with the original tensor but with a different shape.
#         If the new shape is compatible with the original shape (i.e., the number of elements remains the same), the view method can be used.
#         However, if the new shape is not compatible with the original shape (i.e., the number of elements changes), the view method will raise an error.
#
#     reshape method:
#         The reshape method also returns a new tensor with a different shape, but it may copy the data to a new memory location if necessary.
#         It allows reshaping the tensor even when the number of elements changes, as long as the new shape is compatible with the total number of elements in the tensor.
