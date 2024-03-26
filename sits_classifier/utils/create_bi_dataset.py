import numpy as np
import torch
from torch import nn, Tensor
import sits_classifier.utils.csv_utils as csv_utils
import os

# # settings
# PATH = '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/'
PATH = '/home/j/data/'
DATA_DIR = os.path.join(PATH, 'csv_BI_reshaped/')
LABEL_CSV = 'BI_labels_unbalanced.csv'
LABEL_PATH = os.path.join('/home/j/data/', LABEL_CSV)
FINETUNING = False

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
    labels.dropna(inplace = True) # somehow a line is interpreted as NA, most probably the header
    len(labels)
    labels = labels.astype(int)
    x_data, y_data = csv_utils.to_numpy_BI(DATA_DIR, labels)  # turn csv file into numpy dataset while balancing the data based on minority class in dataset
    # x_data = x_data[:, :, 1:12] # 1 - 12 subsets all bands + DOY
    x_set, y_set = numpy_to_tensor(x_data, y_data)  # turn dataset into tensor format
    torch.save(x_set, '/home/j/data/x_set_pxl_bi.pt')
    torch.save(y_set, '/home/j/data/y_set_pxl_bi.pt')