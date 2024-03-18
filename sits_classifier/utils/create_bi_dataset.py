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

def numpy_to_tensor(x_data: np.ndarray, y_data: np.ndarray) -> tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    y_data = y_data.reshape(-1)  # This reshapes the y_data numpy array from a 2-dimensional array with shape (n, 1) to a 1-dimensional array with shape (n, ).
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    x_set = np.nan_to_num(x_set, copy=False, nan=0)
    # standardization:
    sz, seq, num_bands = x_set.shape[0], x_set.shape[1], x_set.shape[2] # retrieve amount of samples, sequence length and num_bands from tensor object
    # see Annex 1
    # batch_norm = nn.BatchNorm1d(num_bands)  # Create a BatchNorm1d layer with `num_bands` as the number of input features.
    # x_set: Tensor = batch_norm(x_set)  # standardization is used to improve convergence, should lead to values between 0 and 1
    s2_cube_np = (x_set - x_set.mean(axis=0)) / (x_set.std(axis=0) + 1e-6) # compare numpy normalization, result: it is identical to nn.BatchNorm1d
    x_set = torch.from_numpy(s2_cube_np).detach()  # The `.detach()` method is necessary here to create a new tensor that is "detached from the computation graph" as we only want to apply this normalization once
    # .detach prevents gradients from flowing backward through a tensor.
    return x_set, y_set


# the thing about standardization is whether to normalize each band separately or all the values at once. in this
# case, nn.BatchNorm1d(num_bands), each band is normalized individually afaik in case all values are merged there is
# the problem of inclusion of date/DOY values at this point of development. These columns should be removed from the
# dataset beforehand.

if __name__ == "__main__":
    balance = False
    labels = csv_utils.balance_labels_subset(LABEL_PATH, DATA_DIR, balance)  # remove y_data with no correspondence in DATA_DIR and optionally
    x_data, y_data = csv_utils.to_numpy_BI(DATA_DIR, labels)  # turn csv file into numpy dataset while balancing the data based on minority class in dataset
    # x_data = x_data[:, :, 1:12] # 1 - 12 subsets all bands + DOY
    x_set, y_set = numpy_to_tensor(x_data, y_data)  # turn dataset into tensor format
    torch.save(x_set, '/home/j/data/x_set_pxl_bi.pt')
    torch.save(y_set, '/home/j/data/y_set_pxl_bi.pt')