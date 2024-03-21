import numpy as np
import torch
from torch import nn, Tensor
import sits_classifier.utils.csv_utils as csv_utils
import os
import pandas as pd

# # settings
PATH = '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data'
DATA_DIR = os.path.join(PATH, 'pxl_unbalanced_buffered_reshaped_standardized')
LABEL_PATH = '/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/pxl_buffered_labels_balanced_species.csv'

def numpy_to_tensor(x_data: np.ndarray, y_data: np.ndarray) -> tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    y_data = y_data.reshape(-1)  # This reshapes the y_data numpy array from a 2-dimensional array with shape (n, 1) to a 1-dimensional array with shape (n, ).
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    x_set = s2_cube_np.view(sz, seq,num_bands).detach()  # sz is the amount of samples, seq is the sequence length, and num_bands is the number of features
    # The `.detach()` method is necessary here to create a new tensor that is "detached from the computation graph" as we only want to apply this normalization once
    # .detach prevents gradients from flowing backward through a tensor.
    return x_set, y_set

if __name__ == "__main__":
    labels = pd.read_csv(LABEL_PATH, sep=',', header=0, index_col=False)
    print(len(labels))
    labels.dropna(inplace = True)
    print(len(labels))
    labels = labels.astype(int)
    x_data, y_data = csv_utils.to_numpy_subset(DATA_DIR, labels)  # turn csv file into numpy dataset while balancing the data based on minority class in dataset
    x_data = x_data[:, :, 1:12] # 1 - 12 subsets all bands + DOY
    x_set, y_set = numpy_to_tensor(x_data, y_data)  # turn dataset into tensor format
    torch.save(x_set, '/home/j/data/x_set_buffered_pxl.pt')
    torch.save(y_set, '/home/j/data/y_set_buffered_pxl.pt')

# Annex 1 tensor.view() vs tensor.reshape()
#     view method:
#         The view method returns a new tensor that shares the same data with the original tensor but with a different shape.
#         If the new shape is compatible with the original shape (i.e., the number of elements remains the same), the view method can be used.
#         However, if the new shape is not compatible with the original shape (i.e., the number of elements changes), the view method will raise an error.
#
#     reshape method:
#         The reshape method also returns a new tensor with a different shape, but it may copy the data to a new memory location if necessary.
#         It allows reshaping the tensor even when the number of elements changes, as long as the new shape is compatible with the total number of elements in the tensor.
