import numpy as np
import torch
from torch import nn, Tensor
import sits_classifier.utils.csv_utils as csv
import os

# # settings
PATH = '/home/j/Nextcloud/my_csv/'
DATA_DIR = os.path.join(PATH, 'multilabel_microdose/')
LABEL_CSV = 'multilabels.my_csv'
LABEL_PATH = os.path.join(PATH, LABEL_CSV)


# LABEL_PATH = '/home/jonathan/data/multilabels.my_csv'
# DATA_DIR = '/home/jonathan/data/multilabel_microdose/'


def numpy_to_tensor(x_data: np.ndarray, y_data: np.ndarray) -> tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    y_data = y_data.reshape(-1)  # This reshapes the y_data numpy array from a 2-dimensional array with shape (n,
    # 1) to a 1-dimensional array with shape (n, ).
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    # standardization:
    sz, seq, num_bands = x_set.size(0), x_set.size(1), x_set.size(2)  # retrieve amount of samples (?) and sequence
    # length from tensor object
    x_set = x_set.view(-1, num_bands)  # use view method to reshape, first arg size of dimension being inferred,
    # arg2 is number of columns in the tensor
    # need to reshape in order to apply batch_norm
    # see Annex 1
    batch_norm = nn.BatchNorm1d(
        num_bands)  # Create a BatchNorm1d layer with `num_bands` as the number of input features.
    x_set: Tensor = batch_norm(
        x_set)  # standardization is used to improve convergence, should lead to values between 0 and 1
    x_set = x_set.view(sz, seq, num_bands).detach()  # sz is the amount of samples, seq is the sequence length,
    # and num_bands is the number of features The `.detach()` method is used here to create a new tensor that is
    # detached from the computation graph. This is done to prevent gradients from flowing backward through this
    # tensor, as it is only used for inference, not for training.
    return x_set, y_set


# the thing about standardization is whether to normalize each band separately or all the values at once. in this
# case, nn.BatchNorm1d(num_bands), each band is normalized individually afaik in case all values are merged there is
# the problem of inclusion of date/DOY values at this point of development. These columns should be removed from the
# dataset beforehand.

if __name__ == "__main__":
    labels = csv.shuffle_labels(LABEL_PATH,
                                DATA_DIR)  # remove y_data with no correspondence in DATA_DIR and optionally balance
    # the data based on minority class in dataset
    x_data, y_data = csv.to_numpy(DATA_DIR, labels)  # turn my_csv file into numpy dataset
    x_data = x_data[:, :, 1:11]
    x_set, y_set = numpy_to_tensor(x_data, y_data)  # turn dataset into tensor format
    torch.save(x_set, '/home/j/data/x_set.pt')
    torch.save(y_set, '/home/j/data/y_set.pt')

label_path = LABEL_PATH
data_dir = DATA_DIR

print("load training data")
labels = csv.load(label_path, None)
x_list = []
y_list = []
for index, row in labels.iterrows():  # i do not understand the use of 2 iterators. for each index and row
    # print(index) # index ends up just being i
    # print(row) # row being the actual labels but not from labels but from data somehow ???
    df_path = os.path.join(data_dir, f'{row[0]}.my_csv')  # retrieve row value to open the correct my_csv
    df = csv.load(df_path, 'OBJECTID', False)  # loading the my_csv, don't know why i need index col
    # and date and a custom load function in the first place when all it does is pd.read_csv
    df = df.iloc[:, 120:]  # remove unwanted columns, such as BST.. geometries, comments etc.
    x = np.array(df).astype(np.float32)
    y = row[:]
    x_list.append(x)
    y_list.append(y)

for i in labels:
    print(i)

for i in labels.iterrows():
    print(i)
    # df_path = os.path.join(data_dir, f'{i[0]}.my_csv')
    # df = my_csv.load(df_path, 'OBJECTID', False)

# i had some problems concatenating x_list so here i do some investigation:
# print(x_list.class)
# i believe the problem is that x has simply too many columns

# turn array list into numpy array
x_data = np.array(x_list)
y_data = np.array(y_list)
print("transfered data to numpy array")
