import pandas as pd
import numpy as np
from typing import Tuple, List
import os
import torch
from torch import Tensor
import torch.utils.data as Data


def balance_labels(label_path:str): # used by ??? i think i do this in R now
    labels = custom_load(label_path, 'ID')
    label_counts = labels["encoded"].value_counts() # find out the least common class in the labels and count the occurrences of each label
    minority_label = label_counts.idxmin() # Get the label with the least occurrences
    minority_count = label_counts[minority_label]     # Get the number of occurrences of the minority label
    dfs = []
    for label in label_counts.index:
        label_df = labels[labels["encoded"] == label]
        if len(label_df) > minority_count:
            label_df = label_df.sample(minority_count, random_state=42)
        dfs.append(label_df)
    balanced_df = pd.concat(dfs) # Concatenate the dataframes
    balanced_df = balanced_df.sample(frac=1, random_state=42) # Shuffle the dataframe
    return balanced_df

def balance_labels_subset(label_path:str, data_dir:str, balance:bool): # used by ??? i think i do this in R now
    file_names = subset_filenames(data_dir)
    labels = pd.read_csv(label_path, sep=',', header=0, index_col=False) # this loads all labels from the csv file
    try:
        labels = labels.drop("Unnamed: 0", axis=1) # just tidying up, removing an unnecessary column
        labels = labels.drop("X", axis=1)  # just tidying up, removing an unnecessary column
    except:
        print(labels)
    labels_subset = labels[labels['ID'].isin(file_names)] # drops all entries from the labels that do not have a corresponding csv file on the drive / in the subset
    label_counts = labels_subset["encoded"].value_counts() # find out least common class in labels and count the occurrences of each label for balancing
    minority_label = label_counts.idxmin()  # Get the label with the least occurrences
    minority_count = label_counts[minority_label] # Get the number of occurrences of the minority label
    labels = labels_subset # just resetting the variable name
    if balance == True:
        dfs = [] # empty list
        for label in label_counts.index:
            label_df = labels[labels["encoded"] == label]
            if len(label_df) > minority_count:
                label_df = label_df.sample(minority_count, random_state=42)
            dfs.append(label_df)
        # Concatenate the dataframes
        balanced_df = pd.concat(dfs)
    else:
        balanced_df = labels
    # Shuffle the dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42)
    return balanced_df

def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int) -> tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader, Tensor]:
    """Build and split dataset, and generate dataloader for training and validation"""
    # automatically split dataset
    dataset = Data.TensorDataset(x_set, y_set) #  'wrapping' tensors: Each sample will be retrieved by indexing tensors along the first dimension.
    # gives me an object containing tuples of tensors of x_set and the labels
    #  x_set: [204, 305, 11] number of files, sequence length, number of bands
    size = len(dataset)
    train_size = round(0.75 * size)
    val_size = round(0.15 * size)
    test_size = size - train_size - val_size
    #train_size, val_size, test_size = round(0.75 * size), round(0.15 * size), round(0.10 * size)
    generator = torch.Generator() # this is for random sampling
    train_dataset, val_dataset, test_dataset = Data.random_split(dataset, [train_size, val_size, test_size], generator) # split the data in train and validation
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False) # Create PyTorch data loaders from the datasets
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    # num_workers is for parallelizing this function, however i need to set it to 1 on the HPC
    # shuffle is True so data will be shuffled in every epoch, this probably is activated to decrease overfitting
    # drop_last = False makes sure, the entirety of the dataset is used even if the remainder of the last samples is fewer than batch_size
    '''
    The DataLoader object now contains n batches of [batch_size, seq_len, num_bands] and can be used for iteration in train()
    '''
    return train_loader, val_loader, test_loader

def custom_load(file_path:str, index_col:str, date:bool=False) -> pd.DataFrame:
    """Load csv file to pandas.Dataframe"""
    if date:
        df = pd.read_csv(file_path, sep=',', header=0, parse_dates = ['date'], index_col=False)
        df.dropna(axis=0, how='any', inplace=True)  # delete date when no available data
    else:
        df = pd.read_csv(file_path, sep=',', header=0, index_col=False)
    return df

def delete(file_path:str) -> None:
    os.remove(file_path)
    print(f'delete file {file_path}')

def export(df:pd.DataFrame, file_path:str, index:bool=True) -> None:
    """Export pandas.Dataframe to csv file"""
    df.to_csv(file_path, index=index)
    print(f'export file {file_path}')

def jitter_numpy(df:np.ndarray, jitter:int=5) -> np.ndarray:
    for sample in range(df.shape[0]):  # Loop through each sample
        max_val = int(np.amax(df[sample, :, 10]))  # Find maximum value in the DOY column
        for i in range(df.shape[2]):  # Loop through each column
            if i == 10:  # Treat DOY column differently
                if max_val >= 100 & max_val < 367: # If max value is less than 367, it means that it is seasonal DOY
                    original_zeros = (df[sample,:, i] == 0)  # Mask for values that are zero before jitter
                    df[sample,:, i] += np.random.randint(-jitter, jitter, (df.shape[1])) # Apply jitter, randint because DOY is an integer
                    df[sample,:, i] = np.clip(df[sample,:, i], None, 366)  # Clip values to maximum of 366
                    df[sample,:, i] = np.where(df[sample,:, i] < 1, 1, df[sample,:, i])  # Ensure values don't go below 1
                    df[sample,:, i] = np.where(original_zeros, 0, df[sample,:, i])  # Set values to 0 if they were originally zero
                else: # it is additive DOY
                    original_zeros = (df[sample,:, i] == 0)  # Mask for values that are zero before jitter
                    df[sample,:, i] += np.random.randint(-jitter, jitter, (df.shape[1]))
                    df[sample,:, i] = np.where(original_zeros, 0, df[:, i])  # Set values to 0 if they were originally zero
            else:
                df[sample,:, i] += np.random.uniform(-jitter, jitter, (df.shape[1])) # Apply jitter to non-DOY columns of sample
        return df

def jitter_tensor(device, batch:Tensor, spectral_jitter:float=0.1, DOY_jitter:int=5) -> Tensor:
    """Add jitter to the tensor after chunking it in batches"""
    batch = batch.to(device) # Move batch to device if not already there, redundant
    # Extract columns 1-10 and column 11
    spectral = batch[..., :10]  # Select all elements along all dimensions up to the 11th dimension
    DOY = batch[..., 10:11]  # Select only the 11th column
    # Add jitter to columns 1-10 and column 11 separately
    jitter_spectral = torch.FloatTensor(spectral.size()).uniform_(-spectral_jitter, spectral_jitter).to(device)
    spectral += jitter_spectral
    original_zeros = (DOY == 0)  # Mask for values that are zero before jitter
    jitter_DOY = torch.randint(low= -DOY_jitter, high=DOY_jitter+1, size=DOY.size()).to(device) # DOY_jitter+1 because randint is exclusive on the upper bound
    DOY += jitter_DOY # Concatenate the two parts back together
    DOY[original_zeros] = 0 # Set values to 0 if they were originally zero
    batch = torch.cat((spectral, DOY), dim=-1)
    return batch

def list_to_dataframe(lst:List[List[float]], cols:List[str], decimal:bool=True) -> pd.DataFrame:
    """Transfer list to pd.DataFrame"""
    df = pd.DataFrame(lst, columns=cols)
    if decimal:
        df = df.round(2)
    else:
        df = df.astype('int')
    return df

def numpy_to_tensor(x_data: np.ndarray, y_data: np.ndarray) -> tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    y_data = y_data.reshape(-1)  # This reshapes the y_data numpy array from a 2-dimensional array with shape (n, 1) to a 1-dimensional array with shape (n, ).
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    sz, seq, num_bands = x_set.size(0), x_set.size(1), x_set.size(2) # retrieve size, sequence length and number of bands
    x_set = x_set.view(sz, seq,num_bands) # sz is the amount of samples, seq is the sequence length, and num_bands is the number of features
    return x_set, y_set

def numpy_to_tensor2(x_data: np.ndarray, y_data: np.ndarray) -> tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    y_data = y_data.reshape(-1)  # This reshapes the y_data numpy array from a 2-dimensional array with shape (n, 1) to a 1-dimensional array with shape (n, ).
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    sz, seq, num_bands = x_set.size(0), x_set.size(1), x_set.size(2) # retrieve size, sequence length and number of bands
    x_set = x_set.view(sz, seq,num_bands) # sz is the amount of samples, seq is the sequence length, and num_bands is the number of features
    return x_set, y_set

def random_sample_pandas(df:pd.DataFrame, sample_size:int=200, winter_start:int=300, winter_end:int=70) -> pd.DataFrame:
    """First drop observations from deep winter, then randomly sample the dataframe"""
    winter_obs = df[(df['DOY2'] > winter_start) | (df['DOY2'] < winter_end)] # find the winter observations with DOY2 > 300 or DOY2 < 90
    n = len(df) # find out the number of rows in the dataframe
    diff = n - len(winter_obs) # find the difference between the number of rows in the dataframe and the amount of winter observations
    if diff > sample_size: # if the difference is greater than the sample size, drop all winter observations
        df = df.drop(winter_obs.index)
    if diff < sample_size: # if the difference is smaller than the sample size, sample the winter observations and drop them from the dataframe
        to_remove = sample_size - diff # calculate the number of winter observations to remove to achieve the desired sample size
        winter_obs = winter_obs.sample(to_remove, random_state=42) # sample the winter observations
        df = df.drop(winter_obs.index) # drop the winter observations from the dataframe
    return df.sample(sample_size) # return a random sample of the desired size

def random_sample_tensor_additive(batch:Tensor, labels:Tensor, sample_size:int=200, winter_start:int=300, winter_end:int=70) -> tuple[Tensor, Tensor]:
    return batch, labels
    
def random_sample_tensor_seasonal(batch:Tensor, sample_size:int=200, winter_start:int=300, winter_end:int=70) -> tuple[Tensor, Tensor]:
    """First drop observations from deep winter, then randomly sample the tensor"""
    """Apply to both, batch and labels"""
    """Lots of redundance, because i always expect batches to be uniform and have the same sequence length for every observation but this way it is safer""" 
    ls = [] # Create an empty list to be my final batch
    for i in range(batch.shape[0]):
        padded_obs = (batch[i,:,10] == 0) # find the observations that are padded with 0s
        todelete = batch.shape[1]-sample_size # calculate the number of samples to delete
        if padded_obs.sum().item() <= todelete: # delete all padded observations if possible
            item = batch[i, ~padded_obs, :]
        else: # in case there are more padded observations than being able to delete
            item = batch[i, :, :]
            indices = torch.where(padded_obs)[0] # find the indices of the padded observations
            indices = indices[torch.randperm(indices.size(0))] # shuffle the indices
            indices = indices[:todelete] # select the indices to drop
            mask = torch.ones(item.shape[0], dtype=bool).cuda()  # Create a mask of True values
            mask[indices] = False  # Set the indices you want to remove to False
            item = item[mask, :]  # Apply the mask to item
        if item.shape[0] > sample_size: # now drop winter observations and randoms until target is achieved
            to_remove = item.shape[0] - sample_size  # calculate the number of winter observations to remove to achieve the desired sample size
            winter_obs = (item[:, -1] > winter_start) | (item[:, -1] < winter_end) # find the winter observations with DOY2 > 300 or DOY2 < 90
            winter_obs = torch.where(winter_obs)[0] # find the indices of the winter observations
            winter_obs = winter_obs[:to_remove] # select the indices to remove
            mask = torch.ones(len(item), dtype=bool).cuda()
            mask[winter_obs] = False
            item = item[mask, :]
            # item = item[~winter_obs,:]
        if item.shape[0] > sample_size: # now drop random observations until target is achieved
            indices = torch.randperm(item.size(0)) # select 200 random indices
            item =  item[indices[:sample_size],:]
        ls.append(item)
        item = None
    new_batch = torch.stack(ls)
    return new_batch

def subset_filenames(data_dir:str):
    # i want to find out which csv files really are existent in my subset/on my drive and only select the matching labels
    import glob
    # Define the pattern to match the CSV files
    file_pattern = data_dir + '/*.csv'
    # Retrieve the filenames that match the pattern
    csv_files = glob.glob(file_pattern)
    # Extract the filenames without the extension
    file_names = [file.split('/')[-1].split('.')[0] for file in csv_files]
    file_names = [int(x) for x in file_names]
    return file_names

def to_numpy(data_dir:str, labels) -> Tuple[np.ndarray, np.ndarray]:
    """Load label and time series data, transfer them to numpy array"""
    """apply end of sequence zero padding"""
    print("load training data")
    labels = labels # TODO i deleted the loading based on colname ID, make sure it works
    # Step 1: find max time steps
    max_len = 0
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{index}.csv') # TODO fix messed up file names
        df = custom_load(df_path, 'date', True)
        max_len = max(max_len, df.shape[0])
    print(f'max sequence length: {max_len}')
    # Step 2: transfer to numpy array
    x_list = []
    y_list = []
    for index, row in labels.iterrows():
        df_path = os.path.join(data_dir, f'{index}.csv') # TODO: fix csv names
        df = custom_load(df_path, 'date', True)
        df = df.drop('date', axis=1) # i decided to drop the date again because i cannot convert it to float32 and i still have DOY for identification
        x = np.array(df).astype(np.float32)
        # use 0 padding make sequence length equal
        padding = np.zeros((max_len - x.shape[0], x.shape[1]))
        x = np.concatenate((x, padding), dtype=np.float32)
        y = row['encoded']
        x_list.append(x)
        y_list.append(y)
    # concatenate array list
    x_data = np.array(x_list)
    y_data = np.array(y_list)
    print("transferred data to numpy array")
    return x_data, y_data

def to_numpy_subset(data_dir:str, labels, SPECIES) -> Tuple[np.ndarray, np.ndarray]:
    """Load label and time series data, transfer them to numpy array"""
    # Step 1: find maximum sequence length of observations
    max_len = 0
    for id in labels['ID']:
        df_path = os.path.join(data_dir, f'{id}.csv')
        df = pd.read_csv(df_path, sep=',', header=0)
        max_len = max(max_len, df.shape[0])
    print(f'max sequence length: {max_len}')
    # Step 2: transfer to numpy array
    x_list = []
    y_list = []
    for tuple in labels.iterrows():
        info = tuple[1] # access the first element of the tuple, which is a <class 'pandas.core.series.Series'>
        ID = info['ID'] # the true value for the ID after NA removal and some messing up is here, this value identifies the csv
        df_path = os.path.join(data_dir, f'{ID}.csv')
        df = pd.read_csv(df_path, sep=',', header=0)
        x = np.array(df).astype(np.float32) # create a new numpy array from the loaded csv file containing spectral values with the dataype float32
        # use 0 padding make sequence length equal
        padding = np.zeros((max_len - x.shape[0], x.shape[1]))
        x = np.concatenate((x, padding), dtype=np.float32) # the 0s are appended to the end, will need to change this in the future to fill in missing observations
        if SPECIES == True:
            y = info['encoded'] # this is the label for the Tree species
        else:
            y = info['evergreen'] # this is the label for wether the observation is evergreen or not
        x_list.append(x)
        y_list.append(y)
    # concatenate array list
    x_data = np.array(x_list)
    y_data = np.array(y_list)
    print("transferred data to numpy array")
    return x_data, y_data

def to_numpy_BI(data_dir:str, labels) -> Tuple[np.ndarray, np.ndarray]:
    """Load label and time series data, transfer them to numpy array"""
    labels = labels
    max_len = 0 # Step 1: find max time steps
    for id in labels['ID']:
        id = int(id)
        df_path = os.path.join(data_dir, f'{id}.csv')
        df = custom_load(df_path, True)
        max_len = max(max_len, df.shape[0])
    print(f'max sequence length: {max_len}')
    # Step 2: transfer to numpy array
    x_list = []
    y_list = []
    for _,row in labels.iterrows():
        ID = int(row.iloc[0])
        # info = tuple[1] # access the first element of the tuple, which is a <class 'pandas.core.series.Series'>
        # ID = int(info[0]) # the true value for the ID after NA removal and some messing up is here, this value identifies the csv
        df_path = os.path.join(data_dir, f'{ID}.csv')
        df = custom_load(df_path, False)
        if 'date' in df.columns:
            df.drop(columns=['date'], inplace=True)
        x = np.array(df).astype(np.float32) # create a new numpy array from the loaded csv file containing spectral values with the dataype float32
        padding = np.zeros((max_len - x.shape[0], x.shape[1]))# use 0 padding make sequence length equal
        x = np.concatenate((x, padding), dtype=np.float32) # the 0s are appended to the end, will need to change this in the future to fill in missing observations
        y = int(row.iloc[2])# this is the label
        x_list.append(x)
        y_list.append(y)
    # concatenate array list
    x_data = np.array(x_list)
    y_data = np.array(y_list)
    print("transferred data to numpy array")
    return x_data, y_data


# oldymcgeezax

# def jitter_pandas(df:np.ndarray, jitter:int=5) -> np.ndarray: # used by main script for static jitter as well as batchwise in training
#     """Add jitter to the training data dataframe after splitting"""
#     """"I expect a df shape of [num_observations, seq_len, num_bands]"""
#     n = df.shape[2]
#     for i in range(n): # apply jitter to spectral values
#         df[:, i] += np.random.uniform(-jitter, jitter, (df.shape[0], 1))
#         df[:, i] = np.where(df[:, i] < 1, 1, df[:, i]) # make sure the values don't go below 1
#         if i == 11: # treat DOY column differently
#             max_val = np.amax(df[:, i]) # find the maximum value in the column
#             if max_val < 367: # find out if the max val is lower than 367, meaning that it is seasonal DOY
#                 df[:, i] = df[:, i].apply(lambda x: max(x, 366)) 
#     return df
