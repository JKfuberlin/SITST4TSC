import argparse # for parsing arguments
import csv
from datetime import datetime # for tracking time and benchmarking
import json
import numpy as np
from sits_classifier.models.transformer import TransformerClassifier
import sits_classifier.utils.validation as val
import sits_classifier.utils.plot as plot
from sits_classifier.utils.pytorchtools import EarlyStopping
import sits_classifier.utils.csv_utils as csvutils # my costum functions
import sys
import torch # Pytorch - DL framework
from torch import nn, optim, Tensor
import torch.utils.data as Data
import os # for creating dirs if needed
from captum.attr import ( # explainable AI
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    FeatureAblation,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Saliency,
    visualization as viz,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer
)
sys.path.append('../') # navigating one level up to access all modules

# flags
PARSE = False
GROMIT = True
SEASONDOY = True # Use the seasonal DOY instead if the multi-year DOY
TRAIN = True 
TESTBI = True # test the model on the BI data
PREJITTER = True # apply static noise to the training data to counter spatial correlation
TSAJ = True # Time series augmentation with jitter 
TSARC = True # Time series augmentation with random time series sampling
FOUNDATION = False # Train or apply a foundation model
FINETUNE = False # Finetune using the BI data
EXPLAIN = False # Explain the model

if GROMIT:
    d_model = 512 
    nhead = 8 # avoid AssertionError: embed_dim must be divisible by num_heads
    num_layers = 3
    dim_feedforward = 4096
    BATCH_SIZE = 16
    UID = 999
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration
    PATH = '/home/j/data/'
    MODEL = 'Transformer'
    if SEASONDOY:
        x_set = np.load('/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/x_set_pxl_buffered_balanced_nonstan_species_season.npy')
    else:
        x_set = np.load('/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/x_set_pxl_buffered_balanced_nonstan_species_years.npy')
    y_set = np.load('/media/j/d56fa91a-1ba4-4e5b-b249-8778a9b4e904/data/y_set_pxl_buffered_balanced_species.npy')
    EPOCH = 420
    LR = 0.00001  # learning rate, which in theory could be within the scope of parameter tuning
else:
    UID = 999
    x_set = torch.load('/home/j/data/x_set.pt')
    y_set = torch.load('/home/j/data/y_set.pt')
    EPOCH = 420  # the maximum amount of epochs i want to train
    LR = 0.00001  # learning rate, which in theory could be within the scope of parameter tuning
    PATH = '/home/jonathan/data/'
    MODEL = 'Transformer'

# general hyperparameters
SEED = 420 # a random seed for reproduction, at some point i should try different random seeds to exclude (un)lucky draws
patience = 5 # early stopping patience; how long to wait after last time validation loss improved.
num_bands = 10 # number of different bands from Sentinel 2
num_classes = 10 # the number of different classes that are supposed to be distinguished
sequence_length = 200 # this is the sequence length i want to work with
esdelta = 0.04 # early stopping delta
WINTERSTART = 300 # the start of the winter season as DOY
WINTEREND = 70 # the end of the winter season as DOY


def timestamp():
    now = datetime.now()
    current_time = now.strftime("%D:%H:%M:%S")
    print("Current Time =", current_time)

def setup_seed(seed:int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True # https://darinabal.medium.com/deep-learning-reproducible-results-using-pytorch-42034da5ad7
    torch.backends.cudnn.benchmark = False # not sure if these lines are needed and non-deterministic algorithms would be used otherwise

def split_data(x_set:np.ndarray, y_set:np.ndarray, seed:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into train, validation and test set"""
    size = len(x_set)
    train_size = round(0.75 * size)
    val_size = round(0.15 * size)
    generator = np.random.default_rng(seed)
    indices = np.arange(size)
    generator.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:] # subset everything that is left after train and validation indices
    if PREJITTER: # apply static noise to the training data to counter spatial correlation
        x_set = csvutils.jitter_numpy(x_set, jitter=5)
    train_dataset = (x_set[train_indices], y_set[train_indices])
    val_dataset = (x_set[val_indices], y_set[val_indices])
    test_dataset = (x_set[test_indices], y_set[test_indices])
    return train_dataset, val_dataset, test_dataset

def validate(model:nn.Module) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        good_pred = 0
        total = 0
        losses = []
        for (inputs, labels) in val_loader:
            batch:Tensor = inputs.to(device)# put the data in gpu
            labels:Tensor = labels.to(device)
            outputs:Tensor = model(batch) # prediction
            loss = criterion(outputs, labels)  # calculating loss by comparing to the y_set
            good_pred += val.true_pred_num(labels, outputs)# recording validation accuracy
            total += labels.size(0)
            losses.append(loss.item()) # record validation loss
        acc = good_pred / total  # average train loss and accuracy for one epoch
        val_loss = np.average(losses)
    print('| Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'.format(val_loss, 100 * acc))
    return val_loss, acc

def test(model:nn.Module, testloader, dataset_name:str, MODEL_NAME) -> None:
    """Test best model"""
    test_loader = testloader
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for (i) in test_loader:
            inputs2:Tensor = i[0]
            labels:Tensor = i[1]
            inputs = csvutils.remove_zero_observations(inputs2)
            inputs = inputs.to(device) # put the data in the gpu
            labels = labels.to(device)
            outputs:Tensor = model(inputs)  # prediction
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()
        classes = ['Spruce', 'Sfir', 'Dougl', 'Pine', 'Oak', 'Redoak', 'Beech', 'Sycamore', 'OtherCon', 'OtherDec']
        plot.draw_confusion_matrix(y_true, y_pred, classes, MODEL_NAME, UID, dataset_name, MODEL_NAME)
    return

if __name__ == "__main__":
    model = torch.load("/home/j/data/outputs/models/Transformer_999_512_8_3_4096_16_SEASONDOY_True_PREJITTER_True_TSAJ_True_TSARC_True_foundation_False_finetune_False_.pkl")


    if TESTBI:
        # test model:
        test_x_set = torch.load('/home/j/data/x_set_pxl_bi.pt')
        test_y_set = torch.load('/home/j/data/y_set_pxl_bi.pt')
        #find unique values of test_y_set
        if SEASONDOY: # if the seasonal DOY is used, the test_x_set needs to be reshaped and the last column removed
            test_x_set = test_x_set[:, :, :-1]
        else: # if the multi-year DOY is used, the test_x_set needs to be reshaped and the second to last column removed
            test_x_set = torch.cat((test_x_set[:, :, :9], test_x_set[:, :, 10:]), dim=2)
        # test_x_set = csvutils.remove_zero_observations(test_x_set)
        test_y_set.unique()
        test_dataset = Data.TensorDataset(test_x_set, test_y_set)
        test_loader_BI = Data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
        test(model, test_loader_BI, "BI", "newmodelarch_masked3")



    # # visualize loss and accuracy during training and validation
    # model.load_state_dict(torch.load(MODEL_PATH))
    # plot.draw_curve(train_epoch_loss, val_epoch_loss, 'loss',method='LSTM', model=MODEL_NAME, uid=UID)
    # plot.draw_curve(train_epoch_acc, val_epoch_acc, 'accuracy',method='LSTM', model=MODEL_NAME, uid=UID)
    # timestamp()
    # # test(model)
    # print('plot results successfully')
    # torch.save(model, f'/home/j/home/jonathan/data/outputs/models/{MODEL_NAME}.pkl')
    # f = open(logfile, 'a')
    # f.write("Model ID: " + str(UID) + "; validation accuracy: " + str(best_acc) + '\n')
    # f.close()
            

# Annex 1 tensor.view() vs tensor.reshape()
#     view method:
#         The view method returns a new tensor that shares the same data with the original tensor but with a different shape.
#         If the new shape is compatible with the original shape (i.e., the number of elements remains the same), the view method can be used.
#         However, if the new shape is not compatible with the original shape (i.e., the number of elements changes), the view method will raise an error.
#
#     reshape method:
#         The reshape method also returns a new tensor with a different shape, but it may copy the data to a new memory location if necessary.
#         It allows reshaping the tensor even when the number of elements changes, as long as the new shape is compatible with the total number of elements in the tensor.
