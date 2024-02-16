import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.utils.data as Data
from typing import Tuple
import os
import json
import sits_classifier.utils.csv as csv
from sits_classifier.models.transformer import TransformerClassifier
import sits_classifier.utils.validation as val
import sits_classifier.utils.plot as plot



# file path
PATH='/mnt/extra/my_volume/'
DATA_DIR = os.path.join(PATH, 'csv_reshape')
LABEL_CSV = 'labels_unbalanced.csv'
METHOD = 'classification'
MODEL = 'transformer'
UID = '5pure'
MODEL_NAME = MODEL + '_' + UID
LABEL_PATH = os.path.join(PATH, LABEL_CSV)
MODEL_PATH = f'../outputs/models/{METHOD}/{MODEL_NAME}.pth'

# general hyperparameters
BATCH_SIZE = 128
LR = 0.001
EPOCH = 20
SEED = 24

# hyperparameters for Transformer model
num_bands = 10
num_classes = 10
d_model = 128
nhead = 8
num_layers = 2
dim_feedforward = 512



def save_hyperparameters() -> None:
    """Save hyperparameters into a json file"""
    params = {
        'general hyperparameters': {
            'batch size': BATCH_SIZE,
            'learning rate': LR,
            'epoch': EPOCH,
            'seed': SEED
        },
        f'{MODEL} hyperparameters': {
            'number of bands': num_bands,
            'embedding size': d_model,
            'number of heads': nhead,
            'number of layers': num_layers,
            'feedforward dimension': dim_feedforward,
            'number of classes': num_classes
        }
    }
    out_path = f'../outputs/models/{METHOD}/{MODEL_NAME}_params.json'
    with open(out_path, 'w') as f:
        data = json.dumps(params, indent=4)
        f.write(data)
    print('saved hyperparameters')


def setup_seed(seed:int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def numpy_to_tensor(x_data:np.ndarray, y_data:np.ndarray) -> Tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    # reduce dimention from (n, 1) to (n, )
    y_data = y_data.reshape(-1)
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)
    return x_set, y_set


def build_dataloader(x_set:Tensor, y_set:Tensor, batch_size:int) -> Tuple[Data.DataLoader, Data.DataLoader]:
    """Build and split dataset, and generate dataloader for training and validation"""
    # automatically split dataset
    dataset = Data.TensorDataset(x_set, y_set) # what does this do? 'wrapping' tensors
    size = len(dataset)
    train_size, val_size = round(0.8 * size), round(0.2 * size)
    generator = torch.Generator() # this is for random sampling
    train_dataset, val_dataset = Data.random_split(dataset, [train_size, val_size], generator)

    # Create PyTorch data loaders from the datasets
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # num_workers is for parallelizing this function
    # shuffle is True so data will be shuffled in every epoch, this probably is activated to decrease overfitting
    # make sure, this does not mess up the proportions of labels
    return train_loader, val_loader


def train(model:nn.Module, epoch:int) -> Tuple[float, float]:
    model.train()
    good_pred = 0
    total = 0
    losses = []
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs[:, :, 1:11]  # this excludes DOY and date
        # exchange dimension 0 and 1 of inputs depending on batch_first or not
        inputs:Tensor = inputs.transpose(0, 1)
        # put the data in gpu
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # recording training accuracy
        good_pred += val.true_pred_num(labels, outputs)
        total += labels.size(0)
        # record training loss
        losses.append(loss.item())
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # average train loss and accuracy for one epoch
    acc = good_pred / total
    train_loss = np.average(losses)
    print('Epoch[{}/{}] | Train Loss: {:.4f} | Train Accuracy: {:.2f}% '
        .format(epoch+1, EPOCH, train_loss, acc * 100), end="")
    return train_loss, acc


def validate(model:nn.Module) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        good_pred = 0
        total = 0
        losses = []
        for (inputs, labels) in val_loader:
            inputs = inputs[:, :, 1:11]  # this excludes DOY and date
            # exchange dimension 0 and 1 of inputs depending on batch_first or not
            inputs:Tensor = inputs.transpose(0, 1)
            # put the data in gpu
            inputs:Tensor = inputs.to(device)
            labels:Tensor = labels.to(device)
            # prediction
            outputs:Tensor = model(inputs)
            loss = criterion(outputs, labels)
            # recording validation accuracy
            good_pred += val.true_pred_num(labels, outputs)
            total += labels.size(0)
            # record validation loss
            losses.append(loss.item())
        # average train loss and accuracy for one epoch
        acc = good_pred / total
        val_loss = np.average(losses)
    print('| Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'
        .format(val_loss, 100 * acc))
    return val_loss, acc


def test(model:nn.Module) -> None:
    """Test best model"""
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for (inputs, labels) in val_loader:
            # exchange dimension 0 and 1 of inputs depending on batch_first or not
            inputs:Tensor = inputs.transpose(0, 1)
            inputs = inputs.to(device)
            labels:Tensor = labels.to(device)
            outputs:Tensor = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()
        # *************************change class here*************************
        classes = ['Spruce', 'Beech', 'Pine', 'Douglas fir', 'Oak']
        # *******************************************************************
        plot.draw_confusion_matrix(y_true, y_pred, classes, MODEL_NAME)

def numpy_to_tensor(x_data:np.ndarray, y_data:np.ndarray) -> Tuple[Tensor, Tensor]:
    """Transfer numpy.ndarray to torch.tensor, and necessary pre-processing like embedding or reshape"""
    # reduce dimension from (n, 1) to (n, )
    y_data = y_data.reshape(-1)
    x_set = torch.from_numpy(x_data)
    y_set = torch.from_numpy(y_data)

    # standardization
    # sz, seq = x_set.size(0), x_set.size(1)
    # x_set = x_set.view(-1, num_bands)
    # batch_norm = nn.BatchNorm1d(num_bands)
    # x_set: Tensor = batch_norm(x_set)
    # x_set = x_set.view(sz, seq, num_bands).detach()

    return x_set, y_set

if __name__ == "__main__":
    # set random seed
    setup_seed(SEED)
   # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # dataset
    labels = csv.balance_labels(LABEL_PATH)
    x_data, y_data = csv.to_numpy(DATA_DIR, labels)
    x_set, y_set = numpy_to_tensor(x_data, y_data)
    train_loader, val_loader = build_dataloader(x_set, y_set, BATCH_SIZE)
    # model
    model = TransformerClassifier(num_bands, num_classes, d_model, nhead, num_layers, dim_feedforward).to(device)
    # save_hyperparameters()
    # loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), LR)
    # evaluate terms
    train_epoch_loss = []
    val_epoch_loss = []
    train_epoch_acc = [0]
    val_epoch_acc = [0]
    # train and validate model
    print("start training")
    for epoch in range(EPOCH):
        train_loss, train_acc = train(model, epoch)
        val_loss, val_acc = validate(model)
        if val_acc > min(val_epoch_acc):
            torch.save(model.state_dict(), MODEL_PATH)
        # record loss and accuracy
        train_epoch_loss.append(train_loss)
        train_epoch_acc.append(train_acc)
        val_epoch_loss.append(val_loss)
        val_epoch_acc.append(val_acc)
    # visualize loss and accuracy during training and validation
    plot.draw_curve(train_epoch_loss, val_epoch_loss, 'loss', METHOD, MODEL_NAME)
    plot.draw_curve(train_epoch_acc, val_epoch_acc, 'accuracy', METHOD, MODEL_NAME)
    # test best model
    print('start testing')
    model.load_state_dict(torch.load(MODEL_PATH))
    test(model)
    print('plot result successfully')