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
TSAJ = False # Time series augmentation with jitter , this is redundant and does the same as PREJITTER
TSARS = True # Time series augmentation with random time series sampling
FOUNDATION = False # Train or apply a foundation model
FINETUNE = False # Finetune using the BI data
EXPLAIN = False # Explain the model

if PARSE:
    parser = argparse.ArgumentParser(description='trains the Transformer with given parameters')
    parser.add_argument('UID', type=int, help='the unique ID of this particular model')
    parser.add_argument('GPU_NUM', type=int, help='which GPU to use, necessary for parallelization')
    parser.add_argument('d_model', type=int, help='d_model')
    parser.add_argument('nhead', type=int, help='number of transformer heads')
    parser.add_argument('num_layers', type=int, help='number of layers')
    parser.add_argument('dim_feedforward', type=int, help='')
    parser.add_argument('batch_size', type=int, help='batch size')
    args = parser.parse_args()
    # hyperparameters for LSTM and argparse
    device = torch.device('cuda:'+str(args.GPU_NUM) if torch.cuda.is_available() else 'cpu') # Device configuration
    d_model = args.d_model  # larger
    nhead = args.nhead  # larger
    num_layers = args.num_layers  # larger
    dim_feedforward = args.dim_feedforward
    BATCH_SIZE = args.batch_size
    UID = str(args.UID)
    print(f"UID = {UID}")
else:
    d_model = 512 
    nhead = 8 # avoid AssertionError: embed_dim must be divisible by num_heads
    num_layers = 3
    dim_feedforward = 4096
    BATCH_SIZE = 16
    UID = 999
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration

MODEL_NAME = 'Transformer' + '_' + str(UID)+'_' + str(d_model)+'_' + str(nhead)+'_' + str(num_layers)+'_' + str(dim_feedforward)+'_' + str(BATCH_SIZE)+'_SEASONDOY_' + str(SEASONDOY) + '_PREJITTER_' + str(PREJITTER)+'_TSAJ_' + str(TSAJ)+'_TSARC_' + str(TSARS)+'_foundation_' + str(FOUNDATION)+'_finetune_' + str(FINETUNE) + '_'
MODEL_PATH = '/home/j/data/outputs/models/' + MODEL_NAME    

if GROMIT:
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
    UID = 998
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
esdelta = 0.02 # early stopping delta
WINTERSTART = 300 # the start of the winter season as DOY
WINTEREND = 70 # the end of the winter season as DOY
    

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
    out_path = f'../../outputs/models/{MODEL_NAME}_params.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        data = json.dumps(params, indent=4)
        f.write(data)
    print('saved hyperparameters')

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
        # find out standard deviation of the x_set
        stdev = np.std(x_set)
        jittervalue = 0.1
        x_set = csvutils.jitter_numpy(x_set, spectraljitter=jittervalue, DOYjitter=5)
    train_dataset = (x_set[train_indices], y_set[train_indices])
    val_dataset = (x_set[val_indices], y_set[val_indices])
    test_dataset = (x_set[test_indices], y_set[test_indices])
    return train_dataset, val_dataset, test_dataset

def train2(model:nn.Module, train_xset:Tensor, train_yset:Tensor, batch_size:int, epoch:int) -> tuple[float, float]:
    model.train()  # sets model into training mode
    good_pred = 0 # initialize variables for accuracy and loss metrics
    total = 0
    losses = []
    train_loader = Data.DataLoader(Data.TensorDataset(train_xset, train_yset), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False) # create a dataloader for the training data
    for (batch, labels) in train_loader: # unclear whether i need to use enumerate(train_loader) or not
        labels = labels.to(device) # tensor [batch_size,] e.g. 32 labels in a tensor
        batch = batch.to(device) # pass the data into the gpu [32, 305, 11] batch_size, sequence max length, num_bands
        if TSAJ:
            batch = csvutils.jitter_tensor(device, batch, spectral_jitter=0.1, DOY_jitter=5) # apply jitter to the input tensor spectral values and DOY
        if TSARS and SEASONDOY:
            batch = csvutils.random_sample_tensor_seasonal(batch, device=device) # apply random time series sampling to the input tensor
        if TSARS and not SEASONDOY:
            batch = csvutils.random_sample_tensor_additive(batch)
        outputs = model(batch)  # applying the model
        # at this point inputs is 305,3,11. 305 [timesteps, batch_size, num_bands]
        loss = criterion(outputs, labels)  # calculating loss by comparing to the y_set
        # recording training accuracy
        predicted_labels = torch.argmax(outputs,dim=1)  # outputs is a vector containing the probabilities for each class so we need to find the corresponding class
        good_pred += torch.sum(predicted_labels == labels).item()  # recording correct predictions
        total += labels.size(0)
        losses.append(loss.item())  # recording training loss
        optimizer.zero_grad() # backward and optimize
        loss.backward()
        optimizer.step()
    acc = good_pred / total
    train_loss = np.average(losses)
    timestamp()
    print('Epoch[{}/{}] | Train Loss: {:.4f} | Train Accuracy: {:.2f}% '.format(epoch + 1, EPOCH, train_loss, acc * 100), end="")
    return train_loss, acc

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
            inputs:Tensor = i[0]
            labels:Tensor = i[1]
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

    train_dataset, val_dataset, test_dataset = split_data(x_set, y_set, SEED) # split the data into train, validation and test set, each is a tensor of two tensors (x and y)
    train_xset, train_yset = csvutils.numpy_to_tensor(train_dataset[0], train_dataset[1]) # convert numpy arrays to tensors
    # split them into x and y because this is the input to the dataloader whithin train()
    val_xset, val_yset = csvutils.numpy_to_tensor(val_dataset[0], val_dataset[1]) # convert numpy arrays to tensors
    test_xset, test_yset = csvutils.numpy_to_tensor(test_dataset[0], test_dataset[1]) # convert numpy arrays to tensors
    val_loader = Data.DataLoader(Data.TensorDataset(val_xset, val_yset), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False) # create a dataloader for the training data
    test_loader = Data.DataLoader(Data.TensorDataset(test_xset, test_yset), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False) # create a dataloader for the training data
    if TRAIN:
        setup_seed(SEED)  # set random seed to ensure reproducibility
        print(device)
        timestamp()
        model = TransformerClassifier(num_bands, num_classes, d_model, nhead, num_layers, dim_feedforward, sequence_length).to(device)
        save_hyperparameters()
        criterion = nn.CrossEntropyLoss().to(device) # define the calculation of loss during training and validation
        optimizer = optim.Adam(model.parameters(), LR) # define how to optimize the model during backpropagation
        softmax = nn.Softmax(dim=1).to(device)
        # evaluate terms
        train_epoch_loss = []
        val_epoch_loss = []
        train_epoch_acc = [0]
        val_epoch_acc = [0]
        # start training and validation of the model
        print("start training")
        timestamp()
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, delta= esdelta, verbose=False)
        logdir = '/home/j/data/prof'
        prof = None
        for epoch in range(EPOCH):
            train_loss, train_acc = train2(model, train_xset, train_yset, BATCH_SIZE, epoch)
            val_loss, val_acc = validate(model)
            if val_acc > max(val_epoch_acc):
                torch.save(model.state_dict(), MODEL_PATH)
                best_acc = val_acc
                best_epoch = epoch
            # record loss and accuracy
            train_epoch_loss.append(train_loss)
            train_epoch_acc.append(train_acc)
            val_epoch_loss.append(val_loss)
            val_epoch_acc.append(val_acc)
            plot.draw_curve(train_epoch_loss, val_epoch_loss, 'loss',method='LSTM', model=MODEL_NAME, uid=UID)
            plot.draw_curve(train_epoch_acc, val_epoch_acc, 'accuracy',method='LSTM', model=MODEL_NAME, uid=UID)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping in epoch " + str(epoch))
                print("Best model in epoch :" + str(best_epoch))
                print("Model ID: " + str(UID) + "; validation loss: " + str(best_acc))
                break
            if epoch == 25 and val_loss < 0.5: # stop in case model is BS early on to save GPU time
                print("something is wrong. check learning rate. Aborting")
                break
            if epoch % 20 == 0:
                print(epoch, '/n', val_acc)
        torch.save(model, f'/home/j/data/outputs/models/{MODEL_NAME}.pkl')
        test(model, test_loader, "FE", MODEL_NAME)    

    if TESTBI:
        # test model:
        test_x_set = torch.load('/home/j/data/x_set_pxl_bi.pt')
        test_y_set = torch.load('/home/j/data/y_set_pxl_bi.pt')
        #find unique values of test_y_set
        if SEASONDOY: # if the seasonal DOY is used, the test_x_set needs to be reshaped and the last column removed
            test_x_set = test_x_set[:, :, :-1]
        else: # if the multi-year DOY is used, the test_x_set needs to be reshaped and the second to last column removed
            test_x_set = torch.cat((test_x_set[:, :, :9], test_x_set[:, :, 10:]), dim=2)
        test_y_set.unique()
        test_dataset = Data.TensorDataset(test_x_set, test_y_set)
        test_loader_BI = Data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
        test(model, test_loader_BI, "BI", MODEL_NAME)

    if EXPLAIN:
        # explain model
        model.zero_grad()
        for iter in range(len(test_dataset)):
            data = test_dataset[iter]
            feature_mask = np.ones(shape=[422, 10])
            for npiter in range(feature_mask.shape[1]):
                feature_mask[:, npiter] = feature_mask[:, npiter] * npiter
                ### convert to pytorch tensor
            feature_mask = torch.tensor(feature_mask).long()

            ### initialize Feature Ablation algorithm
            ablator = FeatureAblation(model)
            attribution = ablator.attribute(
                inputs=data["bert_input"].float().unsqueeze(axis=0).to(device),
                baselines=None,
                target=None,
                additional_forward_args=(
                    data["bert_mask"].long().unsqueeze(axis=0).to(device),
                    data["time"].long().unsqueeze(axis=0).to(device)),
                feature_mask=feature_mask.unsqueeze(axis=0).to(device),
                perturbations_per_eval=NUM_WORKERS,
                show_progress=False
            )

            attribution = attribution.squeeze()
            attribution = pd.DataFrame(attribution.detach().cpu().numpy())

            ### column names:
            df_cols = [os.path.join(band) for band in bands]
            attribution.columns = df_cols

            ### only first row is relevant, all other rows are duplicates
            attribution = attribution.head(1)
            # attribution.shape

            if not os.path.exists(os.path.join(INPUT_DATA_PATH, 'model', 'attr_' + MODEL_NAME, 'feature_ablation')):
                os.mkdir(os.path.join(INPUT_DATA_PATH, 'model', 'attr_' + MODEL_NAME, 'feature_ablation'))

            ### save dataframe to disk
            attribution.to_csv(os.path.join(MODEL_SAVE_PATH, 'attr_' + MODEL_NAME, 'feature_ablation',
                                            str(testdat["plotID"].iloc[iter]) + '_attr_label_' +
                                            str(testdat["test_label"].iloc[iter]) + '_pred_' +
                                            str(testdat["prediction"].iloc[iter]) +
                                            str(np.where(
                                                (testdat["mort_1"].iloc[iter] > 0) & (testdat["prediction"].iloc[iter] == 1)
                                                or (testdat["mort_1"].iloc[iter] == 0) & (
                                                            testdat["prediction"].iloc[iter] == 0),
                                                '_correct', '_false')) +
                                            '_extent_' + str(int(testdat["mort_1"].iloc[iter] * 100)) +
                                            '_featabl.csv'),
                            sep=';', index=False)

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
