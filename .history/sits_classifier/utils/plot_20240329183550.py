from matplotlib import pyplot as plt 
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, r2_score
import pandas as pd
from typing import List
import os

def draw_curve(y_train:list[float], y_val:list[float], name:str, method:str, model:str, uid:str) -> None:
    """
    Visualise the change of loss or accuracy over epochs
    @params:
    y_train: average values from training for each epoch
    y_val: average values from validation for each epoch
    name: Loss or Accuracy
    model: LSTM or Tranformer
    """
    assert len(y_val) == len(y_train), "y_train and y_val must have the same length"
    epoch = len(y_train)
    x = [i for i in range(0, epoch)]
    # plot 2 curves
    plt.plot(x, y_train, color='b', label='train ' + name)
    plt.plot(x, y_val, color='y', label="validation " + name)
    # set label
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # set title and legend
    title = f'{model}_{name}_{uid}'
    plt.title(title)
    plt.legend()
    # create the directory if it doesn't exist
    output_dir = os.path.expanduser('~/data/outputs/plots/')
    os.makedirs(output_dir, exist_ok=True)
    # save figure and clear
    save_path = os.path.join(output_dir, f'{title}.jpg')
    plt.savefig(save_path)
    plt.clf()
    print("Saved plot at:", os.path.abspath(save_path))


def draw_confusion_matrix(y_true:List[int], y_pred:List[int], classes:List[str], model:str, uid:str, TESTBI:bool) -> None:
    """Draw consufion matrix to visualise classification result"""
    BI = TESTBI
    assert len(y_pred) == len(y_true), "y_true and y_pred must have the same length"
    # calculate confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    # draw figure
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    thresh = matrix.max() / 2.
    for i in range(len(matrix)):    
        for j in range(len(matrix[i])):    
            plt.text(j, i, format(matrix[i][j]), 
                    ha="center", va="center", 
                    color="white" if matrix[i, j] >= thresh else "black")
    # set title and label
    indices = range(len(matrix))
    title = f'{uid}_confusion_matrix_BI_{BI}'
    plt.xticks(indices, classes, rotation=45)
    plt.yticks(indices, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    # save figure and clear
    plt.savefig(f'/home/j/data/outputs/pics/{title}.jpg')
    plt.clf()


def draw_scatter_plot(ref:pd.DataFrame, pred:pd.DataFrame, model:str) -> None:
    """Draw scatter plot with r2 for each class"""
    assert ref.shape == pred.shape, "reference and prediction must have the same shape"
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    for i in range(ref.shape[1]):
        header = ref.columns[i]
        y_pred = pred.iloc[:, i]
        y_true = ref.iloc[:, i]
        r2 = r2_score(y_true, y_pred)
        plt.subplot(3, 3, i + 1)
        plt.text(0.5, 0.5, f'r2 = {r2:.2f}', fontdict={'weight':'bold', 'size':15}, ha='center')
        plt.scatter(y_pred, y_true, s = 15, c='lightblue')
        #set title and label
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.title(header)
        if i % 3 == 0:
            plt.ylabel('Reference')
        if i // 3 == 2:
            plt.xlabel('Prediction')
    # save figure and clear
    title = f'{model} scatter plot'
    plt.suptitle(title)
    plt.savefig('../outputs/pics/regression/'+ title +'.jpg')
    plt.clf()


def draw_pie_chart(ref:pd.DataFrame, pred:pd.DataFrame, model:str) -> None:
    """Draw pie chart to show true predicted label number for multi-label classification"""
    assert ref.shape == pred.shape, "reference and prediction must have the same shape"
    # compare reference and prediction
    res = (ref == pred).sum(axis=1)
    # pie chart
    x = []
    labels = []
    gaps = []
    plt.figure(figsize=(8,8))
    for i in range(ref.shape[1]+1):
        num = (res == i).sum()
        x.append(num)
        labels.append(i)
        gaps.append(0.05)
    plt.pie(x, labels=labels, explode=gaps, autopct='%.0f%%', textprops={"size":10})
    # set title and legend
    title = f'{model} true label numbers'
    plt.title(title)
    plt.legend()
    # save figure and clear
    plt.savefig('../outputs/pics/multi_label/'+ title +'.jpg')
    plt.clf()


def draw_multi_confusion_matirx(ref:pd.DataFrame, pred:pd.DataFrame, model:str) -> None:
    """Draw confusion matrix for each binary classification of multi labels"""
    assert ref.shape == pred.shape, "reference and prediction must have the same shape"
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for k in range(ref.shape[1]):
        header = ref.columns[k]
        y_pred = pred.iloc[:, k]
        y_true = ref.iloc[:, k]
        # confusion matrix
        matrix = confusion_matrix(y_true, y_pred)
        # draw figure
        plt.subplot(3, 3, k + 1)
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        thresh = matrix.max() / 2.
        for i in range(len(matrix)):    
            for j in range(len(matrix[i])):    
                plt.text(j, i, format(matrix[i][j]), 
                        ha="center", va="center", 
                        color="white" if matrix[i, j] >= thresh else "black")
        #set title and label
        indices = range(len(matrix))
        plt.xticks(indices)
        plt.yticks(indices)
        if k % 3 == 0:
            plt.ylabel('Reference')
        if k // 3 == 2:
            plt.xlabel('Prediction')
        plt.title(header)
    # save figure and clear
    title = f'{model} multi confusion matrix'
    plt.suptitle(title)
    plt.savefig('../outputs/pics/multi_label/'+ title +'.jpg')
    plt.clf()