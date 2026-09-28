import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

X = torch.arange(0,1,0.02).unsqueeze(dim=1)
y = weight*X +bias

train_split = int(.8*len(X))
X_train,y_train = X[:train_split], y[:train_split]
X_test,y_test = X[train_split:], y[train_split:]

def plot(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
    plt.figure(figsize=(10,10))
    plt.scatter(train_data,train_labels,c='b',s=4,label='Training Data')
    plt.scatter(test_data,test_labels,c='r',s=4,label='Test Data')

    if predictions is not None:
        plt.scatter(test_data,predictions,c='brown',s=4,label='Predictions')
    plt.legend(prop={'size':14})
    plt.show()

plot()