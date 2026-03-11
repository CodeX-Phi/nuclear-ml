import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from helper_functions import plot_decision_boundary, accuracy_fn

import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Hyperparameters for data selections
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Making the data
# Ignore the error
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# Turning data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Device Agnostic Code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Splitting the data
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2,
                                                                        random_state=RANDOM_SEED)

"""
# Plotting the data
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()
plt.close('all')
"""

# Building the model 
class MultiClassifcation(nn.Module):
    def __init__(self, inFeatures, outFeatures, hiddenUnits=16):
        super().__init__()
        self.linear_stack_1 = nn.Sequential(
            nn.Linear(in_features=inFeatures, out_features=hiddenUnits),
            nn.Linear(in_features=hiddenUnits, out_features=hiddenUnits),
            nn.Linear(in_features=hiddenUnits, out_features=outFeatures)
        )
        
        self.linear_stack_2 = nn.Sequential(
            nn.Linear(in_features=inFeatures, out_features=hiddenUnits),
            nn.ReLU(),
            nn.Linear(in_features=hiddenUnits, out_features=hiddenUnits),
            nn.ReLU(),
            nn.Linear(in_features=hiddenUnits, out_features=outFeatures)
        )
    
    def forward(self, x):
        return self.linear_stack_2(x)
    
model_multi_class = MultiClassifcation(2, NUM_CLASSES).to(device)
X_blob_train, X_blob_test = X_blob_train.to(device), X_blob_test.to(device)
y_blob_train, y_blob_test = y_blob_train.to(device), y_blob_test.to(device)


# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_multi_class.parameters(),
                            lr = 0.005)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    model_multi_class.train()
    
    y_blob_logits = model_multi_class(X_blob_train)
    y_blob_pred = torch.argmax(torch.softmax(y_blob_logits, dim=1), dim=1)

    loss = loss_fn(y_blob_logits, y_blob_train)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 50 == 0:
        model_multi_class.eval()
        
        with torch.inference_mode():
            y_test_blobs = model_multi_class(X_blob_test)
            y_test_blob_pred = torch.argmax(torch.softmax(y_test_blobs, dim=1), dim=1)
            
            test_loss = loss_fn(y_test_blobs, y_blob_test)
            test_accuracy = accuracy_fn(y_blob_test, y_test_blob_pred)
            
            print(f"Epoch = {epoch} | Training Loss = {loss} | Test loss = {test_loss} | Test accuracy = {test_accuracy}")
            

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_multi_class, X_blob_train, y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_multi_class, X_blob_test, y_test_blob_pred)
plt.show()
plt.close('all')
            
