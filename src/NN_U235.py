# Imports
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

## We implement a NN for the U-235 data across different energies.
# Setting up the device for PyTorch (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Obtaining the data
data_path_file = "/home/n1rm4l/Desktop/nuclear-ml/data/nuc-data-process/residual_standardised.csv"
df = pd.read_csv(data_path_file)

# Getting the data and splitting it up

# Below is the leave-one-energy-out strat
"""
X_train = df.loc[df["E"] != 14000000, ["std_mass", "std_logE"]].to_numpy(dtype=np.float32)
y_train = df.loc[df["E"] != 14000000, ["Residual"]].to_numpy(dtype=np.float32)
X_test = df.loc[df["E"] == 14000000, ["std_mass", "std_logE"]].to_numpy(dtype=np.float32)
y_test = df.loc[df["E"] == 14000000, ["Residual"]].to_numpy(dtype=np.float32)

X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).view(-1,1).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).view(-1,1).to(device)

dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

X = df[["std_mass", "std_logE", "std_Z", "std_A_CN"]].to_numpy(dtype=np.float32)
X = torch.from_numpy(X).to(device)
y = df[["Residual"]].to_numpy(dtype=np.float32)
y = torch.from_numpy(y).view(-1,1).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Setting up the class
class FissionFragmentsModel(torch.nn.Module):
    def __init__(self, inFeatures, outFeatures, hiddenUnits=128):
        super().__init__()
        self.stack1 = nn.Sequential(
            nn.Linear(in_features=inFeatures, out_features=hiddenUnits),
            nn.ReLU(),
            nn.Linear(in_features=hiddenUnits, out_features=hiddenUnits),
            nn.ReLU(),
            nn.Linear(in_features=hiddenUnits, out_features=outFeatures)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack1(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_nuke = FissionFragmentsModel(inFeatures=4, outFeatures=1).to(device)
print(model_nuke.state_dict())

# Setting up the loss and optimizer
loss_fn = torch.nn.MSELoss() # Mean Absolute Error

optimizer = torch.optim.Adam(params=model_nuke.parameters(), 
                            lr=1e-3, amsgrad=False, weight_decay=1e-5)


epochs = 6000 # Number of loops through the data (hyperparameter)
epoch_list, train_loss, test_loss_val = [], [], []

# 0.
for epoch in range(epochs):
    # Set the model to training mode
    model_nuke.train() # Sets all the parameters that require gradients to require gradients
    
    
    for X_batch, y_batch in train_dataloader:
        # 1. Forward pass
        y_preds = model_nuke(X_batch)

        # 2. Calculate the loss
        loss = loss_fn(y_preds, y_batch)
        """
        # 1. Forward pass
        y_preds = model_nuke(X_train)
        
        # 2. Calculate the loss
        loss = loss_fn(y_preds, y_train)
        """
        # 3. Zero the gradients
        optimizer.zero_grad() # Prevents the accumulation of gradients from previous step 5s

        # 4. Backward propagation on the loss with respect to the parameters of the model
        loss.backward() 

        # 5. Step the optimizer (update the parameters)
        optimizer.step()

    model_nuke.eval()
    
    # Print out the loss every 10 epochs
    if epoch % 500 == 0:
        # Testing mode
        with torch.inference_mode():

            test_pred = model_nuke(X_test)
            test_loss = loss_fn(test_pred, y_test)
        
        print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Test Loss: {test_loss}")
        epoch_list.append(epoch)
        train_loss.append(loss)
        test_loss_val.append(test_loss)

print(model_nuke.state_dict())

def plot_predictions(predictions: torch.Tensor):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0].cpu().detach().numpy(), y_test.cpu().detach().numpy(), color='blue', label='True Values')
    plt.scatter(X_test[:, 0].cpu().detach().numpy(), predictions.cpu().detach().numpy(), color='red', label='Predictions')
    plt.xlabel('Standardized Mass Number')
    plt.ylabel('Residual')
    plt.title('Model Predictions vs True Values')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    
    plt.scatter(epoch_list, [loss.cpu().detach().numpy() for loss in train_loss])
    plt.scatter(epoch_list, [loss.cpu().detach().numpy() for loss in test_loss_val])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    
plot_predictions(predictions=model_nuke(X_test))

model_nuke.eval()

with torch.inference_mode():
    y_preds = model_nuke(X_test)
    y_train_preds = model_nuke(X_train)
    
r2_test = r2_score(y_test.cpu().detach().numpy(), y_preds.cpu().detach().numpy())
r2_train = r2_score(y_train.cpu().detach().numpy(), y_train_preds.cpu().detach().numpy())
print(f"R2 score: {r2_test} | R2 score train: {r2_train}")

"""
# Making predictions with our test data
with torch.inference_mode():
    y_preds_new = model_quad(X_test)

print("New predictions as given below:")
plot_predictions(predictions=y_preds_new)
"""