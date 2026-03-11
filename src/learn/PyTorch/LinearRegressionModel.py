# Imports
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt

## Let's try and build a Linear Regression Model using nn.Linear()
# Also let's get used to writing device-agnostic code

# Generating the data
weight, bias = 0.5, 0.5

start, stop, step = 0, 2, 0.05
X = torch.arange(start, stop, step).unsqueeze(dim=1)
Y = weight * X + bias

# Splitting between training and testing
train_num = int(0.8*len(X))
X_train, X_test = X[:train_num], X[train_num:]
Y_train, Y_test = Y[:train_num], Y[train_num:]

# Predictions graph 
def plot_predictions(train_data=X_train,
                train_labels=Y_train,
                test_data=X_test,
                test_labels=Y_test,
                predictions=None):
    
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    plt.legend(prop={"size":14})
    plt.show()

# Setting up the class
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features=1,
                                            out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
    
# Making the model reproducible
torch.manual_seed(42)
model_lin = LinearRegressionModel()
print("Model Parameters:", model_lin.state_dict())

print("Prediction graph")    
plot_predictions(predictions=model_lin(X_test).detach().numpy())

# Setting up the device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lin = model_lin.to(dev)
## Sending the data to Cuda
X_train = X_train.to(dev)
X_test = X_test.to(dev)
Y_train = Y_train.to(dev)
Y_test = Y_test.to(dev)

# Loss and Optimizer 
loss_fn = torch.nn.L1Loss() # Mean Absolute Error

optimizer = torch.optim.SGD(params=model_lin.parameters(), 
                            lr=0.01)

# Now for the training
epochs = 1000

for epoch in range(epochs):
    # Set the model to training mode
    model_lin.train() # Sets all the parameters that require gradients to require gradients

    # 1. Forward pass
    y_preds = model_lin(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_preds, Y_train)

    # 3. Zero the gradients
    optimizer.zero_grad() # Prevents the accumulation of gradients from previous step 5s

    # 4. Backward propagation on the loss with respect to the parameters of the model
    loss.backward() 

    # 5. Step the optimizer (update the parameters)
    optimizer.step()

    model_lin.eval()
    
    # Print out the loss every 10 epochs
    if epoch % 50 == 0:
        # Testing mode
        with torch.inference_mode():

            test_pred = model_lin(X_test)
            test_loss = loss_fn(test_pred, Y_test)
        
        print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Test Loss: {test_loss}")
        
        #epoch_list.append(epoch)
        #train_loss.append(loss)
        #test_loss_val.append(test_loss)

print(model_lin.state_dict())

# Making predictions with our test data
with torch.inference_mode():
    y_preds_new = model_lin(X_test)

print("New predictions as given below:")
y_preds_new = y_preds_new.cpu()
plot_predictions(predictions=y_preds_new)

plt.close('all')