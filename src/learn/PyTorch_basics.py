# Imports
import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt

## Let's try and build a Polynomial regression model
# Also let's get used to writing device-agnostic code

# Generating the data
weight, bias = 0.5, 1

start, stop, step = -1, 1, 0.05
X = torch.arange(start, stop, step).unsqueeze(dim=1)
Y = weight * (X*X) + bias

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

print("Prediction graph")    
plot_predictions()

# Setting up the class
class QuadraticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(1,                      # starting with randomly assigned weight
                                                      requires_grad=True,     # using gradient descent to work out the weight
                                                      dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.randn(1, 
                                                   requires_grad=True, 
                                                   dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * (x*x) + self.bias

model_quad = QuadraticRegressionModel()
print(model_quad.state_dict())

# Checking what the graph would look like
with torch.inference_mode():
    y_preds = model_quad(X_test)

print("Predictions graph:")
plot_predictions(predictions=y_preds)

# Setting up the loss and optimizer
loss_fn = torch.nn.L1Loss() # Mean Absolute Error

optimizer = torch.optim.SGD(params=model_quad.parameters(), 
                            lr=0.001)


epochs = 3000 # Number of loops through the data (hyperparameter)
epoch_list, train_loss, test_loss_val = [], [], []

# 0.
for epoch in range(epochs):
    # Set the model to training mode
    model_quad.train() # Sets all the parameters that require gradients to require gradients

    # 1. Forward pass
    y_preds = model_quad(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_preds, Y_train)

    # 3. Zero the gradients
    optimizer.zero_grad() # Prevents the accumulation of gradients from previous step 5s

    # 4. Backward propagation on the loss with respect to the parameters of the model
    loss.backward() 

    # 5. Step the optimizer (update the parameters)
    optimizer.step()

    model_quad.eval()
    
    # Print out the loss every 10 epochs
    if epoch % 50 == 0:
        # Testing mode
        with torch.inference_mode():

            test_pred = model_quad(X_test)
            test_loss = loss_fn(test_pred, Y_test)
        
        print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Test Loss: {test_loss}")
        epoch_list.append(epoch)
        train_loss.append(loss)
        test_loss_val.append(test_loss)
