import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim

import ray
from ray import tune

from loss import log_likelihood_loss, mape, mse_loss

print("utils.py loaded")

torch.manual_seed(42)

class MLP(nn.Module):
    # MLP with relu as activation function
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        fc_layers = []
        prev_size = input_size
        for size in hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(nn.ReLU())
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, output_size))
        

        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        x = self.fc_layers(x)
        return x

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP_sin(nn.Module):
    # MLP with sin as activation function
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP_sin, self).__init__()
        
        fc_layers = []
        prev_size = input_size
        for size in hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(SinActivation())
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, output_size))
        

        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        x = self.fc_layers(x)
        return x

def load_data(file_name, return_value='all'):
    X= np.load("Spiro_Mask_dataset\\"+file_name+"_FEATURES_60.npy")
    Y= np.load("Spiro_Mask_dataset\\"+file_name+"_LABELS_60.npy")
    # deleting these 12 rows 
    delete_list  =  [1, 4, 9, 20, 23, 33, 43, 44, 45, 50, 52, 55]

    X= np.delete(X, delete_list, axis=0)
    Y= np.delete(Y,delete_list, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=10)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    if return_value=='all':
        return X_train,X_val, X_test, y_train, y_val, y_test
    else:
        return X,Y

def data_loader(X_train,X_val, X_test, y_train, y_val, y_test):
    batch_size = 4
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader

def model_run_with_dataloader(file_name,num_epochs,hidden_sizes, output_size=1, lr=0.01, activation='relu'):
    torch.manual_seed(42)
    X_train,X_val, X_test, y_train, y_val, y_test = load_data(file_name)  
    train_loader, val_loader, test_loader = data_loader(X_train,X_val, X_test, y_train, y_val, y_test)  
    # Define model, loss function, and optimizer
    if activation=='relu':
        model = MLP(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=output_size)
    else:
        model = MLP_sin(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=output_size)
    loss_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_val_epoch = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_train, y_train in train_loader:
            if output_size==1:
                y_pred = model(X_train)
                y_pred = y_pred[:, 0].reshape(-1,1)
                loss = loss_mse(y_pred, y_train)
            else:
                loss = log_likelihood_loss(X_train, y_train,model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_train.size(0)        
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_val, y_val in val_loader:
                if output_size==1:
                    outputs = model(X_val)
                    outputs = outputs[:, 0].reshape(-1,1)
                    loss = loss_mse(outputs, y_val)
                else:
                    loss = log_likelihood_loss(X_val, y_val,model)
                val_loss += loss.item()  * X_val.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')


        if (epoch+1 ==1) or (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss, val_loss))

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    with torch.no_grad():
        test_loss = 0
        for inputs,targets in test_loader:
            outputs = model(inputs)
            y_pred = outputs[:, 0].reshape(-1,1)
            loss = mape(y_pred, targets)
            test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_loader.dataset)
        print("MAPE test: ", test_loss)
    y_pred = model(X_test)[: , 0].reshape(-1,1)
    if output_size == 1:
        return y_pred
    else:
        var_pred  = model(X_test)[:, 1].reshape(-1,1)
        var_pred = torch.exp(var_pred)
        return y_pred, var_pred

def model_run_without_dataloader(file_name,num_epochs,hidden_sizes, output_size, lr=0.01, activation='relu'):
    torch.manual_seed(42)
    X_train,X_val, X_test, y_train, y_val, y_test = load_data(file_name)    
    # Define model, loss function, and optimizer
    model = MLP(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=output_size)
    loss_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_val_epoch = 0
    for epoch in range(num_epochs):
        
        # Training
        model.train()
        if output_size==1:
            y_pred = model(X_train)
            y_pred = y_pred[:, 0].reshape(-1,1)
            loss = loss_mse(y_pred, y_train)
        else:
            loss = log_likelihood_loss(X_train, y_train,model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if output_size==1:
                y_pred = model(X_val)
                y_pred = y_pred[:, 0].reshape(-1,1)
                val_loss = loss_mse(y_pred, y_val)
            else:
                val_loss = log_likelihood_loss(X_val, y_val,model)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_val_epoch = epoch
                torch.save(model.state_dict(), 'best_model.pt')   

        if (epoch+1 ==1) or (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss.item()))

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    with torch.no_grad():
        output = model(X_val)
        y_pred = output[:, 0].reshape(-1,1)
        print('val loss at Epoch : ', best_val_epoch+1)
        print("MAPE val: ", mape(y_pred, y_val).item())
        print("MSE val: ", mse_loss(y_pred, y_val).item())
        if output_size==2:
            print("Log likelihood Loss val: ", log_likelihood_loss(X_val, y_val,model).item())
    # Evaluate 
    with torch.no_grad():
        output = model(X_test)
        y_pred = output[:, 0].reshape(-1,1)
        if output_size==2:
            var_pred = output[:, 1].reshape(-1,1)
        print("MAPE test: ", mape(y_pred, y_test).item())
        print("MSE test: ", mse_loss(y_pred, y_test).item())
        if output_size==2:
            print("Log likelihood Loss test: ", log_likelihood_loss(X_val, y_test,model).item())    

    if output_size == 1:
        return y_pred
    else:        
        var_pred = torch.exp(var_pred)  
        return y_pred, var_pred

def model_run(file_name,num_epochs,hidden_sizes, output_size=1, lr=0.01, activation='relu', type_run='dataloader'):
    if type_run=='dataloader':
        return model_run_with_dataloader(file_name,num_epochs,hidden_sizes, output_size, lr, activation)
    if type_run=='without_dataloader':
        return model_run_without_dataloader(file_name,num_epochs,hidden_sizes, output_size, lr, activation)

def model_regressor(file_name, regressor_type='RF'):
    torch.manual_seed(42)
    X,y = load_data(file_name, return_value = 'Xy')    
    #LOOCV ONLY
    #All left out samples
    loo = LeaveOneOut()

    #store the predictions here
    y_pred = []

    if regressor_type == 'RF':
        regressor = RandomForestRegressor(n_jobs=-1, bootstrap=True, criterion='mae', 
                                    n_estimators=100,  max_features='auto', max_depth=300,  
                                    min_samples_leaf=1,min_samples_split=5,random_state=42)
    elif regressor_type == 'XGB':
        regressor = XGBRegressor(n_jobs=-1, objective='reg:squarederror', random_state=42)
    elif regressor_type == 'SVR':
        regressor = sklearn.svm.SVR(kernel='rbf', C=10, gamma=10, epsilon=.1)
    # Create a based model
    mpeList = []
    for train_index, test_index in loo.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        

        #for RF and SVR
        reg = regressor.fit(X_train, y_train)
        pred = reg.predict(X_test)
        
        y_pred.append(pred[0])
        
    
        
        #print("True FVC = {}, Predicted FVC = {}".format(y_test,pred))
        mpe = 100*np.mean(np.abs((y_test.reshape(-1) -pred)/y_test.reshape(-1)))
        mpeList.append(mpe)
        #print("Error on Participant ID {} is {:2f}".format(test_index,mpe))
        #print("\n")
    print("for {} Overall MPE is = {}".format(file_name, np.mean(mpeList)))
    #print("Bootstrap = {},Depth = {}, Features = {}, Leaf = {}, Split = {}, Estimator = {}".format(bootstrap,depth,features,leaf,split,estimator))

    y_pred = np.array(y_pred)
    return y_pred



def train_mlp(config):
    global output_size_global
    global raytune_file_name
    X_train,X_val, X_test, y_train, y_val, y_test = load_data(raytune_file_name)
    # best_val_loss = float('inf')
    model = MLP(input_size=X_train.shape[1], hidden_sizes=config["hidden_size"], output_size=output_size_global)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
      # Training
      model.train()
      if output_size_global==1:
        outputs = model(X_train)    
        loss = mse_loss(outputs, y_train)
      else:
        loss = log_likelihood_loss(X_train, y_train,model)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Validation
      model.eval()
      with torch.no_grad():
        if output_size_global==1:
            outputs = model(X_val)
            val_loss = mse_loss(outputs, y_val)
        else:
            val_loss = log_likelihood_loss(X_val, y_val,model)
      tune.report(val_loss = val_loss.item())

def raytune_fun(output_size, file_name):
    
    global raytune_file_name, output_size_global 
    output_size_global = output_size
    raytune_file_name = file_name
    config = {
    "hidden_size": tune.choice([[50,10] ,[60,30,5], [75,35,15,5],[128,80,50,30,16]]),
    'lr':tune.choice([0.01, 0.001]),
    "num_epochs": tune.choice([1])
    }
    ray.shutdown()
    # Initialize Ray
    ray.init()
    analysis = tune.run(
            train_mlp,
            config=config,
            num_samples=10,
            progress_reporter=tune.CLIReporter()
        )    

    return analysis







