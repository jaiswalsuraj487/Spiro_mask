import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.distributions as dist

import ray
from ray import tune



print("utils.py loaded")

torch.manual_seed(42)

class MLP(nn.Module):
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


def loss_tfp(X,y,model):
  out = model(X)
  y_hat = out[:, 0].reshape(-1,1)
  var = torch.exp(out[:, 1]).reshape(-1,1)
  dis = dist.Normal(y_hat, torch.sqrt(var))
  res = -torch.mean(dis.log_prob(y)) 
  return res 


def mape_loss(y_pred, y_true):
    diff = torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=1e-8))
    loss = 100.0 * torch.mean(diff)
    return loss

def mse_loss(y_pred, y_true):
    loss = torch.mean((y_true - y_pred)**2)
    return loss

def load_data(file_name):
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
    return X_train,X_val, X_test, y_train, y_val, y_test


def model_run(file_name,num_epochs,hidden_sizes, output_size, lr=0.01):
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
            loss = loss_tfp(X_train, y_train,model)

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
                val_loss = loss_tfp(X_val, y_val,model)

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
        print("MAPE val: ", mape_loss(y_pred, y_val).item())
        print("MSE val: ", mse_loss(y_pred, y_val).item())
        if output_size==2:
            print("Loss tfp val: ", loss_tfp(X_val, y_val,model).item())
    # Evaluate 
    with torch.no_grad():
        output = model(X_test)
        y_pred = output[:, 0].reshape(-1,1)
        if output_size==2:
            var_pred = output[:, 1].reshape(-1,1)
        print("MAPE test: ", mape_loss(y_pred, y_test).item())
        print("MSE test: ", mse_loss(y_pred, y_test).item())
        if output_size==2:
            print("Loss tfp test: ", loss_tfp(X_val, y_test,model).item())    


    if output_size == 1:
        return y_pred
    else:
        return y_pred, var_pred

def train_mlp(config):
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
        loss = loss_tfp(X_train, y_train,model)
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
            val_loss = loss_tfp(X_val, y_val,model)
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























# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# import torch.distributions as dist

# import ray
# from ray import tune


# global raytune_file_name 
# # raytune_file_name = ''

# print("utils.py loaded")

# torch.manual_seed(42)

# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(MLP, self).__init__()
        
#         fc_layers = []
#         prev_size = input_size
#         for size in hidden_sizes:
#             fc_layers.append(nn.Linear(prev_size, size))
#             fc_layers.append(nn.ReLU())
#             prev_size = size
#         fc_layers.append(nn.Linear(prev_size, output_size))
        

#         self.fc_layers = nn.Sequential(*fc_layers)
        
#     def forward(self, x):
#         x = self.fc_layers(x)
#         return x



#     def loss_tfp(self,X,y,model):
#         out = model(X)
#         y_hat = out[:, 0].reshape(-1,1)
#         var = torch.exp(out[:, 1]).reshape(-1,1)
#         dis = dist.Normal(y_hat, torch.sqrt(var))
#         res = -torch.mean(dis.log_prob(y)) 
#         return res 


#     def mape_loss(self, y_pred, y_true):
#         diff = torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=1e-8))
#         loss = 100.0 * torch.mean(diff)
#         return loss

#     def mse_loss(self, y_pred, y_true):
#         loss = torch.mean((y_true - y_pred)**2)
#         return loss

#     def load_data(self, file_name):
#         X= np.load("D:\\iitgn\\Thesis\\Spiro_Mask2\\Spiro_Mask_dataset\\"+file_name+"_FEATURES_60.npy")
#         Y= np.load("D:\\iitgn\\Thesis\\Spiro_Mask2\\Spiro_Mask_dataset\\"+file_name+"_LABELS_60.npy")
#         # deleting these 12 rows 
#         delete_list  =  [1, 4, 9, 20, 23, 33, 43, 44, 45, 50, 52, 55]

#         X= np.delete(X, delete_list, axis=0)
#         Y= np.delete(Y,delete_list, axis=0)
#         X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

#         X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=10)
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

#         X_val = torch.tensor(X_val, dtype=torch.float32)
#         y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

#         X_test = torch.tensor(X_test, dtype=torch.float32)
#         y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
#         return X_train,X_val, X_test, y_train, y_val, y_test


#     def model_run(self, file_name,num_epochs,hidden_sizes, output_size, lr=0.01):
#         X_train,X_val, X_test, y_train, y_val, y_test = self.load_data(file_name)    
#         # Define model, loss function, and optimizer
#         model = MLP(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, output_size=output_size)
#         loss_mse = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         best_val_loss = float('inf')
#         best_val_epoch = 0
#         for epoch in range(num_epochs):
            
#             # Training
#             model.train()
#             if output_size==1:
#                 y_pred = model(X_train)
#                 y_pred = y_pred[:, 0].reshape(-1,1)
#                 loss = loss_mse(y_pred, y_train)
#             else:
#                 loss = self.loss_tfp(X_train, y_train,model)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 if output_size==1:
#                     y_pred = model(X_val)
#                     y_pred = y_pred[:, 0].reshape(-1,1)
#                     val_loss = loss_mse(y_pred, y_val)
#                 else:
#                     val_loss = self.loss_tfp(X_val, y_val,model)

#                 if val_loss.item() < best_val_loss:
#                     best_val_loss = val_loss.item()
#                     best_val_epoch = epoch
#                     torch.save(model.state_dict(), 'best_model.pt')   

#             if (epoch+1 ==1) or (epoch+1) % 100 == 0:
#                 print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss.item()))

#         model.load_state_dict(torch.load('best_model.pt'))
#         model.eval()

#         with torch.no_grad():
#             output = model(X_val)
#             y_pred = output[:, 0].reshape(-1,1)
#             print('val loss at Epoch : ', best_val_epoch+1)
#             print("MAPE val: ", self.mape_loss(y_pred, y_val).item())
#             print("MSE val: ", self.mse_loss(y_pred, y_val).item())
#             if output_size==2:
#                 print("Loss tfp val: ", self.loss_tfp(X_val, y_val,model).item())
#         # Evaluate 
#         with torch.no_grad():
#             output = model(X_test)
#             y_pred = output[:, 0].reshape(-1,1)
#             if output_size==2:
#                 var_pred = output[:, 1].reshape(-1,1)
#             print("MAPE test: ", self.mape_loss(y_pred, y_test).item())
#             print("MSE test: ", self.mse_loss(y_pred, y_test).item())
#             if output_size==2:
#                 print("Loss tfp test: ", self.loss_tfp(X_val, y_test,model).item())    


#         if output_size == 1:
#             return y_pred
#         else:
#             return y_pred, var_pred

#     def train_mlp_output_1(self, config):
#         X_train,X_val, X_test, y_train, y_val, y_test = self.load_data('FVC')
#         # best_val_loss = float('inf')
#         model = MLP(input_size=X_train.shape[1], hidden_sizes=config["hidden_size"], output_size=1)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=config["lr"])
#         num_epochs = 1
#         for epoch in range(num_epochs):
#             # Training
#             model.train()
#             outputs = model(X_train)
#             loss = criterion(outputs, y_train)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # Validation
#             model.eval()
#             with torch.no_grad():
#                     outputs = model(X_val)
#                     val_loss = criterion(outputs, y_val)
#                     # if val_loss.item() < best_val_loss:
#                     #     best_val_loss = val_loss
#             # tune.report(val_loss = best_val_loss)
#             tune.report(val_loss = val_loss.item())
#             # tune.report(best_val_loss = best_val_loss.item())

#     def train_mlp_output_2(self, config):
#         X_train,X_val, X_test, y_train, y_val, y_test = self.load_data(raytune_file_name)
#         model = MLP(input_size=X_train.shape[1], hidden_sizes=config["hidden_size"], output_size=2)
#         # criterion = loss_tfp
#         optimizer = optim.Adam(model.parameters(), lr=config["lr"])
#         num_epochs = 5
#         for epoch in range(num_epochs):
#             # Training
#             model.train()
#             loss = self.loss_tfp(X_train, y_train, model)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             model.eval()
#             with torch.no_grad():
#                 val_loss = self.loss_tfp(X_val, y_val, model)
#             tune.report(val_loss = val_loss.item())


#     def raytune_fun(self, output_size, file_name):
#         config = {
#             "hidden_size": tune.choice([[50],[50,10], [75,35,15,5],[60,30,5] ]),
#             # "lr": tune.loguniform(1e-5, 1e-1),
#             'lr':tune.choice([0.01, 0.001, 0.0001]),
#             # "num_epochs": tune.choice([2000])
#         }
#         raytune_file_name = file_name

#         ray.shutdown()
#         # Initialize Ray
#         ray.init()

#         if output_size==1:
#             analysis = tune.run(
#                 self.train_mlp_output_1,
#                 config=config,
#                 num_samples=10,
#                 progress_reporter=tune.CLIReporter()
#             )    
#         else:
#             analysis = tune.run(
#                 self.train_mlp_output_2,
#                 config=config,
#                 num_samples=10,
#                 progress_reporter=tune.CLIReporter()
#             )    
            
#         return analysis







