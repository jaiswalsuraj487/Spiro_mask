import torch
import torch.distributions as dist


def log_likelihood_loss(X,y,model):
  out = model(X)
  y_hat = out[:, 0].reshape(-1,1)
  var = torch.exp(out[:, 1]).reshape(-1,1)
  dis = dist.Normal(y_hat, torch.sqrt(var))
  res = -torch.mean(dis.log_prob(y)) 
  return res 


def mape(y_pred, y_true):
    diff = torch.abs((y_true - y_pred) / torch.clamp(torch.abs(y_true), min=1e-8))
    loss = 100.0 * torch.mean(diff)
    return loss

def mse_loss(y_pred, y_true):
    loss = torch.mean((y_true - y_pred)**2)
    return loss
