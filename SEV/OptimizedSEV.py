import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# this function is the basic modules for Optimizing SEV

class SimpleLR(nn.Module):
    # this is the simple logistic regression model
    def __init__(self, n_features,data_map, data_mean, sev_penalty, positive_penalty):
        super(SimpleLR, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(data_map),dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean),dtype=torch.float32).view(-1,1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        # add the sev_penalty to the model
        self.sev_penalty = sev_penalty
        # add the positive penalty to the model
        self.positive_penalty = positive_penalty

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

    def predict(self,x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = torch.sigmoid(self.linear(x_torch))
        return (logits.squeeze()>0.5).detach().cpu().numpy()

    def predict_proba(self,x):
        x_torch = torch.tensor(np.array(x),dtype=torch.float32)
        logits = torch.sigmoid(self.linear(x_torch))
        res = torch.cat([1-logits.T,logits.T]).T
        return res.detach().cpu().numpy()

class SimpleMLP(nn.Module):
    # this is the simple 2-layer MLP model
    def __init__(self, n_features, n_hidden, data_map, data_mean, sev_penalty, positive_penalty):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.activation = nn.ReLU()
        
        self.data_map = torch.tensor(np.array(data_map), dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean), dtype=torch.float32).view(-1, 1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        
        self.sev_penalty = sev_penalty
        self.positive_penalty = positive_penalty

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        logits = self.output(x)
        return torch.sigmoid(logits)

    def predict(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        return (logits.squeeze() > 0.5).detach().cpu().numpy()

    def predict_proba(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        res = torch.cat([1-logits.T,logits.T]).T
        return res.detach().cpu().numpy()

class FourLayerMLP(nn.Module):
    # this is a sample for three layer MLP
    def __init__(self, n_features, n_hidden, data_map, data_mean, sev_penalty, positive_penalty):
        super(FourLayerMLP, self).__init__()
        self.hidden1 = nn.Linear(n_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.activation = nn.ReLU()
        
        self.data_map = torch.tensor(np.array(data_map), dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean), dtype=torch.float32).view(-1, 1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        
        self.sev_penalty = sev_penalty
        self.positive_penalty = positive_penalty
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        logits = self.output(x)
        return torch.sigmoid(logits)
    
    def predict(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        return (logits.squeeze() > 0.5).detach().cpu().numpy()
    
    def predict_proba(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        res = torch.cat([1-logits.T,logits.T]).T
        return res.detach().cpu().numpy()
    
class MoreComplexMLP(nn.Module):
    # this is a sample for ten layer MLP
    def __init__(self, n_features, n_hidden, data_map, data_mean, sev_penalty, positive_penalty):
        super(MoreComplexMLP, self).__init__()
        self.hidden1 = nn.Linear(n_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.hidden4 = nn.Linear(n_hidden, n_hidden)
        self.hidden5 = nn.Linear(n_hidden, n_hidden)
        self.hidden6 = nn.Linear(n_hidden, n_hidden)
        self.hidden7 = nn.Linear(n_hidden, n_hidden)
        self.hidden8 = nn.Linear(n_hidden, n_hidden)
        self.hidden9 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.activation = nn.ReLU()
        
        self.data_map = torch.tensor(np.array(data_map), dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean), dtype=torch.float32).view(-1, 1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        
        self.sev_penalty = sev_penalty
        self.positive_penalty = positive_penalty
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.hidden4(x)
        x = self.activation(x)
        x = self.hidden5(x)
        x = self.activation(x)
        x = self.hidden6(x)
        x = self.activation(x)
        x = self.hidden7(x)
        x = self.activation(x)
        x = self.hidden8(x)
        x = self.activation(x)
        x = self.hidden9(x)
        x = self.activation(x)
        logits = self.output(x)
        return torch.sigmoid(logits)

    def predict(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        return (logits.squeeze() > 0.5).detach().cpu().numpy()
    
    def predict_proba(self, x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        logits = self.forward(x_torch)
        res = torch.cat([1-logits.T,logits.T]).T
        return res.detach().cpu().numpy()

class SimpleGBDT(nn.Module):
    # this is the simple GBDT model, which input a Gradient Boosting Classifier module in sklearn and creates a torch modules with the same structure. All the optimization is done for the weights of each tree created by the sklearn GBDT model.
    def __init__(self, base_model,data_map, data_mean, sev_penalty, positive_penalty):
        super(SimpleGBDT, self).__init__()
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(data_map),dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean),dtype=torch.float32).view(-1,1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        # add the sev_penalty to the model
        self.sev_penalty = sev_penalty
        # add the negative penalty to the model
        self.positive_penalty = positive_penalty
        # add the information for Gradient
        self.estimators = base_model.estimators_
        # create the linear layer for the trees
        self.linear = nn.Linear(len(self.estimators), 1,bias=False)
        # save the linear weights as ones
        self.linear.weights = torch.tensor(np.ones(len(self.estimators))*base_model.learning_rate,dtype=torch.float,requires_grad=True)
        # save the bias predictor to create the bias term of GBDT
        self.bias_predictor = base_model.init_

    def forward(self, x):
        if len(x.shape)==2:
            y_pred = self.linear(torch.tensor([estimator[0].predict(x) for estimator in self.estimators]).transpose(0,1).float()).transpose(0,1)
            bias = torch.tensor(self.bias_predictor.predict_proba(x.detach().numpy())[:,1]).float()
            bias = torch.log(bias/(1-bias))
            out = torch.sigmoid(y_pred+bias)
            return out.view(-1)
        if len(x.shape)==3:
            y_preds = torch.cat([self.linear(torch.tensor([estimator[0].predict(xi) for estimator in self.estimators]).transpose(0,1).float()).transpose(0,1) for xi in x])
            biases = torch.cat([torch.tensor(self.bias_predictor.predict_proba(x.detach().numpy())[:,1]).float()])
            biases = torch.log(biases/(1-biases)).unsqueeze(1)
            out = torch.sigmoid(y_preds+biases)
            return out

    def predict(self,x):
        x_torch = torch.tensor(np.array(x), dtype=torch.float32)
        result = self(x_torch)
        return (result>0.5).detach().cpu().numpy()

    def predict_proba(self,x):
        x_torch = torch.tensor(np.array(x),dtype=torch.float32)
        logits = self(x_torch).detach().numpy()
        res = np.c_[1-logits.T,logits.T]
        return res

# this is the custom dataset class for the pytorch model
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AllOptPlus(nn.Module):
    # this is implementation the loss function allopt+ 
    def __init__(self,model):
        super(AllOptPlus, self).__init__()
        self.model = model

    def SEV_Loss(self,x):
        temp_X1 = torch.unsqueeze(self.model.data_map,2) * torch.transpose(x,0,1)
        # create the SEV+ = 1 datasets
        dataset = torch.stack([torch.tensor(np.where(self.model.data_map==0,self.model.data_mean_map,torch.transpose(i,0,1))) for i in torch.transpose(temp_X1,0,2)],1)
        # get the output of the model
        out = self.model(dataset)
        # get the output of the model
        y_pred = self.model(x)
        # calculate the loss
        loss = -torch.sum(torch.clamp(torch.max(out,dim=0).values-0.5,max= 0.05)*torch.gt(y_pred,0.5))/(torch.sum(torch.gt(y_pred,0.5))+1e-10)
        return self.model.sev_penalty * loss
    
    def BaselinePositiveLoss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,min=-0.05)
        return self.model.positive_penalty * loss

    def forward(self, output, target, x):
        # baseline loss as Binary Cross Entropy Loss
        baseloss = nn.BCELoss()
        # get the BCE Loss
        loss = baseloss(output, target)
        # calculate the AllOpt+ Loss
        sev_loss = self.SEV_Loss(x)
        # calculate the positive baseline loss
        positive_loss = self.BaselinePositiveLoss()
        return loss, loss+ sev_loss + positive_loss, sev_loss

class VolOpt(nn.Module):
    def __init__(self,model):
        super(VolOpt, self).__init__()
        self.model = model

    def SEV_Loss(self,x):
        # calculate the edge length of the model
        edge = torch.abs(self.model.linear(self.model.data_mean.view(1,-1))/self.model.linear.weight)
        # clamp the edge length to avoid the gradient explosion
        edge = torch.clamp(edge,min=1e-10)
        # take the log in order to transfrom the multiplcation of the edge length into the addition of the log of the edge length
        loss = torch.log(edge)
        loss = torch.mean(loss)
        return self.model.sev_penalty * loss
    
    def BaselinePositiveLoss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,min=-0.05)
        return self.model.positive_penalty * loss

    def forward(self, output, target, x):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x)
        positive_loss = self.BaselinePositiveLoss()
        return loss, loss+ sev_loss + positive_loss, sev_loss

class AllOptMinus(nn.Module):
    # this is an implementation of the loss function AllOpt-
    def __init__(self,model):
        super(AllOptMinus, self).__init__()
        self.model = model

    def SEV_Loss(self,x):
        temp_X1 = torch.stack([x for _ in range(self.model.data_map.shape[0])],0)
        # calculate the SEV-=1 datasets
        dataset = torch.stack([torch.tensor(torch.where(self.model.data_map==0,i,self.model.data_mean_map)) for i in torch.transpose(temp_X1,0,1)],1)
        out = self.model(dataset)
        y_pred = self.model(x)
        # calculate the loss funciton
        loss = torch.sum(torch.clamp(torch.min(out,dim=0).values-0.5,min= -0.05)*torch.gt(y_pred,0.5))/(torch.sum(torch.gt(y_pred,0.5))+1e-10)
        return self.model.sev_penalty * loss
    
    def BaselinePositiveLoss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,min=-0.05)
        return self.model.positive_penalty * loss

    def forward(self, output, target, x):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x)
        positive_loss = self.BaselinePositiveLoss()
        return loss, loss+ sev_loss + positive_loss, sev_loss

class AllOptRestricted(nn.Module):
    # this is the implmenation of the alloptR loss function
    def __init__(self,model,choices):
        super(AllOptRestricted, self).__init__()
        self.model = model
        # the choices is a list of the index of the features that we want to select
        self.feature_selection = choices
        
    def SEV_Loss(self,x):
        temp_X1 = torch.stack([x for _ in range(self.model.data_map.shape[0])],0)
        # caculate the SEVR=1 datasets
        dataset = torch.stack([torch.tensor(torch.where(self.model.data_map==0,i,self.model.data_mean_map)) for i in torch.transpose(temp_X1,0,1)],1)
        out = self.model(dataset)
        # remove those features that are restrited
        out = out[self.feature_selection]
        y_pred = self.model(x)
        loss = torch.sum(torch.clamp(torch.min(out,dim=0).values-0.5,min= -0.05)*torch.gt(y_pred,0.5))/(torch.sum(torch.gt(y_pred,0.5))+1e-10)
        return self.model.sev_penalty * loss
    
    def BaselinePositiveLoss(self):
        # make sure the population mean is negative
        loss = torch.clamp(self.model(self.model.data_mean.view(1,-1)) - 0.5,min=-0.05)
        return self.model.positive_penalty * loss

    def forward(self, output, target, x):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x)
        positive_loss = self.BaselinePositiveLoss()
        return loss, loss+ sev_loss + positive_loss, sev_loss

class OriginalLoss(nn.Module):
    # the implementation of the original loss function
    def __init__(self):
        super(OriginalLoss, self).__init__()

    def forward(self, output, target):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        return loss