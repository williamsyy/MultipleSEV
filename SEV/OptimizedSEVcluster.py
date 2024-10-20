import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

class SimpleLR(nn.Module):
    # this is the simple logistic regression model
    def __init__(self, n_features, sev, sev_penalty, positive_penalty):
        super(SimpleLR, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(sev.data_map), dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(sev.cluster_labels),dtype=torch.float32)
        self.data_mean_map = torch.stack([self.data_mean[i,:].view(-1) * torch.ones(self.data_map.shape) for i in range(self.data_mean.shape[0])],dim=0)
        self.sev = sev
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

    # def update_sev(self,sev):
    #     self.sev = deepcopy(sev)

class SimpleMLP(nn.Module):
    # this is the simple 2-layer MLP model
    def __init__(self, n_features, n_hidden, sev,sev_penalty, positive_penalty):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.activation = nn.ReLU()
        
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(sev.data_map), dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(sev.cluster_labels),dtype=torch.float32)
        self.data_mean_map = torch.stack([self.data_mean[i,:].view(-1) * torch.ones(self.data_map.shape) for i in range(self.data_mean.shape[0])],dim=0)
        self.sev = sev
        self.sev_penalty = sev_penalty
        self.positive_penalty = positive_penalty


    def forward(self, x):
        x = self.hidden(x)
        x= self.activation(x)
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

class SimpleGBDT(nn.Module):
    # this is the simple GBDT model, which input a Gradient Boosting Classifier module in sklearn and creates a torch modules with the same structure. All the optimization is done for the weights of each tree created by the sklearn GBDT model.
    def __init__(self, base_model, sev, sev_penalty, positive_penalty):
        super(SimpleGBDT, self).__init__()
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(sev.data_map), dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(sev.cluster_labels),dtype=torch.float32)
        self.data_mean_map = torch.stack([self.data_mean[i,:].view(-1) * torch.ones(self.data_map.shape) for i in range(self.data_mean.shape[0])],dim=0)
        self.sev = sev
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
        self.sev = sev

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
    def __init__(self, X, y, X_neg, sev):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        X_trans = sev.embedding.transform(X,X_neg.values)
        self.labels = sev.cluster.predict(X_trans,X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx],self.labels[idx]

class AllOptMinus(nn.Module):
    # this is an implementation of the loss function AllOpt-
    def __init__(self,model):
        super(AllOptMinus, self).__init__()
        self.model = model

    def SEV_Loss(self,x,labels):
        # get the cluster of the given x
        # original_X = x.detach().numpy()
        # X_trans = self.model.sev.embedding.transform(original_X,self.model.sev.X_neg.values)
        # labels = self.model.sev.cluster.predict(X_trans,original_X)
        # get the SEV = 1 cases
        temp_X1 = torch.stack([x for _ in range(self.model.data_map.shape[0])],0)
        dataset = torch.stack([torch.tensor(torch.where(self.model.data_map[labels[index]]==0,i,self.model.data_mean_map[labels[index]])) for index,i in enumerate(torch.transpose(temp_X1,0,1))],1)
        out = self.model(dataset).squeeze()
        y_pred = self.model(x)
        loss = torch.sum(torch.clamp(torch.min(out,dim=0).values-0.5,min= -0.05)*torch.gt(y_pred,0.5))/(torch.sum(torch.gt(y_pred,0.5))+1e-10)
        # print(loss)
        return self.model.sev_penalty * loss


    def BaselinePositiveLoss(self):
        # make sure the population mean is negative
        loss = 0
        for i in range(self.model.sev.n_clusters):
            loss += torch.clamp(self.model(self.model.data_mean[i].view(1,-1)) - 0.5,min=-0.05)
        return self.model.positive_penalty * loss

    def forward(self, output, target, x,labels):
        baseloss = nn.BCELoss()
        # print(output)
        # print(target)
        loss = baseloss(output, target)
        sev_loss = self.SEV_Loss(x,labels)
        positive_loss = self.BaselinePositiveLoss()
        # print("BaseLoss:",loss)
        # print("SEV Loss", sev_loss)
        # print("Positive Loss",positive_loss)
        return loss, loss+ sev_loss + positive_loss, sev_loss

class OriginalLoss(nn.Module):
    # the implementation of the original loss function
    def __init__(self):
        super(OriginalLoss, self).__init__()

    def forward(self, output, target):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        return loss
    
