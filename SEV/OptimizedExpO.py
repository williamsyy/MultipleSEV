import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import lime
import lime.lime_tabular

# this function is the basic modules for Optimizing SEV

class SimpleLR(nn.Module):
    # this is the simple logistic regression model
    def __init__(self, n_features):
        super(SimpleLR, self).__init__()
        self.linear = nn.Linear(n_features, 1)

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
    def __init__(self, n_features, n_hidden):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
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

class SimpleGBDT(nn.Module):
    # this is the simple GBDT model, which input a Gradient Boosting Classifier module in sklearn and creates a torch modules with the same structure. All the optimization is done for the weights of each tree created by the sklearn GBDT model.
    def __init__(self, base_model):
        super(SimpleGBDT, self).__init__()
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

class OriginalLoss(nn.Module):
    # the implementation of the original loss function
    def __init__(self):
        super(OriginalLoss, self).__init__()

    def forward(self, output, target):
        baseloss = nn.BCELoss()
        loss = baseloss(output, target)
        return loss

class Regularizer(nn.Module):
    def __init__(self, model, n_input, num_samples,stddev = 0.1):
        super(Regularizer, self).__init__()
        self.model = model
        self.n_input = n_input
        self.num_samples = num_samples
        self.stddev = stddev
    
    def neighborhood(self, x):
        num_samples = self.num_samples
        n_input = self.n_input
        with torch.no_grad():
            x_expanded = x.repeat(num_samples, 1)
            constant_term = torch.ones(num_samples, 1)
            noise = torch.randn(num_samples, n_input) * self.stddev
            return torch.cat([x_expanded + noise, constant_term], 1)

    def projection_matrix(self, x_local):
        with torch.no_grad():
            result = torch.matmul(torch.inverse(torch.matmul(x_local.t(), x_local)), x_local.t())
            return result
    # P is [n_input, num_samples], y is [num_samples, n_classes]
    # Output is [num_samples, n_classes]
    def coefficients(self, P, y):
        with torch.no_grad():
            return torch.matmul(P, y)

    # It may be possible to make this computation more efficient by using broadcasting rather than map_fn
    def forward(self, x):
        with torch.no_grad():
            def compute_mse(x):
                x_local = self.neighborhood(x)
                P = self.projection_matrix(x_local)
                y = self.model(x_local[:, :-1])
                B = self.coefficients(P, y)
                y_lin = torch.matmul(x_local, B)
                y_prob = torch.sigmoid(y_lin)
                return torch.nn.BCELoss()(y_prob, y)
            return torch.mean(torch.stack([compute_mse(xi) for xi in x]))

class Regularizer_1D(nn.Module):
    def __init__(self, model, n_input, num_samples, stddev = 0.1):
        super(Regularizer_1D, self).__init__()
        self.model = model
        self.n_input = n_input
        self.num_samples = num_samples
        self.stddev = stddev
    
    def neighborhood(self, x):
        num_samples = self.num_samples
        n_input = self.n_input
        with torch.no_grad():
            x_expanded = x.repeat(num_samples, 1)
            dim = torch.randint(0, n_input, (1,))
            noise = torch.randn(num_samples, 1) * self.stddev
            noise_padded = torch.cat([torch.zeros(num_samples, dim), noise, torch.zeros(num_samples, n_input - dim - 1)], 1)
            return x_expanded + noise_padded, dim

    def coefficients(self, x_chosen, y):
        with torch.no_grad():
            x_bar = torch.mean(x_chosen)
            y_bar = torch.mean(y, dim = 0)
            x_delta = x_chosen - x_bar
            y_delta = y - y_bar
            betas = torch.sum(x_delta * y_delta, dim = 0) / torch.sum(x_delta * x_delta)
            ints = y_bar - betas * x_bar
            return betas, ints

    # It may be possible to make this computation more efficient by using broadcasting rather than map_fn
    def forward(self, x):
        with torch.no_grad():
            def compute_mse(x):
                x_local, dim = self.neighborhood(x)
                x_chosen = x_local[:, dim].squeeze()
                x_chosen.requires_grad = False
                y = self.model(x_local)
                betas, ints = self.coefficients(x_chosen, y)
                # print(x_chosen.shape, betas.shape, ints.shape)
                y_lin = x_chosen * betas + ints
                y_prob = torch.sigmoid(y_lin)
                return torch.nn.BCELoss()(y_prob, y.squeeze())
            return torch.mean(torch.stack([compute_mse(x) for x in x]))

class Stability_Regualrizer(nn.Module):
    def __init__(self, model, n_input, num_samples, train_data, stddev = 0.1):
        super(Stability_Regualrizer, self).__init__()
        self.model = model
        self.n_input = n_input
        self.num_samples = num_samples
        self.stddev = stddev
        self.explainer = lime.lime_tabular.LimeTabularExplainer(train_data, discretize_continuous=False)
    
    def neighborhood(self, x):
        num_samples = self.num_samples
        n_input = self.n_input
        with torch.no_grad():
            x_expanded = x.repeat(num_samples, 1)
            constant_term = torch.ones(num_samples, 1)
            noise = torch.randn(num_samples, n_input) * self.stddev
            return x_expanded + noise
    
    # It may be possible to make this computation more efficient by using broadcasting rather than map_fn
    def forward(self, x):
        with torch.no_grad():
            def compute_mse(x):
                x_local = self.neighborhood(x)
                exp_true = self.explainer.explain_instance(x, self.model.predict_proba)
                exp_neighbor = self.explainer.explain_instance(x_local, self.model.predict_proba)
                return torch.nn.functional.mse_loss(exp_true,exp_neighbor)
            return torch.mean(torch.stack([compute_mse(xi) for xi in x]))

