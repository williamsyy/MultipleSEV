import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../../Data/parkinsons_updrs.data')
y = data['total_UPDRS']
x = data.drop(['total_UPDRS', 'motor_UPDRS'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# set up the model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge

models = [DecisionTreeRegressor(max_depth=None), RandomForestRegressor(max_depth=None,n_estimators=200), GradientBoostingRegressor(n_estimators=200,max_depth=3), LinearRegression(), Lasso(), Ridge()]
model_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Linear Regression', 'Lasso', 'Ridge']


from sklearn.metrics import mean_squared_error

# for i, model in enumerate(models):
#     model.fit(x_train, y_train)
#     print(model_names[i])
#     print('Train Score:', mean_squared_error(y_train, model.predict(x_train)))
#     print('Test Score:', mean_squared_error(y_test, model.predict(x_test)))
#     print()

class RegressionWrapper:
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds
    def fit(self, x, y):
        self.model.fit(x, y)
    def predict(self, x):
        output = self.model.predict(x)
        # check if the output is within the thresholds
        return (output < self.thresholds[0]) | (output > self.thresholds[1])
    def predict_proba(self, x):
        output = self.model.predict(x)
        return np.array([1 - output, output]).T
    
# model = RandomForestRegressor(max_depth=None,n_estimators=200)
# model.fit(x_train, y_train)
# wrapper = RegressionWrapper(model, [0, 15])

import sys
sys.path.append("../SEV/")
import numpy as np
import pandas as pd
from FlexibleSEV import FlexibleSEV
from Encoder import DataEncoder

thresholds = [0,15]

y = data['total_UPDRS']
x = data.drop(['total_UPDRS', 'motor_UPDRS'], axis=1)
x_neg = x[(y > thresholds[0]) * (y < thresholds[1])]

encoder = DataEncoder(standard=True)
encoder.fit(x_neg)
encoded_x = encoder.transform(x)
encoded_X_neg = encoder.transform(x_neg)

# do a Gaussian Mixture Model to get the negative data
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=10, random_state=42)
gmm.fit(encoded_X_neg)
scores = -gmm.score_samples(encoded_X_neg)
selected_x_neg = x_neg[scores < np.percentile(scores, 0.1)]


# do a train_test_split
X_train, X_test, y_train, y_test = train_test_split(encoded_x, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(max_depth=3,n_estimators=200, random_state=42)
# model = RandomForestRegressor(max_depth=None,n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# check the performance
from sklearn.metrics import mean_squared_error, r2_score
print('Train Score:', mean_squared_error(y_train, model.predict(X_train)))
print("Train R2:", r2_score(y_train, model.predict(X_train)))
print('Test Score:', mean_squared_error(y_test, model.predict(X_test)))
print("Test R2:", r2_score(y_test, model.predict(X_test)))


wrapper = RegressionWrapper(model, thresholds)

# build the SEV
sev = FlexibleSEV(wrapper, encoder, encoded_x.columns,encoded_X_neg,tol=0,k=1)

print("Baseline prediction is",model.predict([sev.overall_mean]))
print("New Baseline prediction is", model.predict([selected_x_neg.mean(axis=0)]))


from tqdm import tqdm
import time
# generate the explanations:
flexible_sev = []
L_inf = []
time_lst = []
used_lst = []
for ind,xi in enumerate(tqdm(np.array(X_test))):
    if wrapper.predict([xi]) != 1:
        flexible_sev.append(0)
        continue
    start_time = time.time()
    flexible_sev_num,original_diff,used = sev.sev_cal(xi,mode="minus")
    flexible_sev.append(flexible_sev_num)
    L_inf.append(np.max(np.abs(original_diff)))
    time_lst.append(time.time()-start_time)
    used_lst.append(used)

flexible_sev = np.array(flexible_sev)

print("The average SEV value for the original SEV method is {}".format(np.mean(flexible_sev[flexible_sev!=0])))
print("The detailed distribution of SEV is")
print(pd.Series(flexible_sev).value_counts().sort_index())
print("The average L_inf value for the original SEV method is {}".format(np.mean(L_inf)))
print("The average time for the original SEV method is {}".format(np.mean(time_lst)* 100))
print("The proportion of using flexible SEV is {}".format(np.mean(used_lst)))

# do a pytorch version of linear regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleGBDT(nn.Module):
    # this is the simple GBDT model, which input a Gradient Boosting Classifier module in sklearn and creates a torch modules with the same structure. All the optimization is done for the weights of each tree created by the sklearn GBDT model.
    def __init__(self, base_model,data_map, data_mean, flexible_mean, sev_penalty, positive_penalty):
        super(SimpleGBDT, self).__init__()
        # create the datamap for the dataset
        self.data_map = torch.tensor(np.array(data_map),dtype=torch.float32)
        self.data_mean = torch.tensor(np.array(data_mean),dtype=torch.float32).view(-1,1)
        self.data_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
        self.flexible_mean = torch.tensor(np.array(flexible_mean),dtype=torch.float32).view(-1,1)
        self.flexible_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)
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
    
    def update_quantile(self,quantile_mean):
        self.flexible_mean = torch.tensor(np.array(quantile_mean),dtype=torch.float32).view(-1,1)
        self.flexible_mean_map = self.data_mean.view(-1) * torch.ones(self.data_map.shape)


    

    
