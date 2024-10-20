import sys
sys.path.append("../SEV/")
import warnings
warnings.filterwarnings("ignore")
from carla import Benchmark
from carla.data.catalog import DataCatalog,CsvCatalog

from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog
from carla import Benchmark
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from data_loader import data_loader
from Encoder import DataEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
import argparse
import time
import os
from tqdm import tqdm
from carla.recourse_methods import GrowingSpheres,CCHVAE,Dice,Revise,Wachter
from SEV import SEV


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="adult",type=str)
parser.add_argument("--iterations",default=1,type=int)

args = parser.parse_args()

# load the dataset
X, y, X_neg = data_loader(args.dataset)
print("Working on the dataset {}".format(args.dataset))
# encode the data
encoder = DataEncoder(standard=True)
numerical_feature = X.columns.tolist()
encoder.fit(X_neg)
encoded_X = encoder.transform(X)
# put one column of random zeros and ones in the encoded_X
encoded_X_neg = encoder.transform(X_neg)

final_df = pd.DataFrame(index=["GS","CCHVAE","Dice","Revise","Wachter"],columns =["L_inf","L_0","Proportion of Unrearchable cases"])

cluster_dictionary = {"adult":7,"german":3,"compas":5,"diabetes":4,"fico":4,"mimic":4,"headline1":3,"headline2":2,"headline3":3,"headline_total":2}

for iter in range(args.iterations):
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2,stratify=y, random_state=iter)

    # fit the model
    lr = LogisticRegression(solver='liblinear',penalty='l2',C=1e-2)
    lr.fit(X_train,y_train)
    sev = SEV(lr,encoder,encoded_X.columns)
    continuous_features= [True if i in numerical_feature else False for i in encoded_X.columns]

    # combine X_train and y_train
    train = pd.concat([X_train,y_train],axis=1)
    train.to_csv("../../Data/encoded/{}_encoded.csv".format(args.dataset),index=False)

    # y is the target (which is a pd.Series), get the series name
    data = CsvCatalog(file_path="../../Data/encoded/{}_encoded.csv".format(args.dataset),
                      target=y.name,continuous=encoded_X.columns.tolist(),categorical=[],immutables=[])


    # create a simple LR model in pytorch
    class SimpleLR(torch.nn.Module):
        def __init__(self):
            super(SimpleLR,self).__init__()
            self.linear = torch.nn.Linear(X_train.shape[1],1)
        def forward(self,x):
            x = np.array(x)
            x = torch.tensor(x)
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred
        def predict(self,x):
            x = np.array(x)
            x = torch.tensor(x)
            return torch.sigmoid(self.linear(x))>0.5
        def predict_proba(self,x):
            x = np.array(x)
            x = torch.tensor(x).float()
            return torch.cat([1-torch.sigmoid(self.linear(x)).float(),torch.sigmoid(self.linear(x)).float()]).detach().numpy().reshape(-1,2)

    # put the bias and weights into the model
    model = SimpleLR().cuda()
    model.linear.weight.data = torch.tensor(lr.coef_).float()
    model.linear.bias.data = torch.tensor(lr.intercept_).float()

    

    from carla import MLModel
    class L2LR(MLModel):
        def __init__(self, data):
            super().__init__(data)
            self._feature_input_order = self.data.continuous
            self._mymodel = model
        @property
        def feature_input_order(self):
            # List of the feature order the ml model was trained on
            return self._feature_input_order
        @property
        def backend(self):
            # The ML framework the model was trained on
            return "pytorch"
        @property
        def raw_model(self):
            # The black-box model object
            return self._mymodel
        # The predict function outputs
        # the continuous prediction of the model
        def predict(self, x):
            return self._mymodel.predict(self.get_ordered_features(x))
        # The predict_proba method outputs
        # the prediction as class probabilities
        def predict_proba(self, x):
            return self._mymodel.predict_proba(self.get_ordered_features(x))
        
    # create a simple LR model in pytorch
    class SimpleLR_cchvae(torch.nn.Module):
        def __init__(self):
            super(SimpleLR_cchvae,self).__init__()
            self.linear = torch.nn.Linear(X_train.shape[1],1)
        def forward(self,x):
            x = torch.tensor(x).float()
            y_pred = torch.sigmoid(self.linear(x))
            return torch.cat([1-y_pred,y_pred]).view(-1,2)
        def predict(self,x):
            x = np.array(x)
            x = torch.tensor(x).float()
            return torch.sigmoid(self.linear(x))>0.5
        def predict_proba(self,x):
            try:
                x = np.array(x)
            except:
                x = torch.tensor(x).float()
            x = torch.tensor(x).float()
            return torch.cat([1-torch.sigmoid(self.linear(x)).float(),torch.sigmoid(self.linear(x)).float()]).view(-1,2)
        
    # put the bias and weights into the model
    model_cchvae = SimpleLR_cchvae().cuda()
    model_cchvae.linear.weight.data = torch.tensor(lr.coef_).float().cuda()
    model_cchvae.linear.bias.data = torch.tensor(lr.intercept_).float().cuda()

    from carla import MLModel
    class L2LR_cchave(MLModel):
        def __init__(self, data):
            super().__init__(data)
            self._feature_input_order = self.data.continuous
            self._mymodel = model_cchvae
        @property
        def feature_input_order(self):
            # List of the feature order the ml model was trained on
            return self._feature_input_order
        @property
        def backend(self):
            # The ML framework the model was trained on
            return "pytorch"
        @property
        def raw_model(self):
            # The black-box model object
            return self._mymodel
        # The predict function outputs
        # the continuous prediction of the model
        def predict(self, x):
            return self._mymodel.predict(self.get_ordered_features(x))
        # The predict_proba method outputs
        # the prediction as class probabilities
        def predict_proba(self, x):
            return self._mymodel.predict_proba(self.get_ordered_features(x))

    explain_model = L2LR(data)
    factual = X_test[lr.predict(X_test) > 0.5]
    gs = GrowingSpheres(explain_model)
    

    explain_model_cchvae = L2LR_cchave(data)
    params = {"data_name":args.dataset,"vae_params": {"layers": [encoded_X.shape[1],128,128],"epochs":10}}
    cchvae = CCHVAE(explain_model_cchvae, hyperparams=params)
    dice = Dice(explain_model,{"desired_class":0})
    revise = Revise(explain_model_cchvae,data,hyperparams=params)
    wachter = Wachter(explain_model_cchvae,hyperparams=params)

    final_L_inf = []
    final_L_0 = []
    final_gmm = []

    # build a GMM model for the negative samples
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=cluster_dictionary[args.dataset],random_state=42)
    gmm.fit(encoded_X_neg)

    GS_lst = []
    DiCE_lst = []
    Revise_lst = []
    Wachter_lst = []


    # for i in tqdm(range(
    for i in tqdm(range(factual.shape[0])):
        gs_facts = gs.get_counterfactuals(factual.iloc[[i]])
        # cchvae_facts = cchvae.get_counterfactuals(factual.iloc[[i]])
        dice_facts = dice.get_counterfactuals(factual.iloc[[i]])
        revise_facts = revise.get_counterfactuals(factual.iloc[[i]])
        wachter_facts = wachter.get_counterfactuals(factual.iloc[[i]])

        # get the L_infty for each method
        L_inf_gs = np.nanmax(np.abs(gs_facts.values-factual.iloc[[i]].values)[:,continuous_features])
        # L_inf_cchvae = np.nanmax(np.abs(cchvae_facts.values-factual.iloc[[i]].values)[:,continuous_features])
        L_inf_dice = np.nanmax(np.abs(dice_facts.values-factual.iloc[[i]].values)[:,continuous_features])
        L_inf_revise = np.nanmax(np.abs(revise_facts.values-factual.iloc[[i]].values)[:,continuous_features])
        L_inf_wachter = np.nanmax(np.abs(wachter_facts.values-factual.iloc[[i]].values)[:,continuous_features])
        # get the L_0 for each methods
        L_0_gs = np.sum(sev.data_map.dot(abs((gs_facts.values-factual.iloc[[i]].values).T))>0)
        # L_0_cchvae = np.sum(sev.data_map.dot(abs((cchvae_facts.values-factual.iloc[[i]].values).T>1e-2))>0)
        L_0_dice = np.sum(sev.data_map.dot(abs((dice_facts.values-factual.iloc[[i]].values).T))>0)
        L_0_revise = np.sum(sev.data_map.dot(abs((revise_facts.values-factual.iloc[[i]].values).T))>0)
        L_0_wachter = np.sum(sev.data_map.dot(abs((wachter_facts.values-factual.iloc[[i]].values).T))>0)
        # labels
        GS_lst.append(sev.data_map.dot(abs((gs_facts.values).T)).reshape(-1))
        DiCE_lst.append(sev.data_map.dot(abs((dice_facts.values).T)).reshape(-1))
        Revise_lst.append(sev.data_map.dot(abs((revise_facts.values).T)).reshape(-1))
        Wachter_lst.append(sev.data_map.dot(abs((wachter_facts.values).T)).reshape(-1))
        

        if gs_facts.isnull().any().any():
            gmm_gs = np.nan
        else:
            gmm_gs = gmm.score_samples(gs_facts.values)[0]
        # if cchvae_facts.isnull().any().any():
        #     gmm_cchvae = np.nan
        # else:
        #     gmm_cchvae = gmm.score_samples(cchvae_facts)[0]
        if dice_facts.isnull().any().any():
            gmm_dice = np.nan
        else:
            gmm_dice = gmm.score_samples(dice_facts.values)[0]
        if revise_facts.isnull().any().any():
            gmm_revise = np.nan
        else:
            gmm_revise = gmm.score_samples(revise_facts.values)[0]
        if wachter_facts.isnull().any().any():
            gmm_wachter = np.nan
        else:
            gmm_wachter = gmm.score_samples(wachter_facts.values)[0]
        L_0_lst = [L_0_gs,L_0_dice,L_0_revise,L_0_wachter]
        L_inf_lst = [L_inf_gs,L_inf_dice,L_inf_revise,L_inf_wachter]
        gmm_lst = [gmm_gs,gmm_dice,gmm_revise,gmm_wachter]
        final_L_inf.append(L_inf_lst)
        final_L_0.append(L_0_lst)
        final_gmm.append(gmm_lst)
    # generate a table of L_0 and L_inf
    final_L_inf = np.array(final_L_inf)
    final_L_0 = np.array(final_L_0)
    final_gmm = np.array(final_gmm)
    result_df = pd.DataFrame(index=["GS","Dice","Revise","Wachter"],columns =["L_inf","L_0","Proportion of Unrearchable cases"])
    result_df["L_inf"] = np.nanmean(final_L_inf,axis=0)
    result_df["L_0"] = np.nanmean(final_L_0,axis=0)
    if pd.isnull(final_gmm).all():
        result_df["gmm"] = np.nan
    else:
        result_df["gmm"] = np.nanmedian(final_gmm,axis=0)
    # count the proportion of nan
    result_df["Proportion of Unrearchable cases"] = np.sum(np.isnan(final_L_inf),axis=0)/final_L_inf.shape[0]

    print(result_df)

    from LocalMAP import LocalMAP
    from LocalMAP import LocalMAP
    embedding = LocalMAP()
    # drop the NaN in each of the list
    GS_lst = [i for i in GS_lst if not np.isnan(i).any()]
    DiCE_lst = [i for i in DiCE_lst if not np.isnan(i).any()]
    Revise_lst = [i for i in Revise_lst if not np.isnan(i).any()]
    Wachter_lst = [i for i in Wachter_lst if not np.isnan(i).any()]
    GS_embed = embedding.fit_transform(np.array(GS_lst))
    embedding = LocalMAP()
    DiCE_embed = embedding.fit_transform(np.array(DiCE_lst))
    embedding = LocalMAP()
    Revise_embed = embedding.fit_transform(np.array(Revise_lst))
    embedding = LocalMAP()
    Wachter_embed = embedding.fit_transform(np.array(Wachter_lst))

    # plot in subplots
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(2,2,1)
    ax1.scatter(GS_embed[:,0],GS_embed[:,1])
    ax1.set_title("GS")
    ax2 = fig.add_subplot(2,2,2)
    ax2.scatter(DiCE_embed[:,0],DiCE_embed[:,1])
    ax2.set_title("DiCE")
    ax3 = fig.add_subplot(2,2,3)
    ax3.scatter(Revise_embed[:,0],Revise_embed[:,1])
    ax3.set_title("Revise")
    ax4 = fig.add_subplot(2,2,4)
    ax4.scatter(Wachter_embed[:,0],Wachter_embed[:,1])
    ax4.set_title("Wachter")
    plt.savefig("SEV_{}_values.png".format(args.dataset))
    plt.close()




