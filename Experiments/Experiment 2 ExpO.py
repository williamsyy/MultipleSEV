# Experiment 2 (Flexible SEV): Use AllOpt^{-1} to optimize the SEV value for each data point in the test set
import sys
sys.path.append("../SEV/")
from Encoder import DataEncoder
from OptimizedExpO import CustomDataset, SimpleLR, SimpleMLP, SimpleGBDT, Regularizer, Regularizer_1D
import pandas as pd
import os
import numpy as np
import torch
from data_loader import data_loader
from sklearn.model_selection import train_test_split
import argparse
from FlexibleSEV import FlexibleSEV
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sev_penalty_lst = [0.01]
# positive_penalty_lst = [0.01,0.1,1,10]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', help='dataset name')
parser.add_argument('--model', type=str, default='lr', help='model name')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--warm_up', type=float, default=0.8, help='warm up ratio')
parser.add_argument('--Type', type=str, default='Fed', help='Type of the regularizer')
parser.add_argument('--sample_size', type=int, default=1000, help='sample size for the regularizer')
parser.add_argument('--iterations', type=int, default=10, help='number of iterations')
args = parser.parse_args()

# load the dataset
X, y, X_neg = data_loader(args.dataset)
print("Working on the dataset {}".format(args.dataset))
# encode the data
encoder = DataEncoder(standard=True)
encoder.fit(X_neg)
encoded_X = encoder.transform(X)
encoded_X_neg = encoder.transform(X_neg)

specific_sev = "../Results/csv/Exp2_expo/Exp2_{}_{}_expo.csv".format(args.dataset,args.model)
overall_results = "../Results/csv/Exp2_expo_summary.csv"

# check if the file exists
file_exists = os.path.isfile(specific_sev)
if file_exists:
    specific_sev_df = pd.read_csv(specific_sev)

for iter in range(args.iterations):
    for sev_penalty in sev_penalty_lst:
        print("Iteration {}, sev_penalty {}".format(iter,sev_penalty))
        if file_exists:
            if "Iteration:"+str(iter) + ",sev_penalty:"+str(sev_penalty) in specific_sev_df.columns:
                continue
        # do a train test split
        X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2,stratify=y, random_state=iter)

        print("The shape of the training set is {}".format(X_train.shape))
        print("The shape of the test set is {}".format(X_test.shape))

        # fit the model
        if args.model == "l2lr":
            if args.dataset == "german":
                base_model = LogisticRegression(solver="liblinear",penalty="l2",C=1e-1)
            else:
                base_model = LogisticRegression(solver='liblinear',penalty='l2',C=1e-2)
        elif args.model == "l1lr":
            if args.dataset == "german":
                base_model = LogisticRegression(solver="liblinear",penalty="l1",C=1e-1)
            else:
                base_model = LogisticRegression(solver='liblinear',penalty='l1',C=1e-2)
        elif args.model == "gbdt":
            base_model = GradientBoostingClassifier(n_estimators=200,max_depth=3, random_state=42)
        elif args.model == "mlp":
            base_model = MLPClassifier(hidden_layer_sizes=(128, 128),random_state=42,early_stopping=True)
        else:
            raise ValueError("The model {} is not supported".format(args.model))
        base_model.fit(X_train,y_train)

        sev = FlexibleSEV(base_model,encoder,encoded_X.columns,encoded_X_neg,tol=0,k=1)

        # create the model
        if args.model == "l2lr" or args.model == "l1lr":
            model = SimpleLR(encoded_X.shape[1])
        elif args.model == "gbdt":
            model = SimpleGBDT(base_model)
        elif args.model == "mlp":
            model = SimpleMLP(encoded_X.shape[1],128)
        else:
            raise ValueError("The model {} is not supported".format(args.model))
        
        # create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
        # create the loss function
        if args.Type == "Fed":
            fed_reg = Regularizer(model,encoded_X.shape[1], args.sample_size, stddev=0.1)
        elif args.Type == "1DFed":
            fed_reg = Regularizer_1D(model,encoded_X.shape[1], args.sample_size, stddev=0.1)
        original_loss = torch.nn.BCELoss()
        # create the dataset
        train_dataset = CustomDataset(X_train.values,y_train.values)
        # create the dataloader
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

        train_start_time = time.time()
        # train the model
        for epoch in tqdm(range(100)):
            for i, (X_batch, y_batch) in enumerate(train_loader):
                # forward pass
                y_pred = model(X_batch).squeeze()
                if epoch < 100 * args.warm_up:
                    loss = original_loss(y_pred, y_batch)
                else:
                    loss = original_loss(y_pred, y_batch) + sev_penalty * fed_reg(X_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()
        # get the training time
        train_time = time.time() - train_start_time
        
        sev = FlexibleSEV(base_model,encoder,encoded_X.columns,encoded_X_neg,tol=0,k=1)

        # sev flexible optimization
        sev_lst = []
        L_inf_lst = []
        time_lst = []
        used_lst = []
        for i in tqdm(range(X_test.shape[0])):
            Xi = X_test.iloc[[i]].values
            if base_model.predict(Xi) == 1:
                start_time = time.time()
                sev_value, diff,used = sev.sev_cal(Xi[0], mode="minus",max_depth=6)
                sev_lst.append(sev_value)
                time_lst.append(time.time()-start_time)
                L_inf_lst.append(max(np.abs(diff)))
                used_lst.append(used)
        
        # get the model performance
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

        print("The training accuracy is {}".format(train_acc))
        print("The test accuracy is {}".format(test_acc))
        print("The training auc is {}".format(train_auc))
        print("The test auc is {}".format(test_auc))
        print("The mean SEV for the test dataset is {}".format(np.mean(sev_lst)))
        print("The mean of the time consumption is {}".format(np.mean(time_lst)))
        print("The mean of the L_inf is {}".format(np.mean(L_inf_lst)))
        print("The portion of used flexibility is {}".format(np.mean(used_lst)))

        # # save the results
        # if file_exists:
        #     specific_sev_df["Iteration:"+str(iter) + ",sev_penalty:"+str(sev_penalty)] = sev_lst
        #     specific_sev_df.to_csv(specific_sev,index=False)
        # else:
        #     df = pd.DataFrame()
        #     df["Iteration:"+str(iter) + ",sev_penalty:"+str(sev_penalty)] = sev_lst
        #     df.to_csv(specific_sev,index=False)

        # # save the overall results
        # if os.path.isfile(overall_results):
        #     overall_df = pd.read_csv(overall_results)
        #     overall_df = overall_df.append({"Dataset":args.dataset,"Model":args.model,"Iteration":iter,"Sev_penalty":sev_penalty,"Train_acc":train_acc,"Test_acc":test_acc,"Train_auc":train_auc,"Test_auc":test_auc,"Mean_SEV":np.mean(sev_lst),"Mean_time":np.mean(time_lst),"Mean_L_inf":np.mean(L_inf_lst),"Mean_used":np.mean(used_lst)},ignore_index=True)
        #     overall_df.to_csv(overall_results,index=False)
        # else:
        #     df = pd.DataFrame()
        #     df = df.append({"Dataset":args.dataset,"Model":args.model,"Iteration":iter,"Sev_penalty":sev_penalty,"Train_acc":train_acc,"Test_acc":test_acc,"Train_auc":train_auc,"Test_auc":test_auc,"Mean_SEV":np.mean(sev_lst),"Mean_time":np.mean(time_lst),"Mean_L_inf":np.mean(L_inf_lst),"Mean_used":np.mean(used_lst)},ignore_index=True)
        #     df.to_csv(overall_results,index=False)
    
