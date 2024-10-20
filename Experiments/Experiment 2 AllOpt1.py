# Experiment 2 (Flexible SEV): Use AllOpt^{-1} to optimize the SEV value for each data point in the test set
import sys
sys.path.append("../SEV/")
from Encoder import DataEncoder
from OptimizedSEVflexible import CustomDataset, SimpleLR, SimpleMLP, SimpleGBDT, AllOptMinus, OriginalLoss
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



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', help='dataset name')
parser.add_argument('--model', type=str, default='lr', help='model name')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--warm_up', type=float, default=0.8, help='warm up ratio')
parser.add_argument('--iterations', type=int, default=10, help='number of iterations')
parser.add_argument('--sev_penalty', type=float, default=0.1, help='sev penalty')
parser.add_argument('--positive_penalty', type=float, default=0, help='positive penalty')
args = parser.parse_args()

sev_penalty_lst = [args.sev_penalty]
positive_penalty_lst = [args.positive_penalty]

# load the dataset
X, y, X_neg = data_loader(args.dataset)
print("Working on the dataset {}".format(args.dataset))
# encode the data
encoder = DataEncoder(standard=True)
encoder.fit(X_neg)
encoded_X = encoder.transform(X)
encoded_X_neg = encoder.transform(X_neg)

# specific_sev = "../Results/csv/Exp2_original_opt/Exp2_{}_{}_original_opt.csv".format(args.dataset,args.model)
overall_results = "../Results/csv/Exp2_original_opt_summary_new_gmm.csv"

cluster_dictionary = {"adult":7,"german":3,"compas":5,"diabetes":4,"fico":4,"mimic":4,"headline1":3,"headline2":2,"headline3":3,"headline_total":2}

# check if the file exists
# file_exists = os.path.isfile(specific_sev)
# if file_exists:
#     specific_sev_df = pd.read_csv(specific_sev)

for iter in range(args.iterations):
    for sev_penalty in sev_penalty_lst:
        for positive_penalty in positive_penalty_lst:
            print("Iteration {}, sev_penalty {}, positive_penalty {}".format(iter,sev_penalty,positive_penalty))
            # if file_exists:
            #     if "Iteration:"+str(iter) + ",sev_penalty:"+str(sev_penalty) + ",positive_penalty:"+str(positive_penalty) in specific_sev_df.columns:
            #         continue
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
                model = SimpleLR(encoded_X.shape[1],sev.data_map,sev.overall_mean,sev.final_flexible_mean,sev_penalty,positive_penalty)
            elif args.model == "gbdt":
                model = SimpleGBDT(base_model,sev.data_map,sev.overall_mean,sev.final_flexible_mean,sev_penalty,positive_penalty)
            elif args.model == "mlp":
                model = SimpleMLP(encoded_X.shape[1],128,sev.data_map,sev.overall_mean,sev.final_flexible_mean,sev_penalty,positive_penalty)
            else:
                raise ValueError("The model {} is not supported".format(args.model))
            
            # create the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
            # create the loss function
            loss_fn = OriginalLoss()
            loss_sev = AllOptMinus(model)
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
                        loss = loss_fn(y_pred, y_batch)
                    else:
                        _, loss, _ = loss_sev(y_pred, y_batch, X_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()
            # get the training time
            train_time = time.time() - train_start_time
            
            sev = FlexibleSEV(model,encoder,encoded_X.columns,encoded_X_neg,tol=0,k=1)

            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=cluster_dictionary[args.dataset],random_state=42).fit(encoded_X_neg)

            # sev flexible optimization
            sev_lst = []
            L_inf_lst = []
            time_lst = []
            used_lst = []
            gmm_lst = []
            for i in tqdm(range(min(X_test.shape[0],1000))):
                Xi = X_test.iloc[[i]].values
                if model.predict(Xi) == 1:
                    start_time = time.time()
                    sev_value, diff,used = sev.sev_cal(Xi, mode="minus",max_depth=6)
                    explanations_lst = sev.sev_explain(Xi,sev_value,mode="minus")
                    explanations_lst = [explain[0][(explain[0]!= 0) & (explain[0]!= 1) & (explain[0]!=-1)] for explain in explanations_lst]
                    L_inf_lst.append(np.min([np.max(np.abs(explain)) if len(explain)!= 0 else 0 for explain in explanations_lst]))
                    sev_lst.append(sev_value)
                    time_lst.append(time.time()-start_time)
                    L_inf_lst.append(np.max(np.abs(diff)))
                    used_lst.append(used)
                    gmm_lst.append(gmm.score_samples(Xi[0]+diff))

            L_inf_lst = np.array(L_inf_lst)

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
            print("The mean of the L_inf is {}".format(np.mean(L_inf_lst[L_inf_lst!=0])))
            print("The portion of used flexibility is {}".format(np.mean(used_lst)))
            print("The mean of GMM score is {}".format(np.mean(gmm_lst)))

            # # save the results
            # if file_exists:
            #     specific_sev_df["Iteration:"+str(iter) + ",sev_penalty:"+str(sev_penalty) + ",positive_penalty:"+str(positive_penalty)] = sev_lst
            #     specific_sev_df.to_csv(specific_sev,index=False)
            # else:
            #     df = pd.DataFrame()
            #     df["Iteration:"+str(iter) + ",sev_penalty:"+str(sev_penalty) + ",positive_penalty:"+str(positive_penalty)] = sev_lst
            #     df.to_csv(specific_sev,index=False)

            # save the overall results
            if os.path.isfile(overall_results):
                overall_df = pd.read_csv(overall_results)
                df_extension = pd.DataFrame({"Dataset":args.dataset,"Model":args.model,"Iteration":iter,"Sev_penalty":sev_penalty,"Positive_penalty":positive_penalty,"Train_acc":train_acc,"Test_acc":test_acc,"Train_auc":train_auc,"Test_auc":test_auc,"Mean_SEV":np.mean(sev_lst),"Mean_time":np.mean(time_lst),"Mean_L_inf":np.mean(L_inf_lst),"Mean_used":np.mean(used_lst), "Mean GMM":np.mean(gmm_lst)},index=[0])
                overall_df = pd.concat([overall_df,df_extension],axis=0)
                overall_df.to_csv(overall_results,index=False)
            else:
                df_extension = pd.DataFrame({"Dataset":args.dataset,"Model":args.model,"Iteration":iter,"Sev_penalty":sev_penalty,"Positive_penalty":positive_penalty,"Train_acc":train_acc,"Test_acc":test_acc,"Train_auc":train_auc,"Test_auc":test_auc,"Mean_SEV":np.mean(sev_lst),"Mean_time":np.mean(time_lst),"Mean_L_inf":np.mean(L_inf_lst),"Mean_used":np.mean(used_lst), "Mean GMM":np.mean(gmm_lst)},index=[0])
                df_extension.to_csv(overall_results,index=False)
    
