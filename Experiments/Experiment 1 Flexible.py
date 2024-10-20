# Experiment 1 (Flexible SEV): Calculate the basic SEV value for each data point in the test set
import sys
sys.path.append("../SEV/")
import numpy as np
import pandas as pd
from FlexibleSEV import FlexibleSEV
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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', help='dataset name')
parser.add_argument('--model', type=str, default='lr', help='model name')
parser.add_argument('--iterations', type=int, default=10, help='number of iterations')
parser.add_argument('--tolerance', type=float, default=0.2, help='tolerance')
parser.add_argument('--k', type=int, default=5, help='k')

args = parser.parse_args()

# load the dataset
X, y, X_neg = data_loader(args.dataset)
print("Working on the dataset {}".format(args.dataset))
# encode the data
encoder = DataEncoder(standard=True)
if args.dataset == "diabetes":
    encoder.fit(X)
else:
    encoder.fit(X_neg)
encoded_X = encoder.transform(X)
encoded_X_neg = encoder.transform(X_neg)

# specific_sev = "../Results/csv/Exp1_flexible/Exp1_{}_{}_{}_flexible.csv".format(args.dataset,args.model,args.tolerance)
overall_results = "../Results/csv/Exp1_flexible_summary_final.csv"

# check if the file exists
# file_exists = os.path.isfile(specific_sev)
# if file_exists:
#     specific_sev_df = pd.read_csv(specific_sev)

for iter in range(args.iterations):
    # if file_exists:
    #     if "Iteration:"+str(iter) in specific_sev_df.columns:
    #         continue
    # do a train test split
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2,stratify=y, random_state=iter)

    print("The shape of the training set is {}".format(X_train.shape))
    print("The shape of the test set is {}".format(X_test.shape))

    # fit the model
    if args.model == "l2lr":
        if args.dataset == "german":
            model = LogisticRegression(solver="liblinear",penalty="l2",C=1e-1)
        else:
            model = LogisticRegression(solver='liblinear',penalty='l2',C=1e-2)
    elif args.model == "l1lr":
        if args.dataset == "german":
            model = LogisticRegression(solver="liblinear",penalty="l1",C=1e-1)
        else:
            model = LogisticRegression(solver='liblinear',penalty='l1',C=1e-2)
    elif args.model == "gbdt":
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    elif args.model == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128, 128),random_state=42,early_stopping=True)
    else:
        raise ValueError("The model {} is not supported".format(args.model))

    model.fit(X_train,y_train)

    # evaluate the model
    y_pred_train = model.predict_proba(X_train)[:,1]
    y_pred_test = model.predict_proba(X_test)[:,1]

    train_auc = roc_auc_score(y_train,y_pred_train)
    test_auc = roc_auc_score(y_test,y_pred_test)
    train_acc = accuracy_score(y_train,y_pred_train>0.5)
    test_acc = accuracy_score(y_test,y_pred_test>0.5)
    print("The model performances are as follows:")
    print("The train auc is {}".format(train_auc))
    print("The test auc is {}".format(test_auc))
    print("The train accuracy is {}".format(train_acc))
    print("The test accuracy is {}".format(test_acc))

    # build the SEV
    sev = FlexibleSEV(model, encoder, encoded_X.columns,encoded_X_neg,tol=args.tolerance,k=args.k)

    cluster_dictionary = {"adult":7,"german":3,"compas":5,"diabetes":4,"fico":4,"mimic":4,"headline1":3,"headline2":2,"headline3":3,"headline_total":2}
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=cluster_dictionary[args.dataset],random_state=42).fit(encoded_X_neg)

    # generate the explanations:
    flexible_sev = []
    L_inf = []
    time_lst = []
    used_lst = []
    gmm_lst = []
    for ind,xi in enumerate(tqdm(np.array(X_test))):
        if model.predict([xi]) != 1:
            flexible_sev.append(0)
            continue
        start_time = time.time()
        flexible_sev_num,original_diff,used = sev.sev_cal(xi,mode="minus")
        time_lst.append(time.time()-start_time)
        explanations_lst = sev.sev_explain(xi,flexible_sev_num,mode="minus")
        explanations_lst = [explain[0][(explain[0]!= 0) & (explain[0]!= 1) & (explain[0]!=-1)] for explain in explanations_lst]
        flexible_sev.append(flexible_sev_num)
        L_inf.append(np.min([np.max(np.abs(explain)) if len(explain)!= 0 else 0 for explain in explanations_lst]))
        used_lst.append(used)
        gmm_lst.append(gmm.score_samples(xi+original_diff))

    flexible_sev = np.array(flexible_sev)
    L_inf = np.array(L_inf)

    print("The average SEV value for the original SEV method is {}".format(np.mean(flexible_sev[flexible_sev!=0])))
    print("The detailed distribution of SEV is")
    print(pd.Series(flexible_sev).value_counts().sort_index())
    print("The average L_inf value for the original SEV method is {}".format(np.median(L_inf[L_inf!=0])))
    print("The average time for the original SEV method is {}".format(np.mean(time_lst)* 100))
    print("The proportion of using flexible SEV is {}".format(np.mean(used_lst)))
    print()

    # save the results
    if os.path.isfile(overall_results):
        overall_results_df = pd.read_csv(overall_results)
    else:
        overall_results_df = pd.DataFrame(columns=["Dataset","Model","Tolerance","Iteration","Train AUC","Test AUC","Train Accuracy","Test Accuracy","Average SEV","Average L_inf","Average Time","Proportion of using flexible SEV","Mean of GMM"])
        overall_results_df.to_csv(overall_results,index=False)
    overall_results_df.loc[len(overall_results_df)] = [args.dataset,args.model,args.tolerance,iter,train_auc,test_auc,train_acc,test_acc,flexible_sev[flexible_sev!=0].mean(),np.median(L_inf[L_inf!=0]),np.mean(time_lst)* 100,np.mean(used_lst),np.mean(gmm_lst)]
    overall_results_df.to_csv(overall_results,index=False)

    # if file_exists:
    #     specific_sev_df["Iteration:"+str(iter)] = flexible_sev
    # else:
    #     specific_sev_df = pd.DataFrame(flexible_sev,columns=["Iteration:"+str(iter)])
    #     specific_sev_df.to_csv(specific_sev,index=False)
    #     file_exists = True


    