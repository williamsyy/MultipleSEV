# Experiment 1 (Original SEV): Calculate the basic SEV value for each data point in the test set
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

# specific_sev = "../Results/csv/Exp1_original/Exp1_{}_{}_original.csv".format(args.dataset,args.model)
overall_results = "../Results/csv/Exp1_original_summary_final.csv"
# # check if the file exists
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

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=7)
    gmm.fit(encoded_X_neg)

    # build the original SEV method
    originalSEV = FlexibleSEV(model,encoder,encoded_X.columns,encoded_X_neg,tol=0,k=1)

    cluster_dictionary = {"adult":7,"german":3,"compas":5,"diabetes":4,"fico":4,"mimic":4,"headline1":3,"headline2":2,"headline3":3,"headline_total":2}

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=cluster_dictionary[args.dataset],random_state=42)
    gmm.fit(encoded_X_neg)

    # generate the explanations:
    original_sev = []
    L_inf = []
    time_lst = []
    gmm_lst = []
    for ind,xi in enumerate(tqdm(np.array(X_test[:1000]))):
        if model.predict([xi]) != 1:
            original_sev.append(0)
            continue
        start_time = time.time()
        original_sev_num,original_diff,used = originalSEV.sev_cal(xi,mode="minus")
        time_lst.append(time.time()-start_time)
        explanations_lst = originalSEV.sev_explain(xi,original_sev_num,mode="minus")
        
        # print(original_sev_num)
        # print(explanations_lst)
        explanations_lst = [explain[0][(explain[0]!= 0) & (explain[0]!= 1) & (explain[0]!=-1)] for explain in explanations_lst]
        # for explanation in explanations_lst:
        #     print(explanation)
        L_inf.append(np.min([np.max(np.abs(explain)) if len(explain)!= 0 else 0 for explain in explanations_lst]))
        # print(np.min([np.max(np.abs(explain)) if len(explain)!= 0 else 0 for explain in explanations_lst]))
        original_sev.append(original_sev_num)
        
        gmm_lst.append(gmm.score_samples(xi+original_diff))
    
    original_sev = np.array(original_sev)
    L_inf = np.array(L_inf)

    print("The average SEV value for the original SEV method is {}".format(np.mean(original_sev[original_sev!=0])))
    print("The average L_inf for the original SEV method is {}".format(np.median(L_inf[L_inf != 0])))
    print("The detailed distribution of SEV is")
    print(pd.Series(original_sev).value_counts().sort_index())
    print("The average time for the original SEV method is {}".format(np.mean(time_lst)* 100))
    print("The average GMM score is {}".format(np.median(gmm_lst)))

    # save the results
    if os.path.isfile(overall_results):
        overall_results_df = pd.read_csv(overall_results)
    else:
        overall_results_df = pd.DataFrame(columns=["Dataset","Model","Iteration","Train_Acc","Test_Acc","Train_AUC","Test_AUC","Average SEV","Median L_inf","Average Time", "Average GMM Score"])
    overall_results_df.loc[len(overall_results_df)] = [args.dataset,args.model,iter,train_acc,test_acc,train_auc,test_auc,original_sev[original_sev!=0].mean(),np.mean(L_inf[L_inf != 0]),np.mean(time_lst)* 100, np.mean(gmm_lst)]
    overall_results_df.to_csv(overall_results,index=False)

    # if file_exists:
    #     specific_sev_df["Iteration:"+str(iter)] = original_sev
    #     specific_sev_df.to_csv(specific_sev,index=False)
    # else:
    #     specific_sev_df = pd.DataFrame(original_sev,columns=["Iteration:"+str(iter)])
    #     specific_sev_df.to_csv(specific_sev,index=False)
    #     file_exists = True