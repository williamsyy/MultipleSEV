# Experiment 1 (Flexible SEV): Calculate the basic SEV value for each data point in the test set
import sys
sys.path.append("../SEV/")
import numpy as np
import pandas as pd
from ClusterSEV_Credible import ClusterSEV as ClusterCredible
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
parser.add_argument("--dataset",default="adult",type=str)
parser.add_argument("--model",default="l1lr",type=str)
parser.add_argument("--iterations",default=10,type=int)

args = parser.parse_args()

cluster_dictionary = {"adult":7,"german":3,"compas":5,"diabetes":4,"fico":4,"mimic":4,"headline1":3,"headline2":2,"headline3":3,"headline_total":2}
m_dictionary = {"adult":3,"german":1.01,"compas":2,"diabetes":1.01,"fico":3,"mimic":1.01,"headline1":1.01,"headline2":1.01,"headline3":1.01,"headline_total":1.01}


num_clusters = cluster_dictionary[args.dataset]
m = m_dictionary[args.dataset]

# load the dataset
X, y, X_neg = data_loader(args.dataset)
print("Working on the dataset {}".format(args.dataset))
# encode the data
encoder = DataEncoder(standard=True)
encoder.fit(X_neg)
encoded_X = encoder.transform(X)
encoded_X_neg = encoder.transform(X_neg)


# specific_sev = "../Results/csv/Exp1_cluster/Exp1_{}_{}_cluster.csv".format(args.dataset,args.model)
overall_results = "../Results/csv/Exp1_cluster_credible_summary_final.csv"
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
        model = GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=42)
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
    # calculate the SEV values
    sev = ClusterCredible(model,encoder, encoded_X.columns, encoded_X_neg, n_clusters=num_clusters,m=m,threshold=300)
    X_test_emb = sev.embedding.transform(X_test,encoded_X_neg)

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=num_clusters,random_state=42)
    gmm.fit(encoded_X_neg)
    # generate the explanations
    cluster_sev = []
    L_inf = []
    time_lst = []
    gmm_lst = []
    for i in tqdm(range(X_test.shape[0])):
        Xi = X_test.iloc[i].values.reshape(1,-1)
        if model.predict(Xi) != 1:
            cluster_sev.append(0)
            continue
        start = time.time()
        sev_num,diff = sev.sev_cal(Xi,X_test_emb[i].reshape(1,-1),mode="minus")
        time_lst.append(time.time()-start)
        cluster_sev.append(sev_num)
        gmm_lst.append(gmm.score_samples(Xi+diff))
    
    cluster_sev = np.array(cluster_sev)
    L_inf = np.array(L_inf)

    print("The average SEV for the cluster SEV is {}".format(np.mean(cluster_sev[cluster_sev!=0])))
    print("The average L_inf for the cluster SEV is {}".format(np.median(L_inf[L_inf!=0])))
    print("The detailed distribution of SEV is")
    print(pd.Series(cluster_sev).value_counts())
    print("The average time for the cluster SEV is {}".format(np.mean(time_lst)*100))
    print("The average GMM score is {}".format(np.median(gmm_lst)))

    if os.path.isfile(overall_results):
        overall_results_df = pd.read_csv(overall_results)
    else:
        overall_results_df = pd.DataFrame(columns=["Dataset","Model","Iteration","Train AUC","Test AUC","Train Accuracy","Test Accuracy","Average SEV","Median L_inf","Average Time", "Mean Likelihood"])

    overall_results_df.loc[len(overall_results_df)] = [args.dataset,args.model,iter,train_auc,test_auc,train_acc,test_acc,np.mean(cluster_sev[cluster_sev!=0]),np.median(L_inf[L_inf!=0]),np.mean(time_lst)*100, np.median(gmm_lst)]
    overall_results_df.to_csv(overall_results,index=False)

    # if file_exists:
    #     specific_sev_df["Iteration:"+str(iter)] = cluster_sev
    #     specific_sev_df.to_csv(specific_sev,index=False)
    # else:
    #     specific_sev_df = pd.DataFrame()
    #     specific_sev_df["Iteration:"+str(iter)] = cluster_sev
    #     specific_sev_df.to_csv(specific_sev,index=False)
    #     file_exists = True

