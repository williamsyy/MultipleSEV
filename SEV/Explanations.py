import numpy as np
import pandas as pd
from FlexClustSEV import FlexClustSEV
from FCMCluster import FuzzyCMeans_base, FuzzyCMeans
from ClusterSEV_new import ClusterSEV
from FlexibleSEV import FlexibleSEV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from Encoder import DataEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from TreeSEV import TreeSEV,tree_to_dict,merge_redundant_leaves
from data_loader import data_loader
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--data_name",type=str,default="fico")
args = parse.parse_args()

cluster_dictionary = {"adult":7,"german":3,"compas":5,"diabetes":4,"fico":4,"mimic":4,"headline1":3,"headline2":2,"headline3":3,"headline_total":2}
m_dictionary = {"adult":1.01,"german":1.01,"compas":2,"diabetes":1.01,"fico":3,"mimic":1.01,"headline1":1.01,"headline2":1.01,"headline3":1.01,"headline_total":1.01}

# load the dataset
X,y,X_neg = data_loader(args.data_name)

# encode the data
encoder = DataEncoder(standard=True)
encoder.fit(X_neg)
encoded_X = encoder.transform(X)
encoded_X_neg = encoder.transform(X_neg)

# do a train test split
X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2, random_state=42)

# fit the model
model = LogisticRegression(solver='liblinear',penalty='l2',C=1e-2)
model.fit(X_train,y_train)
tree_model = DecisionTreeClassifier(max_depth=3,random_state=42)
tree_model.fit(X_train,y_train)

# evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# build different SEV methods
originalSEV = FlexibleSEV(model,encoder,encoded_X.columns,encoded_X_neg,tol=0,k=1)
# # build flexible SEV method
flexibleSEV = FlexibleSEV(model,encoder,encoded_X.columns,encoded_X_neg,tol=0.2,k=5)
# build cluster SEV method
clusterSEV = ClusterSEV(model,encoder,encoded_X.columns,encoded_X_neg,n_clusters=cluster_dictionary[args.data_name],m=m_dictionary[args.data_name])
# # build flexclust SEV method
# flexclustSEV = FlexClustSEV(model,encoder,encoded_X.columns,encoded_X_neg,n_clusters=4,m=3,tol=0.2,k=5)
# # build the treeSEV method
tree_sev,explanation_lst = TreeSEV(tree_model,X_test)

def transform_data(X):
    for feature in encoder.original_columns:
        if encoder.columns_types[feature] == "numerical":
            X[feature] = X[feature] * X_neg[feature].std() + X_neg[feature].mean()
    return X

X_embedded = clusterSEV.embedding.transform(X_test,encoded_X_neg)
labels = clusterSEV.cluster.predict(X_embedded,X_test)
final_df = pd.DataFrame(columns=encoded_X.columns)
# generate the explanations:
for ind,xi in enumerate(tqdm(np.array(X_test))):
    if (model.predict([xi]) != 1) or (tree_model.predict([xi]) != 1):
        continue
    output_df = pd.DataFrame([xi],columns=encoded_X.columns)
    final_df = final_df.append(transform_data(output_df))
    original_sev_num,original_diff,used = originalSEV.sev_cal(xi,mode="minus")
    output_df = pd.DataFrame(original_diff+xi,columns=encoded_X.columns)
    final_df = final_df.append(transform_data(output_df))
    flexible_sev_num,flexible_diff,used = flexibleSEV.sev_cal(xi,mode="minus")
    output_df = pd.DataFrame(flexible_diff+xi,columns=encoded_X.columns)
    final_df = final_df.append(transform_data(output_df))
    cluster_sev_num,cluster_diff = clusterSEV.sev_cal(xi,X_embedded[ind],mode="minus")
    output_df = pd.DataFrame(cluster_diff+xi,columns=encoded_X.columns)
    final_df = final_df.append(transform_data(output_df))
    output_df = pd.DataFrame(explanation_lst[ind].values,columns=encoded_X.columns)
    final_df = final_df.append(transform_data(output_df))
    # flexclust_sev_num,flexclust_diff,use = flexclustSEV.sev_cal(xi,X_embedded[ind],mode="minus")
    if cluster_sev_num > original_sev_num and original_sev_num > flexible_sev_num:
        print("The index is ",ind)
        print(original_sev_num,flexible_sev_num,cluster_sev_num,tree_sev[ind])
    


    print(final_df)

final_df.to_csv("final_df_%s.csv"%args.data_name,index=False)




