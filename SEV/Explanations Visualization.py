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
from TreeSEV import TreeSEV,tree_to_dict,merge_redundant_leaves,generate_path

# load the dataset
# data = pd.read_csv("../../Data/oasis_mimiciii.csv").dropna()
# X = data[["age","preiculos","gcs","heartrate_min","heartrate_max","meanbp_min","meanbp_max","resprate_min","resprate_max","tempc_min","tempc_max","urineoutput","mechvent","electivesurgery"]]
# y = data["hospital_expire_flag"]
# # y = np.array(y)
# X_neg = X[y==0]
data = pd.read_csv("../../Data/fico.txt")
target = "RiskPerformance"
X = data[[i for i in data.columns if i != target]]
y = data[target]
# y = np.array(y)
X_neg = X[y==0]

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
tree_model = DecisionTreeClassifier(max_depth=5,random_state=42)
tree_model.fit(X_train,y_train)

# evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# build different SEV methods
originalSEV = FlexibleSEV(model,encoder,X.columns,encoded_X_neg,tol=0,k=1)
# build flexible SEV method
flexibleSEV = FlexibleSEV(model,encoder,X.columns,encoded_X_neg,tol=0.2,k=5)
# build cluster SEV method
clusterSEV = ClusterSEV(model,encoder,encoded_X.columns,encoded_X_neg,n_clusters=4,m=3)
# build flexclust SEV method
flexclustSEV = FlexClustSEV(model,encoder,encoded_X.columns,encoded_X_neg,n_clusters=4,m=3,tol=0.2,k=5)
# build the treeSEV method
tree_sev,explanation_lst = TreeSEV(tree_model,X_test)

print(X_train)

tree_dict = tree_to_dict(tree_model,X_test.columns)
tree_dict = merge_redundant_leaves(tree_dict)


X_embedded = clusterSEV.embedding.transform(X_test,encoded_X_neg)

labels = np.argmin([np.linalg.norm(X_embedded - flexclustSEV.cluster_centers_transformed[i],axis=1) for i in range(4)],axis=0)

selected_index = [850]

def transformation(xi):
    # standardize the x_neg and transform back to xi
    xi = xi * X_neg.std(axis=0) + X_neg.mean(axis=0)
    return xi

explanation_df = pd.DataFrame(columns= X_train.columns)

def get_node_influence(tree_dict,path):
    current_node = tree_dict
    for node in path:
        if node == "L":
            current_node = current_node["true"]
        else:
            current_node = current_node["false"]
    return current_node["feature"], current_node["reference"] * X_neg[current_node["feature"]].std() + X_neg[current_node["feature"]].mean()

# load a X_test
X_test = pd.read_csv("Explanations copy.csv").set_index("Unnamed: 0")
X_test = encoder.transform(X_test)
selected_index = [0]

for selected_ind in selected_index:
    xi = X_test.values[selected_ind]
    # add xi to explanation df
    explanation_df = explanation_df.append(pd.Series(transformation(xi),index=X_train.columns),ignore_index=True)
    original_sev_num,original_diff,used = originalSEV.sev_cal(xi,mode="minus")
    # flexible_sev_num,flexible_diff,used = flexibleSEV.sev_cal(xi,mode="minus")
    cluster_sev_num,cluster_diff = clusterSEV.sev_cal(xi,X_embedded[selected_ind],mode="minus")
    # flexclust_sev_num,flexclust_diff,use = flexclustSEV.sev_cal(xi,X_embedded[selected_ind],mode="minus")

    print("The original difference is")
    print(transformation(originalSEV.final_flexible_mean).values[0]*(original_diff[0]!= 0))
    explanation_df = explanation_df.append(pd.Series(transformation(originalSEV.final_flexible_mean).values[0]*(original_diff[0]!= 0),index=X_train.columns),ignore_index=True)

    # print("The flexible difference is")
    # print(transformation(flexibleSEV.final_flexible_mean).values[0]*(flexible_diff[0]!= 0))
    # explanation_df = explanation_df.append(pd.Series(transformation(flexibleSEV.final_flexible_mean).values[0]*(flexible_diff[0]!= 0),index=X_train.columns),ignore_index=True)

    print("The cluster difference is")
    print(transformation(clusterSEV.cluster_labels[labels[selected_ind]]).values[0]*(cluster_diff[0]!= 0))
    explanation_df = explanation_df.append(pd.Series(transformation(clusterSEV.cluster_labels[labels[selected_ind]]).values*(cluster_diff[0]!= 0),index=X_train.columns),ignore_index=True)

    # print("The flexclust difference is")
    # print(transformation(flexclustSEV.cluster_centers_flexible[labels[selected_ind]].values) * (flexclust_diff[0] != 0))
    # explanation_df = explanation_df.append(pd.Series(transformation(flexclustSEV.cluster_centers_flexible[labels[selected_ind]].values).values * (flexclust_diff[0] != 0),index=X_train.columns),ignore_index=True)

    # print("The tree difference is")
    # print(transformation(explanation_lst[selected_ind]).values[0])
    # explanation_df = explanation_df.append(pd.Series(transformation(explanation_lst[selected_ind]).values[0],index=X_train.columns)-xi,ignore_index=True)

    # explanation_df = explanation_df.append(pd.Series(transformation(clusterSEV.cluster_labels[labels[selected_ind]]).values,index=X_train.columns),ignore_index=True)
    # feature,reference = get_node_influence(tree_dict,generate_path(tree_dict,X_test.iloc[selected_ind].to_frame().T)[:-1])
    # print("The feature is ",feature)
    # print("The reference is ",reference)
    # values = pd.DataFrame(np.zeros((1,len(X_train.columns))),columns=X_train.columns)
    # values[feature] = reference
    # explanation_df = explanation_df.append(values,ignore_index=True)


   

print("The explanations are:")
print(explanation_df)
explanation_df.to_csv("Explanations.csv")
    
# # replace 0 as np.nan
# explanation_df = explanation_df.replace(0,np.nan)
# # fill the nan with the index 0
# explanation_df = explanation_df.fillna(explanation_df.iloc[0])

# explanation_df_transformed = encoder.transform(explanation_df)

# # do a feature embedding for explanation
# explanation_embedding = clusterSEV.embedding.transform(explanation_df_transformed,encoded_X_neg)

# # do a overall embedding for explanation
# overall_embedding = clusterSEV.embedding.transform(encoded_X,encoded_X_neg)

# # plot those points
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,6))
# # plt.scatter(overall_embedding[:, 0], overall_embedding[:, 1],s=10,alpha=0.2,c="grey")
# label_lst = ["Query","SEV^1","SEV^F","SEV^C","SEV^C+F","SEV^T"]
# for i in range(len(explanation_embedding)):
#     plt.scatter(explanation_embedding[i,0],explanation_embedding[i,1],s=10,label=label_lst[i])
# plt.legend()
# plt.savefig("overall_embedding.png")



    

