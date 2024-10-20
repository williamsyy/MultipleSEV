import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from itertools import product,combinations
import time
import sys
sys.path.append("./SEV")

import numpy as np
import pandas as pd
from pacmap import PaCMAP
from FCMCluster import FuzzyCMeans, FuzzyCMeans_base
from itertools import combinations

class ClusterSEV:
    def __init__(self, model, data_encoder, X_col, X_neg, n_clusters = 3, m=2, strict=None):
        self.model = model
        self.data_encoder = data_encoder
        self.data_mean = {}
        self.data_map = pd.DataFrame(np.zeros((len(self.data_encoder.original_columns),len(X_col))),columns = X_col, index = self.data_encoder.original_columns)
        # the least unit for the SEV version starting from the least output values
        self.least_unit = None
        self.overall_mean = []
        self.choices = list(np.arange(len(self.data_encoder.original_columns)))
        # use the dataencoder to get the mean/median value for each type of features
        for index,feature in enumerate(self.data_encoder.original_columns):
            # for numerical feature, the mean is just the mean
            if self.data_encoder.columns_types[feature] == "numerical":
                self.data_mean[feature] = self.data_encoder.columns_mean[feature]
                self.overall_mean.append(self.data_encoder.columns_mean[feature])
                self.data_map.loc[feature,feature] = 1
            # for binary feature, the mean is the mode of the feature
            elif self.data_encoder.columns_types[feature] == "binary":
                try:
                    self.data_mean[feature] = self.data_encoder.columns_labelencoder[feature].transform(np.array([self.data_encoder.columns_mean[feature]]).reshape(-1,1)).toarray()[0,0]
                    self.overall_mean.append(self.data_encoder.columns_labelencoder[feature].transform(np.array([self.data_encoder.columns_mean[feature]]).reshape(-1,1)).toarray()[0,0])
                except:
                    self.data_mean[feature] = self.data_encoder.columns_mean[feature]
                    self.overall_mean.append(self.data_encoder.columns_mean[feature])
                self.data_map.loc[feature, feature] = 1
            elif self.data_encoder.columns_types[feature] == "category":
                mode_value = self.data_encoder.columns_mean[feature]
                if feature in self.data_encoder.merge_dict.keys():
                    result = self.data_encoder.columns_onehotdecoder[feature].reverse_transform(mode_value)
                    self.data_mean[feature] = result
                    self.data_map.loc[feature, self.data_encoder.merge_dict[feature]] = 1
                    self.overall_mean += list(result)
                else:
                    result = self.data_encoder.columns_labelencoder[feature].transform([[mode_value]]).toarray()[0]
                    self.data_mean[feature] = result
                    self.overall_mean += list(result)
                    cats = [str(feature) + "=" + str(cat) for cat in self.data_encoder.columns_labelencoder[feature].categories_[0]]
                    self.data_map.loc[feature, cats] = 1
            
            
            if (strict is not None) and (feature in strict):
                self.choices.remove(index)
        # save the data map
        self.data_map = np.array(self.data_map)
        # get the overall mean
        self.overall_mean = np.array(self.overall_mean)
        # get the pacmap transformed samples
        self.embedding = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=1, FP_ratio=2.0,random_state=42)
        self.X_transformed = self.embedding.fit_transform(X_neg)

        # cluster the X_transformed
        self.cluster = FuzzyCMeans(model,n_clusters=n_clusters, m = m).fit(self.X_transformed, X_neg.values)
        self.cluster_labels = []
        self.cluster_labels_transformed = []
        # list out the remained cluster index
        self.remain_cluster = []
        self.is_all_negative = True
        for i in range(n_clusters):
            sample_cluster = X_neg[(self.cluster.cluster_labels==i)&(model.predict(X_neg)==0)]
            X_med = sample_cluster.median(axis=0).values
            # print("X_med's shape is",X_med.shape,"The X_median is", X_med)
            X_med_transformed = self.embedding.transform(X_med.reshape(1,-1),X_neg)
            self.cluster_labels.append(X_med)
            self.cluster_labels_transformed.append(X_med_transformed)
            if sample_cluster.shape[0] != 0:
                self.remain_cluster.append(i)
            if self.model is not None:
                if sample_cluster.shape[0] != 0:
                    try:
                        X_med_predict = self.model.predict(X_med.reshape(1,-1))[0]
                    except:
                        X_med_predict = self.model.predict(X_med.reshape(1, -1))
                    if X_med_predict == 1:
                        print("Warning: Cluster {} is positive".format(i), "The probability is", self.model.predict_proba(X_med.reshape(1,-1))[0,1])
                        self.is_all_negative = False
        self.cluster_labels_transformed = np.array(self.cluster_labels_transformed).reshape(n_clusters,-1)
        self.n_clusters = n_clusters
        self.m = m

        self.X_neg = X_neg
        # self.X_trans = X_transformed
        # check if the cluster labels are all negative
        self.result = {}

    # a method for update the cluster in the SEV
    def update_cluster(self,model):
        self.model= model
        self.cluster = FuzzyCMeans(model,n_clusters=self.n_clusters, m = self.m).fit(self.X_transformed, self.X_neg.values)
        self.cluster_labels = []
        self.cluster_labels_transformed = []
        # list out the remained cluster index
        self.remain_cluster = []
        self.is_all_negative = True
        for i in range(self.n_clusters):
            sample_cluster = self.X_neg[(self.cluster.cluster_labels==i)&(model.predict(self.X_neg)==0)]
            X_med = sample_cluster.median(axis=0).values.reshape(1,-1)
            # print("X_med's shape is",X_med.shape,"The X_median is", X_med)
            X_med_transformed = self.embedding.transform(X_med.reshape(1,-1),self.X_neg)
            self.cluster_labels.append(X_med)
            self.cluster_labels_transformed.append(X_med_transformed)
            if sample_cluster.shape[0] != 0:
                self.remain_cluster.append(i)
            if self.model is not None:
                if sample_cluster.shape[0] != 0:
                    try:
                        X_med_predict = self.model.predict(X_med.reshape(1,-1))[0]
                    except:
                        X_med_predict = self.model.predict(X_med.reshape(1, -1))
                    if X_med_predict == 1:
                        print("Warning: Cluster {} is positive".format(i), "The probability is", self.model.predict_proba(X_med.reshape(1,-1))[0,1])
                        self.is_all_negative = False

        self.cluster_labels_transformed = np.array(self.cluster_labels_transformed).reshape(self.n_clusters,-1)


    def transform(self, Xi, conditions,cluster_label):
        """
        This function aims to transfer Xi based on its boolean vector
        :param Xi: a DataFrame row for training dataset
        :param conditions: a boolean vector represents which feature should take the mean
        :return: Xi_temp: the transferred Xi
        """
        remain_columns = conditions.dot(self.data_map)
        Xi_temp = Xi.reshape(1,-1)[0]*remain_columns + self.cluster_labels[cluster_label] *(1-remain_columns)
        return Xi_temp.reshape(1, -1)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sev_cal(self,Xi, X_emb, mode = "plus",max_depth=None):
        """
        Calculate the SEV value for Xi
        :param Xi: a DataFrame row for training dataset
        :param mode {'Mean','NegativeMost', 'Counterfactual'} default: 'Mean': the parameter to control
        what kind of SEV that we would like to calculate, 'Mean' represents to search from (0,0,0) and find
        the shortest path from it to first postive term, 'NegativeMost' represents to search the shortest path
        from the least negative outcomes value node to the first postive term and the 'Counterfactual' means to
        search from (1,1,1) to the first negative term value.
        :return: The selected SEV
        """
        if max_depth is None:
            max_depth = len(self.data_encoder.original_columns)
        choices = self.choices
        # find the closest cluster label
        # cluster_label = np.argmin(np.linalg.norm(Xi-self.cluster_labels,axis=1))
        # X_trans = self.embedding.transform(Xi.reshape(1,-1),self.X_neg.values)

        # cluster_label = self.cluster.predict(X_emb.reshape(1,-1), Xi.reshape(1,-1))[0]
        # print(X_emb.shape)
        # print(self.cluster_labels_transformed.shape)
        cluster_label = np.argmin(np.linalg.norm(X_emb.reshape(1,-1)-self.cluster_labels_transformed,axis=1,ord=2))
        # print(cluster_label)
        # BFS process
        for choice in  range(1,len(self.data_encoder.original_columns)+1):
            if choice > max_depth:
                return choice,X_trans-Xi
            combs = combinations(choices,choice)
            for comb in combs:
                if mode == "plus":
                    pointer = np.zeros(len(self.data_encoder.original_columns))
                elif mode == "minus":
                    pointer = np.ones(len(self.data_encoder.original_columns))
                pointer[np.array(comb)] = 1-pointer[np.array(comb)]
                # print(pointer)
                # try to collect the score from the result dictionary if it is already calculated
                try:
                    score = self.result[tuple(pointer)]
                except:
                    X_trans = self.transform(Xi, pointer,cluster_label)
                    score = self.model.predict_proba(X_trans)[0, 1]
                # for counterfactual the score should be negative
                if mode == "minus":
                    if score < 0.5:
                        return len(comb),X_trans-Xi
                else:
                    if score >= 0.5:
                        return len(comb),X_trans-Xi
        print("Warning: This query is unreachable")
        return len(comb), X_trans - Xi
    
    def sev_explain(self,Xi, X_emb, depth, mode = "plus"):
        choices = self.choices
        cluster_label = np.argmin(np.linalg.norm(X_emb.reshape(1,-1)-self.cluster_labels_transformed,axis=1,ord=2))
        choice = depth
        all_explanations = []
        combs = combinations(choices,choice)
        for comb in combs:
            if mode == "plus":
                pointer = np.zeros(len(self.data_encoder.original_columns))
            elif mode == "minus":
                pointer = np.ones(len(self.data_encoder.original_columns))
            pointer[np.array(comb)] = 1-pointer[np.array(comb)]
            # print(pointer)
            # try to collect the score from the result dictionary if it is already calculated
            try:
                score = self.result[tuple(pointer)]
            except:
                X_trans = self.transform(Xi, pointer,cluster_label)
                score = self.model.predict_proba(X_trans)[0, 1]
            # for counterfactual the score should be negative
            if mode == "minus":
                if score < 0.5:
                    all_explanations.append(X_trans - Xi)
            else:
                if score >= 0.5:
                    all_explanations.append(X_trans - Xi)
        all_explanations = np.array(all_explanations)
        return all_explanations

        

if __name__ == "__main__":
    data = pd.read_csv("../../Data/fico.txt")
    target = "RiskPerformance"
    X = data[[i for i in data.columns if i != target]]
    y = data[target]
    # y = np.array(y)
    X_neg = X[y==0]

    # do a data encoder
    from Encoder import DataEncoder
    encoder = DataEncoder(standard=True)
    encoder.fit(X)
    encoded_X_neg = encoder.transform(X_neg)
    encoded_X = encoder.transform(X)

    # do a train test split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(encoded_X,y,test_size=0.2,stratify=y)

    # load the model
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    model = LogisticRegression(solver="liblinear",C=1e-2)
    # model = MLPClassifier(hidden_layer_sizes=(128,128),early_stopping=True)
    # model = GradientBoostingClassifier(n_estimators=200,max_depth=3)
    model.fit(X_train,y_train)
    # get the accuracy and auc
    from sklearn.metrics import accuracy_score,roc_auc_score
    y_pred = model.predict(X_test)
    print("The accuracy is",accuracy_score(y_test,y_pred))
    print("The auc is",roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))
   

    print ("For clusterSEV SEV:")
    # get the SEV value
    sev = ClusterSEV(model,encoder,encoded_X.columns,encoded_X_neg,n_clusters=3,m=3)
    print("The cluster labels are shown below:")
    print(sev.cluster.cluster_labels)

    # X_neg_med = pd.DataFrame(columns=X_neg.columns)
    # for i in range(3):
    #     print("The cluster {} is".format(i))
    #     # print("The X_neg is",X_neg[sev.cluster.cluster_labels==i].median(axis=0))
    #     X_neg_med.loc[i] = X_neg[sev.cluster.cluster_labels==i].median(axis=0)

    # print(X_neg_med.to_csv("FICO Group.csv"))

    X_emb = sev.embedding.transform(X_test.values,encoded_X_neg)
    sev_lst = []
    flexible_used = []
    diff_lst = []
    for ind,xi in enumerate(tqdm(np.array(X_test))):
        if model.predict([xi]) != 1:
            continue
        sev_num,diff = sev.sev_cal(xi,X_emb[ind],mode="minus")
        sev_lst.append(sev_num)
        diff_lst.append(np.max(diff))

    # report the result
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV is",np.sum(sev_lst)/len(sev_lst))
    print("The mean diff is",np.mean(diff_lst))

        