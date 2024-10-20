import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

class FuzzyCMeans(BaseEstimator, ClusterMixin):
    def __init__(self, model, n_clusters=3, m=3, error=1e-10, max_iter=1000):
        super().__init__()
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.max_iter = max_iter
        self.model = model

    def fit(self, X, original_X):
        self.n_samples = X.shape[0]
        r = np.random.RandomState(42)
        self.u = r.rand(self.n_samples, self.n_clusters)
        self.u = self.u / self.u.sum(axis=1, keepdims=True)
        predictions = self.model.predict_proba(original_X)[:, 1]
        score = np.array([0.5 if i>0.5 else 0.01 for i in predictions])
        modified_m = (2 * self.m * score + 1).reshape(-1,1)
        for _ in range(self.max_iter):
            u_old = self.u.copy()
            self.centroids = self._update_centroids(X,modified_m)
            self.u = self._update_membership_degrees(X,modified_m)
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.cluster_labels = self.predict(X,original_X)
        return self

    def _update_centroids(self, X,modified_m):
        um = self.u ** modified_m
        return (X.T @ um / np.sum(um, axis=0)).T

    def _update_membership_degrees(self, X,modified_m):
        """
        Update the fuzzy membership degrees matrix
        """
        temp = cdist(X, self.centroids)
        power = (2. / (modified_m- 1)).reshape(-1,1,1)
        # power = (2. / (self.m - 1))
        denominator = np.sum((temp[:, :, None] / temp[:, None, :]) ** (power), axis=-1)
        return 1. / denominator

    def predict(self, X,original_X):
        predictions = self.model.predict_proba(original_X)[:, 1]
        score = np.array([0.5 if i>0.5 else 0.01 for i in predictions])
        modified_m = (2 * self.m * score + 1).reshape(-1,1)
        if hasattr(self, 'centroids'):
            return np.argmax(self._update_membership_degrees(X,modified_m), axis=1)
        else:
            raise AttributeError("Model not yet fitted.")
        

class FuzzyCMeans_base(BaseEstimator, ClusterMixin):
    def __init__(self, model, n_clusters=3, m=3, error=1e-10, max_iter=1000):
        super().__init__()
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.max_iter = max_iter
        self.model = model

    def fit(self, X, original_X):
        self.n_samples = X.shape[0]
        r = np.random.RandomState(42)
        self.u = r.rand(self.n_samples, self.n_clusters)
        self.u = self.u / self.u.sum(axis=1, keepdims=True)
        modified_m = self.m
        for _ in range(self.max_iter):
            u_old = self.u.copy()
            self.centroids = self._update_centroids(X,modified_m)
            self.u = self._update_membership_degrees(X,modified_m)
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.cluster_labels = self.predict(X,original_X)
        return self

    def _update_centroids(self, X,modified_m):
        um = self.u ** modified_m
        # um = self.u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def _update_membership_degrees(self, X,modified_m):
        """
        Update the fuzzy membership degrees matrix
        """
        temp = cdist(X, self.centroids)
        power = 2. / (modified_m- 1)
        denominator = np.sum((temp[:, :, None] / temp[:, None, :]) ** (power), axis=-1)
        return 1. / denominator

    def predict(self, X,original_X):
        modified_m = self.m
        if hasattr(self, 'centroids'):
            return np.argmax(self._update_membership_degrees(X,modified_m), axis=1)
        else:
            raise AttributeError("Model not yet fitted.")



if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    # from data_loader_test import data_loader
    from sklearn.model_selection import train_test_split
    from Encoder import DataEncoder
    import pacmap
    import pandas as pd

    data = pd.read_csv("../../Data/fico.txt")
    target = "RiskPerformance"
    X = data[[i for i in data.columns if i != target]]
    y = data[target]
    # y = np.array(y)
    X_neg = X[y==0]

    # encode the data
    encoder = DataEncoder(standard=True)
    encoder.fit(X_neg)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    encoded_X_train = encoder.transform(X_train).values
    encoded_X_test = encoder.transform(X_test).values
    encoded_X_neg = encoder.transform(X_neg).values
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0,random_state=42)
    X_transformed = embedding.fit_transform(encoded_X_neg)

    from sklearn.ensemble import GradientBoostingClassifier
    lr = GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=42)
    lr = LogisticRegression(penalty="l2",C=1e-2, solver="liblinear")
    lr.fit(encoded_X_train, y_train)
    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    sc= plt.scatter(X_transformed[:, 0], X_transformed[:, 1],c=lr.predict_proba(encoded_X_neg)[:,1],s=10,cmap="RdBu_r")
    plt.colorbar(sc,label="score for each sample")
    # plt.savefig("../../MultipleSEV/cluster/test_%s.png"%data_name)
    # plt.close()
    plt.subplot(1,2,2)
    
    fcm = FuzzyCMeans(model=lr, n_clusters=4, m=1.1)
    # fcm = FuzzyCMeans_base(model=lr, n_clusters=4, m=3)
    fcm.fit(X_transformed, encoded_X_neg)
    labels = fcm.predict(X_transformed,encoded_X_neg)
    print(labels)
    for i in range(4):
        if X_transformed[labels == i].shape[0] == 0:
            continue
        plt.scatter(X_transformed[labels == i, 0], X_transformed[labels == i, 1],alpha=0.2,label="cluster {}".format(i),s=10)
    # plot the centroids
    # plt.scatter(fcm.centroids[:, 0], fcm.centroids[:, 1], marker='*', s=20, c='red',label="centroids")

    cluster_lst = []
    predict_lst = []
    for i in range(4):
        # select the encoded_X_neg that are in the cluster
        real_clusters = np.median(encoded_X_neg[labels==i],axis=0).reshape(1,-1)
        # give a prediction tothe real clusters
        # try:
        print(i,lr.predict_proba(real_clusters)[0,1],encoded_X_neg[labels==i].shape[0])
        predict_lst.append(lr.predict_proba(real_clusters)[0,1])
        cluster_lst.append(real_clusters[0])
        # except:
        #     print("No training data in cluster {}".format(i))

    cluster_arr = np.array(cluster_lst)
    # do a pacmap on the cluster arr
    cluster_transform = embedding.transform(cluster_arr,basis=encoded_X_neg)
    sc2 = plt.scatter(cluster_transform[:, 0], cluster_transform[:, 1], marker='*', s=100, c="blue",label="centroid")
    # ignore the axis
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(sc2,label="centroid predicted label")

    plt.legend()
    plt.savefig("test_fico.png")
    plt.close()