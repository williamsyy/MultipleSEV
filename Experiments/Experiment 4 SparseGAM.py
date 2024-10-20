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

import fastsparsegams

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', help='dataset name')
parser.add_argument('--model', type=str, default='lr', help='model name')
parser.add_argument('--iterations', type=int, default=10, help='number of iterations')
parser.add_argument('--tolerance', type=float, default=0, help='tolerance')
parser.add_argument('--k', type=int, default=1, help='k')

args = parser.parse_args()

# load the dataset
X, y, X_neg = data_loader(args.dataset)
print("Working on the dataset {}".format(args.dataset))
# encode the data
encoder = DataEncoder(standard=True)
encoder.fit(X_neg)
encoded_X = encoder.transform(X)
encoded_X_neg = encoder.transform(X_neg)

# do a train test split
X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2,stratify=y)

model = fastsparsegams.fit(X_train.values, y_train.values.astype(float), penalty="L0", max_support_size=20, loss="Logistic")


# y_pred_train = model.predict_proba(X_train)[:,1]
# y_pred_test = model.predict_proba(X_test)[:,1]

y_pred_train = model.predict(x=X_train.values, lambda_0=model.lambda_0[0][31])
y_pred_test = model.predict(x=X_test.values, lambda_0=model.lambda_0[0][31])

train_acc = accuracy_score(y_train,y_pred_train>0.5)
test_acc = accuracy_score(y_test,y_pred_test>0.5)

print("The model performances are as follows:")
print("The train accuracy is {}".format(train_acc))
print("The test accuracy is {}".format(test_acc))
sev = FlexibleSEV(model, encoder, encoded_X.columns,encoded_X_neg,tol=args.tolerance,k=args.k)

# generate the explanations:
flexible_sev = []
L_inf = []
time_lst = []
used_lst = []
for ind,xi in enumerate(tqdm(np.array(X_test))):
    if model.predict(np.array([xi]),lambda_0=model.lambda_0[0][31]) != 1:
        flexible_sev.append(0)
        continue
    start_time = time.time()
    flexible_sev_num,original_diff,used = sev.sev_cal(xi,mode="minus")
    flexible_sev.append(flexible_sev_num)
    L_inf.append(np.max(np.abs(original_diff)))
    time_lst.append(time.time()-start_time)
    used_lst.append(used)

flexible_sev = np.array(flexible_sev)

print("The average SEV value for the original SEV method is {}".format(np.mean(flexible_sev[flexible_sev!=0])))
print("The detailed distribution of SEV is")
print(pd.Series(flexible_sev).value_counts().sort_index())
print("The average L_inf value for the original SEV method is {}".format(np.mean(L_inf)))
print("The average time for the original SEV method is {}".format(np.mean(time_lst)* 100))
print("The proportion of using flexible SEV is {}".format(np.mean(used_lst)))
