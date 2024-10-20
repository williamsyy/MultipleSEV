from sklearn.tree import DecisionTreeClassifier
from gosdt.model.threshold_guess import compute_thresholds, cut
import sys
sys.path.append("../SEV/")
import numpy as np
import pandas as pd
from data_loader import data_loader
from Encoder import DataEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
import argparse
import time
import os
from tqdm import tqdm
from treefarms import TREEFARMS
import os, sys
import warnings
from TreeSEV import TreeSEV
warnings.filterwarnings("ignore")
# ignore gosdt's print information 
class ignorePrint: 
    def __enter__(self): 
        self._original_stdout = sys.stdout 
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb): 
        sys.stdout.close() 
        sys.stdout = self._original_stdout
        

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="adult",type=str)
parser.add_argument("--iterations",default=10,type=int)

args = parser.parse_args()

# load the dataset
X, y, X_neg = data_loader(args.dataset)
print("Working on the dataset {}".format(args.dataset))
# encode the data
encoder = DataEncoder(standard=True)
encoder.fit(X_neg)
encoded_X = encoder.transform(X)
encoded_X_neg = encoder.transform(X_neg)

file_name = "../Results/csv/TOpt.csv".format(args.dataset)

performance_lst = []

for iter in range(1):

    cart = DecisionTreeClassifier(max_depth=4,criterion="entropy")
    c45 = DecisionTreeClassifier(max_depth=4,criterion="gini")
    
    # do a train test split
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2,stratify=y, random_state=iter)

    cart.fit(X_train,y_train)
    c45.fit(X_train,y_train)

    # do a threshold guess
    X_guessed, thresholds, header, threshold_guess_time = compute_thresholds(encoded_X.copy(), y, 50,1)
    # do another train test split
    X_train_guessed, X_test_guessed, y_train, y_test = train_test_split(X_guessed, y, test_size=0.2,stratify=y, random_state=iter)

    

    # build the treeFarm on the guessed data
    cart.fit(X_train_guessed,y_train)
    c45.fit(X_train_guessed,y_train)
    
    # train TREEFARMS model
    config = {
        "regularization": 0.02,  # regularization penalizes the tree with more leaves. We recommend to set it to relative high value to find a sparse tree.
        "rashomon_bound_multiplier": 0.01,  # rashomon bound multiplier indicates how large of a Rashomon set would you like to get
    }

    
    # ignore all the warnings for this specific following line

    # models = TREEFARMS(config)
    # models.fit(X_train_guessed, y_train)

    # print("The number of model is {}".format(models.model_set.get_tree_count()))

    # do a model evaluation on the train and test set
    # calculate the probability
    y_prob_train_cart = cart.predict_proba(X_train_guessed)[:,1]
    y_prob_test_cart = cart.predict_proba(X_test_guessed)[:,1]
    y_prob_train_c45 = c45.predict_proba(X_train_guessed)[:,1]
    y_prob_test_c45 = c45.predict_proba(X_test_guessed)[:,1]

    # calculate the metrics
    train_auc_cart = roc_auc_score(y_train,y_prob_train_cart)
    test_auc_cart = roc_auc_score(y_test,y_prob_test_cart)
    train_auc_c45 = roc_auc_score(y_train,y_prob_train_c45)
    test_auc_c45 = roc_auc_score(y_test,y_prob_test_c45)
    train_acc_cart = accuracy_score(y_train,cart.predict(X_train_guessed))
    test_acc_cart = accuracy_score(y_test,cart.predict(X_test_guessed))
    train_acc_c45 = accuracy_score(y_train,c45.predict(X_train_guessed))
    test_acc_c45 = accuracy_score(y_test,c45.predict(X_test_guessed))

    # select the positive instances
    selected_X_test_pos_cart = X_test_guessed[cart.predict(X_test_guessed)==1]
    selected_X_test_pos_c45 = X_test_guessed[c45.predict(X_test_guessed)==1]

    selected_X_test_pos_cart_original = X_test[cart.predict(X_test_guessed)==1]
    selected_X_test_pos_c45_original = X_test[c45.predict(X_test_guessed)==1]


    # calculate the X_test's SEV
    sev_cart,explanation_lst_cart = TreeSEV(cart,selected_X_test_pos_cart,backend="sklearn")
    sev_c45, explanation_lst_c45 = TreeSEV(c45,selected_X_test_pos_c45,backend="sklearn")

    L_inf_cart = []
    for i in range(len(explanation_lst_cart)):
        # select columns name that is not zero
        columns = header[explanation_lst_cart[i].values[0]!=0]
        Xi = selected_X_test_pos_cart_original.iloc[i].copy()
        for column in columns:
            reference,value = column.split("<=")
            value = float(value)
            # replace the original 
            Xi[reference] = value
        # print("L_inf distance is {}".format(np.max(np.abs(Xi-selected_X_test_pos_cart_original.iloc[i]))))
        L_inf_cart.append(np.max(np.abs(Xi-selected_X_test_pos_cart_original.iloc[i])))

    L_inf_c45 = []
    for i in range(len(explanation_lst_c45)):
        # select columns name that is not zero
        columns = header[explanation_lst_c45[i].values[0]!=0]
        Xi = selected_X_test_pos_c45_original.iloc[i].copy()
        for column in columns:
            reference,value = column.split("<=")
            value = float(value)
            # replace the original 
            Xi[reference] = value
        # print("L_inf distance is {}".format(np.max(np.abs(Xi-selected_X_test_pos_cart_original.iloc[i]))))
        L_inf_c45.append(np.max(np.abs(Xi-selected_X_test_pos_c45_original.iloc[i])))


    print("The SEV of CART is {}".format(np.mean(sev_cart)))
    print("The SEV of C45 is {}".format(np.mean(sev_c45)))
    print("The L_inf of CART is {}".format(np.mean(L_inf_cart)))
    print("The L_inf of C45 is {}".format(np.mean(L_inf_c45)))
    print("The Train ACC of CART is {}".format(train_acc_cart))
    print("The Test ACC of CART is {}".format(test_acc_cart))
    print("The Train ACC of C45 is {}".format(train_acc_c45))
    print("The Test ACC of C45 is {}".format(test_acc_c45))

    print("------- Try on the GOSDT Model -------")

    # find the best model in the model set
    all_models = []
    all_sev = []
    all_acc_train = []
    all_acc_test = []
    all_L_inf = []
    total_L_inf = []
    for model_index  in tqdm(range(models.model_set.get_tree_count())):
        model = models[model_index]
        y_train_pred = model.predict(X_train_guessed)
        y_test_pred = model.predict(X_test_guessed)
        selected_X_test_pos_gosdt = X_test_guessed[y_test_pred==1]
        selected_X_test_pos_gosdt_original = X_test[y_test_pred==1]
        sev_lst,explanation_lst_temp = TreeSEV(model,selected_X_test_pos_gosdt,backend="gosdt")
        L_inf_temp = []
        for i in range(len(explanation_lst_temp)):
            # select columns name that is not zero
            columns = header[explanation_lst_temp[i].values[0]!=0]
            Xi = selected_X_test_pos_gosdt_original.iloc[i].copy()
            for column in columns:
                reference,value = column.split("<=")
                value = float(value)
                # replace the original 
                Xi[reference] = value
            # print("L_inf distance is {}".format(np.max(np.abs(Xi-selected_X_test_pos_cart_original.iloc[i]))))
            L_inf_temp.append(np.max(np.abs(Xi-selected_X_test_pos_gosdt_original.iloc[i])))

        if len(sev_lst) == 0:
            continue
        sev = np.mean(np.array(sev_lst))
        all_models.append(model)
        all_sev.append(sev)
        all_L_inf.append(L_inf_temp)
        total_L_inf.append(np.mean(L_inf_temp))
        all_acc_train.append(accuracy_score(y_train,y_train_pred))
        all_acc_test.append(accuracy_score(y_test,y_test_pred))
        
    all_acc_test = np.array(all_acc_test)
    # find the best model with the lowest SEV but with the highest accuracy
    best_model_index = np.argmin([all_sev,-all_acc_test],axis=1)[1]
    best_model = all_models[best_model_index]
    best_sev = all_sev[best_model_index]
    best_L_inf = total_L_inf[best_model_index]

    print("The best SEV is {}".format(best_sev))
    print("The best L_inf is",best_L_inf)
    print("The best Train ACC is {}".format(all_acc_train[best_model_index]))
    print("The best Test ACC is {}".format(all_acc_test[best_model_index]))

    print("The mean SEV is {}".format(np.mean(all_sev)))
    print("The mean L_inf is",np.mean(total_L_inf))
    print("The mean Train ACC is {}".format(np.mean(all_acc_train)))
    print("The mean Test ACC is {}".format(np.mean(all_acc_test)))
    

    # save the results
    performance_lst.append([args.dataset,"CART",np.mean(sev_cart),train_acc_cart,test_acc_cart,train_auc_cart,test_auc_cart,"Original",iter])
    performance_lst.append([args.dataset,"C45",np.mean(sev_c45),train_acc_c45,test_acc_c45,train_auc_c45,test_auc_c45,"Original",iter])
    performance_lst.append([args.dataset,"TreeFarm",best_sev,all_acc_train[best_model_index],all_acc_test[best_model_index],0,0,"GOSDT",iter])
    performance_lst.append([args.dataset,])


    # print("The best SEV is {}".format(np.min(all_sev)))
    # print("The best L_inf is",np.min(total_L_inf))
    # print("The best Train ACC is {}".format(all_acc_train[np.argmin(all_sev)]))
    # print("The best Test ACC is {}".format(all_acc_test[np.argmin(all_sev)]))

    # print("The mean SEV is {}".format(np.mean(all_sev)))
    # print("The mean L_inf is",np.mean(total_L_inf))
    # print("The mean Train ACC is {}".format(np.mean(all_acc_train)))
    # print("The mean Test ACC is {}".format(np.mean(all_acc_test)))

    # # save the results
    # if not os.path.exists(file_name):
    #     df = pd.DataFrame(columns=["Dataset","Model","SEV","Train ACC","Test ACC","Train AUC","Test AUC","Model Type","Iteration"])

    # else:
    #     df = pd.read_csv(file_name)
    
    # df = df.append({"Dataset":args.dataset,"Model":"CART","SEV":np.mean(sev_cart),"Train ACC":train_acc_cart,"Test ACC":test_acc_cart,"Train AUC":train_auc_cart,"Test AUC":test_auc_cart,"Model Type":"Original","Iteration":iter},ignore_index=True)
    # df = df.append({"Dataset":args.dataset,"Model":"C45","SEV":np.mean(sev_c45),"Train ACC":train_acc_c45,"Test ACC":test_acc_c45,"Train AUC":train_auc_c45,"Test AUC":test_auc_c45,"Model Type":"Original","Iteration":iter},ignore_index=True)
    # df = df.append({"Dataset":args.dataset,"Model":"GOSDT","SEV":np.min(all_sev),"Train ACC":all_acc_train[np.argmin(all_sev)],"Test ACC":all_acc_test[np.argmin(all_sev)],"Train AUC":0,"Test AUC":0,"Model Type":"GOSDT","Iteration":iter},ignore_index=True)
    # df.to_csv(file_name,index=False)
    
    


