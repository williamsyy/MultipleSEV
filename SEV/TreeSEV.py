# In this coding, we would implement the decision tree classifier for CART and C4.5
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from Encoder import DataEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import tree as _tree
import copy

# transfer the tree into dictionary format and save the left and right children node into the dictionary
def tree_to_dict(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node, depth):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_child = tree_.children_left[node]
            right_child = tree_.children_right[node]
            return {"feature": name,
                    "reference": threshold,
                    "relation": "<=",
                    "true": recurse(left_child, depth + 1),
                    "false": recurse(right_child, depth + 1)}
        else:
            return {"prediction": np.argmax(tree_.value[node])}
    return recurse(0, 1)

def generate_path(tree,sample):
    nodes = [tree]
    path = ""

    while len(nodes) > 0:
        node = nodes.pop(0)
        if 'prediction' in node:
            return path
        else:
            if node["feature"] not in sample.columns:
                value = sample.values[0,node["feature"]]
            else:
                value = sample[node["feature"]].values[0]
            reference = node["reference"]
            if reference == "true":
                reference = True
            if reference == "false":
                reference = False
            if node["relation"] == "==":
                if value == reference:
                    nodes.append(node["true"])
                    path += "L"
                else:
                    nodes.append(node["false"])
                    path += "R"
            elif node["relation"] == ">=":
                if value >= reference:
                    nodes.append(node["true"])
                    path += "L"
                else:
                    nodes.append(node["false"])
                    path += "R"
            elif node["relation"] == "<=":
                if value <= reference:
                    nodes.append(node["true"])
                    path += "L"
                else:
                    nodes.append(node["false"])
                    path += "R"
            elif node["relation"] == ">":
                if value > reference:
                    nodes.append(node["true"])
                    path += "L"
                else:
                    nodes.append(node["false"])
                    path += "R"
            elif node["relation"] == "<":
                if value < reference:
                    nodes.append(node["true"])
                    path += "L"
                else:
                    nodes.append(node["false"])
                    path += "R"
            else:
                raise "Unsupported relational operator {}".format(node["relation"])

def possible_sev_cal(start_node, path, sample):
        sev = 1
        explanation = sample.copy()
        curr_node = start_node.copy()
        for index,step in enumerate(path):
            go_left = True if step == "L" else False
            if index == 0:
                if curr_node["feature"] in sample.columns:
                    explanation[curr_node["feature"]] = curr_node["reference"]
                else:
                    explanation.iloc[0,curr_node["feature"]] = curr_node["reference"]
                if go_left:
                    curr_node = curr_node["true"]
                else:
                    curr_node = curr_node["false"]
                continue
            # check if the sample fits the condition
            try:
                value = sample[curr_node["feature"]].values[0]
            except:
                value = sample.values[0,curr_node["feature"]]
            reference = curr_node["reference"]
            if reference == "true":
                reference = True
            if reference == "false":
                reference = False
            if curr_node["relation"] == "<=":
                if go_left:
                    if value <= reference:
                        curr_node = curr_node["true"]
                    else:
                        if curr_node["feature"] in sample.columns:
                            explanation[curr_node["feature"]] = curr_node["reference"]
                        else:
                            explanation.iloc[0,curr_node["feature"]] = curr_node["reference"]
                        curr_node = curr_node["true"]
                        sev += 1
                else:
                    if value <= reference:
                        if curr_node["feature"] in sample.columns:
                            explanation[curr_node["feature"]] = curr_node["reference"]
                        else:
                            explanation.iloc[0,curr_node["feature"]] = curr_node["reference"]
                        curr_node = curr_node["false"]
                        sev += 1
                    else:
                        curr_node = curr_node["false"]
            elif curr_node["relation"] == "==":
                if go_left:
                    if value == reference:
                        curr_node = curr_node["true"]
                    else:
                        if curr_node["feature"] in sample.columns:
                            explanation[curr_node["feature"]] = curr_node["reference"]
                        else:
                            explanation.iloc[0,curr_node["feature"]] = curr_node["reference"]
                        curr_node = curr_node["true"]
                        sev += 1
                else:
                    if value == reference:
                        if curr_node["feature"] in sample.columns:
                            explanation[curr_node["feature"]] = curr_node["reference"]
                        else:
                            explanation.iloc[0,curr_node["feature"]] = curr_node["reference"]
                        curr_node = curr_node["false"]
                        sev += 1
                    else:
                        curr_node = curr_node["false"]
            else:
                raise ValueError("Unsupported relational operator {}".format(curr_node["relation"]))
            
        if 'prediction' in curr_node:
            assert (curr_node['prediction'] == 0)
            return sev, explanation
        else:
            raise ValueError("The path does not end with a leaf node")
        
        

def TreeSEV(model,X,backend="sklearn"):
    if X.shape[0] == 1:
        X = X.reshape(1,-1)
    if backend == "sklearn":
        tree_dict = tree_to_dict(model, X.columns)
    else:
        tree_dict = model.source

    nodes = [(tree_dict,"")]

    # add the negative path to the list
    negative_path = []

    # list out the node index
    # print("Start to list out the node index")
    tree_index = {"":tree_dict}
    

    while len(nodes) > 0:
        node,path = nodes.pop(0)
        if 'prediction' in node:
            if node['prediction'] == 0:
                negative_path.append(path)
        else:
            tree_index[path+"L"] = node["true"]
            tree_index[path+"R"] = node["false"]
            nodes.append((node['true'],path+"L"))
            nodes.append((node['false'],path+"R"))
    # assign the nodes with the negative path based on the negative path
    negative_path_dict = {}
    for path in negative_path:
        for i in range(len(path)):
            temp_node = path[:i]
            if temp_node not in negative_path_dict:
                negative_path_dict[temp_node] = [a[len(temp_node):] for a in negative_path if a.startswith(temp_node)]

    sev_lst = []
    explanation_lst = []
    # loop through all the samples
    for i in range(X.shape[0]):
    # for i in range(X_test_guessed.shape[0]):
        Xi = X.iloc[[i]]
        if model.predict(Xi)[0] == 1:
            path = generate_path(tree_dict, Xi)
            sev_val = None
            curr_explanation = None
            for index in range(len(path)):
                start_node_index = path[:index]
                # opposite_direction
                opposite_direction = "L" if path[index] == "R" else "R"
                start_node = tree_index[start_node_index]
                if start_node_index not in negative_path_dict:
                    continue
                paths_need_to_be_checked = [a for a in negative_path_dict[start_node_index] if a.startswith(opposite_direction)]
                for temp_path in paths_need_to_be_checked:
                    # print(start_node_index,start_node,path,Xi)
                    curr_sev, explanation = possible_sev_cal(start_node, temp_path, Xi)
                    if sev_val is None:
                        sev_val = curr_sev
                        curr_explanation = explanation
                    elif sev_val > curr_sev:
                        sev_val = curr_sev
                        curr_explanation = explanation
                if sev_val == 1:
                    break
            explanation_lst.append(curr_explanation)
            sev_lst.append(sev_val)
        else:
            sev_lst.append(0)
            explanation_lst.append(0)
            
    return sev_lst, explanation_lst

# get which leaf node the sample belongs to
def node_assigned(model,Xi):
    tree = model.tree_
    node_index = tree.apply(Xi)
    return node_index

# get the prediction of the sample
def predict(model,Xi):
    tree = model.tree_
    node_index = tree.apply(Xi)
    return tree.value[node_index]

# get the decision process of the sample
def decision_path(tree,path):
    node = tree
    features_lst = []
    for node in path:
        if node == "L":
            features_lst.append(tree["feature"]+" "+tree["relation"]+" "+str(tree["reference"]))
        else:
            features_lst.append(tree["feature"]+" "+">"+" "+str(tree["reference"]))
        if node == "L":
            tree = tree["true"]
        else:
            tree = tree["false"]
    return features_lst
    
def merge_redundant_leaves(node):
    # Check if the node is a leaf node (base case for recursion)
    if 'prediction' in node:
        return node

    # Recurse on the 'true' and 'false' branches
    node['true'] = merge_redundant_leaves(node['true'])
    node['false'] = merge_redundant_leaves(node['false'])

    # Check if both branches are leaf nodes and have the same prediction
    if ('prediction' in node['true'] and 'prediction' in node['false'] and 
        node['true']['prediction'] == node['false']['prediction']):
        # Merge the nodes
        return {'prediction': node['true']['prediction']}
    
    return node


if __name__ == "__main__":
    # from treefarms.model.threshold_guess import compute_thresholds, cut
    from gosdt.model.threshold_guess import compute_thresholds, cut
    data = pd.read_csv("../../Data/fico.txt")
    target = "RiskPerformance"
    X = data[[i for i in data.columns if i != target]]
    y = data[target]
    # y = np.array(y)
    X_neg = X[y==0]

    # data = pd.read_csv("../../Data/headline_total.csv").dropna()
    # data.columns = data.columns.astype(str)
    # target = 'y'
    # X = data[[i for i in data.columns if i != target]]
    # y = data[target]
    # for col in X.columns:
    #     if (X[col].nunique() == 2) or (X[col].nunique() > 10):
    #         X[col] = X[col].astype(int)
    #     else:
    #         X[col] = X[col].astype(str)
    # # y = np.array(y)
    # X_neg = X[y==0]

    # data = pd.read_csv("../../Data/oasis_mimiciii.csv").dropna()
    # X = data[["age","preiculos","gcs","heartrate_min","heartrate_max","meanbp_min","meanbp_max","resprate_min","resprate_max","tempc_min","tempc_max","urineoutput","mechvent","electivesurgery"]]
    # y = data["hospital_expire_flag"]
    # # y = np.array(y)
    # X_neg = X[y==0]

    # data = pd.read_csv("../../Data/german.data",header=None,sep="\s+")
    # data.columns = data.columns.astype(str)
    # target = '20'
    # X = data[[i for i in data.columns if i != target]]
    # y = data[target].map({1:0,2:1})
    # # y = np.array(y)
    # X_neg = X[y==0]

    

    # encode the dataset
    encoder = DataEncoder(standard=True)
    encoder.fit(X_neg)
    encoded_X = encoder.transform(X)
    encoded_X_neg = encoder.transform(X_neg)

    # # guess the threshold
    X_guessed,_,_,_ = compute_thresholds(encoded_X.copy(), y,100,1)
    # split the dataset
    X_train,X_test,y_train,y_test = train_test_split(X_guessed,y,test_size=0.2,random_state=42,stratify=y)

    print("The training dataset has {} samples and {} features".format(X_train.shape[0],X_train.shape[1]))
    print("The testing dataset has {} samples and {} features".format(X_test.shape[0],X_test.shape[1]))
    # # train the model
    model = DecisionTreeClassifier(random_state=42,max_depth=4)
    model.fit(X_train,y_train)
    tree_dict = tree_to_dict(model, X_train.columns)
    # # print(print_decision_tree(tree_dict))
    tree_dict = merge_redundant_leaves(tree_dict)

    print("The training accracy is {}".format(accuracy_score(y_train,model.predict(X_train))))
    print("The testing accracy is {}".format(accuracy_score(y_test,model.predict(X_test))))
    print("The training AUC is {}".format(roc_auc_score(y_train,model.predict_proba(X_train)[:,1])))
    print("The testing AUC is {}".format(roc_auc_score(y_test,model.predict_proba(X_test)[:,1])))

    # calculate the SEV
    sev_lst = TreeSEV(model,X_guessed)

    print("The mean SEV is {}".format(np.mean(sev_lst)))
    print("The distribution of SEV is")
    print(pd.Series(sev_lst).value_counts())


    for ind,i in enumerate(sev_lst):
        if i == 2:
            print(ind)
            print(X.iloc[[ind]])
            print(predict(model,X.iloc[[ind]]).values[0])
            # print out the dec
            break