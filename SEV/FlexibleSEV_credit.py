import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from itertools import product,combinations
import time
from sklearn.mixture import GaussianMixture


class FlexibleSEV:
    """
    This is the overall SEV Calculation and the calculation procedure should be divided to two parts
    fit and search, it uses the DataEncoder and the training model as an input and initialize to get
    the map between the columns information and its features used for adjusting the Xi into its mean
    value.
    :param model: The input trained model that have fit() and predict() methods
    :param data_encoder: The fitted DataEncoder for the training dataset
    """
    def __init__(self, model, data_encoder, X_col, X_neg, tol=0.2, k=5, strict=None, overall_mean=None, n_components=2, threshold=0):
        self.model = model
        self.data_encoder = data_encoder
        self.data_mean = {}
        self.data_map = pd.DataFrame(np.zeros((len(self.data_encoder.original_columns),len(X_col))),columns = X_col, index = self.data_encoder.original_columns)
        # the least unit for the SEV version starting from the least output values
        self.least_unit = None
        self.tol = tol
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
            # for the categorical features, suppose it is getting from the one-hot feature,
            # then reverse transform back to the one-hot encoded version, suppose it is getting
            # from the categorical features directly, use the OneEncoder to transform the value
            # into one-hot version
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

        if overall_mean is not None:
            # get the overall mean
            self.overall_mean = np.array(overall_mean)
        else:
            self.overall_mean = np.array(self.overall_mean)


        # flexible mean copied from the overall mean and converted to the pd.DataFrame
        self.flexible_mean = pd.DataFrame(self.overall_mean.reshape(1,-1),columns = X_col)

        # get the final flexible mean
        self.final_flexible_mean = pd.DataFrame(self.flexible_mean.copy())

        # do a quantile transformation for the numerical features
        # collect the quantiles for the numerical features' means
        quantile_dict = {}
        for index,feature in enumerate(self.data_encoder.original_columns):
            if self.data_encoder.columns_types[feature] == "numerical":
                quantile_loc = (X_neg[feature]<X_neg[feature].mean()).mean()
                # get the upper bound of the feature
                quantile_upper = np.quantile(X_neg[feature],min(quantile_loc+tol,1))
                # get the lower bound of the feature
                quantile_lower = np.quantile(X_neg[feature],max(quantile_loc-tol,0))
                # do a linespace to sample 5 points from the lower bound to the upper bound(include)
                quantile_dict[feature] = np.linspace(quantile_lower,quantile_upper,k)
                self.adjusted_mean = []
                # for each value in the quantile_dict[feature], replace the value in the overall_mean with it
                # and calculate the score, then choose the furthest point as the adjusted mean
                for value in quantile_dict[feature]:
                    # copy a new flexible mean
                    flexible_mean_temp = self.flexible_mean.copy()
                    # replace the value in the flexible mean
                    flexible_mean_temp[feature] = value
                    # calculate the score
                    try:
                        score = self.model.predict_proba(flexible_mean_temp)[0,1]
                    except:
                        score = 1
                    # if the score is the smallest, then replace the adjusted mean
                    if (len(self.adjusted_mean) == 0) or (score < self.adjusted_mean[1]):
                        self.adjusted_mean = [value,score]
                # replace the value in the flexible mean
                self.final_flexible_mean[feature] = self.adjusted_mean[0]
        
        self.gmm = GaussianMixture(n_components=n_components,random_state=42).fit(X_neg)
        self.threshold = threshold

        # print("The original overall mean is",self.overall_mean)
        # print("The original flexible mean is",self.final_flexible_mean.values)
        

                
        # save the results
        self.result = {}

    def transform(self, Xi, conditions):
        """
        This function aims to transfer Xi based on its boolean vector
        :param Xi: a DataFrame row for training dataset
        :param conditions: a boolean vector represents which feature should take the mean
        :return: Xi_temp: the transferred Xi
        """
        
        remain_columns = conditions.dot(self.data_map)
        Xi_temp = Xi*remain_columns + self.overall_mean *(1-remain_columns)
        return Xi_temp.reshape(1, -1)
    
    def flexible_transform(self, Xi, conditions):
        """
        This function aims to transfer Xi based on its boolean vector
        :param Xi: a DataFrame row for training dataset
        :param conditions: a boolean vector represents which feature should take the mean
        :return: Xi_temp: the transferred Xi
        """
        remain_columns = conditions.dot(self.data_map)
        Xi_temp = Xi*remain_columns + self.final_flexible_mean.values *(1-remain_columns)
        return Xi_temp.reshape(1, -1)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def calculate_all(self,Xi):
        """
        Calculate all the possible combinations for the result
        :param Xi: a numpy array row in the dataset
        :return: The least value Node in the boolean lattice
        """
        # list out all the possible combinations
        combinations = list(product([0,1],repeat = len(self.data_encoder.original_columns)))
        # initialize the least score
        least_score = None
        for combination in combinations:
            # calculate the score of Xi
            score = self.model.predict_proba(self.transform(Xi,np.array(combination)))[0,1]
            # get the least score and the least score population
            if (least_score is None) or (score < least_score):
                self.least_unit = combination
                least_score = score
            # save the combination result in the dictionary
            self.result[combination]= score
        # return the least unit
        return self.least_unit

    def sev_cal(self,Xi, mode = "plus",max_depth=None):
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

        prev_choice = 1
        # BFS process
        for choice in  range(1,len(self.data_encoder.original_columns)+1):
            if choice > max_depth:
                return choice,Xi-self.final_flexible_mean.values,True
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
                    score = self.model.predict(self.transform(Xi, pointer))[0]
                # for counterfactual the score should be negative
                if mode == "minus":
                    if score < 0.5:
                        if self.gmm.score_samples(self.flexible_transform(Xi, pointer)) > self.threshold:
                            return len(comb),self.transform(Xi, pointer) - Xi,False
                        else:
                            if 'save_explanation' not in locals():
                                save_explanation = self.transform(Xi, pointer) - Xi
                                save_sev = len(comb)
                                save_flexible = False
                else:
                    if score >= 0.5:
                        if self.gmm.score_samples(self.flexible_transform(Xi, pointer)) > self.threshold:
                            return len(comb),self.transform(Xi, pointer) - Xi,False
                        else:
                            if 'save_explanation' not in locals():
                                save_explanation = self.transform(Xi, pointer) - Xi
                                save_sev = len(comb)
                                save_flexible = False
            if prev_choice == choice-1 and self.tol != 0:
                prev_choice = choice
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
                        score = self.model.predict(self.flexible_transform(Xi, pointer))[0]
                    # for counterfactual the score should be negative
                    if mode == "minus":
                        if score < 0.5:
                            if self.gmm.score_samples(self.flexible_transform(Xi, pointer)) > self.threshold:
                                return len(comb),self.flexible_transform(Xi, pointer) - Xi,True
                            else:
                                if 'save_explanation' not in locals():
                                    save_explanation = self.flexible_transform(Xi, pointer)- Xi
                                    save_sev = len(comb)
                                    save_flexible = True
                    else:
                        if score >= 0.5:
                            if self.gmm.score_samples(self.flexible_transform(Xi, pointer)) > self.threshold:
                                return len(comb),self.flexible_transform(Xi, pointer) - Xi,True
                            else:
                                if 'save_explanation' not in locals():
                                    save_explanation = self.flexible_transform(Xi, pointer)-Xi
                                    save_sev = len(comb)
                                    save_flexible = True
        return save_sev,save_explanation, save_flexible
        
    def sev_explain(self,Xi, depth, mode="plus"):
        choice = depth
        explanations = []
        combs = combinations(self.choices,choice)
        for comb in combs:
            flag = False
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
                score = self.model.predict(self.transform(Xi, pointer))[0]
            # for counterfactual the score should be negative
            if mode == "minus":
                if score < 0.5:
                    explanations.append(self.transform(Xi, pointer) - Xi)
                    flag = True
            else:
                if score >= 0.5:
                    explanations.append(self.transform(Xi, pointer) - Xi)
                    flag = True
            if flag == False and self.tol != 0:
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
                    score = self.model.predict(self.flexible_transform(Xi, pointer))[0]
                # for counterfactual the score should be negative
                if mode == "minus":
                    if score < 0.5:
                        explanations.append(self.flexible_transform(Xi, pointer) - Xi)
                else:
                    if score >= 0.5:
                        explanations.append(self.flexible_transform(Xi, pointer) - Xi)
        return explanations

                

def SEVPlot(model,encoder, encoded_data,mode, savefig=None,max_depth = None, strict=[],max_time=14400):
    """
    This function is used to plot the SEV values for encoded
    :param model: The trained model
    :param encoder: The DataEncoder for the training dataset
    :param encoded_data: The fitted one-hot encoded dataset
    :param mode: The hyperparamter in SEV choosing the way of calculating SEV
    :param savefig (default: None): If the savefig is a string, then it would save the figure as
    the string
    :return: The list of SEV values for the encoded data
    """
    y_pred = model.predict(encoded_data)
    print("There are",len(y_pred), "samples in the test dataset,",np.sum(y_pred),"are predicted as positive.")
    sev = SEV(model, encoder, encoded_data.columns,strict=strict)
    sev_lst = []
    count = 0
    start_time = time.time()
    for ind,xi in enumerate(tqdm(np.array(encoded_data))):
        if (max_time is not None) and (time.time() - start_time > max_time):
            print("Time Out!")
            sev_lst.append(0)
            continue
        if y_pred[ind] == 1:
            sev_num = sev.sev_cal(xi, mode=mode,max_depth=max_depth)  # calculate the SEV
            sev_lst.append(sev_num)
            count+=1
        else:
            sev_lst.append(0)
    plt.hist(sev_lst, alpha=0.5)
    if savefig is not None:
        plt.savefig(savefig)
    plt.cla()
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV for",mode,"is",np.sum(sev_lst)/count)
    return sev_lst,count

def SEVCount(model,encoder, encoded_data,mode, savefig=None,max_depth = None,strict=[],unique=False,max_time=14400):
    """
    This function is used to count the SEV values for encoded
    :param model: The trained model
    :param encoder: The DataEncoder for the training dataset
    :param encoded_data: The fitted one-hot encoded dataset
    :param mode: The hyperparamter in SEV choosing the way of calculating SEV
    :param savefig (default: None): If the savefig is a string, then it would save the figure as
    the string
    :return: The list of SEV values for the encoded data
    """
    elements_counts = {}
    y_pred = model.predict(encoded_data)
    print("There are",len(y_pred), "samples in the test dataset,",np.sum(y_pred),"are predicted as positive.")
    sev = SEV(model, encoder, encoded_data.columns,strict=strict)
    sev_lst = []
    count = 0
    start_time = time.time()
    for ind,xi in enumerate(tqdm(np.array(encoded_data))):
        if y_pred[ind] == 1:
            sev_num = sev.sev_cal(xi, mode=mode,max_depth=max_depth)  # calculate the SEV
            sev_lst.append(sev_num)
            features = sev.sev_count(xi,mode=mode,choice=sev_num)
            for feature in features:
                # if feature in strict:
                #     continue
                if unique:
                    if len(feature) != 1:
                        continue
                    if feature not in elements_counts.keys():
                        elements_counts[feature] = 1
                    else:
                        elements_counts[feature] += 1
                else:
                    if feature not in elements_counts.keys():
                        elements_counts[feature] = 1
                    else:
                        elements_counts[feature] += 1
            count+=1
        else:
            sev_lst.append(0)
        if time.time() - start_time > max_time:
            print("Time Out!")
            sev_lst = np.zeros(encoded_data.shape[0])
            break
    plt.hist(sev_lst, alpha=0.5)
    if savefig is not None:
        plt.savefig(savefig)
    plt.cla()
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV for",mode,"is",np.sum(sev_lst)/count)
    print("The count of each features are")
    print(elements_counts)
    return sev_lst

if __name__ == "__main__":
    # # load the data
    # data = pd.read_csv("../../Data/adult.data",header=None)
    # data.columns = data.columns.astype(str)
    # target = '14'
    # X = data[[i for i in data.columns if i != target]]
    # y = data[target].map({" <=50K":0," >50K":1})
    # # y = np.array(y)
    # X_neg = X[y==0]

    data = pd.read_csv("../../Data/fico.txt")
    target = "RiskPerformance"
    X = data[[i for i in data.columns if i != target]]
    y = data[target]
    # y = np.array(y)
    X_neg = X[y==0]

    # encode the data
    from Encoder import DataEncoder
    encoder = DataEncoder(standard=True)
    encoder.fit(X_neg)
    X = encoder.transform(X)
    X_neg = encoder.transform(X_neg)

    # do a train test split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)

    

    # load the model
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    # model = LogisticRegression(solver="liblinear",C=1e-2)
    # model = MLPClassifier(hidden_layer_sizes=(128,128),early_stopping=True)
    model = GradientBoostingClassifier(n_estimators=200,max_depth=3)
    model.fit(X,y)
    # get the accuracy and auc
    from sklearn.metrics import accuracy_score,roc_auc_score
    y_pred = model.predict(X_test)
    print("The accuracy is",accuracy_score(y_test,y_pred))
    print("The auc is",roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))

    # sample 100 from the test dataset
    X_test = X_test.sample(100,random_state=0)

    print ("For non-flexible SEV:")
    # get the SEV value
    sev = FlexibleSEV(model,encoder,X.columns,X_neg,tol=0,k=1)    
    sev_lst = []
    flexible_used = []
    diff_lst = []
    for xi in tqdm(np.array(X_test)):
        sev_num,diff,used = sev.sev_cal(xi,mode="minus")
        sev_lst.append(sev_num)
        flexible_used.append(used)
        diff_lst.append(np.max(diff))

    # report the result
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV is",np.sum(sev_lst)/len(sev_lst))
    print("The flexible used is",np.sum(flexible_used)/len(flexible_used))
    print("The mean diff is",np.mean(diff_lst))

    print ("For flexible SEV with tolerance 0.05:")
    # get the SEV value
    sev = FlexibleSEV(model,encoder,X.columns,X_neg,tol=0.05,k=5)
    sev_lst = []
    flexible_used = []
    diff_lst = []
    for xi in tqdm(np.array(X_test)):
        sev_num,diff,used = sev.sev_cal(xi,mode="minus")
        sev_lst.append(sev_num)
        flexible_used.append(used)
        diff_lst.append(np.max(diff))
    
    # report the result
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV is",np.sum(sev_lst)/len(sev_lst))
    print("The flexible used is",np.sum(flexible_used)/len(flexible_used))
    print("The mean diff is",np.mean(diff_lst))

    print ("For flexible SEV with tolerance 0.1:")
    # get the SEV value
    sev = FlexibleSEV(model,encoder,X.columns,X_neg,tol=0.1,k=5)
    sev_lst = []
    flexible_used = []
    diff_lst = []
    for xi in tqdm(np.array(X_test)):
        sev_num,diff,used = sev.sev_cal(xi,mode="minus")
        sev_lst.append(sev_num)
        flexible_used.append(used)
        diff_lst.append(np.max(diff))
        
    
    # report the result
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV is",np.sum(sev_lst)/len(sev_lst))
    print("The flexible used is",np.sum(flexible_used)/len(flexible_used))
    print("The mean diff is",np.mean(diff_lst))

    print ("For flexible SEV with tolerance 0.2:")
    # get the SEV value
    sev = FlexibleSEV(model,encoder,X.columns,X_neg,tol=0.2,k=5)
    sev_lst = []
    flexible_used = []
    diff_lst = []
    for xi in tqdm(np.array(X_test)):
        sev_num,diff,used = sev.sev_cal(xi,mode="minus")
        sev_lst.append(sev_num)
        flexible_used.append(used)
        diff_lst.append(np.max(diff))
    
    # report the result
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV is",np.sum(sev_lst)/len(sev_lst))
    print("The flexible used is",np.sum(flexible_used)/len(flexible_used))
    print("The mean diff is",np.mean(diff_lst))
    
    