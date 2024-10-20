import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from warnings import filterwarnings
filterwarnings("ignore")

class OneHotDecoder:
    """
    This class is a Decoder to transform the One-hot Encoded Features to numerical values
    The idea of this class is to use two mappings between one-hot encoding and numerical values
    and it is used mostly for the transform between population mean and its one-hot encoded version
    """
    def __init__(self):
        self.one_hot_mapping = None # from one-hot tuple to encoded value
        self.one_hot_mapping_reverse = None # from encoded value to the one-hot tuple
        self.column_name = None # save the columns that needs to be encoded
    def fit(self, X):
        index_X = X.groupby(list(X.columns)).count().index # get the one-hot encoded columns
        # create the mapping
        self.one_hot_mapping = dict(zip(index_X,np.arange(len(index_X))))
        self.one_hot_mapping_reverse = dict(zip(np.arange(len(index_X)),index_X))
        # save the column name for future transform
        self.column_name = X.columns
    def transform(self, X):
        # check if the one-hot encoding is fitted
        if self.one_hot_mapping is None:
            raise ValueError("Haven't fitted yet!")
        # save the processed X in a list
        output = []
        for Xi in np.array(X[self.column_name]):
            try:
                output.append(self.one_hot_mapping[tuple(Xi)]) # map the one-hot value to numerical
            except:
                output.append(-1)
        return output
    def reverse_transform(self,value):
        return self.one_hot_mapping_reverse[value] # map the numerical value to one-hot

class DataEncoder:
    """
    This class enables us to input a dataset transform the features into correct version and
    save its mean value
    - Numerical Features: No extra pre-processing needed, save Mean value as the population mean
    - Binary Features: The Standard of Binary Features is the number of unique value in this feature
    equals to 2, it would encode the feature into one column using OnehotEncoder(drop="ifbinary"), \
    and save the mode value as the population mean
    - Categorical Features: The Standard of Binary Features is the number of unique value n in this features
    greater than 2 and the column type is object (if you want some ordinal value be processed in one-hot
    version, you can convert the column type of that feature into string), it would encode the feature
    into n columns for n unique values and the mode value as the population mean.
    - One-hot encoded Features: In some case, the input dataset is already under one-hot encoded version,
    thus, it is hard to get the population mean in this case, the encoder can input a dictionary with the
    merged feature name as the key and the categories in the features as a list. It would use a One-hot
    decoder to transform the one-hot value to a numerical one and then encode it just as the categorical
    features.
    """
    def __init__(self, standard=False):
        # initialization
        self.original_columns = [] # column names after the one-hot features are merged
        self.columns_types = {} # column types for each featurs in the original columns
        self.columns_labelencoder = {} # map for all OnehotEncoders of categorical features
        self.columns_mean = {} # map for all features and its population mean
        self.merge_dict = {} # map for one-encoded features that need to be merged
        self.columns_onehotdecoder = {} # the one-hot decoder of one-hot encoded features in raw data

        self.standard = standard
        if self.standard:
            self.numerical_standard_encoder = {} # dictionary of standard encoder

    def fit(self,df,categorical_dict ={}):
        """
        The fitting function in the DataEncoder can be divided into three main steps:
        1. Merge the one-hot encoded features into features in corresponding to based on OnehotDecoder
        2. Label each features based on their unique values and column types and use the OneHotEncoder
        to encode all the features
        3.
        :param df: The input dataset
        :param categorical_dict: The map between the merged features names and the one-hot
        encoded feature name
        :return: The dataset with merged features based on the One-hot Decoder
        """
        # print("Start to merge the features...")
        self.merge_dict = categorical_dict
        for key,values in self.merge_dict.items():
            # check if the select features are one-hot encoded
            # if df.groupby(values).count().shape[0] != len(values):
            #     raise ValueError("This is not a one-hot vector!")
            self.columns_onehotdecoder[key] = OneHotDecoder()
            self.columns_onehotdecoder[key].fit(df[values]) # fit the one-hot decoder fot data

        # merge the columns representing the same feature
        df_merged = df.copy()
        for key, values in self.merge_dict.items():
            df_merged[key] = self.columns_onehotdecoder[key].transform(df[values])
            # delete the original dataset
            for value in values:
                del df_merged[value]

        # label each columns based on its value details are mentioned in description
        for data_col in df_merged.columns:
            if df_merged[data_col].nunique() == 2:
                self.columns_types[data_col] = "binary"
            elif (df_merged[data_col].dtype == object) and (df_merged[data_col].nunique() > 2):
                self.columns_types[data_col] = "category"
            elif df_merged[data_col].nunique() == 1:
                self.columns_types[data_col] = "constant"
            elif data_col in self.merge_dict.keys():
                self.columns_types[data_col] = "category"
            else:
                self.columns_types[data_col] = "numerical"
        # do the label encoding for the different types of features
        for data_col in df_merged.columns:
            if (self.columns_types[data_col] == "binary"):
                # fit the columns by One-Hot Encoder for binary features
                self.columns_labelencoder[data_col] = OneHotEncoder(drop="if_binary")
                self.columns_labelencoder[data_col].fit(np.array(df_merged[data_col]).reshape(-1, 1))
                df_merged[data_col] = self.columns_labelencoder[data_col].transform(np.array(df_merged[data_col]).reshape(-1, 1)).toarray()
            elif (self.columns_types[data_col] == "category"):
                # fit the columns by One-Hot Encoder for categorical features
                self.columns_labelencoder[data_col] = OneHotEncoder()
                self.columns_labelencoder[data_col].fit(np.array(df_merged[data_col]).reshape(-1, 1))
            elif (self.columns_types[data_col] == "numerical"):
                if self.standard:
                    self.numerical_standard_encoder[data_col] = StandardScaler()
                    self.numerical_standard_encoder[data_col].fit(np.array(df_merged[data_col]).reshape(-1, 1))
                    df_merged[data_col] = self.numerical_standard_encoder[data_col].transform(np.array(df_merged[data_col]).reshape(-1, 1))
            elif self.columns_types[data_col] == "constant":
                # ignore the constant columns
                continue
            self.original_columns.append(data_col)

        # print("The features are merged! The remaining features are",self.original_columns)

        # print("Start calculating the mean of the features...")

        # calculate the population mean for each features
        for feature,type in self.columns_types.items():
            if type == "category":
                self.columns_mean[feature] = np.array(df_merged[feature].mode())[0]
            elif type == "binary":
                self.columns_mean[feature] = np.array(df_merged[feature].mode())[0]
            elif type == "numerical":
                self.columns_mean[feature] = 0
        # print("Successfuly get the mean of the features.")
        return df_merged

    def transform(self,df):
        # first step: transform the dataset to the merged form
        df_output = df.copy() # copy the dataset to be a newer version for processing
        # the difference between merge version and the original version is the merge of one-hot
        # encoded features

        # second step: do the label encoding for categorical and binary features
        for feature, type in self.columns_types.items():
            if type == "numerical":
                if self.standard:
                    df_output[feature] = self.numerical_standard_encoder[feature].transform(np.array(df_output[feature]).reshape(-1,1))
                else:
                    df_output[feature] = df_output[feature]
            elif type == "category":
                if feature in self.merge_dict.keys():
                    continue
                for cate_type in self.columns_labelencoder[feature].categories_[0]:
                    df_output[str(feature)+"="+str(cate_type)] = (df_output[feature] == cate_type)+0
                del df_output[feature]
            elif type == "binary":
                df_output[feature] = self.columns_labelencoder[feature].transform(np.array(df[feature]).reshape(-1, 1)).toarray().astype(int)

        return df_output
        
if __name__ == "__main__":
    from data_loader_test import data_loader

    X,y,X_neg = data_loader("headline_total")

    merge_dict = {"ethnity":["ethnicity_1","ethnicity_2","ethnicity_3","ethnicity_4","ethnicity_5","ethnicity_6"]}


    encoder = DataEncoder(standard=True)
    encoder.fit(X_neg,merge_dict)
    encoded_X = encoder.transform(X)
    print(X.columns)
    print(encoded_X.columns)
    # encoded_X_neg = encoder.transform(X_neg)
    # print(encoded_X_neg.head())
    print(encoded_X.head())