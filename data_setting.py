import pandas as pd
from sklearn.preprocessing import RobustScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def data_cleaning():
    csv = pd.read_csv('weather.csv')
    data = csv.copy()

    data_categoric = data.select_dtypes(include='object')
    data_numeric = data.select_dtypes(exclude='object')

    simp_cat = SimpleImputer(strategy='most_frequent')
    cat = simp_cat.fit_transform(data_categoric)
    imputed_cat = pd.DataFrame(cat, columns= data_categoric.columns)
    data.drop(data_categoric,axis=1,inplace=True)

    simp_num = SimpleImputer(strategy='median')
    num = simp_num.fit_transform(data_numeric)
    imputed_num = pd.DataFrame(num, columns=data_numeric.columns)
    new_data = pd.concat((imputed_cat,imputed_num),axis=1)

    numbers = new_data.select_dtypes(exclude='object')
    standard = RobustScaler()
    standardize = standard.fit_transform(numbers)
    df_standard = pd.DataFrame(standardize,columns=numbers.columns)


    labeling = new_data.select_dtypes(include='object')
    label = LabelEncoder()

    for i in labeling.columns:
        labeling[i] = label.fit_transform(labeling[i])

    data_final = pd.concat((labeling,df_standard),axis=1)
    data_final['Target'] = data_final['RainTomorrow']
    data_final.drop(['Date','Location','RainTomorrow'],axis=1,inplace=True)

    X = data_final.iloc[:,:-1]
    y = data_final.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test

def transforming():
    X_tr, X_te, y_tra, y_te = data_cleaning()
    X_train = np.array(X_tr)
    X_test = np.array(X_te)
    y_train = np.array(y_tra)
    y_test = np.array(y_te)

    x_train_tensor = torch.tensor(X_train,dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return x_train_tensor,x_test_tensor, y_train_tensor, y_test_tensor


class dataset:
    x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = transforming()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

