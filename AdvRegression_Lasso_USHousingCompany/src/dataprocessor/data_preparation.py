import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def numeric_catagoric(housing, cols):
    numeric_cols = housing[housing.columns[housing.columns.isin(cols)]].columns
    categoric_cols = housing.columns[~housing.columns.isin(numeric_cols)]
    return numeric_cols, categoric_cols

def dummy_variables(housing, categoric_cols):
    for col in categoric_cols:
        d2 = pd.get_dummies(housing[col], prefix=col, drop_first=True)
        # Adding the results to the master dataframe
        housing = pd.concat([housing, d2], axis=1)

        # Dropping the repeated variables
        housing = housing.drop([col], axis=1)

    return housing

def select_types(X):
    num_cols = X.select_dtypes(include=['float64', 'int64'])
    cat_cols = X.select_dtypes(include=['object'])
    return num_cols, cat_cols

def find_scalar(housing, num_cols):
    scaler = StandardScaler()

    housing[num_cols.columns] = scaler.fit_transform(housing[num_cols.columns])
    return housing

def train_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
    return X_train, X_test, y_train, y_test