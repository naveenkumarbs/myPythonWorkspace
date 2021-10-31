from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic

def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + np.log(n) * num_params
    return bic


def linear_regression(X_train, X_test, y_train, y_test):
    features = X_train.columns

    regression = LinearRegression()

    selected_features = []
    min_aic = np.inf
    for step in range(0, 10, 1):
        for feature in features:
            testing_features = selected_features + [feature]
            regression.fit(X_train[testing_features], y_train)
            y_pred = regression.predict(X_test[testing_features])
            mse = metrics.mean_squared_error(y_test, y_pred)
            num_params = len(regression.coef_) + 1  # features and intercept
            n = X_train.shape[0]
            aic = calculate_aic(n, mse, num_params)
            bic = calculate_bic(n, mse, num_params)
            if aic < min_aic:
                min_aic = aic
                current_bic = bic
                best_feature = feature
        selected_features = selected_features + [best_feature]
        print("Selected features: ", selected_features, "\nAIC: ", min_aic, "\nBIC: ", current_bic)

    y_pred = regression.predict(X_test[selected_features])
    return y_pred


def logistic_regression(X_train, y_train):
    logreg = LogisticRegression()

    from sklearn.feature_selection import RFE
    rfe = RFE(logreg, 50)  # running RFE with 50 variables as output
    rfe = rfe.fit(X_train, y_train)

    list(zip(X_train.columns, rfe.support_, rfe.ranking_))

    col_rfe = X_train.columns[rfe.support_]

    return col_rfe, rfe


def lasso_regression(params, folds):
    lasso = Lasso()

    # cross validation
    model_cv = GridSearchCV(estimator=lasso,
                            param_grid=params,
                            scoring='neg_mean_absolute_error',
                            cv=folds,
                            return_train_score=True,
                            verbose=1)

    return model_cv

def R2_score(lasso, X_train, y_train, X_test, y_test):
    y_train_pred = lasso.predict(X_train)
    train = metrics.r2_score(y_true=y_train, y_pred=y_train_pred)

    y_test_pred = lasso.predict(X_test)
    test=metrics.r2_score(y_true=y_test, y_pred=y_test_pred)
    return train, test