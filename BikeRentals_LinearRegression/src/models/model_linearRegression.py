import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def generateLinRegTrainPredictions(model, X, y):
    y_pred = model.predict(X)
    y_df = pd.DataFrame(y)
    y_pred.reset_index(drop=True, inplace=True)
    y_df.reset_index(drop=True, inplace=True)
    y_pred_final = pd.concat([y_df, y_pred], axis=1)
    y_pred_final = y_pred_final.rename(columns={0: 'predicted'})
    return y_pred_final


def generateLinRegTestPredictions(model,X,y):
    y_df = pd.DataFrame(y)
    y_pred = []
    for i in range(len(X)):
        predVal = 0
        for col in model.params.index:
            if col != 'const':
                predVal = predVal + model.params[col] * X[col].iloc[i]
        predVal = predVal + model.params['const']
        y_pred.append(predVal)
    y_pred = pd.DataFrame(y_pred)
    y_df.reset_index(drop=True, inplace=True)
    y_pred_final = pd.concat([y_df, y_pred], axis=1)
    y_pred_final = y_pred_final.rename(columns={0: 'predicted'})

    return y_pred_final




def generateLinRegModel(X, y):
    X_lm = generateSMConstant(X)

    # Create a first fitted model
    lr = sm.OLS(y, X_lm).fit()
    return lr, X_lm


def generateSMConstant(X):
    return sm.add_constant(X)


def generateVIF(X):
    vif = pd.DataFrame()
    vif['features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by='VIF', ascending=False)
    return vif
