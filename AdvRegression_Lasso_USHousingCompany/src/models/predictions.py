import pandas as pd

def generateLinRegPredictions(model,X,y):
    y_df = pd.DataFrame(y)
    y_pred = []
    model_updated = model.set_index('colName')
    for i in range(len(X)):
        predVal = 0
        for col in model_updated.index[:]:
            predVal = predVal + model_updated.at[col, 'Coef'] * X[col].iloc[i]
        y_pred.append(predVal)
    y_pred = pd.DataFrame(y_pred)
    y_df.reset_index(drop=True, inplace=True)
    y_pred_final = pd.concat([y_df, y_pred], axis=1)
    y_pred_final = y_pred_final.rename(columns={0: 'predicted'})
    return y_pred_final