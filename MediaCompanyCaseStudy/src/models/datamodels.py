from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np

def data_model(X, y):
    lm = LinearRegression()
    lm.fit(X, y)
    X = sm.add_constant(X)
    # create a fitted model in one line
    lm_1 = sm.OLS(y, X).fit()
    return lm_1

def cond(i):
    if i % 7 == 5: return 1
    elif i % 7 == 4: return 1
    else :return 0
    return i

def weekend_variable(media):
    media['weekend']=[cond(i) for i in media['day']]
    return media

def lag_variable(media):
    media['Lag_Views'] = np.roll(media['Views_show'], 1)
    media.Lag_Views.replace(108961, 0, inplace=True)
    return media

