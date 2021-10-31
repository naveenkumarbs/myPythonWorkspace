import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def predictions(media, lm_10, X):
    # X = media[['weekend', 'Character_A', 'ad_impression_million']]
    X = sm.add_constant(X)
    Predicted_views = lm_10.predict(X)
    mse = mean_squared_error(media.Views_show, Predicted_views)
    r_squared = r2_score(media.Views_show, Predicted_views)

    return mse, r_squared, Predicted_views

# def predictions_lm6(media, lm_6):
#     X = media[['weekend', 'Character_A', 'Visitors']]
#     X = sm.add_constant(X)
#     Predicted_views = lm_6.predict(X)
#     mse = mean_squared_error(media.Views_show, Predicted_views)
#     r_squared = r2_score(media.Views_show, Predicted_views)
#     return mse, r_squared, Predicted_views