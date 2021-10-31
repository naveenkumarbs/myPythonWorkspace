from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


def generateLogRegPredictions(model, X):
    predictedProbabilityOnTest = model.predict(X)
    return predictedProbabilityOnTest


def generateLogRegModel(X_train, y_train):
    param_random = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    learner = LogisticRegression(class_weight = 'balanced')
    random_search_log = RandomizedSearchCV(learner,
                                           param_distributions=param_random,
                                           cv=4, n_jobs=-1, verbose=1, scoring="recall")
    random_search_log.fit(X_train, y_train)
    rf_best_pca_log = random_search_log.best_estimator_
    model = rf_best_pca_log.fit(X_train, y_train)
    return model

