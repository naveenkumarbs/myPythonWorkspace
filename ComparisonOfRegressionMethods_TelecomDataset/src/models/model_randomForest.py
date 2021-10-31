
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def getBestEstimatorForRandomForest(params, X_train, y_train):
    classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    random_search_rf = RandomizedSearchCV(estimator=classifier_rf, param_distributions=params,
                                          cv=4, n_jobs=-1, verbose=1, scoring="recall")
    random_search_rf.fit(X_train, y_train)
    rf_best = random_search_rf.best_estimator_
    rf_best_params = random_search_rf.best_params_
    return rf_best, rf_best_params
