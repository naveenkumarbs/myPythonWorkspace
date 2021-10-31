from sklearn.linear_model import LogisticRegression


def generateLogRegPredictions(model, X):
    predictedProbabilityOnTest = model.predict_proba(X)
    return predictedProbabilityOnTest


def generateLogRegModel(X_train, y_train):
    learner = LogisticRegression()
    model = learner.fit(X_train, y_train)
    return model

