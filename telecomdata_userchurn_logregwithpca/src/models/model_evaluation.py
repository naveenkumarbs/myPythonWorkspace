from sklearn import metrics


def generateAccuracyScore(actual, predicted):
    accuracyScore = "{:2.2f}".format(metrics.accuracy_score(actual, predicted))
    return accuracyScore


def generateROC_AUC_Score(actual, predicted):
    roc_auc_score = "{:2.2f}".format(metrics.roc_auc_score(actual, predicted))
    return roc_auc_score