from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report
import pandas as pd


def generateAccuracyScore(actual, predicted):
    accuracyScore = "{:2.2f}".format(metrics.accuracy_score(actual, predicted))
    return accuracyScore


def generateROC_AUC_Score(actual, predicted):
    roc_auc_score = "{:2.2f}".format(metrics.roc_auc_score(actual, predicted))
    return roc_auc_score


def evaluate_model(dt_classifier, X_train, y_train, X_test, y_test):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Recall :", recall_score(y_train, dt_classifier.predict(X_train)))
    print("Train Precision :", precision_score(y_train, dt_classifier.predict(X_train)))
    print("Train F1 :", f1_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print(classification_report(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Recall :", recall_score(y_test, dt_classifier.predict(X_test)))
    print("Test Precision :", precision_score(y_test, dt_classifier.predict(X_test)))
    print("Test F1 :", f1_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))
    print(classification_report(y_test, dt_classifier.predict(X_test)))


def evaluate_pca(model_pca_random, df_train_pca, y_train, df_test_pca, y_test):
    print("Train Accuracy :", accuracy_score(y_train, model_pca_random.predict(df_train_pca)))
    print("Train Recall :", recall_score(y_train, model_pca_random.predict(df_train_pca)))
    print("Train Precision :", precision_score(y_train, model_pca_random.predict(df_train_pca)))
    print("Train F1 :", f1_score(y_train, model_pca_random.predict(df_train_pca)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, model_pca_random.predict(df_train_pca)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, model_pca_random.predict(df_test_pca)))
    print("Test Recall :", recall_score(y_test, model_pca_random.predict(df_test_pca)))
    print("Test Precision :", precision_score(y_test, model_pca_random.predict(df_test_pca)))
    print("Test F1 :", f1_score(y_test, model_pca_random.predict(df_test_pca)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, model_pca_random.predict(df_test_pca)))


def updateResults(nextResults, results):
    results = pd.concat([results, nextResults])
    results = results[['Method', 'Test_Accuracy', 'Test_Recall', 'Test_Precision', 'Test_F1']]
    return results

