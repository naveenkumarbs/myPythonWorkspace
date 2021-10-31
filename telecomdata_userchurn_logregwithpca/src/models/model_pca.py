from sklearn.decomposition import PCA, IncrementalPCA

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.logregpca.config import pca_path


def generatePCA(df, n_comp=None):
    pca = PCA(random_state=42, n_components=n_comp)
    pca.fit(df)
    return pca


def generateScreePlot(pca):
    var_cumu =  np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=[12, 8])
    plt.vlines(x=16, ymax=1, ymin=0, colors="r", linestyles="--")
    plt.hlines(y=0.95, xmax=30, xmin=0, colors="g", linestyles="--")
    plt.plot(var_cumu)
    plt.ylabel("Cumulative variance explained")
    plt.savefig(pca_path + "/screen_plot.png")


def generateIncrementalPCA(df, n_comp=None):
    pca_final = IncrementalPCA(n_components=n_comp)
    df_train_pca = pca_final.fit_transform(df)
    return pca_final, df_train_pca


def generateCorrPlotPCA(df, suffix=""):
    plt.figure(figsize=[15, 15])
    corrmat = np.corrcoef(df.transpose())
    sns.heatmap(corrmat, annot=True)
    plt.savefig(pca_path + "/corr_"+suffix+".png")


def generateTestSetPCA(df, pca):
    result = pca.transform(df)
    return result

