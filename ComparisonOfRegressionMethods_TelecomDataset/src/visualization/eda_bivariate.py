import math

import seaborn as sns
import matplotlib.pyplot as plt

from src.compareregressions.config import eda_path
from src.compareregressions.loggers import Loggers

logger = Loggers.__call__().get_logger()


def draw_boxPlot(df, xcol, ycol):
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(xcol + 'vs'+ ycol )
    sns.boxplot(x = df[xcol], y = df[ycol]);


def bivar_num_plots(df, cols):
    plt.ioff()
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (5, 5)})
    sns.pairplot(df[cols])
    plt.savefig(eda_path + "/bivar_pairplot.png")


def bivar_cat_plots(df, num_cols, cat_cols):
    logger.info("Bivariate categorical plots")
    plt.figure(figsize=(15, 10))
    l = len(cat_cols) * len(num_cols)
    l = math.ceil(math.sqrt(l))
    i = 1
    for ncol in num_cols:
        for ccol1 in cat_cols:
            logger.info("Bivariate analysis for num column: " + ncol + " and cat column " + ccol1)
            plt.subplot(l, l, i)
            draw_boxPlot(df, ncol, ccol1)
            i = i+1
    plt.savefig(eda_path + "/bivar_cat_subplots.png")


def bivar_corr_plot(df, cols, colssuffix=""):
    plt.subplots(figsize=(11, 9))
    with sns.axes_style("white"):
        ax = sns.heatmap(df[cols].corr(), annot=True);
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(eda_path + "/bivar_corr_" + colssuffix + ".png")

