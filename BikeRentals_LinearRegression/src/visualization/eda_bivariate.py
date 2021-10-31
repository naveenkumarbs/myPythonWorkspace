import seaborn as sns
import matplotlib.pyplot as plt
import math

from src.linreg.config import eda_path
from src.linreg.loggers import Loggers

logger = Loggers.__call__().get_logger()


def bivar_num_plots(df, cols):
    plt.ioff()
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (5, 5)})
    sns.pairplot(df[cols])
    plt.savefig(eda_path + "/bivar_pairplot.png")


def bivar_cat_plots(df, num_cols, cat_cols):
    logger.info("Bivariate categorical plots")
    for ncol in num_cols:
        for ccol1 in cat_cols:
            sns.reset_orig()
            sns.set(rc={'figure.figsize': (5, 5)})
            logger.info("Bivariate analysis for num column: " + ncol + " and cat column " + ccol1)
            sns.boxplot(x=ccol1, y=ncol, data=df)
            plt.savefig(eda_path + "/bivar_" + ncol + "_" + ccol1 + ".png")


def bivar_cat_subplots(df, num_cols, cat_cols):
    plt.figure(figsize=(15, 10))
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (15, 10)})
    l = len(cat_cols) * len(num_cols)
    l = math.ceil(math.sqrt(l))
    i = 1
    for ncol in num_cols:
        for ccol1 in cat_cols:

            logger.info("Bivariate analysis for num column: " + ncol + " and cat column " + ccol1)
            plt.subplot(l, l, i)
            sns.boxplot(x=ccol1, y=ncol, data=df)
            i = i + 1

    plt.savefig(eda_path + "/bivar_cat_subplots.png")
    plt.show()


def bivar_corr_plot(df, suffix=""):
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (20, 20)})
    sns.heatmap(df.corr(), annot=True)
    plt.savefig(eda_path + "/bivar_corr_" + suffix + ".png")
