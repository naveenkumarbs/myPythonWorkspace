import seaborn as sns
import matplotlib.pyplot as plt

from src.logregpca.config import eda_path
from src.logregpca.loggers import Loggers

logger = Loggers.__call__().get_logger()


def bivar_num_plots(df, cols):
    plt.ioff()
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (5, 5)})
    sns.pairplot(df[cols])
    plt.savefig(eda_path + "/bivar_pairplot.png")


def bivar_cat_plots(df, num_cols, cat_cols):
    logger.info("Bivariate categorical plots")
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (5, 5)})
    for ncol in num_cols:
        for ccol1 in cat_cols:
            logger.info("Bivariate analysis for num column: " + ncol + " and cat column " + ccol1)
            sns.stripplot(x=ccol1, y=ncol, data=df, jitter=True, dodge=True)
            plt.savefig(eda_path + "/bivar_" + ncol + "_" + ccol1 + ".png")


def bivar_corr_plot(df, suffix=""):
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (20, 20)})
    sns.heatmap(df.corr(), annot=True)
    plt.savefig(eda_path + "/bivar_corr_" + suffix + ".png")

