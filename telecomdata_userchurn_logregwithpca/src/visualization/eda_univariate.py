import seaborn as sns
import matplotlib.pyplot as plt

from src.logregpca.config import eda_path
from src.logregpca.loggers import Loggers

logger = Loggers.__call__().get_logger()


def univar_num_plots(df, cols):
    plt.ioff()
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (5, 5)})
    for col in cols:
        logger.info("Univariate analysis for columns: " + col)
        sns.distplot(df[col])
        plt.savefig(eda_path+"/univar_"+col+".png")


def univar_cat_plots(df, cols):
    logger.info("univariate categorical plots")
    sns.reset_orig()
    sns.set(rc={'figure.figsize': (5, 5)})
    for col in cols:
        logger.info("Univariate analysis for columns: " + col)
        sns.countplot(df[col])
        plt.savefig(eda_path+"/univar_"+col+".png")
