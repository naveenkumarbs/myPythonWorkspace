import seaborn as sns
import matplotlib.pyplot as plt

from src.linreg.config import eda_path
from src.linreg.loggers import Loggers

logger = Loggers.__call__().get_logger()


def univar_num_plots(df, cols):
    plt.ioff()
    for col in cols:
        logger.info("Univariate analysis for columns: " + col)
        sns.reset_orig()
        sns.set(rc={'figure.figsize': (5, 5)})
        sns.distplot(df[col])
        plt.savefig(eda_path+"/univar_"+col+".png")


def univar_cat_plots(df, cols):
    logger.info("univariate categorical plots")

    for col in cols:
        logger.info("Univariate analysis for columns: " + col)
        sns.reset_orig()
        sns.set(rc={'figure.figsize': (5, 5)})
        sns.countplot(df[col])
        plt.savefig(eda_path+"/univar_"+col+".png")
