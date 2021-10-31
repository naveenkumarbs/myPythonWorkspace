import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

from src.compareregressions.config import eda_path
from src.compareregressions.loggers import Loggers

logger = Loggers.__call__().get_logger()


def plotCounts(df, col):
    ax = sns.countplot(data = df, y = col )
    total = len(df[col])
    for p in ax.patches:
            percentage = '{:.3f}%'.format(100 * p.get_width()/total)
            x = p.get_x() + p.get_width() + 0.2
            y = p.get_y() + p.get_height()/2
            ax.annotate(percentage, (x, y))


def draw_histPlot(df, col):
    b = np.arange(0,df[col].max()+1,10)
    plt.xlabel(col)
    plt.title(col)
    df[col].plot.hist(bins = b)



def univar_num_plots(df, cols):
    plt.ioff()
    plt.figure(figsize = (15,15))
    l = len(cols)
    l = math.ceil(math.sqrt(l))
    i = 1
    for col in cols:
        logger.info("Univariate analysis for columns: " + col)
        plt.subplot(l, l, i)
        draw_histPlot(df, col)
        i = i + 1
    plt.savefig(eda_path + "/univar_num_subplots.png")


def univar_cat_plots(df, cols):
    logger.info("univariate categorical plots")
    plt.figure(figsize=(15, 12))
    l = len(cols)
    l = math.ceil(math.sqrt(l))
    i = 1
    for col in cols:
        logger.info("Univariate analysis for columns: " + col)
        plt.subplot(l, l, i)
        plotCounts(df, col)
        i = i + 1
    plt.savefig(eda_path+"/univar_cat_subplots.png")
