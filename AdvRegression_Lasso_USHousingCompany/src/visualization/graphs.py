import matplotlib.pyplot as plt
import seaborn as sns
from src.ushousingcompany.config import fig_path


def saleprice(df, val):
    plt.figure(figsize=[10,10])
    sns.distplot(df[val])
    plt.savefig(fig_path+'saleprice.png')


def univariate_graph(df, cols):
    plt.ioff()
    fig = plt.figure(figsize=(20, 80))
    fig.tight_layout()
    i = 1
    for col in cols:
        plt.subplot(13, 3, i)
        i = i + 1
        plt.title(col)
        # sns.distplot(housing[col])
        plt.hist(df[col], alpha=0.5)

    plt.savefig(fig_path+'univariate.png')


def pairplot_graph(housing, numeric_cols):
    sns.pairplot(housing[numeric_cols])
    plt.interactive(False)
    sns_plot = sns.pairplot(housing[numeric_cols], size=2.0)
    sns_plot.savefig(fig_path+"pairplot.png")
    # return 'pairplot.png'

def univariate_cate_graph(housing, numeric_cols):
    fig = plt.figure(figsize=(20, 80))
    fig.tight_layout()
    i = 1
    for col in housing.columns[~housing.columns.isin(numeric_cols)]:
        plt.subplot(13, 4, i)
        i = i + 1
        plt.title(col)
        sns.countplot(x=col, data=housing)

    plt.savefig(fig_path + 'univariate_cate.png')


def univariate_analysis_graph(housing, numeric_cols):
    fig = plt.figure(figsize=(20, 80))
    fig.tight_layout()
    i = 1
    for col in housing.columns[~housing.columns.isin(numeric_cols)]:
        plt.subplot(13, 4, i)
        i = i + 1
        plt.title(col)
        sns.countplot(x=col, data=housing)
    plt.savefig(fig_path + 'univariate_analysis.png')

def bivariate_analysis(housing):
    fig = plt.figure(figsize=(50, 50))
    vg_corr = housing.corr()
    sns.heatmap(vg_corr,
                xticklabels=vg_corr.columns.values,
                yticklabels=vg_corr.columns.values,
                annot=True);
    plt.savefig(fig_path + 'bivariate_analysis.png')


def plot_train_test(cv_results):
    cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

    # plotting
    plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
    plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
    plt.xlabel('alpha')
    plt.ylabel('Negative Mean Absolute Error')

    plt.title("Negative Mean Absolute Error and alpha")
    plt.legend(['train score', 'test score'], loc='upper left')
    plt.savefig(fig_path+'lasso_mean_train_test.png')