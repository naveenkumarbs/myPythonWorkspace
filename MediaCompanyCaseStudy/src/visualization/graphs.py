import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.mediacompanycasestudy.config import fig_path

def day_view_show(media):
    plt.plot(media['day'], media['Views_show'])
    plt.title('Day vs View_show')
    plt.xlabel('day')
    plt.ylabel('Views_show')
    plt.savefig(fig_path+'day_vs_show.png')

def scatter_plot(media):
    colors = (0, 0, 0)
    area = np.pi * 3
    plt.scatter(media.day, media.Views_show, s=area, c=colors, alpha=0.5)
    plt.title('Scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(fig_path+'scatter_plot.png')

def day_viewshow_ad_impression(media):
    fig = plt.figure()
    host = fig.add_subplot(111)

    par1 = host.twinx()
    par2 = host.twinx()

    host.set_xlabel("Day")
    host.set_ylabel("View_Show")
    par1.set_ylabel("Ad_impression")

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(.9)

    p1, = host.plot(media.day, media.Views_show, color=color1, label="View_Show")
    p2, = par1.plot(media.day, media.Ad_impression, color=color2, label="Ad_impression")

    lns = [p1, p2]
    host.legend(handles=lns, loc='best')

    par2.spines['right'].set_position(('outward', 60))

    par2.xaxis.set_ticks([])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    plt.savefig(fig_path+"pyplot_multiple_y-axis.png", bbox_inches='tight')


def corelation_graph(media):
    plt.figure(figsize=(20, 10))  # Size of the figure
    sns.heatmap(media.corr(), annot=True)
    plt.savefig(fig_path+'corelation.png')


def actual_predictions(media, Predicted_views):
    c = [i for i in range(1, 81, 1)]
    fig = plt.figure()
    plt.plot(c, media.Views_show, color="blue", linewidth=2.5, linestyle="-")
    plt.plot(c, Predicted_views, color="red", linewidth=2.5, linestyle="-")
    fig.suptitle('Actual and Predicted', fontsize=20)  # Plot heading
    plt.xlabel('Index', fontsize=18)  # X-label
    plt.ylabel('Views', fontsize=16)  # Y-label
    plt.savefig(fig_path+'actual_predict_lm10.png')

def actual_predict_errors(media, Predicted_views):
    c = [i for i in range(1, 81, 1)]
    fig = plt.figure()
    plt.plot(c, media.Views_show - Predicted_views, color="blue", linewidth=2.5, linestyle="-")
    fig.suptitle('Error Terms', fontsize=20)  # Plot heading
    plt.xlabel('Index', fontsize=18)  # X-label
    plt.ylabel('Views_show-Predicted_views', fontsize=16)  # Y-label
    plt.savefig(fig_path+'actual_predict_lm10_error.png')
