from IPython.display import Image, display
from glob import glob
from src.linreg.config import eda_path


def eda_print_univar_graphs():
    eda_print_graphs('univar_')


def eda_print_bivar_graphs():
    eda_print_graphs('bivar_')


def eda_print_graphs(suffix=""):
    for imageName in glob(eda_path+'/'+suffix+'*.png', recursive=True):
        display(Image(filename=imageName))
        print(imageName)


