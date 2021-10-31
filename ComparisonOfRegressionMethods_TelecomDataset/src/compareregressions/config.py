import configparser
from pathlib import Path


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# python 3 style
class Configurations(object, metaclass=SingletonType):
    # __metaclass__ = SingletonType   # python 2 Style
    conf = None

    def __init__(self):
        self.conf = configparser.ConfigParser()

        # below path is set with respect to the notebook
        self.conf.read('../src/configurations.ini')

    def get_configuration(self):
        return self.conf

    def get_data_path_external(self):
        return self.conf['data']['path_external']


# paths are wrt notebook location

# data
data_dir_external = Path(Configurations.__call__().get_configuration()['data']['path_external'])
filename = Configurations.__call__().get_configuration()['data']['filename']

# serialisation
serialization_path = Path(Configurations.__call__().get_configuration()['serialisation']['path'])
serialization_filename = Configurations.__call__().get_configuration()['serialisation']['filename']
serialization_pca_filename = Configurations.__call__().get_configuration()['serialisation']['pca_filename']

# logging
log_path = Configurations.__call__().get_configuration()['logging']['path']
logfilename = Configurations.__call__().get_configuration()['logging']['filename']

# eda reports
eda_path = Configurations.__call__().get_configuration()['reports']['eda_path']
pca_path = Configurations.__call__().get_configuration()['reports']['pca_path']
