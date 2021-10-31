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
flower_classes = [x for x in Configurations.__call__().get_configuration()['data']['flowers_cls'].split(',')]

# serialisation
serialization_path = Path(Configurations.__call__().get_configuration()['serialisation']['path'])
serialization_filename = Configurations.__call__().get_configuration()['serialisation']['filename']

# logging
log_path = Configurations.__call__().get_configuration()['logging']['path']
logfilename = Configurations.__call__().get_configuration()['logging']['filename']

# cnn network
img_channels = int(Configurations.__call__().get_configuration()['network']['img_channels'])
img_rows = int(Configurations.__call__().get_configuration()['network']['img_rows'])
img_cols = int(Configurations.__call__().get_configuration()['network']['img_cols'])
nb_classes = int(Configurations.__call__().get_configuration()['network']['nb_classes'])
hyper_parameters_for_lr = [float(x) for x in Configurations.__call__().get_configuration()['network']['hyper_parameters_for_lr'].split(',')]