from sklearn.metrics import roc_auc_score
import keras
import numpy as np

class roc_callback(keras.callbacks.Callback):

    def __init__(self, vg):
        self.validation_generator = vg

    def on_train_begin(self, logs={}):
        logs = {}
        logs['val_auc'] = 0

    def on_epoch_end(self, epoch,logs={}):
        y_p = []
        y_v = []
        for i in range(len(self.validation_generator)):
            x_val, y_val = self.validation_generator[i]
            y_pred = self.model.predict(x_val)
            y_p.append(y_pred)
            y_v.append(y_val)
        y_p = np.concatenate(y_p)
        y_v = np.concatenate(y_v)
        roc_auc = roc_auc_score(y_v, y_p)
        print('\nVal AUC for epoch{}: {}'.format(epoch, roc_auc))
        logs['val_auc'] = roc_auc
