from keras import optimizers
import keras
import matplotlib.pyplot as plt
import numpy as np

from src.cnn import resnet
from src.cnn.config import img_cols, img_rows, img_channels, nb_classes, flower_classes
from src.dataprocessor.DataGenerator import DataGenerator
from src.dataprocessor.AugumentedDataGenerator import AugmentedDataGenerator
from src.models.model_LossHistory import LossHistory
from src.models.model_decayLR import DecayLR
from src.models.model_metrics import roc_callback


# python 3 style
class ModelBuilder(object):
    model = None
    lr = None
    epochs = None

    def __init__(self, lr=0.1, epochs=3):
        self.lr = lr
        self.epochs = epochs
        self.model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
        sgd = optimizers.SGD(lr=self.lr, clipnorm=1.)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,
                           metrics=['accuracy'])

    def get_model(self):
        if self.model is None:
            self.model = ModelBuilder()
        return self.model

    def run_ablation(self, ablation=100, epochs_count=1):
        # using resnet 18
        # create data generator objects in train and val mode
        # specify ablation=number of data points to train on
        training_generator = DataGenerator('train', ablation=ablation)
        validation_generator = DataGenerator('val', ablation=ablation)

        # fit: this will fit the net on 'ablation' samples, only 1 epoch
        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 epochs=epochs_count)

    def run_overfitting(self, ablation=20, epochs_count=20):
        self.run_ablation(ablation, epochs_count)

    def model_train(self, lr, ablation=32, epochs_count=3):
        # model
        training_generator = AugmentedDataGenerator('train', ablation=ablation)
        validation_generator = AugmentedDataGenerator('val', ablation=ablation)
        history = LossHistory()
        decay = DecayLR(base_lr=lr)
        # checkpoint
        filepath = '../models/best_model.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                     mode='max')
        auc_logger = roc_callback(validation_generator)

        # fit
        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 epochs=epochs_count, callbacks=[auc_logger, history, decay, checkpoint])

        # self.model.fit_generator(generator=training_generator,
        #                         validation_data=validation_generator,
        #                         epochs=epochs_count, callbacks=[auc_logger, history, decay])

    def model_predictions(self, img):
        h, w, _ = img.shape
        img = img[int(h / 2) - 50:int(h / 2) + 50, int(w / 2) - 50:int(w / 2) + 50, :]
        plt.imshow(img)
        probs = self.model.predict(img[np.newaxis, :])
        max_index_col = np.argmax(probs[0], axis=0)
        return flower_classes[max_index_col], round((probs[0][max_index_col]), 2)
