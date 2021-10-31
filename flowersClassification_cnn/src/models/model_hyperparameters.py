from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

from src.cnn import resnet
from src.cnn.config import img_channels, img_rows, img_cols, nb_classes, hyper_parameters_for_lr
from src.dataprocessor.DataGenerator import DataGenerator
from src.models.model_LossHistory import LossHistory
from src.models.model_decayLR import DecayLR


def hyperParameterTuning(epochs_count=3):
    # range of learning rates to tune
    # instantiate a LossHistory() object to store histories
    history = LossHistory()
    plot_data = {}

    # for each hyperparam: train the model and plot loss history
    for lr in hyper_parameters_for_lr:
        print('\n\n' + '==' * 20 + '   Checking for LR={}  '.format(lr) + '==' * 20)
        sgd = optimizers.SGD(lr=lr, clipnorm=1.)

        # model and generators
        model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])
        training_generator = DataGenerator('train', ablation=100)
        validation_generator = DataGenerator('val', ablation=100)
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            epochs=epochs_count, callbacks=[history])

        # plot loss history
        plot_data[lr] = history.losses
    return plot_data


def plot_LearningRate_hyperParameter(plot_data):
    # plot loss history for each value of hyperparameter
    f, axes = plt.subplots(1, 3, sharey=True)
    f.set_figwidth(15)

    plt.setp(axes, xticks=np.arange(0, len(plot_data[0.01]), 1) + 1)

    for i, lr in enumerate(plot_data.keys()):
        axes[i].plot(np.arange(len(plot_data[lr])) + 1, plot_data[lr])


def hyperParameterTuningWithDecayLR(lr, epoch_count=3):
    # to store loss history
    history = LossHistory()
    plot_data = {}

    # start with lr=0.1
    decay = DecayLR(base_lr=lr)

    # model
    sgd = optimizers.SGD()
    model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    training_generator = DataGenerator('train', ablation=100)
    validation_generator = DataGenerator('val', ablation=100)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epoch_count, callbacks=[history, decay])

    plot_data[lr] = decay.lr_history
