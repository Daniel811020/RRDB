import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from Config import Cfg
from tensorflow.python.keras import backend as K
from tensorflow.python import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from Config import Cfg

def create_callback_new(
    train_record_path,
    warmup_epoch,
    sample_count,
    batchsize,
    epoch,
    learning_rate_base,
    warmup_learning_rate,
):

    # Warm up
    warmup_steps = int(warmup_epoch * sample_count / batchsize)
    total_steps = int(epoch * sample_count / batchsize)

    warm_up_lr = WarmUpCosineDecayScheduler(
        learning_rate_base=learning_rate_base,
        total_steps=total_steps,
        warmup_learning_rate=warmup_learning_rate,
        warmup_steps=warmup_steps,
        hold_base_rate_steps=0,
        num_cycles=1,
        verbose=0
    )

    csv_logger = CSVLogger(train_record_path + '/log.csv', append=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.2,
        patience=2, min_lr=Cfg.min_lr)

    if Cfg.reduce_LR:
        callbacks = [
            reduce_lr,
            csv_logger
        ]
    else:
        callbacks = [
            warm_up_lr,
            csv_logger
        ]

    return callbacks

def create_callback(
    train_sess,
    saver,
    train_record_path,
    warmup_epoch,
    sample_count,
    batchsize,
    epoch,
    learning_rate_base,
    warmup_learning_rate,
):
    class SaveCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            saver.save(
                train_sess,
                train_record_path + "/aware_augmix-%05d" % epoch
            )

    # Warm up
    warmup_steps = int(warmup_epoch * sample_count / batchsize)
    total_steps = int(epoch * sample_count / batchsize)

    warm_up_lr = WarmUpCosineDecayScheduler(
        learning_rate_base=learning_rate_base,
        total_steps=total_steps,
        warmup_learning_rate=warmup_learning_rate,
        warmup_steps=warmup_steps,
        hold_base_rate_steps=0,
        num_cycles=1,
        verbose=0
    )

    # earlystopping = EarlyStopping(
    #     monitor='loss',
    #     min_delta=0,
    #     patience=5,
    #     verbose=1,
    #     mode='auto',
    #     baseline=None,
    #     restore_best_weights=False
    # )

    modelcheckpoint = ModelCheckpoint(
        filepath=train_record_path + '/best.h5',
        verbose=0,
        save_freq="epoch",
        save_best_only=False,
        save_weights_only=True,

    )

    # tensorboard = TensorBoard(
    #     log_dir=train_record_path,
    #     histogram_freq=0,
    #     batch_size=batchsize,
    #     write_graph=True,
    #     write_grads=False,
    #     write_images=False,
    #     embeddings_freq=0,
    #     embeddings_layer_names=None,
    #     embeddings_metadata=None,
    #     embeddings_data=None,
    #     update_freq='epoch'
    # )

    csv_logger = CSVLogger(train_record_path + '/log.csv', append=True)

    callbacks = [
        # earlystopping,
        # modelcheckpoint,
        # SaveCheckpoint(),
        warm_up_lr,
        csv_logger
    ]

    return callbacks


def create_float_callback(
    train_record_path,
    warmup_epoch,
    sample_count,
    batchsize,
    epoch,
    learning_rate_base,
    warmup_learning_rate,
):
    # Warm up
    warmup_steps = int(warmup_epoch * sample_count / batchsize)
    total_steps = int(epoch * sample_count / batchsize)

    warm_up_lr = WarmUpCosineDecayScheduler(
        learning_rate_base=learning_rate_base,
        total_steps=total_steps,
        warmup_learning_rate=warmup_learning_rate,
        warmup_steps=warmup_steps,
        hold_base_rate_steps=0,
        num_cycles=1,
        verbose=0
    )

    modelcheckpoint = ModelCheckpoint(
        filepath=train_record_path + '/best.h5',
        verbose=0,
        save_freq="epoch",
        save_best_only=False,
        save_weights_only=True,

    )
    csv_logger = CSVLogger(train_record_path + '/log.csv', append=True)

    callbacks = [
        # earlystopping,
        modelcheckpoint,
        warm_up_lr,
        csv_logger
    ]

    return callbacks


def create_callback_2(
    train_sess,
    saver,
    train_record_path,
    warmup_epoch,
    sample_count,
    batchsize,
    epoch,
    learning_rate_base,
    warmup_learning_rate,
):
    class SaveCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            saver.save(
                train_sess,
                train_record_path + "/aware_augmix-%05d" % epoch
            )

    modelcheckpoint = ModelCheckpoint(
        filepath=train_record_path +
        '/best_{}.h5'.format(time.strftime("%Y_%m_%d_%H_%M_%S",
                                           time.localtime())),
        verbose=0,
        save_freq="epoch",
        save_best_only=False,
        save_weights_only=True,

    )

    csv_logger = CSVLogger(train_record_path + '/log.csv', append=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=2, min_lr=1e-6)
    callbacks = [
        # modelcheckpoint,
        reduce_lr,
        SaveCheckpoint(),
        csv_logger
    ]

    return callbacks


def cosine_with_hard_restarts_schedule_with_warmup(
    epoch,
    total_epochs,
    warmup_epochs=0,
    lr_start=1e-4,
    lr_max=1e-3,
    lr_min=1e-4,
    num_cycles=1.
):

    if epoch < warmup_epochs:
        lr = (lr_max - lr_start) / warmup_epochs * epoch + lr_start
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_max * \
            (0.5 *
                (1.0 + tf.math.cos(math.pi * ((num_cycles * progress) % 1.0))))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, lr)
    return lr


def cosine_decay_with_warmup(
    global_step,
    learning_rate_base,
    total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=0,
    hold_base_rate_steps=0,
    num_cycles=1
):
    if total_steps < warmup_steps:
        raise ValueError(
            'total_steps must be larger or equal to warmup_steps.')

    if num_cycles == 1:
        learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
            np.pi *
            (global_step - warmup_steps - hold_base_rate_steps
             ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    else:
        all_step = float(total_steps - warmup_steps - hold_base_rate_steps)
        cycle_step = (all_step/num_cycles)
        if global_step > warmup_steps:
            if int(
                (global_step - warmup_steps - hold_base_rate_steps) %
                cycle_step
            ) == 0:
                learning_rate = learning_rate_base
            else:
                learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
                    np.pi *
                    int((global_step - warmup_steps - hold_base_rate_steps) %
                        cycle_step) / float(cycle_step)
                ))
        else:
            learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
                np.pi *
                (global_step - warmup_steps - hold_base_rate_steps
                 ) / float(total_steps - warmup_steps - hold_base_rate_steps)))

    if hold_base_rate_steps > 0:
        learning_rate = np.where(
            global_step > warmup_steps + hold_base_rate_steps,
            learning_rate,
            learning_rate_base
        )
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(
        self,
        learning_rate_base,
        total_steps,
        global_step_init=0,
        warmup_learning_rate=0.0,
        warmup_steps=0,
        hold_base_rate_steps=0,
        num_cycles=1,
        verbose=0
    ):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.num_cycles = num_cycles
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(
            global_step=self.global_step,
            learning_rate_base=self.learning_rate_base,
            total_steps=self.total_steps,
            warmup_learning_rate=self.warmup_learning_rate,
            warmup_steps=self.warmup_steps,
            hold_base_rate_steps=self.hold_base_rate_steps,
            num_cycles=self.num_cycles
        )
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning rate to %s.' %
                  (self.global_step + 1, lr))


class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, mom=0.9, stop_multiplier=None,
                 reload_weights=True, batches_lr_update=5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20*self.mom/3 + 10  # 4 if mom=0.9
            # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier

    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p['epochs']*p['samples']//p['batch_size']
        except Exception:
            n_iterations = p['steps']*p['epochs']

        self.learning_rates = np.geomspace(
            self.min_lr,
            self.max_lr,
            num=n_iterations//self.batches_lr_update+1)
        self.losses = []
        self.iteration = 0
        self.best_loss = 0
        if self.reload_weights:
            self.model.save_weights('Lr_finder/tmp.hdf5')

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')

        if self.iteration != 0:  # Make loss smoother using momentum
            loss = self.losses[-1]*self.mom+loss*(1-self.mom)

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss

        if self.iteration % self.batches_lr_update == 0:
            # Evaluate each lr over 5 epochs

            if self.reload_weights:
                self.model.load_weights('Lr_finder/tmp.hdf5')

            lr = self.learning_rates[self.iteration//self.batches_lr_update]
            K.set_value(self.model.optimizer.lr, lr)

            self.losses.append(loss)

        if loss > self.best_loss*self.stop_multiplier:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights('Lr_finder/tmp.hdf5')

        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')

        plt.savefig('Lr_finder/plot.png')
        plt.show()
