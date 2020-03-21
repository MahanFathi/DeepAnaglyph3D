from .base import BaseOptimizer

import tensorflow as tf


class FConv(BaseOptimizer):
    optimizer = tf.keras.optimizers.Adam()

    def __init__(self, cfg, model):
        super(FConv, self).__init__(cfg, model)
        self.optimizer.learning_rate = cfg.OPTIMIZER.LR
        self.loss_object = tf.keras.losses.MSE

    def loss(self, x, y_hat):
        y = self.model(x, training=True)
        return loss_object(y_true=y_hat, y_pred=y)
