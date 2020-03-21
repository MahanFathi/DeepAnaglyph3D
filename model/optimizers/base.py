import tensorflow as tf


class BaseOptimizer(object):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    @property
    def optimizer(self, ):
        raise NotImplementedError

    def loss(self, x, y_hat):
        raise NotImplementedError

    def step(self, x, y_hat):
        with tf.GradientTape() as tape:
            loss = self.loss(x, y_hat)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
