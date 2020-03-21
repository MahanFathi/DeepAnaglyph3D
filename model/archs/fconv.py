import tensorflow as tf


class FConv(tf.keras.Model):

    def __init__(self, cfg):
        super(FConv, self).__init__(name='FullyConv')

        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), 2,
                                   padding="same", activation='relu'),
            tf.keras.layers.Conv2D(16, (2, 2), 2,
                                   padding="same", activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), 2,
                                   padding="same", activation='relu'),
            tf.keras.layers.Conv2D(64, (2, 2), 2,
                                   padding="same", activation='relu'),
        ])
        # (None, 32, 32, 32)
        self.deconv = tf.keras.Sequential([
            tf.keras.layers.Convolution2DTranspose(32, (3, 3), 2,
                                   padding="same", activation='relu'),
            tf.keras.layers.Convolution2DTranspose(16, (2, 2), 2,
                                   padding="same", activation='relu'),
            tf.keras.layers.Convolution2DTranspose(8, (3, 3), 2,
                                   padding="same", activation='relu'),
            tf.keras.layers.Convolution2DTranspose(3, (2, 2), 2,
                                   padding="same"),
        ])

    def call(self, input_tensor, training=False):
        latent = self.conv(input_tensor, training=training)
        output = self.deconv(latent, training=training)
        return output
