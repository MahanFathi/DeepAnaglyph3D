import math
import tensorflow as tf

from random import random
from data.datasets.objet import settings
from data.datasets.objet.util import Env


def objet(cfg):
    dataset = tf.data.Dataset.from_generator(
        data_generator_factory(cfg),
        output_types=(
            tf.int32,
            tf.int32,
        ),
        output_shapes=(
            (settings.WIDTH, settings.HEIGHT, 3),
            (settings.WIDTH, settings.HEIGHT, 3),
        ),
    )
    return dataset


def data_generator_factory(cfg):

    class DataGenerator(object):
        def __init__(self):
            self.env = Env(cfg)

        def __iter__(self):
            return self

        def __next__(self):
            theta = random() * math.pi
            self.env.set_cam(theta)
            image = self.env.get_image()
            anaglyph = self.env.get_anaglyph()
            return image, anaglyph

    return DataGenerator

