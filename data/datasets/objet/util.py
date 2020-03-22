import json
import numpy as np

from pyobjet import Objet
from stereoscopy import create_anaglyph
from PIL import Image

from data.datasets.objet import settings

class Env(object):

    def __init__(self, cfg):
        json_path = cfg.DATASET.PATH_TO_JSON
        self.objet = Objet(
            json_path,
            settings.WIDTH,
            settings.HEIGHT,
        )
        self.analyze_json(json_path)
        self.default_cam_radius = settings.CAM_RADIUS
        self.default_cam_height = settings.CAM_HEIGHT
        self.cam_theta = None
        self.cam_radius = None
        self.cam_height = None
        self.cam_delta_theta = 0.02
        return


    def analyze_json(self, path_to_json):

        with open(path_to_json) as f:
            data_json = json.load(f)

        self.object_names = [obj['name'] for obj in data_json['objects']]


    def set_cam(self, theta, radius=None, height=None):
        self.cam_height = self.default_cam_height or height
        self.cam_radius = self.default_cam_radius or radius
        self.cam_theta = theta


    def get_image(self, ):
        y = self.cam_height
        x = self.cam_radius * np.sin(self.cam_theta)
        z = self.cam_radius * np.cos(self.cam_theta)
        self.objet.set_camera(
            [x, y, z],
            [.0, .0, .0],
        )
        self.objet.draw()
        img = self.objet.get_image()
        return img


    def get_right_image(self, ):
        self.cam_theta += self.cam_delta_theta
        img = self.get_image()
        self.cam_theta -= self.cam_delta_theta
        return img


    def get_left_image(self, ):
        self.cam_theta -= self.cam_delta_theta
        img = self.get_image()
        self.cam_theta += self.cam_delta_theta
        return img


    def get_anaglyph(self, ):
        left = self.get_left_image()
        right = self.get_right_image()
        anaglyph = create_anaglyph(
        [Image.fromarray(right.astype(np.uint8), 'RGB'),
         Image.fromarray(left.astype(np.uint8), 'RGB')],
        'half-color',
        'red-cyan',
        )
        return np.array(anaglyph)
