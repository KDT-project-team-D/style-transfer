from django.db import models

# Create your models here.
from content.cyclegan import CycleGAN
from PIL import Image
from django.conf import settings

import numpy as np
import tensorflow as tf
import io, base64


class Photo(models.Model):
    image = models.ImageField(upload_to='images/')

    def vangogh_predict(self):

        cycle_gan = CycleGAN()
        img_data = self.image.read()
        img_bin = io.BytesIO(img_data)
        image = Image.open(img_bin).convert("RGB")
        translated_img = cycle_gan.predict(image, 0)

        return translated_img

    def cezanne_predict(self):

        cycle_gan = CycleGAN()
        img_data = self.image.read()
        img_bin = io.BytesIO(img_data)
        image = Image.open(img_bin).convert("RGB")
        translated_img = cycle_gan.predict(image, 1)

        return translated_img

    def monet_predict(self):

        cycle_gan = CycleGAN()
        img_data = self.image.read()
        img_bin = io.BytesIO(img_data)
        image = Image.open(img_bin).convert("RGB")
        translated_img = cycle_gan.predict(image, 2)

        return translated_img
    '''
        global graph

        with graph.as_default():
    '''

