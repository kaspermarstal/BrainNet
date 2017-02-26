# Util
import time

# Images
import numpy as np
from numpy.random import random, randint
from SimpleITK import Extract, GetArrayFromImage, ReadImage
from skimage.transform import resize

# Keras
from keras.utils.np_utils import to_categorical

# The number of classes is tied to the dataset
nb_classes = 4

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s.' % (time.time() - self.tstart)

def generator(images, labels, input_shape, patch_size=32, batch_size=32):
    while True:
        X = np.empty((batch_size, input_shape[0], input_shape[1], input_shape[2]), dtype=float)
        y = np.empty(batch_size, dtype=int)

        for i in range(0, batch_size):
            subject_id = randint(0, len(images))
            X[i, :, :], y[i] = generate_one(images[subject_id], labels[subject_id], input_shape, patch_size)

        y = to_categorical(y, nb_classes=nb_classes)

        yield X, y

def generate_one(image, label, input_shape, patch_size):
    p = random()

    # TODO: Obtain patches at abitrary angles
    if p > 0.66:
        patch, label = axial_patch_generator(image, label, patch_size)
    elif p > 0.33:
        patch, label = coronal_patch_generator(image, label, patch_size)
    else:
        patch, label = sagittal_patch_generator(image, label, patch_size)

    patch = GetArrayFromImage(patch)

    if random() > 0.5:
        patch = np.fliplr(patch)

    if random() > 0.5:
        patch = np.flipud(patch)

    patch = resize(patch, input_shape)

    return patch, label

def axial_patch_generator(image, label, patch_size):
    image_size = image.GetSize()
    assert(image_size == label.GetSize())

    point = (randint(0, image_size[0] - patch_size), randint(0, image_size[1]), randint(0, image_size[2] - patch_size))
    patch = Extract(image, (patch_size, 0, patch_size), point)

    return patch, label.GetPixel(point)

def coronal_patch_generator(image, label, patch_size):
    image_size = image.GetSize()
    assert (image_size == label.GetSize())

    point = (randint(0, image_size[0] - patch_size), randint(0, image_size[1] - patch_size), randint(0, image_size[2]))
    patch = Extract(image, (patch_size, patch_size, 0), point)

    return patch, label.GetPixel(point)

def sagittal_patch_generator(image, label, patch_size):
        image_size = image.GetSize()
        assert (image_size == label.GetSize())

        point = (randint(0, image_size[0]), randint(0, image_size[1] - patch_size), randint(0, image_size[2] - patch_size))
        patch = Extract(image, (0, patch_size, patch_size), point)

        return patch, label.GetPixel(point)


