import numpy as np
import SimpleITK as sitk
from glob import glob
import itertools
from numpy.random import randint, random, permutation
import time
from keras.utils.np_utils import to_categorical
from skimage.transform import resize

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s.' % (time.time() - self.tstart)

def generator_2d(images, labels, input_shape, nb_classes=1001, patch_size=32, batch_size=32):
    while True:
        X = np.empty((batch_size, input_shape[0], input_shape[1]), dtype=float)
        y = np.empty(batch_size, dtype=int)

        for i in range(0, batch_size):
            subject_id = randint(0, len(images))
            X[i, :, :], y[i] = patch_generator_2d(images[subject_id], labels[subject_id], input_shape, patch_size)

        X = np.expand_dims(X, axis=3)
        y = to_categorical(y, nb_classes=nb_classes)

        yield X, y

def patch_generator_2d(image, label, input_shape, patch_size):
    p = random()

    if p > 0.66:
        patch, label = axial_patch_generator(image, label, patch_size)
    elif p > 0.33:
        patch, label = coronal_patch_generator(image, label, patch_size)
    else:
        patch, label = sagittal_patch_generator(image, label, patch_size)

    patch = sitk.GetArrayFromImage(patch)

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
    patch = sitk.Extract(image, (patch_size, 0, patch_size), point)

    return patch, label.GetPixel(point)

def coronal_patch_generator(image, label, patch_size):
    image_size = image.GetSize()
    assert (image_size == label.GetSize())

    point = (randint(0, image_size[0] - patch_size), randint(0, image_size[1] - patch_size), randint(0, image_size[2]))
    patch = sitk.Extract(image, (patch_size, patch_size, 0), point)

    return patch, label.GetPixel(point)

def sagittal_patch_generator(image, label, patch_size):
        image_size = image.GetSize()
        assert (image_size == label.GetSize())

        point = (randint(0, image_size[0]), randint(0, image_size[1] - patch_size), randint(0, image_size[2] - patch_size))
        patch = sitk.Extract(image, (0, patch_size, patch_size), point)

        return patch, label.GetPixel(point)

if __name__ == "__main__":

    print("Globbing ... "),
    with Timer():
        image_filenames = glob('/Users/kasper/Data/OASIS/disc*/OAS1_*_MR1/PROCESSED/MPRAGE/T88_111/OAS1_*_MR1_mpr_n4_anon_111_t88_gfc.hdr')
        label_filenames = glob('/Users/kasper/Data/OASIS/disc*/OAS1_*_MR1/FSL_SEG/OAS1_*_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr')

    print('Loading %i images ...' % len(image_filenames)),
    with Timer():
        images = [sitk.ReadImage(image_filename) for image_filename in image_filenames]
        labels = [sitk.ReadImage(label_filename) for label_filename in label_filenames]

    print('Generating 1000 batches ...'),
    batch_size = 1000
    patch_size = 32
    with Timer():
        n = 0
        while n < 10:
            x_train, y_train = generator_2d(images, labels, (299, 299), patch_size, batch_size)
            n += 1

    print('Saving a patch to disk ...'),
    sitk.WriteImage(sitk.GetImageFromArray(x_train), 'patches.nii.gz')

