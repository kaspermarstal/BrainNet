# Util
import os
import sys
import argparse
from glob import glob

# Data
from data import generator, nb_classes, ReadImage, Timer
from sklearn.model_selection import train_test_split

# Keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop

# Model
from inception_v4 import create_inception_v4, input_shape

# TensorFlow
from tensorflow.python.platform import app

def main(argv):

  print('Finding data ...'),
  with Timer():
      image_filenames = glob(os.path.join(FLAGS.data_dir, 'disc*/OAS1_*_MR1/PROCESSED/MPRAGE/T88_111/OAS1_*_MR1_mpr_n4_anon_111_t88_gfc.hdr'))
      label_filenames = glob(os.path.join(FLAGS.data_dir, 'disc*/OAS1_*_MR1/FSL_SEG/OAS1_*_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'))
      assert(len(image_filenames) == len(label_filenames))
      print('Found %i images.' % len(image_filenames))

  print('Loading images ...'),
  with Timer():
      images = [ReadImage(image_filename) for image_filename in image_filenames]
      labels = [ReadImage(label_filename) for label_filename in label_filenames]
      images_train, images_test, labels_train, labels_test = train_test_split(images, labels, train_size=0.66)

  tensor_board = TensorBoard(log_dir='./TensorBoard')
  early_stopping = EarlyStopping(monitor='acc', patience=2, verbose=1)

  model = create_inception_v4(nb_classes=nb_classes, load_weights=False)
  model.compile(optimizer=RMSprop(lr=0.045, rho=0.94, epsilon=1., decay=0.9), loss='categorical_crossentropy', metrics=['acc'])
  model.fit_generator(generator(images_train, labels_train, input_shape, nb_classes, FLAGS.patch_size, FLAGS.batch_size),
                      samples_per_epoch=FLAGS.samples_per_epoch, nb_epoch=FLAGS.nb_epochs, callbacks=[tensor_board, early_stopping],
                      verbose=1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-d',
      '--data-dir',
      dest='data_dir',
      help='Path to data directory.',
      required=True,
  )
  parser.add_argument(
      '-p',
      '--patch-size',
      default=32,
      type=int,
      dest='patch_size',
      help='Size of the p-by-p patch in millimetre (mm).',
  )
  parser.add_argument(
      '-b',
      '--batch-size',
      default=32,
      type=int,
      dest='batch_size',
      help='Batch size.',
  )
  parser.add_argument(
      '-e',
      '--nb-epochs',
      default=8,
      type=int,
      dest='nb_epochs',
      help='Number of epochs.',
  )
  parser.add_argument(
      '-s',
      '--samples-per-epoch',
      default=1024,
      type=int,
      dest='samples_per_epoch',
      help='Number of samples per epoch.',
  )

  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
