import random
import os
import numpy as np
from scipy import misc
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

num_train = 0
num_classes = 256
num_validation = 0
num_test = 0
img_dim = 40
num_epochs = 1
batch_size = 200
learning_rate = 0.01
momentum = 0.9

def load_data(fold):
  X = []
  Y = []
  for i in range(1,num_classes+1):
    folder = '../dataset/' + fold + '/' + str(i) + '/'
    print 'In folder ', i
    listing = os.listdir(folder)
    for fl in listing:
      if not fl.endswith('.bmp'):
        continue
      im = misc.imread(folder+fl, flatten=True)
      im = misc.imresize(im, (img_dim,img_dim))
      X.append(np.array(im))
      Y.append(i)
  return np.asarray(X), np.asarray(Y)

def process_data():
  global num_classes, num_train
  X_train , Y = load_data('Train')
  num_train = X_train.shape[0]
  Y_train = np.zeros((num_train,num_classes))
  Y_train[range(num_train),(Y-1)] = 1
  X_train , Y_train = shuffle(X_train, Y_train)

  X_test , Y = load_data('Test')
  num_test = X_test.shape[0]
  Y_test = np.zeros((num_test,num_classes))
  Y_test[range(num_test),(Y-1)] = 1
  X_test , Y_test = shuffle(X_test, Y_test)

  print 'Training X shape :- ', X_train.shape
  print 'Training Y shape :- ', Y_train.shape
  print 'Testing X shape :- ', X_test.shape
  print 'Testing Y shape :- ', Y_test.shape

# X_train ,Y_train , X_test , Y_test = process_data()
convNet = NeuralNet(
  # layer construction
  layers = [
    ('input',layers.InputLayer),
    ('conv1',layers.Conv2DLayer),
    ('pool1',layers.MaxPool2DLayer),
    ('conv2',layers.Conv2DLayer),
    ('pool2',layers.MaxPool2DLayer),
    ('hiden1',layers.DenseLayer),
    ('hiden2',layers.DenseLayer),
    ('output',layers.DenseLayer),
  ],

  # layer parameters
  input_shape = (None, 1, img_dim, img_dim),
  conv1_num_filters=32, conv1_filter_size=(5, 5), conv1_nonlinearity=lasagne.nonlinearities.rectify,
  pool1_pool_size=(2, 2),
  conv2_num_filters=64, conv2_filter_size=(5, 5), conv2_nonlinearity=lasagne.nonlinearities.rectify,
  pool2_pool_size=(2, 2),
  hidden1_num_units=256, hidden1_nonlinearity=lasagne.nonlinearities.rectify,
  hidden2_num_units=500, hidden2_nonlinearity=lasagne.nonlinearities.rectify,
  output_num_units=num_classes, output_nonlinearity=lasagne.nonlinearities.softmax,

  # ConvNet Params
  update_learning_rate = learning_rate,
  update_momentum = momentum,
  max_epochs = num_epochs,

)






