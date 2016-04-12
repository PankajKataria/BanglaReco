import random
import os
import time
import numpy as np
from scipy import misc
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.utils import shuffle

num_train = 0
num_classes = 256
num_test = 0
img_dim = 32
num_epochs = 300
learning_rate = 0.00008
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
  global num_classes, num_train, num_test

  X_train , Y_train = load_data('Train')
  X_test , Y_test = load_data('Test')
  X_train = X_train.astype(np.float64)
  X_test = X_test.astype(np.float64)
  num_train = X_train.shape[0]
  num_test = X_test.shape[0]

  mean_image = np.mean(X_train,axis=0)
  X_train -= mean_image
  X_test -= mean_image

  X_train = X_train.reshape(-1, 1, img_dim, img_dim)
  Y_train -= 1
  X_train , Y_train = shuffle(X_train, Y_train)

  X_test = X_test.reshape(-1, 1, img_dim, img_dim)
  Y_test -= 1
  X_test , Y_test = shuffle(X_test, Y_test)

  print 'Training X shape :- ', X_train.shape
  print 'Training Y shape :- ', Y_train.shape
  print 'Testing X shape :- ', X_test.shape
  print 'Testing Y shape :- ', Y_test.shape

  return X_train, Y_train, X_test, Y_test

X_train ,Y_train , X_test , Y_test = process_data()

dataset = dict(
  X_train=lasagne.utils.floatX(X_train),
  Y_train=Y_train.astype(np.int32),
  X_test=lasagne.utils.floatX(X_test),
  num_examples_train=X_train.shape[0],
  num_examples_test=X_test.shape[0],
  input_height=X_train.shape[1],
  input_width=X_train.shape[2],
  output_dim=num_classes,
)

convNet = NeuralNet(
  # layer construction
  layers = [
    ('input',layers.InputLayer),
    ('conv1',layers.Conv2DLayer),
    ('pool1',layers.MaxPool2DLayer),
    ('conv2',layers.Conv2DLayer),
    ('pool2',layers.MaxPool2DLayer),
    ('hidden1',layers.DenseLayer),
    # ('hidden2',layers.DenseLayer),
    ('dropout',layers.DropoutLayer),
    ('output',layers.DenseLayer),
  ],

  # layer parameters
  input_shape = (None, 1, img_dim, img_dim),
  conv1_num_filters=32, conv1_filter_size=(5, 5), conv1_nonlinearity=lasagne.nonlinearities.rectify,
  pool1_pool_size=(2, 2),
  conv2_num_filters=64, conv2_filter_size=(5, 5), conv2_nonlinearity=lasagne.nonlinearities.rectify,
  pool2_pool_size=(2, 2),
  hidden1_num_units=256, hidden1_nonlinearity=lasagne.nonlinearities.rectify,
  # hidden2_num_units=256, hidden2_nonlinearity=lasagne.nonlinearities.rectify,
  dropout_p=0.5,
  output_num_units=num_classes, output_nonlinearity=lasagne.nonlinearities.softmax,

  # ConvNet Params
  update = nesterov_momentum,
  update_learning_rate = learning_rate,
  update_momentum = momentum,
  max_epochs = num_epochs,
  verbose = 1,

)

tic = time.time()
convNet.fit(dataset['X_train'], dataset['Y_train'])
toc = time.time()
y_pred = convNet.predict(dataset['X_test'])

print y_pred

num_correct = np.sum(Y_test==y_pred)
accuracy = float(num_correct)/num_test
print 'Accuracy is :- ', accuracy
print 'Time taken to train the data :- ', toc-tic

