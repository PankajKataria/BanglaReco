import random
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

num_train = 0
num_classes = 256
num_validation = 0
num_test = 0
img_dim = 32

def load_data(fold):
  X = []
  Y = []
  for i in range(1,1+1):
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
  Y_train[range(num_train),Y] = 1

  X_test , Y = load_data('Test')
  num_test = X_test.shape[0]
  Y_test = np.zeros((num_test,num_classes))
  Y_test[range(num_test),temp_Y] = 1

  print 'Training X shape :- ', X_train.shape
  print 'Training Y shape :- ', Y_train.shape
  print 'Testing X shape :- ', X_test.shape
  print 'Testing Y shape :- ', Y_test.shape

X_train ,Y_train , X_test , Y_test = process_data()


