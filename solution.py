import random
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

num_train = 0
num_classes = 256
num_validation = 0
num_test = 0
img_dim = 32

def load_data(fold):
  X = []
  Y = []
  for i in range(1,2):
    folder = './dataset/' + fold + '/' + str(i) + '/'
    print 'In folder ', i
    listing = os.listdir(folder)
    for fl in listing:
      if not fl.endswith('.bmp'):
        continue
      im = Image.open(folder+fl)
      im = im.resize((img_dim,img_dim), Image.ANTIALIAS)
      X.append(np.array(im))
      Y.append(np.array(i))
  print np.asarray(X).shape
  return np.asarray(X), np.asarray(Y)

def refine_data():
  global X_train, Y_train, X_test, Y_test
  print 'Training X shape :- ', X_train.shape
  print 'Training Y shape :- ', Y_train.shape
  print 'Testing X shape :- ', X_test.shape
  print 'Testing Y shape :- ', Y_test.shape

  X_train = np.reshape(X_train, (X_train.shape[0],-1))
  X_test = np.reshape(X_test, (X_test.shape[0],-1))

  print 'Training X shape :- ', X_train.shape
  print 'Training Y shape :- ', Y_train.shape
  print 'Testing X shape :- ', X_test.shape
  print 'Testing Y shape :- ', Y_test.shape


X_train ,Y_train = load_data('Train')
X_test , Y_test = load_data('Test')
refine_data()