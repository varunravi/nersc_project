# Varun Ravi
# 9/21

import sys
import time
from astropy.table import Table
import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
import random
from sklearn.utils import shuffle
import re
import shutil
import datetime
from sklearn.model_selection import train_test_split
from resnet_classifier import deeplens_classifier
import os

x_train = np.load('./test/x_train.npy')
y_train = np.load('./test/y_train.npy')
x_test = np.load('./test/x_test.npy')
y_test = np.load('./test/y_test.npy')

model = deeplens_classifier(learning_rate=0.001, learning_rate_steps=3, learning_rate_drop=0.1, batch_size=128, n_epochs=5)

model.fit(x_train,y_train,x_test,y_test)
print("DONE :)")


