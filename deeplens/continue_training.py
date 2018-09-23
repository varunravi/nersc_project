import sys
sys.path.append('..')
import time
from astropy.table import Table
import pyfits as fits
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
home = expanduser("~")

user_model = raw_input("Please input which model you would like to load: ")

user_epochs = raw_input("Please input # of aditional epochs to train: ")

user_imgs = raw_input("please input the number of origional training images: ")

model = deeplens_classifier()

model.load('/Users/Chris/CMUDeepLens/notebooks/Trained_Sets/deeplens_params_'+str(user_model)+'.npy', x, y)


imgs = int(user_imgs)
n_epochs = int(user_epochs)

# download_path=home+'/Desktop/' 
# download_path='/Users/Chris/CMUDeepLens/' 

export_path=home+'/Desktop/'   # To be adjusted on your machine



d = Table.read(export_path+'catalogs_'+str(imgs)+'.hdf5', path='/ground')  # Path to be adjusted on your machine




x = np.asarray(d['image']).reshape((-1,4,101,101))
# print x.shape

y = np.asarray(d['is_lens']).reshape((-1,1))
# print y.shape

xval = np.asarray(d['image'][int(imgs*.75):]).reshape((-1,4,101,101))
yval = np.asarray(d['is_lens'][int(imgs*.75):]).reshape((-1,1))
# print xval.shape
# print yval.shape




vmin=-1e-9
vmax=1e-9
scale=100

mask = np.where(x == 100)
mask_val = np.where(xval == 100)

x[mask] = 0
xval[mask_val] = 0


x = np.clip(x, vmin, vmax)/vmax * scale
xval = np.clip(xval, vmin, vmax)/vmax * scale 

x[mask] = 0
xval[mask_val] = 0


model.fit(x,y,xval,yval)


model.save('deeplens_params_final.npy')



model.eval_purity_completeness(xval,yval)



tpr,fpr,th = model.eval_ROC(xval,yval)




plt.title('ROC on Training set')
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(0,1); 
plt.ylim(0,1.)
plt.grid(True)



p = model.predict_proba(xval)
plt.show()