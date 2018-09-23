import numpy as np
import sys
import os 
import pyfits as fits

def save_numpy(array, name, path="/global/homes/v/vrav2/LBNL_key_items/notebooks/deeplens/test/"):
    np.save((path+name), array)
    print(name, 'saved')

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Error: The arguments required are: csv_file & output_filename")
