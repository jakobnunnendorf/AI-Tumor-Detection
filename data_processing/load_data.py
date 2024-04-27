from sklearn.utils import shuffle
import cv2
import numpy as np
from os import listdir
from crop import *

def load_data(directories, image_size):
 
    # load all images in a directory
        
    x = [] # (image, width, height, channel) 
    y = [] # (image, 1)
    image_width, image_height = image_size
    
    for directory in directories:
        print(directory)
        for filename in listdir(directory):
            # load image
            path = directory + '/' + filename
            image = cv2.imread(path)
            image = crop(image, plot=False)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize
            image = image / 255.
            # convert image to numpy array and append it to x
            x.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])

            else:
                y.append([0])
        
                
    x = np.array(x)
    y = np.array(y)
    
    # Shuffle the data
    x, y = shuffle(x, y)
    
    print(f'Number of examples is: {len(x)}')
    print(f'x shape is: {x.shape}')
    print(f'y shape is: {y.shape}')
    
    return x, y