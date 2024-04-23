
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

def crop_brain_contour(image, plot=False):
    """Crop the brain contour from a MRI image."""
    # Convert image to grayscale, apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold image, then erode/dilate to clean up
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours, get largest one
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop image using the four extreme points
    crop = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]  

    if plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(crop)
        plt.axis('off')
        plt.title('Cropped Image')
        plt.show()

    return crop


ex_img = cv2.imread('yes/Y1.jpg')
ex_new_img = crop_brain_contour(ex_img, True)

print("After the y1")


def load_data(dir_list, image_size):
    print(dir_list, image_size)
 
    # load all images in a directory
        
    x = [] # x: A numpy array with shape = (#_examples, image_width, image_height, #_channels) 
    y = [] # y: A numpy array with shape = (#_examples, 1)
    image_width, image_height = image_size
    
    for directory in dir_list:
        print(directory)
        for filename in listdir(directory):
            print(filename)
            # load the image
            path = directory + '/' + filename
            print(path)
            image = cv2.imread(path)
            # crop the brain and ignore the unnecessary rest part of the image
            image = crop_brain_contour(image, plot=False)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
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

print("Line 120")


# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes =  'yes' 
augmented_no =  'no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)
x, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))

import numpy as np
import matplotlib.pyplot as plt

def plot_sample_images(x, y, n=50):
    """
    Plots n sample images for both values of y (labels).
    
    Arguments:
        x: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
        n: Number of images to plot for each class.
    """
    for label in [0, 1]:
        # Select the images with the corresponding labels
        images = x[y.flatten() == label][:n]
        
        # Determine the number of columns for subplots
        columns_n = min(n, 10)
        rows_n = int(np.ceil(n / columns_n))

        # Create subplots
        plt.figure(figsize=(20, 10))
        
        # Plot images
        for i, image in enumerate(images):
            plt.subplot(rows_n, columns_n, i + 1)
            plt.imshow(image)
            
            # Remove ticks
            plt.tick_params(axis='both', which='both', 
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        # Set title
        label_str = "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_str}")
        plt.show()


plot_sample_images(x, y)

def split_data(features, targets, test_size=0.2):
    """Split data into train, val, and test sets."""
    train_features, val_features, train_targets, val_targets = \
        train_test_split(features, targets, test_size=test_size)
    test_features, val_features, test_targets, val_targets = \
        train_test_split(val_features, val_targets, test_size=0.5)
    return train_features, train_targets, val_features, val_targets, test_features, test_targets

x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y, test_size=0.3)

print ("number of training examples = " + str(x_train.shape[0]))
print ("number of development examples = " + str(x_val.shape[0]))
print ("number of test examples = " + str(x_test.shape[0]))
print ("x_train shape: " + str(x_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("x_val (dev) shape: " + str(x_val.shape))
print ("Y_val (dev) shape: " + str(y_val.shape))
print ("x_test shape: " + str(x_test.shape))
print ("Y_test shape: " + str(y_test.shape))

# Nicely formatted time string
def format_elapsed_time(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)
    score = f1_score(y_true, y_pred)
    
    return score

def build_model(input_shape):
    
    # Define the input placeholder as a tensor with shape input_shape. 
    x_input = Input(input_shape) # shape=(?, 240, 240, 3)
    
    # Zero-Padding: pads the border of x_input with zeroes
    x = ZeroPadding2D((2, 2))(x_input) # shape=(?, 244, 244, 3)
    
    # CONV -> BN -> RELU Block applied to x
    x = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(x)
    x = BatchNormalization(axis = 3, name = 'bn0')(x)
    x = Activation('relu')(x) # shape=(?, 238, 238, 32)
    
    # MAxPOOL
    x = MaxPooling2D((4, 4), name='max_pool0')(x) # shape=(?, 59, 59, 32) 
    
    # MAxPOOL
    x = MaxPooling2D((4, 4), name='max_pool1')(x) # shape=(?, 14, 14, 32)
    
    # FLATTEN x 
    x = Flatten()(x) # shape=(?, 6272)
    # FULLYCONNECTED
    x = Dense(1, activation='sigmoid', name='fc')(x) # shape=(?, 1)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = x_input, outputs = x, name='BrainDetectionModel')
    
    return model

IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
model = build_model(IMG_SHAPE)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# tensorboard
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')
# checkpoint
# unique file name that will include the epoch and the validation (development) accuracy
filepath = "models/cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}.keras"

# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

start_time = time.time()

model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {execution_time}")

history = model.history.history

for key in history.keys():
    print(key)

def plot_metrics(history):
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

plot_metrics(history) 