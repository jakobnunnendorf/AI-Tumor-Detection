from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, ZeroPadding2D, MaxPooling2D, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,classification_report, roc_auc_score,roc_curve
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

width, height = (240, 240)
def crop(image):
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold image then clean up
    threshold = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.erode(threshold, None, iterations=2)
    threshold = cv2.dilate(threshold, None, iterations=2)

    # Find contours, get largest one
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contour = max(contours, key=cv2.contourArea)

    # Find extreme points
    left = tuple(contour[contour[:, :, 0].argmin()][0])
    right = tuple(contour[contour[:, :, 0].argmax()][0])
    top = tuple(contour[contour[:, :, 1].argmin()][0])
    bottom = tuple(contour[contour[:, :, 1].argmax()][0])

    # Crop using the extreme points
    cropped = image[top[1]:bottom[1], left[0]:right[0]]  

    return cropped


def load_data(directories, image_size):
 
    # load all images in a directory
        
    x = [] # (image, width, height, directory) 
    y = [] # (image, 1) (cancerous)
    image_width, image_height = image_size
    
    for directory in directories:
        for filename in listdir(directory):
            # load images
            path = directory + '/' + filename
            image = cv2.imread(path)
            image = crop(image)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize
            image = image / 255.
            x.append(image)
            # append a value of 1 if image in "yes" 
            if 'yes' in directory:
                y.append([1])

            else:
                y.append([0])
        
                
    x = np.array(x)
    y = np.array(y)
    
    # randomize order
    x, y = shuffle(x, y)
    
    # samples and shape
    print(f'Number of examples is: {len(x)}')
    print(f'shape of x is: {x.shape}')
    print(f'shape of y is: {y.shape}')
    
    return x, y


x, y = load_data(['yes', 'no'], (width, height))

def plot_sample_images(x, y, n=50):
    
    for label in [0, 1]:
        # Select the images with corresponding labels
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
        plt.suptitle(f"Cancer: {label_str}")
        plt.show()


plot_sample_images(x, y)

def split_data(x, y, test_size=0.2):
    #Split data to training, validation, and test sets
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=test_size)
    test_x, val_x, test_y, val_y = train_test_split(val_x, val_y, test_size=0.5)
    return train_x, train_y, val_x, val_y, test_x, test_y


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

# format time string
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
    inputs = Input(shape=input_shape)
    
    # First block
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Second block
    y = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01))(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Dropout(0.3)(y)

    # Flatten and dense layers
    z = Flatten()(y)
    z = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(z)
    z = Dropout(0.5)(z)
    outputs = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=inputs, outputs=outputs, name="CancerDetection")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

shape = (width, height, 3)
model = build_model(shape)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# tensorboard
log_file = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file}')
# checkpoint
# file name that includes the epoch and the validation accuracy
filepath = "models/cnn-parameters-{epoch:02d}-{val_accuracy:.2f}.keras"

# save the model with the best validation  accuracy till now
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

start_time = time.time()

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {format_elapsed_time(execution_time)}")

history = model.history.history


def trainingHistory(history):
    """Plot the training history"""

    # Training and validation loss
    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title("Loss")
    plt.legend()

    # Training and validation accuracy
    plt.figure()
    plt.plot(history["accuracy"], label="Training accuracy")
    plt.plot(history["val_accuracy"], label="Validation accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.show()


trainingHistory(history) 


best_model = load_model(filepath='models/cnn-parameters-40-0.79.keras')
best_model.metrics_names
loss, acc = best_model.evaluate(x=x_test, y=y_test)

print (f"Test Loss = {loss}")
print (f"Test Accuracy = {acc}")

y_test_prob = best_model.predict(x_test)
f1score = compute_f1_score(y_test, y_test_prob)
print(f"F1 score: {f1score}")

y_val_prob = best_model.predict(x_val)
f1score_val = compute_f1_score(y_val, y_val_prob)
print(f"F1 score: {f1score_val}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).ravel()
    y_pred_class = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred_class))

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

evaluate_model(model, x_test, y_test)

def data_percentage(y):
    total_count = len(y)
    positive_count = np.sum(y)

    positive_percentage = 100 * positive_count / total_count
    negative_percentage = 100 * (total_count - positive_count) / total_count

    print(f"Number of examples: {total_count}")
    print(f"Percentage of positive examples: {positive_percentage:.2f}%")
    print(f"Percentage of negative examples: {negative_percentage:.2f}%")


# the whole data
data_percentage(y)

print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)
