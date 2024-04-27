from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, ZeroPadding2D, MaxPooling2D, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import time
from crop import *
from load_data import *
from plot_sample_images import *
from split_data import *

width, height = (240, 240)

x, y = load_data(['yes', 'no'], (width, height))

# plot_sample_images(x, y)

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
    x_input = Input(input_shape)  # shape=(None, 240, 240, 3)
    
    # Zero-Padding: pads the border of x_input with zeroes
    x = ZeroPadding2D((3, 3))(x_input)  # shape=(None, 246, 246, 3)
    
    # First CONV -> BN -> RELU Layer
    x = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(x)
    x = BatchNormalization(axis=3, name='bn0')(x)
    x = Activation('relu')(x)  # shape=(None, 240, 240, 32)
    x = MaxPooling2D((4, 4), name='max_pool0')(x)  # shape=(None, 60, 60, 32)
    
    # Second CONV -> BN -> RELU Layer
    x = Conv2D(64, (5, 5), strides=(1, 1), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Activation('relu')(x)  # shape=(None, 56, 56, 64)
    x = MaxPooling2D((2, 2), name='max_pool1')(x)  # shape=(None, 28, 28, 64)

    # Third CONV -> BN -> RELU Layer
    x = Conv2D(128, (3, 3), strides=(1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, name='bn2')(x)
    x = Activation('relu')(x)  # shape=(None, 26, 26, 128)
    x = MaxPooling2D((2, 2), name='max_pool2')(x)  # shape=(None, 13, 13, 128)

    # Flatten the data to a 1-D vector
    x = Flatten()(x)  # shape=(None, 21632)

    # Add a fully connected layer
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)  # Dropout added for regularization

    # Final fully connected layer with sigmoid activation for binary classification
    x = Dense(1, activation='sigmoid', name='fc2')(x)  # shape=(None, 1)
    
    # Create the Keras model instance
    model = Model(inputs=x_input, outputs=x, name='BrainDetectionModel')
    
    return model

shape = (width, height, 3)
model = build_model(shape)
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
# changed epochs from 10 to 2 to speed up refactoring
model.fit(x=x_train, y=y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {format_elapsed_time(execution_time)}")

history = model.history.history


def trainingHistory(history):
    
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

trainingHistory(history) 


best_model = load_model(filepath='models/cnn-parameters-improvement-09-0.84.keras')
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

def data_percentage(y):
    
    m=len(y)
    n_positive = np.sum(y)
    n_negative = m - n_positive
    
    pos_prec = (n_positive* 100.0)/ m
    neg_prec = (n_negative* 100.0)/ m
    
    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {n_positive}") 
    print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {n_negative}") 

# the whole data
data_percentage(y)

print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)