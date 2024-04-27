import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import imutils
from os import listdir
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Conv2D, ZeroPadding2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import shuffle
import time

# Define image dimensions
width, height = 240, 240

def crop(image, plot=False):
    """Crop the brain contour from an MRI image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
        plt.subplot(122), plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), plt.title('Cropped')
        plt.show()
    return cropped

def load_data(directories, image_size):
    x, y = [], []
    for directory in directories:
        for filename in listdir(directory):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            if image is not None:
                image = crop(image)
                image = cv2.resize(image, image_size)
                image = image / 255.0
                x.append(image)
                y.append(1 if 'yes' in directory else 0)
    x, y = np.array(x), np.array(y)
    return shuffle(x, y)

# Load data with augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Applying the same data generator to validation data without augmentation
validation_datagen = ImageDataGenerator()

x, y = load_data(['yes', 'no'], (width, height))
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, stratify=y_val)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
val_generator = validation_datagen.flow(x_val, y_val, batch_size=32)

def build_model(input_shape):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the convolutional base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model((height, width, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tensorboard = TensorBoard(log_dir=f'logs/brain_tumor_{int(time.time())}')
checkpoint = ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model using the data generator
model.fit(train_generator, epochs=10, validation_data=val_generator,
          callbacks=[checkpoint, tensorboard, early_stopping])

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test).ravel()
    y_pred_label = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred_label))
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
    else:
        auc = "Undefined (only one class present in y_true)"
        print("AUC is undefined because only one class is present in y_true.")

evaluate_model(model, x_test, y_test)

# Function to show the percentage of data distribution
def data_percentage(y):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    total = len(y)
    print(f"Total samples: {total}, Positive: {pos} ({100 * pos / total:.2f}%), Negative: {neg} ({100 * neg / total:.2f}%)")

print("Training Data Distribution:")
data_percentage(y_train)
print("Validation Data Distribution:")
data_percentage(y_val)
print("Test Data Distribution:")
data_percentage(y_test)

# Plotting function for training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Fit the model and plot history
history = model.fit(train_generator, epochs=10, validation_data=val_generator,
                    callbacks=[checkpoint, tensorboard, early_stopping])
plot_training_history(history)

# Load the best model and evaluate it on the test set
best_model = load_model('models/best_model.keras')
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
