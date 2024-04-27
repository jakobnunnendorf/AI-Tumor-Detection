from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, ZeroPadding2D, MaxPooling2D, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import time
from crop import *
from load_data import *
from plot_sample_images import *

def split_data(features, targets, test_size=0.2):
    train_features, val_features, train_targets, val_targets = \
        train_test_split(features, targets, test_size=test_size)
    test_features, val_features, test_targets, val_targets = \
        train_test_split(val_features, val_targets, test_size=0.5)
    return train_features, train_targets, val_features, val_targets, test_features, test_targets