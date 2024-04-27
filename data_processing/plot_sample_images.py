import numpy as np
import matplotlib.pyplot as plt
from crop import *
from load_data import *

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