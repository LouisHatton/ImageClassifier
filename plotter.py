import numpy as np
import matplotlib.pyplot as plt

def display_multiple_img(images, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title][0])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

def handle_predictions(images, predictions, actual, classes):
    if len(predictions) < 6: return None

    # Convert provided info into data structure for display_multiple_img
    data = {f'Pred: {classes[predictions[i]]}, Actual: {classes[actual[i]]}': images[i] for i in range(6)}
    display_multiple_img(data, rows=3, cols=2)