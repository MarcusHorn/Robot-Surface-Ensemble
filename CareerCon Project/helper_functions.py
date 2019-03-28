
# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    # Retrieved from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def quaternion_to_euler(x, y, z, w):
    # Retrieved from https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
    # Returns a radian measurement for roll, pitch, and yaw
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def normalize(data):
    # Normalizes the orientation parameters in the quaternion for an input dataset 'data'
    data['mod_quat'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2 + data['orientation_W']**2)**.5
    data['norm_orientation_X'] = data['orientation_X']/data['mod_quat']
    data['norm_orientation_Y'] = data['orientation_X']/data['mod_quat']
    data['norm_orientation_Z'] = data['orientation_X']/data['mod_quat']
    data['norm_orientation_W'] = data['orientation_X']/data['mod_quat']
    return data

def add_euler_angles(data):
    # Derives Euler angles from the quaternion for an input dataset 'data'
    # *Requires normalized quaternion orientations first*
    x = data['norm_orientation_X'].tolist()
    y = data['norm_orientation_Y'].tolist()
    z = data['norm_orientation_Z'].tolist()
    w = data['norm_orientation_W'].tolist()
    eX, eY, eZ = [],[],[]
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        eX.append(xx)
        eY.append(yy)
        eZ.append(zz)
    data['euler_X'] = eX
    data['euler_Y'] = eY
    data['euler_Z'] = eZ
    return data

def add_direction_vectors(data):
    # Derives unit direction vectors from Euler angles in dataset 'data'
    roll = data['euler_X'].tolist()
    pitch = data['euler_Y'].tolist()
    yaw = data['euler_Z'].tolist()
    uX, uY, uZ = [],[],[]
    for i in range(len(roll)):
        xx = math.cos(yaw[i])*math.cos(pitch[i])
        yy = math.sin(yaw[i])+math.cos(pitch[i])
        zz = math.sin(pitch[i])
    return data

def descriptive_statistics(features, data, stats):
    # Creates descriptive statistics such as max, min, std. dev, mean, median, etc. from
    # features 'stats' in dataset 'data' and stores these in 'features'
    for stat in stats:
        features[stat + '_min'] = 0
    return features

def drop_features(data, drops):
    # Drops a supplied list of dropped features 'drops' from dataset 'data'
    #for drop in drops:
        #data[drop].remove
    return data