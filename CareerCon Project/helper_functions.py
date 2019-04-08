
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
    # Normalizes the direction-dependent parameters for an input dataset 'data'
    # Specifically, creates unit vectors for orientation, velocity, and acceleration
    data['mod_quat'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2 + data['orientation_W']**2)**.5
    data['norm_orientation_X'] = data['orientation_X']/data['mod_quat']
    data['norm_orientation_Y'] = data['orientation_Y']/data['mod_quat']
    data['norm_orientation_Z'] = data['orientation_Z']/data['mod_quat']
    data['norm_orientation_W'] = data['orientation_W']/data['mod_quat']
    
    data['mod_angular_velocity'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)**.5
    data['norm_velocity_X'] = data['angular_velocity_X']/data['mod_angular_velocity']
    data['norm_velocity_Y'] = data['angular_velocity_Y']/data['mod_angular_velocity']
    data['norm_velocity_Z'] = data['angular_velocity_Z']/data['mod_angular_velocity']
    
    data['mod_linear_acceleration'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**.5
    data['norm_acceleration_X'] = data['linear_acceleration_X']/data['mod_linear_acceleration']
    data['norm_acceleration_Y'] = data['linear_acceleration_Y']/data['mod_linear_acceleration']
    data['norm_acceleration_Z'] = data['linear_acceleration_Z']/data['mod_linear_acceleration']
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
        uX.append(xx)
        uY.append(yy)
        uZ.append(zz)
    data['orientation_vector_X'] = uX
    data['orientation_vector_Y'] = uY
    data['orientation_vector_Z'] = uZ
    return data

def eng_data(data):
    # Creates engineered features within dataset 'data'
    # Intended for use on the raw X data
    
    # Idea 1: Ratios
    data['ratio_velocity-acceleration'] = data['mod_angular_velocity'] / data['mod_linear_acceleration']
    
    return data

def descriptive_features(features, data, stats):
    # Creates descriptive statistics such as max, min, std. dev, mean, median, etc. from
    # features 'stats' in dataset 'data' and stores these in 'features'
    for col in data.columns:
        if col not in stats:
            continue
        # Base statistics
        colData = data.groupby(['series_id'])[col]
        features[col + '_min'] = colData.min()
        features[col + '_max'] = colData.max()
        features[col + '_std'] = colData.std()
        features[col + '_mean'] = colData.mean()
        features[col + '_median'] = colData.median()
        features[col + '_range'] = features[col + '_max']-features[col + '_min']
    return features

def eng_features(features):
    # Creates engineered features within dataset 'features'
    # Intended for use on the modified X data
    
    # Idea 1: Dot and cross products of mean unit direction vectors
    # Note: np.dot and np.cross are very slow to perform on large sets of data,
    # minimize iterations used of them
    Ox = features['orientation_vector_X_mean']
    Oy = features['orientation_vector_Y_mean']
    Oz = features['orientation_vector_Z_mean']
    Vx = features['norm_velocity_X_mean']
    Vy = features['norm_velocity_Y_mean']
    Vz = features['norm_velocity_Z_mean']
    Ax = features['norm_acceleration_X_mean']
    Ay = features['norm_acceleration_Y_mean']
    Az = features['norm_acceleration_Z_mean']
    
    oDv,oDa,vDa = [],[],[]
    oCv_x,oCv_y,oCv_z = [],[],[]
    oCa_x,oCa_y,oCa_z = [],[],[]
    vCa_x,vCa_y,vCa_z = [],[],[]
    for i in range(len(Ox)):
        oDv.append(np.dot([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]]))
        oCv = np.cross([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]])
        oCv_x.append(oCv[0])
        oCv_y.append(oCv[1])
        oCv_z.append(oCv[2])
        oDa.append(np.dot([Ox[i],Oy[i],Oz[i]],[Ax[i],Ay[i],Az[i]]))
        oCa = np.cross([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]])
        oCa_x.append(oCa[0])
        oCa_y.append(oCa[1])
        oCa_z.append(oCa[2])
        vDa.append(np.dot([Vx[i],Vy[i],Vz[i]],[Ax[i],Ay[i],Az[i]]))
        vCa = np.cross([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]])
        vCa_x.append(vCa[0])
        vCa_y.append(vCa[1])
        vCa_z.append(vCa[2])
        
    features['orientation_dot_velocity'] = oDv
    features['orientation_cross_velocity_X'] = oCv_x
    features['orientation_cross_velocity_Y'] = oCv_y
    features['orientation_cross_velocity_Z'] = oCv_z
    features['orientation_dot_acceleration'] = oDa
    features['orientation_cross_acceleration_X'] = oCa_x
    features['orientation_cross_acceleration_Y'] = oCa_y
    features['orientation_cross_acceleration_Z'] = oCa_z
    features['velocity_dot_acceleration'] = vDa
    features['velocity_cross_acceleration_X'] = vCa_x
    features['velocity_cross_acceleration_Y'] = vCa_y
    features['velocity_cross_acceleration_Z'] = vCa_y
    
    # Idea 2: 
    return features
    
def drop_features(data, drops):
    # Drops a supplied list of dropped features 'drops' from dataset 'data'
    #for drop in drops:
        #data[drop].remove
    return data