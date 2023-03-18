import numpy as np

import tensorflow as tf2
#print("TF version =", tf2.__version__)
# we want TF 2.x
assert tf2.__version__ >= "2.0"
# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

# basic data preparation
epsilon = 1.0e-08
def normalize_data(x_raw, y_raw, dydx_raw=None, crop=None):
    if isinstance(x_raw, tf.Tensor):
        # crop dataset
        m = crop if crop is not None else x_raw.shape[0]
        x_cropped = x_raw[:m]
        y_cropped = y_raw[:m]
        dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None
        
        # normalize dataset
        x_mean = tf.math.reduce_mean(x_cropped, axis=0)
        x_std = tf.math.reduce_std(x_cropped, axis=0) + epsilon
        x = (x_cropped - x_mean) / x_std
        y_mean = tf.math.reduce_mean(y_cropped, axis=0)
        y_std = tf.math.reduce_std(y_cropped, axis=0) + epsilon
        y = (y_cropped - y_mean) / y_std
        
        # normalize derivatives too
        if dycropped_dxcropped is not None:
            dy_dx = dycropped_dxcropped / y_std * x_std
            # weights of derivatives in cost function = (quad) mean size
            lambda_j = 1.0 / tf.sqrt((dy_dx ** 2).mean(axis=0)).numpy().reshape(1, -1)
        else:
            dy_dx = None
            lambda_j = None        
        return x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j
    
    else:
        # crop dataset
        m = crop if crop is not None else x_raw.shape[0]
        x_cropped = x_raw[:m]
        y_cropped = y_raw[:m]
        dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None
        
        # normalize dataset
        x_mean = x_cropped.mean(axis=0)
        x_std = x_cropped.std(axis=0) + epsilon
        x = (x_cropped- x_mean) / x_std
        y_mean = y_cropped.mean(axis=0)
        y_std = y_cropped.std(axis=0) + epsilon
        y = (y_cropped-y_mean) / y_std
        
        # normalize derivatives too
        if dycropped_dxcropped is not None:
            dy_dx = dycropped_dxcropped / y_std * x_std 
            # weights of derivatives in cost function = (quad) mean size
            lambda_j = 1.0 / np.sqrt((dy_dx ** 2).mean(axis=0)).reshape(1, -1)
        else:
            dy_dx = None
            lambda_j = None
        
        return x_mean, x_std, x, y_mean, y_std, y, dy_dx, lambda_j

def normalize_data_NewData(x_raw, y_raw, x_mean, x_std, y_mean, y_std, dydx_raw=None, crop=None):
    
    if isinstance(x_raw, tf.Tensor):
        # crop dataset
        m = crop if crop is not None else x_raw.shape[0]
        x_cropped = x_raw[:m]
        y_cropped = y_raw[:m]
        dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None
        
        # normalize dataset
        x = (x_cropped - x_mean) / x_std
        y = (y_cropped - y_mean) / y_std
        
        # normalize derivatives too
        if dycropped_dxcropped is not None:
            dy_dx = dycropped_dxcropped / y_std * x_std 
            # weights of derivatives in cost function = (quad) mean size
            lambda_j = 1.0 / tf.math.sqrt(tf.math.reduce_mean(tf.math.square(dy_dx), axis=0, keepdims=True))
        else:
            dy_dx = None
            lambda_j = None
        
        return x, y, dy_dx, lambda_j
    else:
        # crop dataset
        m = crop if crop is not None else x_raw.shape[0]
        x_cropped = x_raw[:m]
        y_cropped = y_raw[:m]
        dycropped_dxcropped = dydx_raw[:m] if dydx_raw is not None else None
        
        # normalize dataset
        x_mean = x_cropped.mean(axis=0)
        x_std = x_cropped.std(axis=0) + epsilon
        x = (x_cropped- x_mean) / x_std
        y_mean = y_cropped.mean(axis=0)
        y_std = y_cropped.std(axis=0) + epsilon
        y = (y_cropped-y_mean) / y_std
        
        # normalize derivatives too
        if dycropped_dxcropped is not None:
            dy_dx = dycropped_dxcropped / y_std * x_std 
            # weights of derivatives in cost function = (quad) mean size
            lambda_j = 1.0 / np.sqrt((dy_dx ** 2).mean(axis=0)).reshape(1, -1)
        else:
            dy_dx = None
            lambda_j = None
        
        return x, y, dy_dx, lambda_j
