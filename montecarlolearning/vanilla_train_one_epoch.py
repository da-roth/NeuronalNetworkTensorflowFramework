import tensorflow as tf2
#print("TF version =", tf2.__version__)
# we want TF 2.x
assert tf2.__version__ >= "2.0"
# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

import numpy as np

# training loop for one epoch
def vanilla_train_one_epoch(# training graph from vanilla_training_graph()
                            Regressor,                          
                            # params, left to client code
                            learning_rate, batch_size):        
    
    m, n = Regressor.x.shape

    if isinstance(Regressor.x_raw, tf.Tensor):
        Regressor.session.run(Regressor.minimizer,feed_dict = {Regressor.learning_rate: learning_rate, Regressor.predictionsTest: np.random.rand(1, Regressor.n), Regressor.isTraining: True})
    else:
        first = 0 #index
        last = min(batch_size, m) #Assume batch size is 256, than first index = 0, after one loop first is at 256. If last is 750 (again index) we will ahve 3 iterations in while
        Regressor.session.run(Regressor.minimizer, feed_dict = {
            Regressor.inputs: Regressor.x[first:last], 
            Regressor.labels: Regressor.y[first:last],
            Regressor.learning_rate: learning_rate
        })
        first = last
        last = min(first + batch_size, m)