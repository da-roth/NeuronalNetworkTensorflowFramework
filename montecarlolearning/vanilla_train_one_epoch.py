import tensorflow as tf2
#print("TF version =", tf2.__version__)
# we want TF 2.x
assert tf2.__version__ >= "2.0"
# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

# training loop for one epoch
def vanilla_train_one_epoch(# training graph from vanilla_training_graph()
                            Regressor,                          
                            # params, left to client code
                            learning_rate, batch_size):        
    
    if isinstance(Regressor.x, tf.Tensor):
        input_tensor = tf.convert_to_tensor(Regressor.x, dtype=tf.float32)
        label_tensor = tf.convert_to_tensor(Regressor.y, dtype=tf.float32)
        Regressor.session.run(Regressor.minimizer,feed_dict = {Regressor.learning_rate: learning_rate})
    else:
        m, n = Regressor.x.shape
        # minimization loop over mini-batches
        Regressor.session.run(Regressor.minimizer, feed_dict = {
                Regressor.inputs: Regressor.x, 
                Regressor.labels: Regressor.y,
                Regressor.learning_rate: learning_rate
            })
    
