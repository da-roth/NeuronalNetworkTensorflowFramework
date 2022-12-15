import tensorflow as tf2
print("TF version =", tf2.__version__)

# we want TF 2.x
assert tf2.__version__ >= "2.0"

# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

real_type = tf.float32

def vanilla_net(
    input_dim,           # dimension of inputs, e.g. 10
    hiddenNeurons,       # units in hidden layers, assumed constant, e.g. 20
    hiddenLayers,        # number of hidden layers, e.g. 4
    activationFunctions, # tensorflow activation functions for hidden layer
    seed):               # seed for initialization or None for random
    
    # set seed
    tf.set_random_seed(seed)
    
    # input layer
    xs = tf.placeholder(shape=[None, input_dim], dtype=real_type)
    
    # connection weights and biases of hidden layers
    ws = [None]
    bs = [None]
    # layer 0 (input) has no parameters
    
    # layer 0 = input layer
    zs = [xs] # eq.3, l=0
    
    # first hidden layer (index 1)
    # weight matrix
    ws.append(tf.get_variable("w1", [input_dim, hiddenNeurons], \
        initializer = tf.variance_scaling_initializer(), dtype=real_type))
    # bias vector
    bs.append(tf.get_variable("b1", [hiddenNeurons], \
        initializer = tf.zeros_initializer(), dtype=real_type))
    # graph
    zs.append(zs[0] @ ws[1] + bs[1]) # eq. 3, l=1
    
    # second hidden layer (index 2) to last (index hiddenLayers)
    activationFunction = activationFunctions
    for l in range(1, hiddenLayers): 
        ws.append(tf.get_variable("w%d"%(l+1), [hiddenNeurons, hiddenNeurons], \
            initializer = tf.variance_scaling_initializer(), dtype=real_type))
        bs.append(tf.get_variable("b%d"%(l+1), [hiddenNeurons], \
            initializer = tf.zeros_initializer(), dtype=real_type))
        zs.append(activationFunction(zs[l]) @ ws[l+1] + bs[l+1]) # eq. 3, l=2..L-1

    # output layer (index hiddenLayers+1)
    ws.append(tf.get_variable("w"+str(hiddenLayers+1), [hiddenNeurons, 1], \
            initializer = tf.variance_scaling_initializer(), dtype=real_type))
    bs.append(tf.get_variable("b"+str(hiddenLayers+1), [1], \
        initializer = tf.zeros_initializer(), dtype=real_type))
    # eq. 3, l=L
    zs.append(tf.nn.softplus(zs[hiddenLayers]) @ ws[hiddenLayers+1] + bs[hiddenLayers+1]) 
    #zs.append(zs[hiddenLayers] @ ws[hiddenLayers+1] + bs[hiddenLayers+1]) 
    
    # result = output layer
    ys = zs[hiddenLayers+1]
    
    # return input layer, (parameters = weight matrices and bias vectors), 
    # [all layers] and output layer
    return xs, (ws, bs), zs, ys
