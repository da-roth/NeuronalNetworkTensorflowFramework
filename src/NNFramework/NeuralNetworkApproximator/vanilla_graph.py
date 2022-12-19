from vanilla_net import *
from vanilla_net_biasNeuron import *
from backprop import *

def vanilla_training_graph(input_dim, hiddenNeurons, hiddenLayers, activationFunctionsHidden, activationFunctionOutput, seed, biasNeuron = False):
    
    # net
    if biasNeuron:
        inputs, weights_and_biases, layers, predictions = \
            vanilla_net_biasNeuron(input_dim, hiddenNeurons, hiddenLayers, activationFunctionsHidden,activationFunctionOutput , seed)
    else:
        inputs, weights_and_biases, layers, predictions = \
            vanilla_net(input_dim, hiddenNeurons, hiddenLayers, activationFunctionsHidden,activationFunctionOutput , seed)
    
    # backprop even though we are not USING differentials for training
    # we still need them to predict derivatives dy_dx 
    derivs_predictions = backprop(weights_and_biases, layers, activationFunctionsHidden, activationFunctionOutput)
    
    # placeholder for labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    
    # loss 
    loss = tf.losses.mean_squared_error(labels, predictions)
    #loss = tf.losses.absolute_difference(labels, predictions)
    
    # optimizer
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    #optimizer = tf.train.AdamOptimizer()
    
    # return all necessary 
    return inputs, labels, predictions, derivs_predictions, learning_rate, loss, optimizer.minimize(loss)


