###Packages
try:
    from vanilla_net import *
except:
    from montecarlolearning.vanilla_net import *
import tensorflow as tf2
#print("TF version =", tf2.__version__)
# we want TF 2.x
assert tf2.__version__ >= "2.0"
# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

### Backpropagation
# compute d_output/d_inputs by (explicit) backprop in vanilla net
def backprop(
    weights_and_biases, # 2nd output from vanilla_net() 
    zs,
    activationFunctionsHidden,  # tensorflow activation functions for hidden layer
    activationFunctionOutput,  # tensorflow activation functions for the output layer
    ):                # 3rd output from vanilla_net()
    
    ws, bs = weights_and_biases
    L = len(zs) - 1
    
    # if activationFunctionsHidden is only one function use it in each hidden layer, otherwise each entry per layer
    if type(activationFunctionsHidden) == list:
        if len(activationFunctionsHidden) == 1:
            activationFunctionHidden = [activationFunctionsHidden[0]] * (L-1)
        elif len(activationFunctionsHidden) < (L-1):
            print('amount of activation functions must be one or amount of hidden layers')
        else:
            activationFunctionHidden = activationFunctionsHidden
    else:
        activationFunctionHidden = [activationFunctionsHidden] * (L-1)
    
    # backpropagation, eq. 4, l=L..1
    zbar = tf.ones_like(zs[L]) # zbar_L = 1
    # output layer
    zbar = (zbar @ tf.transpose(ws[L])) * activationFunctionOutput(zs[L-1])
    for l in range(L-2, 0, -1):
        zbar = (zbar @ tf.transpose(ws[l+1])) * activationFunctionHidden[l](zs[l]) # eq. 4
    # for l=0
    zbar = zbar @ tf.transpose(ws[1]) # eq. 4
    
    xbar = zbar # xbar = zbar_0
    
    # dz[L] / dx
    return xbar    

# combined graph for valuation and differentiation
def twin_net(input_dim, hidden_units, hidden_layers, activationFunctionsHidden, activationFunctionOutput,  seed):
    
    # first, build the feedforward net
    xs, (ws, bs), zs, ys, isTraining = vanilla_net(input_dim, hidden_units, hidden_layers, activationFunctionsHidden, activationFunctionOutput, seed)
    
    # then, build its differentiation by backprop
    xbar = backprop((ws, bs), zs, activationFunctionsHidden, activationFunctionOutput)
    
    # return input x, output y and differentials d_y/d_z
    return xs, ys, xbar