try:
    from vanilla_graph import *
    from vanilla_train_one_epoch import *
    from diff_training_graph import *
except:
    from montecarlolearning.vanilla_graph import *
    from montecarlolearning.vanilla_train_one_epoch import *
    from montecarlolearning.diff_training_graph import *

import numpy as np

def train(description,
          # neural approximator
          approximator,              
          # training params
          epochs, 
          # one-cycle learning rate schedule
          learning_rate_schedule,
          batches_per_epoch,
          min_batch_size,
          reinit=True, 
          # callback function and when to call it
          callback=None,           # arbitrary callable
          callback_epochs=[],   # call after what epochs, e.g. [5, 20]
          xTest=None,
          yTest=None
          ):     
              
    # batching
    batch_size = max(min_batch_size, approximator.m // batches_per_epoch)
    
    # one-cycle learning rate sechedule
    lr_schedule_epochs, lr_schedule_rates = zip(*learning_rate_schedule)
            
    # reset
    if reinit:
        approximator.session.run(approximator.initializer)
    
    # callback on epoch 0, if requested
    if callback and 0 in callback_epochs:
        callback(approximator, 0)
        
    # Frequency counter    
    i = 0
    testFrequency = epochs/10
    
    # Print training setting
    #print('Training will be done with the following settings:\n')
    #print('min_batch_size: ' + str(min_batch_size))
    #print('batches per epoch: ' + str(batches_per_epoch))
    print('Used batch_size will be: ' + str(batch_size))
    #print('epochs: ' + str(epochs))
    #print('learning_rate_schedule: ' + str(learning_rate_schedule))
    
    #print('\n Train started:')
    # loop on epochs, with progress bar (tqdm)
    for epoch in range(epochs):
        
        # interpolate learning rate in cycle
        learning_rate = np.interp(epoch / epochs, lr_schedule_epochs, lr_schedule_rates)
        
        # train one epoch
        if not approximator.differential:
        
            vanilla_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                learning_rate, 
                batch_size, 
                approximator.session)
            i = i +1
            if i % testFrequency == 0:
                #print('learning_rate for the last epoch was' + str(learning_rate))
                predictions = approximator.predict_values(approximator.x_raw)
                errors = predictions - approximator.y_raw
                rmse = np.sqrt((errors ** 2).mean())
                if not (type(xTest) == None):
                    predictionsTest = approximator.predict_values(xTest)
                    errorsTest = predictionsTest - yTest
                    rmseTest = np.sqrt((errorsTest ** 2).mean())
                    #print('RMSE on training data after ' + str(i) + ' epochs is '  + str(rmse) + '. RMSE on test data: ' + str(rmseTest) )
                #else:
                #    #print('RMSE on training data after ' + str(i) + ' epochs is '  + str(rmse) )
        else:
        
            diff_train_one_epoch(
                approximator.inputs, 
                approximator.labels, 
                approximator.derivs_labels,
                approximator.learning_rate, 
                approximator.minimizer, 
                approximator.x, 
                approximator.y, 
                approximator.dy_dx,
                learning_rate, 
                batch_size, 
                approximator.session)
        
        # callback, if requested
        if callback and epoch in callback_epochs:
            callback(approximator, epoch)

    #print('Training done.')
    # final callback, if requested
    if callback and epochs in callback_epochs:
        callback(approximator, epochs)        

