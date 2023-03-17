try:
    from vanilla_graph import *
    from vanilla_train_one_epoch import *
    from diff_training_graph import *
    from TrainingSettings import *
    from TrainingOptionEnums import *
except:
    from montecarlolearning.vanilla_graph import *
    from montecarlolearning.vanilla_train_one_epoch import *
    from montecarlolearning.diff_training_graph import *
    from montecarlolearning.TrainingSettings import *
    from montecarlolearning.TrainingOptionEnums import *

import numpy as np

def train(description,
          # neural Regressor
          Regressor,              
          # training params
          TrainingSettings,
          reinit=True, 
          # callback function and when to call it
          callback=None,           # arbitrary callable
          callback_epochs=[],   # call after what TrainingSettings.epochs, e.g. [5, 20]
          xTest=None,
          yTest=None
          ):     
              
    # batching
    batch_size = max(TrainingSettings.minBatchSize, Regressor.m // TrainingSettings.batchesPerEpoch)
           
    # one-cycle learning rate sechedule
    if TrainingSettings.usingExponentialDecay: 
        # The non-staircase version of the exponential decay learning rate schedule
        #lr_schedule_rates = [TrainingSettings.InitialLearningRate * TrainingSettings.DecayRate ** (i / TrainingSettings.DecaySteps) for i in range(TrainingSettings.TrainingSteps+1)]
        # The staircase version of the exponential decay learning rate schedule
        lr_schedule_rates = [TrainingSettings.InitialLearningRate * TrainingSettings.DecayRate ** (i // ( TrainingSettings.DecaySteps)) for i in range(TrainingSettings.TrainingSteps+1)]
        lr_schedule_epochs = np.linspace(0, 1, TrainingSettings.TrainingSteps+1)
    else:    
        lr_schedule_epochs, lr_schedule_rates = zip(*TrainingSettings.learningRateSchedule)
            

    # reset
    if reinit:
        Regressor.session.run(Regressor.initializer)
    
    # callback on epoch 0, if requested
    if callback and 0 in callback_epochs:
        callback(Regressor, 0)
        
    # Frequency counter    
    i = 0

    # Print training setting
    #print('Training will be done with the following settings:\n')
    #print('TrainingSettings.minBatchSize: ' + str(TrainingSettings.minBatchSize))
    #print('batches per epoch: ' + str(TrainingSettings.epochsTrainingSettings.learningRateScheduleTrainingSettings.batchesPerEpoch))
    #print('Used batch_size will be: ' + str(batch_size))
    #print('TrainingSettings.epochs: ' + str(TrainingSettings.epochs))
    #print('TrainingSettings.epochsTrainingSettings.learningRateSchedule: ' + str(TrainingSettings.epochsTrainingSettings.learningRateSchedule))
    
    #print('\n Train started:')
    # loop on TrainingSettings.epochs, with progress bar (tqdm)
    for epoch in range(TrainingSettings.epochs):
        
        # interpolate learning rate in cycle
        if Regressor.Generator.TrainMethod == TrainingMethod.Standard:
            learning_rate = np.interp(epoch / TrainingSettings.epochs, lr_schedule_epochs, lr_schedule_rates)
        elif Regressor.Generator.TrainMethod == TrainingMethod.GenerateDataDuringTraining:
            learning_rate = np.interp(TrainingSettings.madeSteps / TrainingSettings.TrainingSteps, lr_schedule_epochs, lr_schedule_rates)
            TrainingSettings.increaseMadeSteps()
        
        # train one epoch
        if not Regressor._Generator.Differential:
        
            vanilla_train_one_epoch(
                Regressor.inputs, 
                Regressor.labels, 
                Regressor.learning_rate, 
                Regressor.minimizer, 
                Regressor.x, 
                Regressor.y, 
                learning_rate, 
                batch_size, 
                Regressor.session)
            i = i +1
            if i % TrainingSettings.testFrequency == 0:
                #print('learning_rate for the last epoch was' + str(learning_rate))
                predictions = Regressor.predict_values(Regressor.x_raw)
                errors = predictions - Regressor.y_raw
                rmse = np.sqrt((errors ** 2).mean())
                if not (xTest == None):
                    predictionsTest = Regressor.predict_values(xTest)
                    errorsTest = predictionsTest - yTest
                    rmseTest = np.sqrt((errorsTest ** 2).mean())
                    #print('RMSE on training data after ' + str(i) + ' TrainingSettings.epochs is '  + str(rmse) + '. RMSE on test data: ' + str(rmseTest) )
                #else:
                #    #print('RMSE on training data after ' + str(i) + ' TrainingSettings.epochs is '  + str(rmse) )
        else:
        
            diff_train_one_epoch(
                Regressor.inputs, 
                Regressor.labels, 
                Regressor.derivs_labels,
                Regressor.learning_rate, 
                Regressor.minimizer, 
                Regressor.x, 
                Regressor.y, 
                Regressor.dy_dx,
                learning_rate, 
                batch_size, 
                Regressor.session)
        
        # callback, if requested
        if callback and epoch in callback_epochs:
            callback(Regressor, epoch)

    #print('Training done.')
    # final callback, if requested
    if callback and TrainingSettings.epochs in callback_epochs:
        callback(Regressor, TrainingSettings.epochs)        

