# training loop for one epoch
def vanilla_train_one_epoch(# training graph from vanilla_training_graph()
                            Regressor,                          
                            # params, left to client code
                            learning_rate, batch_size):        
    
    m, n = Regressor.x.shape
    
    # minimization loop over mini-batches
    Regressor.session.run(Regressor.minimizer, feed_dict = {
            Regressor.inputs: Regressor.x, 
            Regressor.labels: Regressor.y,
            Regressor.learning_rate: learning_rate
        })