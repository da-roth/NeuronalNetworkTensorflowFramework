# training loop for one epoch
def vanilla_train_one_epoch(# training graph from vanilla_training_graph()
                            Regressor                           
                            # params, left to client code
                            learning_rate, batch_size):        
    
    m, n = Regressor.x.shape
    
    # minimization loop over mini-batches
    first = 0 #index
    last = min(batch_size, m) #Assume batch size is 256, than first index = 0, after one loop first is at 256. If last is 750 (again index) we will ahve 3 iterations in while
    while first < m:
        Regressor.session.run(Regressor.minimizer, feed_dict = {
            Regressor.inputs: Regressor.x[first:last], 
            Regressor.labels: Regressor.y[first:last],
            Regressor.learning_rate: learning_rate
        })
        first = last
        last = min(first + batch_size, m)