import matplotlib.pyplot as plt
import numpy as np


try:
    from montecarlolearning.TrainingOptionEnums import *
except ModuleNotFoundError:
    from TrainingOptionEnums import *

def plot_results(title, 
          predictions, 
          xTest, 
          xAxisName, 
          yAxisName, 
          targets, 
          sizes, 
          computeRmse=False, 
          differentialML = False,
          weights=None,
          trainingMethod = TrainingMethod.Standard):
    

    if differentialML:
        displayResults = enumerate(["standard", "differential"])
        numCols = 2
    else:
        displayResults = enumerate(["standard"])
        numCols = 1
    numRows = len(sizes)
    
    fig, ax = plt.subplots(numRows, numCols, squeeze=False)
    fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)
    ax[0,0].set_title("standard")

    for i, size in enumerate(sizes):
        ax[i,0].annotate("size %d" % size, xy=(0, 0.5), 
          xytext=(-ax[i,0].yaxis.labelpad-5, 0),
          xycoords=ax[i,0].yaxis.label, textcoords='offset points',
          ha='right', va='center')
    
    if differentialML:
        displayResults = enumerate(["standard", "differential"])
        ax[0,1].set_title("differential")
    else:
        displayResults = enumerate(["standard"])
    
    if trainingMethod == TrainingMethod.Standard:
        for j, regType, in displayResults:
            for i, size in enumerate(sizes):        
                if computeRmse:
                    errors = (predictions[(regType,size)]-targets)
                    if weights is not None:
                        errors /= weights
                    rmse = np.sqrt((errors ** 2).mean(axis=0))
                    t = "rmse %.2f" % rmse
                else:
                    t = xAxisName
                    
                ax[i,j].set_xlabel(t)            
                ax[i,j].set_ylabel(yAxisName)

                # ax[i,j].plot(xTest, predictions[(regType, size)], 'co', \
                #             markersize=2, markerfacecolor='white', label="predicted")
                # ax[i,j].plot(xTest, targets, 'r.', markersize=0.5, label='targets')
                ax[i,j].plot(predictions[(regType, size)], xTest,'co', \
                            markersize=2, markerfacecolor='white', label="predicted")
                ax[i,j].plot(targets,xTest, 'r.', markersize=0.5, label='targets')

                ax[i,j].legend(prop={'size': 8}, loc='upper left')
    elif trainingMethod == TrainingMethod.GenerateDataDuringTraining:
        for j, regType, in displayResults:     
            if computeRmse:
                errors = predictions[("standard",targets.size)]-targets
                if weights is not None:
                    errors /= weights
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                t = "rmse %.2f" % rmse
            else:
                t = xAxisName
                
            ax[i,j].set_xlabel(t)            
            ax[i,j].set_ylabel(yAxisName)

            ax[i,j].plot(xTest, predictions[("standard",targets.size)], 'co', \
                        markersize=2, markerfacecolor='white', label="predicted")
            ax[i,j].plot(xTest, targets, 'r.', markersize=0.5, label='targets')

            ax[i,j].legend(prop={'size': 8}, loc='upper left')
    else:
       print('Training method not recognized')
       
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("% s -- %s" % (title, yAxisName), fontsize=16)
    plt.show()
    
    
