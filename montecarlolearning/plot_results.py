import matplotlib.pyplot as plt
import numpy as np


try:
    from montecarlolearning.TrainingOptionEnums import *
except ModuleNotFoundError:
    from TrainingOptionEnums import *

def plot_results(title, 
          yPredicted, 
          xTest, 
          yTest,
          Generator,
          computeRmse=True, 
          weights=None):
    

    if Generator.Differential:
        displayResults = enumerate(["standard", "differential"])
        numCols = 2
    else:
        displayResults = enumerate(["standard"])
        numCols = 1

    
    if Generator.TrainMethod == Generator.TrainMethod.Standard:
        numRows = len(Generator.trainingSetSizes)
            
        fig, ax = plt.subplots(numRows, numCols, squeeze=False)
        fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)
        ax[0,0].set_title("standard")

        for i, size in enumerate(Generator.trainingSetSizes):
            ax[i,0].annotate("size %d" % size, xy=(0, 0.5), 
            xytext=(-ax[i,0].yaxis.labelpad-5, 0),
            xycoords=ax[i,0].yaxis.label, textcoords='offset points',
            ha='right', va='center')
        
        if Generator.Differential:
            displayResults = enumerate(["standard", "differential"])
            ax[0,1].set_title("differential")
        else:
            displayResults = enumerate(["standard"])

        for j, regType, in displayResults:
            for i, size in enumerate(Generator.trainingSetSizes):        
                if computeRmse:
                    errors = (yPredicted[(regType,size)]-yTest)
                    if weights is not None:
                        errors /= weights
                    rmse = np.sqrt((errors ** 2).mean(axis=0))
                    t = "rmse %.2f" % rmse
                else:
                    t = Generator.inputName
                    
                ax[i,j].set_xlabel(t)            
                ax[i,j].set_ylabel(Generator.outputName)

                # ax[i,j].plot(xTest, yPredicted[(regType, size)], 'co', \
                #             markersize=2, markerfacecolor='white', label="predicted")
                # ax[i,j].plot(xTest, yTest, 'r.', markersize=0.5, label='yTest')
                ax[i,j].plot(yPredicted[(regType, size)], xTest,'co', \
                            markersize=2, markerfacecolor='white', label="predicted")
                ax[i,j].plot(yTest,xTest, 'r.', markersize=0.5, label='yTest')

                ax[i,j].legend(prop={'size': 8}, loc='upper left')
    elif Generator.TrainMethod == Generator.TrainMethod.GenerateDataDuringTraining:
        fig, ax = plt.subplots()
        fig.set_size_inches(4 + 1.5, 4 )
        ax.set_title("standard")
        ax.annotate("y-axis",xy=(0, 0.5), 
        xytext=(-ax.yaxis.labelpad-5, 0),
        xycoords=ax.yaxis.label, textcoords='offset points',
        ha='right', va='center')


        for j, regType, in displayResults:     
            if computeRmse:
                errors = yPredicted[("standard",yTest.size)]-yTest
                if weights is not None:
                    errors /= weights
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                t = "rmse %.2f" % rmse
            else:
                t = Generator.inputName
                
            ax.set_xlabel(t)            
            ax.set_ylabel(Generator.outputName)

            ax.plot(xTest[:,0], yPredicted[("standard",yTest.size)], 'co', \
                        markersize=2, markerfacecolor='white', label="predicted")
            ax.plot(xTest[:,0], yTest, 'r.', markersize=0.5, label='yTest')

            ax.legend(prop={'size': 8}, loc='upper left')
    else:
       print('Training method not recognized')
       
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("% s -- %s" % (title, Generator.outputName), fontsize=16)
    plt.show()
    
    
