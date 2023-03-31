class TrainingSettings:

    def __init__(self):
        # Mandatory for all:
        # self._learning_rate_schedule = [
        #     (0.0, 0.01), 
        #     (0.2, 0.001), 
        #     (0.4, 0.0001), 
        #     (0.6, 0.00001), 
        #     (0.8, 0.000001)]
        
        # 1. a) For TrainingMethod.Standard
        self._epochs = 1
        self._batches_per_epoch = 10
        self._min_batch_size = 20

        # 1. b) For TrainingMethod.GenerateDataDuringTraining
        self._testFrequency = 1000
        self._nTest = 1000
        self._samplesPerStep = 1000
        self._trainingSteps = 100

    @property
    def epochs(self):
        return self._epochs

    def set_epochs(self, value):
        self._epochs = value

    @property
    def learningRateSchedule(self):
        return self._learning_rate_schedule

    def set_learning_rate_schedule(self, value):
        self._learning_rate_schedule = value

    @property
    def batchesPerEpoch(self):
        return self._batches_per_epoch

    def set_batches_per_epoch(self, value):
        self._batches_per_epoch = value

    @property
    def minBatchSize(self):
        return self._min_batch_size

    def set_min_batch_size(self, value):
        self._min_batch_size = value

    @property
    def testFrequency(self):
        return self._testFrequency
    
    def set_test_frequency(self, value):
        self._testFrequency = value

    @property
    def nTest(self):
        return self._nTest

    def set_nTest(self, value):
        self._nTest = value

    @property
    def TrainingSteps(self):
        return self._trainingSteps
    
    def set_trainingSteps(self, trainingSetSizes):
        self._trainingSteps = trainingSetSizes

    @property
    def SamplesPerStep(self):
        return self._samplesPerStep
    
    def set_samplesPerStep(self, trainingSetSizes):
        self._samplesPerStep = trainingSetSizes
