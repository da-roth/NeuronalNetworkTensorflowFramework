class TrainingSettings:
    
    """
    A class for storing training settings.

    Attributes:
    ----------
    _learning_rate_schedule (list): A list of tuples representing the learning rate schedule.
    _epochs (int): The number of epochs for standard training.
    _batches_per_epoch (int): The number of batches per epoch for standard training.
    _min_batch_size (int): The minimum batch size for standard training.
    _madeSteps (int): The number of training steps made for generate data during training method.
    _testFrequency (int): The frequency of testing during generate data during training method.
    _nTest (int): The number of test cases during generate data during training method.
    _mcRounds (int): The number of Monte Carlo rounds during generate data during training method.
    _samplesPerStep (int): The number of samples per step during generate data during training method.
    _trainingSteps (int): The number of training steps during generate data during training method.
    _useExponentialDecay (bool): A flag indicating whether or not to use exponential decay.
    _initial_learning_rate (float): The initial learning rate for exponential decay.
    _decay_steps (int): The number of decay steps for exponential decay.
    _decay_rate (float): The decay rate for exponential decay.

    Methods:
    -------
    increaseMadeSteps(): Increases the made steps.
    useExponentialDecay(initial_learning_rate, decay_rate, decay_steps): Sets up the exponential decay.
    set_epochs(value): Sets the number of epochs.
    set_fileName(value): Sets the file name.
    set_learning_rate_schedule(value): Sets the learning rate schedule.
    set_batches_per_epoch(value): Sets the number of batches per epoch.
    set_min_batch_size(value): Sets the minimum batch size.
    set_test_frequency(value): Sets the test frequency.
    set_nTest(value): Sets the number of test cases.
    set_mcRounds(value): Sets the number of Monte Carlo rounds.
    set_trainingSteps(trainingSetSizes): Sets the number of training steps.
    set_samplesPerStep(trainingSetSizes): Sets the number of samples per step.
    """

    def __init__(self):
        # Mandatory for all:
        self._learning_rate_schedule = [
            (0.0, 0.01), 
            (0.2, 0.001), 
            (0.4, 0.0001), 
            (0.6, 0.00001), 
            (0.8, 0.000001)]
        
        # 1. a) For TrainingMethod.Standard
        self._epochs = 1
        self._batches_per_epoch = 1
        self._min_batch_size = 20

        # 1. b) For TrainingMethod.GenerateDataDuringTraining
        self._madeSteps = 0
        self._testFrequency = 1000
        self._nTest = 1000              # batch_size_approx
        self._mcRounds = 1              # How many rounds with "Samples for testing"
        self._samplesPerStep = 1000
        self._trainingSteps = 100
        
        #2. Learning rate schedule
        self._useExponentialDecay = False

    @property
    def madeSteps(self):
        return self._madeSteps

    def increaseMadeSteps(self):
        self._madeSteps = self._madeSteps + 1

    @property
    def usingExponentialDecay(self):
        return self._useExponentialDecay
    
    def useExponentialDecay(self, initial_learning_rate, decay_rate, decay_steps):
        self._useExponentialDecay = True
        self._initial_learning_rate = initial_learning_rate
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate

    @property
    def InitialLearningRate(self):
        return self._initial_learning_rate
    
    @property
    def DecaySteps(self):
        return self._decay_steps
    
    @property
    def DecayRate(self):
        return self._decay_rate

    @property
    def epochs(self):
        return self._epochs

    def set_epochs(self, value):
        self._epochs = value

    @property
    def FileName(self):
        return self._fileName

    def set_fileName(self, value):
        self._fileName = value


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
    def mcRounds(self):
        return self._mcRounds

    def set_mcRounds(self, value):
        self._mcRounds = value

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
