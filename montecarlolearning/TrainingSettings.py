class TrainingSettings:

    _epochs = None
    _learning_rate_schedule = None
    _batches_per_epoch = None
    _min_batch_size = None

    def __init__(self):
        self._epochs = 1
        self._learning_rate_schedule = [
            (0.0, 0.01), 
            (0.2, 0.001), 
            (0.4, 0.0001), 
            (0.6, 0.00001), 
            (0.8, 0.000001)]
        self._batches_per_epoch = 10
        self._min_batch_size = 20

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