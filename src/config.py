import torch


class ConstSingleton:
    _instance = None

    @classmethod
    def __getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls._instance

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise Exception("Can't rebind const(%s)" % name)
        self.__dict__[name] = value


class Config(ConstSingleton):
    """Set Hyper-parameters of models in here.
    """
    def __init__(self):
        # [Data]
        self.TEST_SET_SIZE = 100

        # [Train]
        self.LEARNING_RATE = 0.001
        self.MAX_EPOCH = 300
        self.WEIGHT_DECAY = 0.0003
        self.CRITERION = torch.nn.MSELoss()

        # [Model]
        self.INPUT_SIZE = 32
        self.HIDDEN_SIZE = 12
        self.OUTPUT_SIZE = 1
        self.BATCH_SIZE = 2

        # [ETC]
        self.DEBUG_MODE = False
        self.LOGGING_ENABLE = False
        self.CHECKPOINT_ENABLE = False
        self.TRAIN_MODE = True
