import os
import torch
import torch.utils.data as data
from datasets import StudentPerformance


class BaseData(object):
    """ Base class of the data model.
    """
    def __init__(self):
        self.cuda = torch.cuda.is_available()


class UCIStudentPerformance(BaseData):
    root = os.path.join("data", "student")

    def __init__(self, batch_size, is_eval, debug):
        BaseData.__init__(self)
        self.batch_size = batch_size
        self.data = StudentPerformance(
            self.root,
            train=not is_eval,
            debug_mode=debug,
        )

    def load(self):
        additional_options = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            **additional_options)

