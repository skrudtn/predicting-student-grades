import torch.nn as nn
import torch.nn.functional as F
from config import Config


class ANN(nn.Module):
    config = Config.instance()

    def __init__(self, is_training_mode):
        super(ANN, self).__init__()
        self.is_training_mode = is_training_mode
        self.linear1 = nn.Linear(self.config.INPUT_SIZE, self.config.HIDDEN_SIZE)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(self.config.HIDDEN_SIZE, self.config.OUTPUT_SIZE)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.ReLU(x)
        x = F.dropout(x, training=self.is_training_mode)
        x = self.linear2(x)
        return x

