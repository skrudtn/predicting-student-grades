import os
import sys
import time
import glob
import torch
from model import Linear
from loader import UCIStudentPerformance
from config import Config


class Trainer(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_folder = "checkpoints"

    def __init__(self, model, data_loader, optimizer):
        self.data_loader = data_loader.load()
        self.model = model
        self.optimizer = optimizer
        self.prefix = model.__class__.__name__ + "_"
        self.checkpoint_filename = self.prefix + str(int(time.time())) + ".pt"

    # TODO(kyungsoo): Make checkpoint for log
    def save_checkpoint(self, checkpoint):
        try:
            os.mkdir(self.checkpoint_folder)
        except FileExistsError:
            pass
        torch.save(checkpoint, os.path.join(self.checkpoint_folder, self.checkpoint_filename))

    def load_checkpoint(self):
        root = os.path.dirname(sys.modules['__main__'].__file__)
        file_names = glob.glob(os.path.join(root, self.checkpoint_folder, self.prefix + "*.pt"))
        if len(file_names) == 0:
            print("[!] Checkpoint not found.")
            return {}
        file_name = file_names[-1]  # Pick the most recent file.
        print("[+] Checkpoint Loaded. '{}'".format(file_name))
        return torch.load(file_name)

    def train(self, max_epoch, batch_size):
        raise NotImplementedError

    def evaluate(self, batch_size):
        raise NotImplementedError

    @staticmethod
    def get_accuracy(target, output):
        # print("target", target.squeeze())
        # print("output", output.squeeze())
        accuracy = 1-(abs(target.squeeze() - output.squeeze()))/20
        # print("accuracy: ", accuracy)
        return accuracy


class LinearTrainer(Trainer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.instance()

    def __init__(self, model, data_loader, optimizer):
        super().__init__(model, data_loader, optimizer)
        if self.config.LOGGING_ENABLE:
            from tensor_board_logger import TensorBoardLogger
            self.logger = TensorBoardLogger(os.path.join("logs", model.__class__.__name__))

        self.current_epoch = 0

    def train(self, max_epoch, batch_size):
        print("Training started")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.train()
        epoch_resume = 0
        if self.config.CHECKPOINT_ENABLE:
            checkpoint = self.load_checkpoint()
            try:
                epoch_resume = checkpoint["epoch"]
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.model.load_state_dict(checkpoint["model"])
            except KeyError:
                # There is no checkpoint
                pass

        for epoch in range(epoch_resume, max_epoch):
            accuracy_sum = 0
            loss_sum = 0
            self.current_epoch = epoch
            for batch_idx, (data, target) in enumerate(self.data_loader):
                # Transpose vector to make it (num of words / batch size) * batch size * index size(1).
                # _data = np.transpose(_data, (1, 0, 2))
                data, target = data.type(torch.FloatTensor).to(device=self.device), target.type(torch.FloatTensor).to(device=self.device)
                output = self.model(data)
                loss = self.config.CRITERION(output, target)
                loss.backward()
                self.optimizer.step()
                # if self.config.DEBUG_MODE:
                #     print("Train Epoch: {}/{} [{}/{} ({:.0f}%)]".format(
                #         epoch, max_epoch, batch_idx * len(data),
                #         len(self.data_loader.dataset), 100. * batch_idx / len(self.data_loader)))
                #     print("Loss: {:.6f}".format(loss.item()))
                #     print("target : ", target)
                #     print("output : ", output, end="\n\n")
                accuracy = self.get_accuracy(target, output)
                accuracy_sum += accuracy
                loss_sum += loss

            print("accuracy avg : ", accuracy_sum/len(self.data_loader))
            print("loss avg : ", loss_sum/len(self.data_loader))

        print("End")


def main():
    config = Config.instance()
    loader = UCIStudentPerformance(
        batch_size=config.BATCH_SIZE,
        is_eval=False,
        debug=config.DEBUG_MODE)

    model = Linear(
        config.INPUT_SIZE,
        config.HIDDEN_SIZE,
        config.OUTPUT_SIZE,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    trainer = LinearTrainer(model, loader, optimizer)
    trainer.train(config.MAX_EPOCH, config.BATCH_SIZE)


if __name__ == "__main__":
    main()
