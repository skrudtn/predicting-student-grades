import csv
import os
import torch
import torch.utils.data as data
import numpy as np
from helper import Helper
from config import Config


class StudentPerformance(data.Dataset):
    config = Config()
    helper = Helper()

    def __init__(self, root, train=True, debug_mode=False, subject='mat'):
        self.root = os.path.expanduser(root)
        self.train = train
        self.debug = debug_mode
        self.processed_data_path = os.path.join(self.root, 'processed')

        if subject is 'mat':
            self.csv_data_file = os.path.join(self.root, 'student-mat.csv')
            self.test_set_size = self.config.MAT_TEST_SET_SIZE
            self.training_file = os.path.join(self.processed_data_path, 'training-mat.pt')
            self.test_file = os.path.join(self.processed_data_path, 'test-mat.pt')
        elif subject is 'por':
            self.csv_data_file = os.path.join(self.root, 'student-por.csv')
            self.test_set_size = self.config.POR_TEST_SET_SIZE
            self.training_file = os.path.join(self.processed_data_path, 'training-por.pt')
            self.test_file = os.path.join(self.processed_data_path, 'test-por.pt')
        else:
            print('file not found' + subject)
            return

        if not self.helper.check_path_exists(self.processed_data_path):
            self.pre_process()

        if self.train:
            self.loaded_data = torch.load(self.training_file)
        else:
            self.loaded_data = torch.load(self.test_file)

    def pre_process(self):
        print('preprocess start')
        mode = 'training'
        preprocessed = {
            'training': {
                'data': [],
                'targets': [],
            },
            'test': {
                'data': [],
                'targets': [],
            }
        }

        with open(self.csv_data_file, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            lines = []
            for line in rdr:
                line = line[0]
                line = line.replace('\"', '').replace('\'', '')\
                    .replace('yes', '1').replace('no', '0')\
                    .replace('GP', '1').replace('MS', '0')\
                    .replace('GT3', '1').replace('LE3', '0')\
                    .replace('GP', '1').replace('MS', '0')\
                    .replace('F', '1').replace('M', '0')\
                    .replace('U', '1').replace('R', '0')\
                    .replace('T', '1').replace('A', '0')\
                    .replace('mother', '2').replace('father', '1').replace('other', '0')\
                    .replace('teacher', '4').replace('health', '3').replace('services', '2')\
                    .replace('at_home', '1').replace('other', '0').replace('teacher', '4')\
                    .replace('home', '3').replace('reputation', '2')\
                    .replace('course', '1').replace('other', '0')\
                    .replace('home', '3').replace('reputation', '2')\
                    .replace('course', '1').replace('other', '0')

                line = line.split(";")
                lines.append(line)
            lines = lines[1:]

            for i in range(len(lines)):
                for j in range(len(lines[i])):
                    lines[i][j] = float(lines[i][j])

            for line in lines:
                preprocessed[mode]['data'].append(line[:-1])
                preprocessed[mode]['targets'].append(line[-1:])

            preprocessed[mode]['targets'] = torch.from_numpy(np.array(preprocessed[mode]['targets']))
            preprocessed[mode]['data'] = torch.from_numpy(np.array(preprocessed[mode]['data']))
            f.close()

        try:
            os.mkdir(self.processed_data_path)
        except FileExistsError:
            pass
        with open(self.training_file, 'wb') as f:
            torch.save(preprocessed['training'], f)
        with open(self.test_file, 'wb') as f:
            torch.save(preprocessed['test'], f)

        print('preprocess done')

    def __getitem__(self, idx):
        return self.loaded_data['data'][idx], self.loaded_data['targets'][idx]

    def __len__(self):
        return len(self.loaded_data['data'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
