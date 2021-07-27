
import os
import cv2
import time
import pickle
import numpy as np
import xarray as xr
import os.path as osp
import torch.utils.data as tordata

from .utils import TripletSampler, collate_fn

__all__ = ['CASIA', 'OUMVLP']


def CASIA(batch_size=[8, 16], test_batch_size=1, num_workers=3,
          dataset_path='./data/CASIA/processed/',
          list_path='./data/CASIA/list/', frame_num=30,
          resolution=64, pid_num=73, pid_shuffle=False, cache=True):

    return _CASIA_OUMVLP(batch_size, test_batch_size, num_workers,
                         dataset_path, list_path, frame_num, resolution,
                         pid_num, pid_shuffle, cache)


def OUMVLP(batch_size=[32, 16], test_batch_size=1, num_workers=3,
           dataset_path='./data/OUMVLP/processed/',
           list_path='./data/OUMVLP/list/', frame_num=30,
           resolution=64, pid_num=5153, pid_shuffle=False, cache=True):

    return _CASIA_OUMVLP(batch_size, test_batch_size, num_workers,
                         dataset_path, list_path, frame_num, resolution,
                         pid_num, pid_shuffle, cache)


def _CASIA_OUMVLP(batch_size=[8, 16], test_batch_size=1, num_workers=3,
                  dataset_path='./data/CASIA/processed/',
                  list_path='./data/CASIA/list/', frame_num=30,
                  resolution=64, pid_num=73, pid_shuffle=False, cache=True):
    r""" Construct CASIA's trainset and testset

    Args:
        dataset_path:
            directory of dataset
        resolution: 64
        pid_num:
            number of subjects for training, the rest is for testing. For
            CASIA-B, 73 subjects for training and 40 subjects for testing.
        pid_shuffle:
            whether shuffle the order of id list. Default: False

    Variable:
        pid_list:
            partition of train and test set. pid_list[0] is a list of
            training subjects, pid_list[1] is a list of testing subjects.
        label:
            list of every sequence's ID. len(label) == num_sequence
        seq_dir:
            list of every sequence's directory.
        view:
            list of every sequence's view-point.
        seq_type:
            list of every sequence's condition.

    """
    train_list_path = osp.join(list_path, '{}_{}_train_list.npy'.format(
        pid_num, pid_shuffle))
    test_list_path = osp.join(list_path, '{}_{}_test_list.npy'.format(
        pid_num, pid_shuffle))

    if osp.exists(train_list_path) and osp.exists(test_list_path):
        train_list = np.load(train_list_path, allow_pickle=True)
        test_list = np.load(test_list_path, allow_pickle=True)

    else:
    # either train or test list file does not exist
        seq_dir = list()
        view = list()
        seq_type = list()
        label = list()
        for _label in sorted(list(os.listdir(dataset_path))):
            # In CASIA-B, data of subject #5 is incomplete.
            # Thus, we ignore it in training.
            if 'CASIA' in dataset_path and _label == '005':
                continue
            label_path = osp.join(dataset_path, _label)
            for _seq_type in sorted(list(os.listdir(label_path))):
                seq_type_path = osp.join(label_path, _seq_type)
                for _view in sorted(list(os.listdir(seq_type_path))):
                    _seq_dir = osp.join(seq_type_path, _view)
                    seqs = os.listdir(_seq_dir)
                    if len(seqs) > 0:
                        seq_dir.append([_seq_dir])
                        label.append(_label)
                        seq_type.append(_seq_type)
                        view.append(_view)

        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]

        train_list = np.array([
            [seq_dir[i] for i, l in enumerate(label) if l in pid_list[0]],
            [label[i] for i, l in enumerate(label) if l in pid_list[0]],
            [seq_type[i] for i, l in enumerate(label) if l in pid_list[0]],
            [view[i] for i, l in enumerate(label) if l in pid_list[0]],
        ])
        np.save(train_list_path, train_list)

        test_list = np.array([
            [seq_dir[i] for i, l in enumerate(label) if l in pid_list[1]],
            [label[i] for i, l in enumerate(label) if l in pid_list[1]],
            [seq_type[i] for i, l in enumerate(label) if l in pid_list[1]],
            [view[i] for i, l in enumerate(label) if l in pid_list[1]],
        ])
        np.save(test_list_path, test_list)

    train_index_path = osp.join(list_path, '{}_{}_train_index.npy'.format(
        pid_num, pid_shuffle))
    test_index_path = osp.join(list_path, '{}_{}_test_index.npy'.format(
        pid_num, pid_shuffle))

    if osp.exists(train_index_path) and osp.exists(test_index_path):
        train_index = np.load(train_index_path, allow_pickle=True)
        test_index = np.load(test_index_path, allow_pickle=True)

        train_source = DataSet(train_list[0].tolist(),
                               train_list[1].tolist(),
                               train_list[2].tolist(),
                               train_list[3].tolist(),
                               cache, resolution,
                               train_index)

        test_source = DataSet(test_list[0].tolist(),
                              test_list[1].tolist(),
                              test_list[2].tolist(),
                              test_list[3].tolist(),
                              cache, resolution,
                              test_index)

    else:
        train_source = DataSet(train_list[0].tolist(),
                               train_list[1].tolist(),
                               train_list[2].tolist(),
                               train_list[3].tolist(),
                               cache, resolution)

        test_source = DataSet(test_list[0].tolist(),
                              test_list[1].tolist(),
                              test_list[2].tolist(),
                              test_list[3].tolist(),
                              cache, resolution)
        np.save(train_index_path, train_source.index_dict.values)
        np.save(test_index_path, test_source.index_dict.values)

    # construct data loader
    sampler = TripletSampler(train_source, batch_size)
    fn = lambda x: collate_fn(x, frame_num, 'random')
    train_loader = tordata.DataLoader(
        dataset=train_source,
        batch_sampler=sampler,
        collate_fn=fn,
        num_workers=num_workers)

    # sampler=tordata.sampler.SequentialSampler(test_source),
    fn = lambda x: collate_fn(x, frame_num, 'all')
    test_loader = tordata.DataLoader(
        dataset=test_source,
        batch_size=test_batch_size,
        collate_fn=fn,
        num_workers=num_workers)

    return train_loader, test_loader



class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution,
                 index_dict=None):
        r"""
        Attributes:
            data_size: number of sequences
            seq_dir: list of every sequence's directory.
            view: list of every sequence's view-points.
            seq_type: list of every sequence's condition.
            label: list of every sequence's ID.

        Notes:
            In the '__getitem__' method, the returned label is the index of
            specifed label list, not the original ID. Refers to:
            target_label = [train_label_set.index(l) for l in label]
            This will simplify the softmax loss trainning.

        """

        end = time.time()
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = sorted(list(set(self.label)))
        self.seq_type_set = set(self.seq_type)
        self.view_set = set(self.view)

        if index_dict is None:
            index_dict = np.zeros((len(self.label_set),
                          len(self.seq_type_set),
                          len(self.view_set))).astype('int')
            index_dict -= 1
            self.index_dict = xr.DataArray(
                index_dict,
                coords={'label': sorted(list(self.label_set)),
                        'seq_type': sorted(list(self.seq_type_set)),
                        'view': sorted(list(self.view_set))},
                dims=['label', 'seq_type', 'view'])

            for i in range(self.data_size):
                _label = self.label[i]
                _seq_type = self.seq_type[i]
                _view = self.view[i]
                self.index_dict.loc[_label, _seq_type, _view] = i
        else:
            self.index_dict = xr.DataArray(
                index_dict,
                coords={'label': sorted(list(self.label_set)),
                        'seq_type': sorted(list(self.seq_type_set)),
                        'view': sorted(list(self.view_set))},
                dims=['label', 'seq_type', 'view'])


    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    def __getitem__(self, index):
        # pose sequence sampling
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return (data, frame_set, self.view[index], self.seq_type[index],
                self.label_set.index(self.label[index]))

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)
