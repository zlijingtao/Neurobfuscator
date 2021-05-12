import os
import sys
import re
import numpy as np

import random
import math
import torch.utils.data as data
import pickle
from getSample import all_select_seg

def sb_normalize(list_array):
    for n in range(len(list_array)):
        list_array[n] = list_array[n][:, :, :-1]
        mean = np.mean(list_array[n])
        std = np.std(list_array[n])
        list_array[n] = (list_array[n] - mean)/std
    return list_array
def smart_normalize(list_array):
    np.errstate(invalid='ignore', divide='ignore')
    for n in range(len(list_array)):
        list_array[n] = list_array[n][:, :, :-1]
        norm = np.linalg.norm(list_array[n], axis = 1)
        mean = np.mean(list_array[n], axis = 1)
        list_array[n] = np.nan_to_num((list_array[n]- mean)/norm)
        # list_array[n] = np.nan_to_num(list_array[n]/norm)
        
    
    return list_array
#ToDO: use numpy normalize/and its tf corresponding.

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, meta_filename, normalize = "sb"):
        """Reads source and target sequences from txt files."""

        with open(meta_filename, 'rb') as handle:
            train_data_dict = pickle.load(handle)
            self.train_inputs = train_data_dict['train_inputs_list']
            self.train_targets_sparse = train_data_dict['train_targets_sparse_list']
            self.train_seq_len_list = train_data_dict['train_seq_len_list']
            self.data_index = train_data_dict['index_list']
            self.original = train_data_dict['original_list']
            self.num_total_seqs = len(self.data_index)
            c = list(zip(self.train_inputs, self.train_targets_sparse, self.train_seq_len_list, self.data_index, self.original))
            random.shuffle(c)
            self.train_inputs, self.train_targets_sparse, self.train_seq_len_list, self.data_index, self.original = zip(*c)
            #tuple to list
            self.train_inputs = list(self.train_inputs)
            self.train_targets_sparse = list(self.train_targets_sparse)
            self.train_seq_len_list = list(self.train_seq_len_list)
            self.data_index = list(self.data_index)
            self.original = list(self.original)
            if normalize == "sb":
                self.train_inputs = sb_normalize(self.train_inputs)
            elif normalize == "smart":
                self.train_inputs = smart_normalize(self.train_inputs)
            # print(len(self.train_inputs))
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        train_inputs = self.train_inputs[index]
        # print(train_inputs.shape)
        train_targets_sparse = self.train_targets_sparse[index]
        # print(train_targets_sparse)
        train_seq_len_list = self.train_seq_len_list[index]
        # print(train_seq_len_list)
        data_index = self.data_index[index]
        # print(data_index)
        original = self.original[index]
        # print(original)
        return train_inputs, train_targets_sparse, train_seq_len_list, data_index, original
    def __len__(self):
        return self.num_total_seqs

class DataFeeder(object):

    def __init__(self, meta_filename):
        self.data_dir = os.path.dirname(meta_filename)
        self.meta_filename = meta_filename

        self.sample_list = []
        with open(meta_filename, 'r') as infile:
            for line in infile:
                sample = line.strip().split('|')
                index, feats_filename, feats_n_frames, labels_filename, label_n_frames, segs_filename, seg_length = sample
                index = int(index)
                feats_filename = os.path.join(self.data_dir, feats_filename)
                labels_filename = os.path.join(self.data_dir, labels_filename)
                segs_filename = os.path.join(self.data_dir, segs_filename)
                self.sample_list.append((index,
                                        feats_filename,
                                        feats_n_frames,
                                        labels_filename,
                                        label_n_frames,
                                        segs_filename,
                                        seg_length))

        print('load %d samples' % len(self.sample_list))
        self.training_ratio = 0.8
        self.training_index = math.floor(len(self.sample_list) * 0.8)
        self.training_set = self.sample_list[:self.training_index]
        self.testing_set = self.sample_list[self.training_index:]
        self.n_training_set = len(self.training_set)
        print('training_set has %d samples' % len(self.training_set))
        print('testing_set has %d samples' % len(self.testing_set))

        self.shuffle = True


    def to_pickle(self, save_path = "."):
        i = 0
        train_inputs_list = []
        train_targets_sparse_list = []
        train_seq_len_list = []
        index_list = []
        original_list = []
        for sample in self.sample_list:
            index, feats_filename, feats_n_frames, labels_filename, label_n_frames, segs_filename, seg_length = sample
            train_inputs = np.load(feats_filename)  # 3-D
            seg_table = np.load(segs_filename)
            train_targets = np.load(labels_filename)
            train_inputs, train_targets_sparse, original = all_select_seg(train_inputs, train_targets, seg_table)  # 0 for scheduler
            train_seq_len = [train_inputs.shape[1]]
            # train_inputs = train_inputs[:, :, :5]
            train_inputs_list.append(train_inputs)
            train_targets_sparse_list.append(train_targets_sparse)
            train_seq_len_list.append(train_seq_len)
            index_list.append(index)
            original_list.append(original)
        train_dict = {}
        train_dict['train_inputs_list'] = train_inputs_list
        train_dict['train_targets_sparse_list'] = train_targets_sparse_list
        train_dict['train_seq_len_list'] = train_seq_len_list
        train_dict['index_list'] = index_list
        train_dict['original_list'] = original_list
        with open(save_path + '/train_data_dict.pickle', 'wb') as handle:
            pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 0

    def next_training_batch(self):
        while True:
            if self.shuffle:
                random.shuffle(self.training_set)
            for sample in self.training_set:
                index, feats_filename, feats_n_frames, labels_filename, label_n_frames, segs_filename, seg_length = sample
                # print('training:', index, os.path.basename(feats_filename), os.path.basename(labels_filename))
                train_inputs = np.load(feats_filename)  # 3-D
                seg_table = np.load(segs_filename)
                train_targets = np.load(labels_filename)
                #print('input shape:', train_inputs.shape)
                train_inputs, train_targets_sparse, original = all_select_seg(train_inputs, train_targets, seg_table)  # 0 for scheduler
                train_seq_len = [train_inputs.shape[1]]
                #train_targets = np.load(labels_filename)
                #print('target shape:', train_targets.shape)
                #TODO: cut the region of interest according to the seg_table
                yield train_inputs, train_targets_sparse, train_seq_len, index, original

    def next_testing_batch(self):
        while True:
            if self.shuffle:     # the testing set has no need to shuffle
                random.shuffle(self.testing_set)
            for sample in self.testing_set:
                index, feats_filename, feats_n_frames, labels_filename, label_n_frames, segs_filename, seg_length = sample
                print('testing:', index, os.path.basename(feats_filename), os.path.basename(labels_filename))
                train_inputs = np.load(feats_filename)  # 3-D
                seg_table = np.load(segs_filename)
                train_targets = np.load(labels_filename)


                train_inputs, train_targets_sparse, original = all_select_seg(train_inputs, train_targets, seg_table) # 1 for complete
                train_seq_len = [train_inputs.shape[1]]


                yield train_inputs, train_targets_sparse, train_seq_len, index, original

if __name__ == '__main__':
    # datafeeder = DataFeeder('./deepsniffer/training_randomgraphs/train.txt')
    # datafeeder.to_pickle(save_path = "./deepsniffer/dataset")
    
    '''Check content of current dataset'''
    datafeeder = Dataset('./obfuscator/dataset/train_data_dict.pickle')
    train_inputs, train_targets_sparse, train_seq_len_list, data_index, original = datafeeder.__getitem__(0)
    print(train_inputs.shape)