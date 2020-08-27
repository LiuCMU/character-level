import json
import numpy as np
from collections import Counter

from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from . import utils # import modules from sibling packages https://docs.python.org/3.4/tutorial/modules.html#importing-from-a-package

import torch


def get_sample_weights(labels):
    '''
    For example, input labels = [1, 1, 0], returns [0.5 0.5 1. ]
    '''

    counter = Counter(labels) # for a list of labels, elements are stored as dictinary keys and their counts are stored as dictionary values
    counter = dict(counter)
    for k in counter:
        counter[k] = 1 / counter[k]
    sample_weights = np.array([counter[l] for l in labels])
    return sample_weights


def load_data(args):
    # chunk your dataframes in small portions
    chunks = pd.read_csv(args.data_path,
                         usecols=[args.text_column, args.label_column],
                         chunksize=args.chunksize,
                         encoding=args.encoding,
                         nrows=args.max_rows,
                         sep=args.sep)
    texts = []
    labels = []
    for df_chunk in tqdm(chunks):
        aux_df = df_chunk.copy()
        aux_df = aux_df.sample(frac=1) # return a random sample of the dataframe, used to SCHUFFLE here
        aux_df = aux_df[~aux_df[args.text_column].isnull()] # removing rows containing no text
        aux_df = aux_df[(aux_df[args.text_column].map(len) > 1)] # removing rows containing only 1 letter.
        aux_df['processed_text'] = (aux_df[args.text_column]
                                    .map(lambda text: utils.process_text(args.steps, text)))
        texts += aux_df['processed_text'].tolist() # transform a dataframe column into a list
        labels += aux_df[args.label_column].tolist()

    if bool(args.group_labels): #bool(1) == True, bool(0) == False

        if bool(args.ignore_center):

            label_ignored = args.label_ignored

            clean_data = [(text, label) for (text, label) in zip(
                texts, labels) if label not in [label_ignored]] # ignore some data points if they contain certain labels

            texts = [text for (text, label) in clean_data]
            labels = [label for (text, label) in clean_data]

        #     labels = list(
        #         map(lambda l: {1: 0, 2: 0, 4: 1, 5: 1}[l], labels)) # This is not a good method to map labels.

        # else:
        #     labels = list(
        #         map(lambda l: {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}[l], labels))
        
    if bool(args.balance):

        counter = Counter(labels)
        keys = list(counter.keys())
        values = list(counter.values())
        count_minority = np.min(values)

        balanced_labels = []
        balanced_texts = []

        for key in keys: 
            balanced_texts += [text for text, label in zip(texts, labels) if label == key][:int(args.ratio * count_minority)] #slicing is a good
            # to keep balanced labels. It's worth learning.
            balanced_labels += [label for text, label in zip(texts, labels) if label == key][:int(args.ratio * count_minority)] 

        texts = balanced_texts
        labels = balanced_labels

    number_of_classes = len(set(labels))

    print(
        f'data loaded successfully with {len(texts)} rows and {number_of_classes} labels')
    print('Distribution of the classes', Counter(labels))

    sample_weights = get_sample_weights(labels)

    return texts, labels, number_of_classes, sample_weights


class MyDataset(Dataset):
    def __init__(self, texts, labels, args):
        self.texts = texts
        self.labels = labels
        self.length = len(self.texts)

        self.vocabulary = args.alphabet + args.extra_characters
        self.number_of_characters = args.number_of_characters + \
            len(args.extra_characters)
        self.max_length = args.max_length
        self.preprocessing_steps = args.steps
        self.identity_mat = np.identity(self.number_of_characters) # identity array is an array with ones in diaganol and zeros elsewhere.
        # Input is the #rows = # columns

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]

        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text)[::-1] if i in self.vocabulary],
                        dtype=np.float32)                  # -1 denotes starting from the end to beginning when looping string
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros(
                (self.max_length, self.number_of_characters), dtype=np.float32)

        label = self.labels[index]
        data = torch.Tensor(data)

        return data, label
