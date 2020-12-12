"""
Module for text processing.
"""

import numpy as np
import spacy
import json


nlp_english = spacy.load('en_core_web_sm')


def simple_proc(text):
    """Applies a simple processing step to text.

    Args:
        text (str): a paragraph text

    Returns:
        list: tokens
    """
    r_list = []
    for token in nlp_english(text):
        if token.is_alpha and not token.is_stop:
            r_list.append(token.lemma_.lower())
    return r_list


class TextProc:
    """Text processing.

    It has two modes: training mode and evaluation mode.

    Attributes:
        count_pair_list (list of tuple): the tuple has 2 elements word string and frequency count.
            The list is sorted in the descending order by frequency count.
        words_collection (list of list): the inner list is of tokens.
        mode (str): value of 'train' or 'eval' that gives the mode.
        is_trained (bool): whether the text processing object is trained or not.
        top_num (int, optional): number of top words in count_pair_list used in processing. If None, all words
            are used.

    """
    def __init__(self, text_corpus=None, ordered_count_pairs=None, top_num=None):
        """Inits an text processing object.

        Either text_corpus is provided or ordered_count_pairs and top_num are provided.
        The first case creates an object in training mode and the second case creates
        an object in evaluation mode.

        text_corpus (list of str, optional): text collection
        ordered_count_pairs (list of tuple, optional): the tuple has 2 elements word string and frequency count.
            The list is sorted in the descending order by frequency count.
        top_num (int, optional): number of top words in count_pair_list used in processing. If None, all words
            are used.

        """
        if text_corpus is not None:
            words_collection = [simple_proc(tmp) for tmp in text_corpus]

            self.words_collection = words_collection

            word_count_dict = {}
            for word_list in words_collection:
                for word in word_list:
                    if word in word_count_dict:
                        word_count_dict[word] += 1
                    else:
                        word_count_dict[word] = 1

            count_pair_list = []
            for key, value in word_count_dict.items():
                count_pair_list.append((key, value))

            count_pair_list.sort(key=lambda x: x[1], reverse=True)

            self.count_pair_list = count_pair_list
            self.mode = 'train'
            self.is_trained = False
            self.top_num = None
        else:
            self.words_collection = None
            self.count_pair_list = ordered_count_pairs
            self.mode = 'eval'
            self.is_trained = True
            self.top_num = top_num

    @classmethod
    def from_load_wcount_pair(cls, path):
        """Loads json word count data to initialize a text processing object.

        It is the loading function that corresponds to persistence function save_wcount. The json
        has two keys: wordcount, top_num
        """
        with open(path, 'r') as infile:
            data_dict = json.load(infile)

        count_pair_list = []
        for key, value in data_dict['wordcount'].items():
            count_pair_list.append((key, value))

        count_pair_list.sort(key=lambda x: x[1], reverse=True)

        return cls(ordered_count_pairs=count_pair_list, top_num=data_dict['top_num'])

    def save_wcount(self, path):
        """Saves word count info into json for persisting trained text processing object."""
        data_dict = {}
        count_dict = {_[0] : _[1] for _ in self.count_pair_list}
        data_dict['top_num'] = self.top_num
        data_dict['wordcount'] = count_dict
        with open(path, 'w') as outfile:
            json.dump(data_dict, outfile)

    def trainmode(self):
        """Switches to training mode"""
        self.mode = 'train'

    def evalmode(self):
        """Switches to evaluation mode"""
        self.mode = 'eval'

    def get_selectedwords(self):
        """Gets selected word vocabulary.

        Returns:
            list
        """
        if self.mode != 'evel':
            raise Exception('The mode must be in "evel".')

        if self.top_num is not None:
            selected_words = [item[0] for item in self.count_pair_list[:self.top_num]]
        else:
            selected_words = [item[0] for item in self.count_pair_list]

        return selected_words

    def process(self, text_corpus=None, top_num=None):
        """Processes data.

        It behaves differently in two different modes: 'train' and 'eval'. When self is in mode 'train',
        text_corpus must be None and only top_num applies. When self is in mode 'test', text_corpus must
        be not None and top_num not applies.

        text_corpus (list of str, optional): text collection
        top_num (int, optional): number of top words in count_pair_list used in processing. If None, all words
            are used.

        Returns:
            tuple:
                list of str lists: processed word collections
                list of strs: word vocabulary

        """
        if self.mode == 'train':
            self.top_num = top_num
            if text_corpus is not None:
                raise ValueError('If text_corpus is given, please do not use train mode. Use method evalmode to change mode.')

            words_collection = self.words_collection
        else:
            if not self.is_trained:
                raise Exception('Function needs to be called first in train mode.')

            words_collection = [simple_proc(tmp) for tmp in text_corpus]

        if self.top_num is not None:
            selected_words = [item[0] for item in self.count_pair_list[:self.top_num]]
        else:
            selected_words = [item[0] for item in self.count_pair_list]

        selected_word_set = set(selected_words)

        if self.mode == 'train':
            self.is_trained = True

        if self.top_num is None:
            return words_collection, selected_word_set

        masked_words_collection = []

        for word_list in words_collection:
            masked_word_list = []

            for word in word_list:
                if word in selected_word_set:
                    masked_word_list.append(word)

            masked_words_collection.append(masked_word_list)

        return masked_words_collection, selected_words
