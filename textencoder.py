"""
Module for text encoder that transforms word collections to numeric features.
"""

import numpy as np


class TextEncoder:
    """Abstract class for text encoder.
    """
    def encode(self, words_collection):
        raise NotImplementedError('Implementation for method "encode" is required.')


class OneHotEncoder(TextEncoder):
    """One hot encoder for text.
    
    Attributes:
        word_to_idx (dict): mapping from word to index.
        idx_to_word (dict): mapping from index to word.

    """
    
    def __init__(self, word_vocab):
        """Inits one hot encoder object.
        
        Args:
            word_vocab (list of strs): the words are not repeated as a vocabulary.

        """
        # index 0 is reserved for word not in word_vocab
        self.word_to_idx = {word : (idx+1) for idx, word in enumerate(word_vocab)}
        self.idx_to_word = {(idx+1) : word for idx, word in enumerate(word_vocab)}
        
    def encode(self, words_collection):
        """Encodes word collections to be one-hot vectors.
        
        Args:
            words_collection (list of str lists): each string is considered as a word
        
        Returns:
            2d numpy array

        """
        num_text = len(words_collection)
        
        # the index zero is kept 
        code_vecs = np.zeros((num_text, len(self.word_to_idx) + 1)).astype('int')
        
        for idx, word_list in enumerate(words_collection):
            for word in word_list:
                if word in self.word_to_idx:
                    code_vecs[idx, self.word_to_idx[word]] = 1
                else:
                    code_vecs[idx, 0] = 0

        return code_vecs
    

class IndexEncoder(TextEncoder):
    """Index encoder for text.
    
    Attributes:
        word_to_idx (dict): mapping from word to index.
        idx_to_word (dict): mapping from index to word.

    """
    def __init__(self, word_vocab):
        # index 0 is reserved for word not in word_vocab
        self.word_to_idx = {word : (idx+1) for idx, word in enumerate(word_vocab)}
        self.idx_to_word = {(idx+1) : word for idx, word in enumerate(word_vocab)}
        
    def encode(self, words_collection):
        """Encodes word collections to be vectors of index integers.
        
        Args:
            words_collection (list of str lists): each string is considered as a word
        
        Returns:
            list of int lists

        """
        idx_vecs = []
        
        for word_list in words_collection:
            curr_idx_list = []
            for word in word_list:
                if word in self.word_to_idx:
                    curr_idx_list.append(self.word_to_idx[word])
                else:
                    curr_idx_list.append(0)

            idx_vecs.append(curr_idx_list)

        return idx_vecs
