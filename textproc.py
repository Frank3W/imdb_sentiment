import numpy as np
import spacy


nlp_english = spacy.load('en_core_web_sm')

def simple_proc(text):
    r_list = []
    for token in nlp_english(text):
        if token.is_alpha and not token.is_stop:
            r_list.append(token.lemma_.lower())
    return r_list


class TextProc:

    def __init__(self, text_corpus):
        
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
        
    def trainmode(self):
        self.mode = 'train'
        
    def evalmode(self):
        self.mode = 'eval'
        
    def process(self, text_corpus=None, top_num=None):
        if self.mode == 'train':
            if top_num is not None:
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


class WordEncoder:
    
    def __init__(self, word_list):
        self.word_num = len(word_list)
        # index 0 is reserved for word not in word_list
        self.word_to_idx = {word : (idx+1) for idx, word in enumerate(word_list)}
        self.idx_to_word = {(idx+1) : word for idx, word in enumerate(word_list)}
        
    def onehot_encode(self, words_collection):
        """Encodes word collections to be one-hot vectors.
        
        Args:
            words_collection: list of string list where each string is considered as a word
        
        Returns:
            a 2d numpy array
        """
        num_text = len(words_collection)
        
        # the index zero is kept 
        code_vecs = np.zeros((num_text, self.word_num + 1)).astype('int')
        
        for idx, word_list in enumerate(words_collection):
            for word in word_list:
                if word in self.word_to_idx:
                    code_vecs[idx, self.word_to_idx[word]] = 1
                else:
                    code_vecs[idx, 0] = 0

        return code_vecs
    
    def idx_encode(self, words_collection):
        """Encodes word collections to be vectors of index integers.
        
        Args:
            words_collection: list of string list where each string is considered as a word
        
        Returns:
            a list of integer lists
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
