
#%%
import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional
from vocabulary import Vocabulary
from sequence_vocabulary import SequenceVocabulary
from data_preprocess import preprocess_texts_to_tokens



#%% Vectorizer
class ReviewVectorizer(object):
    """coordinate the vocabulary and put them to use"""
    def __init__(self, review_vocab, category_vocab) -> None:
        self.review_vocab = review_vocab
        self.category_vocab = category_vocab
        
    def vectorize(self, review, vector_length=-1):
        indices = [self.review_vocab.begin_seq_index]
        indices.extend(self.review_vocab.lookup_token(token) for token in preprocess_texts_to_tokens(review))
        indices.append(self.review_vocab.end_seq_index)
        
        if vector_length < 0:
            vector_length = len(indices)
            
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.review_vocab.mask_index
        
        return out_vector
    
    @classmethod
    def from_dataframe(cls, review_df, cutoff=0):
        category_vocab = Vocabulary()
        for category in sorted(set(review_df['reviews.doRecommend'])):
            category_vocab.add_token(category)
            
        word_counts = Counter()
        for review in review_df['reviews.text']:
            # for token in title.split(" "):
            #     if token not in string.punctuation:
            #         word_counts[token] += 1
            
            for token in preprocess_texts_to_tokens(review):
                word_counts[token] += 1
                    
        review_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                review_vocab.add_token(word)
                
        return cls(review_vocab, category_vocab)
    
    @classmethod
    def from_serializable(cls, contents):
        review_vocab = SequenceVocabulary.from_serializable(contents['title_vocab'])
        
        category_vocab = Vocabulary.from_serializable(contents['category_vocab'])
        
        return cls(review_vocab=review_vocab, category_vocab=category_vocab)
    
    #@classmethod
    def to_serializable(self):
        return {'review_vocab': self.review_vocab.to_serializable(),
                'category_vocab': self.category_vocab.to_serializable()}
    



