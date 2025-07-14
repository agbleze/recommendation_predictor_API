import os
from argparse import Namespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import requests
from ..model_store.artefacts import model_path, vector_path

args = Namespace(
    # Data and Path hyper parameters
    data_csv=None,
    vectorizer_file=vector_path,
    model_file=model_path,
    save_dir="model_store",
    # Model hyper parameters
    glove_filepath='glove/glove.6B.100d.txt', 
    use_glove=False,
    embedding_size=100, 
    hidden_dim=100, 
    num_channels=100, 
    # Training hyper parameter
    seed=1337, 
    learning_rate=0.001, 
    dropout_p=0.1, 
    batch_size=128, 
    num_epochs=10, 
    early_stopping_criteria=5, 
    # Runtime option
    cuda=True, 
    catch_keyboard_interrupt=True, 
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
    device='cpu',
    max_seq_length=1646, # This is the max length of the sequence
) 


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100        
        
       
def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
        
def load_glove_from_file(glove_filepath):
    """Load the glove embeddings
    
    Args:
        glove_filepath (str): path to the glove embedding file
        
    Returns:
        word_to_idx (dict): embeddings (numpy.ndarray)
    
    """
    word_to_index = {}
    embeddings = []    
    with open(glove_filepath) as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line word num1 num2 ...
            word_to_index[line[0]] = index
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)
            
        
def make_embedding_matrix(glove_filepath, words):
    """create embedding matrix from a specific set of words
    
    Args:
        glove_fliepath (str): fle path to the glove embeddings
        words (list): list of words in the dataset
    
    """        
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    
    final_embeddings = np.zeros((len(words), embedding_size))
    
    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i
            
    return final_embeddings

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
        
# def predict_category(review, classifier, vectorizer, max_length):
#     """Predicts a news category for a new title
    
#     Args:
#         review (str): a raw title string
#         classifier (ReviewClassifier): an instanve of the trained classifier
#         vectorizer (ReviewVectorizer): the corresponding vectorizer
#         max_length (int): the max sequence length
#             CNN are sensitive to the input data tensor size, 
#             This ensures to keep it the same size as the training data
#     """
#     vectorized_review = torch.tensor(vectorizer.vectorize(review, vector_length=max_length))
#     result = classifier(vectorized_review.unsqueeze(0), apply_softmax=True)
#     probability_values, indices = result.max(dim=1)
#     predicted_category = vectorizer.category_vocab.lookup_index(indices.item())
    
#     return {'category': predicted_category,
#             'probability': probability_values.item()
#             }


import numpy as np
from scipy.special import softmax

def predict_category(review, session, vectorizer, max_length):
    """
    Predicts a news category for a new title using ONNX Runtime
    
    Args:
        review (str): A raw title string
        session (onnxruntime.InferenceSession): Loaded ONNX session
        vectorizer (ReviewVectorizer): Corresponding vectorizer instance
        max_length (int): Required input length for CNN
        
    Returns:
        dict: {'category': str, 'probability': float}
    """
    # Step 1: Tokenize + pad/truncate input
    token_ids = vectorizer.vectorize(review, vector_length=max_length)
    input_array = np.array([token_ids], dtype=np.int64)  # shape: [1, max_length]
    
    # Step 2: Run ONNX inference
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_array})[0]  # shape: [1, num_classes]
    
    # Step 3: Apply softmax to get probabilities
    probs = softmax(logits, axis=1)  # shape: [1, num_classes]
    
    # Step 4: Get highest probability class
    predicted_idx = np.argmax(probs[0])
    predicted_category = vectorizer.category_vocab.lookup_index(predicted_idx)
    confidence = probs[0][predicted_idx]
    
    return {
        'category': predicted_category,
        'probability': float(confidence)
    }
 
def request_prediction(URL: str, review_data: str):
    in_data = {'review': review_data}
    req = requests.post(url = URL, json=in_data)
    response = req.content
    prediction = json.loads(response)
    return prediction

