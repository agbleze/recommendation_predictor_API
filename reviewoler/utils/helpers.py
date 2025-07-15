from argparse import Namespace
import json
import requests
from ..model_store.artefacts import model_path, vector_path
import numpy as np
from scipy.special import softmax

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

def predict_category(review, session, vectorizer, max_length):
    """
    Predicts recommendation status  from product review text using ONNX Runtime
    
    Args:
        review (str): Review text
        session (onnxruntime.InferenceSession): Loaded ONNX session
        vectorizer (ReviewVectorizer): Corresponding vectorizer instance
        max_length (int): Rthe max sequence length
#             CNN are sensitive to the input data tensor size, 
#             This ensures to keep it the same size as the training data
        
    Returns:
        dict: {'category': str, 'probability': float}
    """
    token_ids = vectorizer.vectorize(review, vector_length=max_length)
    input_array = np.array([token_ids], dtype=np.int64)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_array})[0]
    probs = softmax(logits, axis=1)  
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

