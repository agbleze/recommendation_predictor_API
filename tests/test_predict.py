import onnxruntime as ort
from reviewoler.utils.helpers import predict_category
import json
from reviewoler.utils.helpers import (args, predict_category)
from reviewoler.preprocess.vectorizer import ReviewVectorizer
from reviewoler.model_store.artefacts import model_path, vector_path
import pytest


@pytest.fixture()
def vocab():
    file = open(vector_path)
    _vocab = json.load(file)
    return _vocab

@pytest.fixture()
def vectorizer(vocab):
    _vectorizer = ReviewVectorizer.from_serializable(contents=vocab)
    return _vectorizer

@pytest.fixture()
def model_session():
    session = ort.InferenceSession(model_path)
    return session


@pytest.fixture()
def prediction(vectorizer, model_session):
    review = "i love this great product. value for money is great"
    result = predict_category(review=review, session=model_session,
                            vectorizer=vectorizer, 
                            max_length=args.max_seq_length + 2 # +2 for the begin and end sequence tokens
                            ) 
    return result   

def test_data_type(prediction):
    assert isinstance(prediction, dict)

def test_expected_keys_are_available(prediction):
    assert "category" in prediction
    assert "probability" in prediction
    
def test_prediction_sanity(prediction):
    pred = prediction["category"]
    assert pred == "true"