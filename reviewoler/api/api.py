from flask import Flask, request
from flask_restful import Resource, Api
from ..utils.helpers import predict_category
import json
from ..utils.helpers import (args, predict_category)
from ..preprocess.vectorizer import ReviewVectorizer
import onnxruntime as ort

file = open(args.vectorizer_file)
vectorizer_file = json.load(file)

review_vocab = vectorizer_file['review_vocab']['token_to_idx'].keys()
review_outcome = vectorizer_file['category_vocab']['token_to_idx']  


vectorizer = ReviewVectorizer.from_serializable(contents=vectorizer_file)
session = ort.InferenceSession(args.model_file)

class RecommendPredictor(Resource):
    @staticmethod
    def post():
        review = request.get_json()['review']
        
        result = predict_category(review=review, session=session,
                                vectorizer=vectorizer, 
                                max_length=args.max_seq_length + 2 # +2 for the begin and end sequence tokens
                                )       
        return result
    
class Entrypoint(Resource):
    @staticmethod
    def get():
        return {'message': '''This is a Deep Learning API for predicting whether a product product will be recommended based on reviews'''}            

app = Flask(__name__)
api = Api(app)

api.add_resource(Entrypoint, '/')
api.add_resource(RecommendPredictor, '/predict')

if __name__ == '__main__':
    app.run()