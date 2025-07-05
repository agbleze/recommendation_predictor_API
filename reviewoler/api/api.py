from flask import Flask, request
from flask_restful import Resource, Api
from ..utils.helpers import predict_category
from ..model.classifier import ReviewClassifier
import torch
import json
from ..utils.helpers import (args, predict_category)
from ..preprocess.vectorizer import ReviewVectorizer
from ..utils.embedding_matrix import EmbeddingMatrixMaker
from ..model_store.artefacts import embedding_path

file = open(args.vectorizer_file)
vectorizer_file = json.load(file)

review_vocab = vectorizer_file['title_vocab']['token_to_idx'].keys()
review_outcome = vectorizer_file['category_vocab']['token_to_idx']  

embedding_matrix = EmbeddingMatrixMaker(glove_filepath=args.glove_filepath,
                                        words=review_vocab
                                    )

embeddings = embedding_matrix.load_embedding(embedding_path)
classifier = ReviewClassifier(embedding_size=args.embedding_size,
                            num_embeddings=len(review_vocab),
                            num_channels=args.num_channels,
                            hidden_dim=args.hidden_dim,
                            num_classes=len(review_outcome),
                            dropout_p=args.dropout_p,
                            pretrained_embeddings=embeddings,
                            padding_idx=0
                            )

classifier.load_state_dict(torch.load(args.model_state_file))

classifier = classifier.to(args.device)
vectorizer = ReviewVectorizer.from_serializable(contents=vectorizer_file)
class RecommendPredictor(Resource):
    @staticmethod
    def post():
        review = request.get_json()['review']
        
        result = predict_category(review=review, classifier=classifier,
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