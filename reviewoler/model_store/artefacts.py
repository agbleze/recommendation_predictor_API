import os
from ..data.review_dataset import ReviewDataset


__all__ = ['model_path',
            'vector_path',
            'embedding_path'
            ]

_dirname = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(_dirname, "model.pth")
vector_path = os.path.join(_dirname, 'vectorizer.json')
embedding_path = os.path.join(_dirname, "embeddings.npy")
