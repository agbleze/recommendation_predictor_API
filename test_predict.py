#%%
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from reviewoler.utils.helpers import predict_category
import json
from reviewoler.utils.helpers import (args, predict_category)
from reviewoler.preprocess.vectorizer import ReviewVectorizer
from reviewoler.model_store.artefacts import model_path, vector_path


#%%
file = open(vector_path)
vectorizer_file = json.load(file)

review_vocab = vectorizer_file['review_vocab']['token_to_idx'].keys()
review_outcome = vectorizer_file['category_vocab']['token_to_idx']  

#%%

vectorizer = ReviewVectorizer.from_serializable(contents=vectorizer_file)
session = ort.InferenceSession(model_path)
#input_tensor = sequence.reshape(1, 1, -1)
# inputs = {session.get_inputs()[0].name: sequence}

# outputs = session.run(None, inputs)
# outputs
# #%%

# len(token_ids)

# #%%

# logits = softmax(outputs[0], axis=1)
# logits
# #%%
# np.argmax(logits)
#%%
review = "i recommend"
result = predict_category(review=review, session=session,
                        vectorizer=vectorizer, 
                        max_length=args.max_seq_length + 2 # +2 for the begin and end sequence tokens
                        )       


#%%

result


#%%  
token_ids = [847, 23, 112, 592, 9, 847, 23, 112, 592, 
             847, 23, 112, 23, 112, 592, 9, 
             847, 23, 112, 592, 847, 23, 112,
             23, 112, 592, 9, 847, 23,
             847, 23, 112, 592, 847, 23, 112,
             23, 112, 592, 9, 847, 23
             ]
sequence = np.array([token_ids], dtype=np.int64)
model_path = "/home/lin/codebase/review_classifier/model_store/model_state.onnx"
op_model_path = "/home/lin/codebase/review_classifier/model_store/optimized_model.onnx"
session = ort.InferenceSession(op_model_path)
#input_tensor = sequence.reshape(1, 1, -1)
inputs = {session.get_inputs()[0].name: sequence}

outputs = session.run(None, inputs)
outputs
#%%

len(token_ids)

#%%

logits = softmax(outputs[0], axis=1)
logits
#%%
np.argmax(logits)


#%%

softmax([10, 10])

# %%
for i in session.get_inputs():
    print("Name:", i.name)
    print("Shape:", i.shape)
    print("Type:", i.type)

# %%
session._model_meta

# %%
