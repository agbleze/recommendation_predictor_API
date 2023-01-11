
#%%
import requests



#requests.post(url='http://127.0.0.1:8000/predict', json={'review': 'this is a greate product'})
# %%

#requests.get(url='http://127.0.0.1:8000')

# %%
res = requests.post(url='https://f696-138-246-3-71.ngrok.io/predict', json={'review': 'I recommend product'})

res.content




# %%
