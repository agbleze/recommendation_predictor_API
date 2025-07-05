
#%%
import requests



#requests.post(url='http://127.0.0.1:8000/predict', json={'review': 'this is a greate product'})
# %%

#requests.get(url='http://127.0.0.1:8000')

# %%
# res = requests.post(url='https://fbfb-138-246-3-71.ngrok.io/predict', 
#                     json={'review': 'recommend product'})

# print(res.content)


#https://bewjff2ygzhmhe7luyxlbgk3am.srv.us/

# %%
import json
def request_prediction(URL: str, review_data: str):
    in_data = {'review': review_data}
    req = requests.post(url = URL, json=in_data)
    response = req.content
    prediction = json.loads(response)#['category']
    return prediction
  
# %%
review_data = ['its a complete scam', 'such an asshole, I will never buy this product again',
                'Just another bad buy', 'It arrrive on time', 'it is a great',
                'It arrrive on time and it is a great', ' this is a wonderful product',
                ' I hate the product', 'I dont like it. I hate the product',
                'I love the product', 'recommend product', 'love it'
                ]
request_prediction(URL='http://127.0.0.1:5000/predict', 
                   review_data=review_data[-1]
                   )
# %%
