# Reviewoler API: Predicting Clients Recommendation from Product Reviews
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/agbleze/recommendation_predictor_API/.github%2Fworkflows%2Fci.yml)
![Static Badge](https://img.shields.io/badge/reviewoler-product%20review%20analyzer-yellow)
![Static Badge](https://img.shields.io/badge/reviewoler-v0.2.5-red)
![GitHub Tag](https://img.shields.io/github/v/tag/agbleze/recommendation_predictor_API)
![GitHub Release](https://img.shields.io/github/v/release/agbleze/recommendation_predictor_API)
![GitHub License](https://img.shields.io/github/license/agbleze/recommendation_predictor_API)


## Project Description

Reviewoler ia an api powered by Deep Learning for predicting whether client will recommend a product and the likelihood of that based on product reviews. This service is containerized with Docker and ready for deployment

The actual development of the model is in a ![different repo](https://github.com/agbleze/reviews_recommender.git)
Thus this repo is meant to serve the model developed from that repo as a self contained api that can be deloyed and used independently.


## 📌 Table of Contents

- [Features](#-features)
- [Quickstart](#quickstart)
- [Docker Deployment](#docker-deployment)
- [Usage Tutorial](#usage-tutorial)
- [License](#license)
- [Contributing](#contributing)

---

## 🚀 Features

- 🔍 Predicts product recommendation likelihood from raw review text
- 🌐 RESTful API with Flask + Flask-RESTful  
- 🐳 Dockerized with Gunicorn for scalable deployment  
---

## Quickstart

### To run the package locally 

1. Create and activate a virtual enviornment. This can be done on unbuntu based system as follows:

- I. Virtual environment called env (name it with your preference)

``` python -m venv env ```

- II. Activate the virtual environment

``` source env/bin/activate ```


2. Clone the repository as follows

```git clone https://github.com/agbleze/recommendation_predictor_API.git ```


3. Install Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

4. Install reviewoler package as follows

``` pip install . ```

5. Run reviewoler

``` python -m reviewoler ```


By default, the api is served at port number 8080 hence you need to ensure no application is already using that port else you have to use a different port number. 
The URL is expected to default to http://192.168.0.168:8080  or http://127.0.0.1:8080. 

The root endpoint when called returns a message describing the api - ```"{"message": "This is a Deep Learning API for predicting whether a product product will be recommended based on reviews"}"```


## Docker deployment

Dockerfile is provided for building and running reviewoler as a docker container.

Assumming Docker is already installed, build your image with:

#### Build Docker image
```bash
docker build -t your_image_name:your_image_tag .
```

#### Run the Docker Container

To run the container and expose the API on port 5000, use:

```bash
docker run --rm -p 8080:8080 your_image_name:your_image_tag
```

This command maps port 8080 of your machine to the container’s port 8080. 
Hence, that port number has to be free. Once the container is running, you can test it just like local API.


## Usage Tutorial

### How to use the reviewoler to predict whether client/customer/user will recommend the product

reviewoler provides helper function that accepts a review text and makes prediction.

```python

from reviewoler.utils.helpers import request_prediction

URL='http://127.0.0.1:8080/predict'

review = 'its a complete scam',


result = request_prediction(URL=URL, 
                            review_data=review
                            )
print(f"Result: {result}")
```

## Test reviewoler

To test the package after cloning it, use the command below

```pytest```


## LICENSE

MIT License. See LICENSE for details.

## CONTRIBUTING

Read CONTRIBUTING.md for details on how to contribute.