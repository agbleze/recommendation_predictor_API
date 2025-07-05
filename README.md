# Reviewoler: Deep learning powered API for predicting whether clients will recommend product based on product reviews
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/agbleze/recommendation_predictor_API/.github%2Fworkflows%2Fci.yml)
![Static Badge](https://img.shields.io/badge/reviewoler-product%20review%20analyzer-yellow)



This repo contains code for an API backed by Deep Learning with Natural Language Processing 


## Project Description

This is a Natural Language Processing project for predicting whether 
a product will be recommended based on review

The value this project seeks to deliver is to enable businesses analyze their product reviews and 
and determine whether or not a client or customer is likely to recommend their 
product or service. The tool can be used on large scale to:

1. Determine after-sales and purchase intent of your customers

2. To augment other sales data for prediction tasks

3. To improve customer satisfaction by automatically identifying and resolving issues

4. To augment and enrich other text datasets for advance analysis


## Dataset and variables used

Dataset: Product reviews provided by customers and clients after ordering products or patronizing services.
The dataset encompass variety of products and services by businesses without focus on a single category.
                                     

Target variable: Recommendation status with binary values depicting whether or not a product was recommended

1. Predictor variable: Review of product which is text provided by a client or customer who purchased the 
product or patronized the service


## Method of analysis

The reviews were preprocessed, vectorized and an embedding matrix developed. 
A neural network architecture was developed and feed with processed data for training and validation of model
                                 

## Result and use

1. A Deep learning API that accepts a product review and predict whether client is likely to recommend the product as a response.
                                   
