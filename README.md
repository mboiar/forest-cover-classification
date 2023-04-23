# Evaluation of different classification models on CoverType dataset

## Classifiers

- DecisionTree (Acc. 88%)
- KNN (Acc. 87%)
- NaiveBayes
- Neural Network

## Usage

To obtain a classifier's prediction using RestAPI, send a POST request to the `/classify` endpoint with the following data:
```{"feature_data": <feature vector>, "model": "<model name: DecisionTree|KNN|NN|NaiveBayes>"}```
