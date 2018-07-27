import pandas as pd
import nltk
import tensorflow
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import os

from LogisticRegresionModel import LogisticRegresionModel
from DataProcessing import DataProcessing

trainData = pd.read_csv(
    'C:/Users/Sava/Documents/Movie-sentiment-review/Data Parsing/labeledData.csv')
testData = pd.read_csv(
    'C:/Users/Sava/Documents/Movie-sentiment-review/Data Parsing/TestLabeledData.csv')

sample_size=5
permutation = list(np.random.permutation(trainData.shape[0]))[:sample_size]
reviews = trainData.iloc[permutation, 0]
grades = trainData.iloc[permutation, 1]

testReviews = testData.iloc[permutation, 0]
testGrades = testData.iloc[permutation, 1]

dataProcessing = DataProcessing(reviews, testReviews, grades, testGrades, True)
dataProcessing.cleanData()
dataProcessing.tokenize()

trainData, testData, trainLable, testLable = dataProcessing.getData()
print(trainData[0])
logisticRegresionModel = LogisticRegresionModel(trainData, testData, trainLable, testLable)
logisticRegresionModel.model()
#print(trainData)
