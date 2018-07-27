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

sample_size=1000
reviews = trainData.iloc[:sample_size, 0]
grades = trainData.iloc[:sample_size, 1]

testReviews = testData.iloc[:sample_size, 0]
testGrades = testData.iloc[:sample_size, 1]

dataProcessing = DataProcessing(reviews, testReviews, grades, testGrades, True)
dataProcessing.cleanData()
dataProcessing.tokenize()

trainData, testData, trainLable, testLable = dataProcessing.getData()
print(trainData[0])
logisticRegresionModel = LogisticRegresionModel(trainData, testData, trainLable, testLable)
logisticRegresionModel.model()
#print(trainData)
