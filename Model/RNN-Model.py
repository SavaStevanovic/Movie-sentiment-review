import pandas as pd
import nltk
import tensorflow
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import os

from DataProcessing import DataProcessing

trainData = pd.read_csv(
    'C:/Users/Sava/Documents/Movie-sentiment-review/Data Parsing/labeledData.csv')
testData = pd.read_csv(
    'C:/Users/Sava/Documents/Movie-sentiment-review/Data Parsing/TestLabeledData.csv')

reviews = trainData.iloc[:10, 0]
grades = trainData.iloc[:10, 1]

testReviews = testData.iloc[:10, 0]
testGrades = testData.iloc[:10, 1]

dataProcessing = DataProcessing(reviews, testReviews, grades, testGrades, True)
dataProcessing.cleanData()
dataProcessing.tokenize()

trainData, testData, trainLable, testLable = dataProcessing.getData()
print(trainData)
