import pandas as pd
import nltk
import tensorflow
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import os

from LogisticRegresionModel import LogisticRegresionModel
from DataProcessing import DataProcessing

dirname = os.path.dirname(__file__)
dirname=os.path.abspath(os.path.join(dirname, os.pardir))
trainData = pd.read_csv(os.path.join(dirname,
    'Data Parsing\\labeledData.csv'))
testData = pd.read_csv(os.path.join(dirname,
    'Data Parsing\\TestLabeledData.csv'))

np.random.seed(5)

sample_size=trainData.shape[0]
permutation = list(np.random.permutation(trainData.shape[0]))[:sample_size]
reviews = trainData.iloc[permutation, 0]
grades = trainData.iloc[permutation, 1]

permutation = list(np.random.permutation(testData.shape[0]))[:sample_size]
testReviews = testData.iloc[permutation, 0]
testGrades = testData.iloc[permutation, 1]

dataProcessing = DataProcessing(reviews, testReviews, grades, testGrades, True)
dataProcessing.cleanData()
dataProcessing.tokenize()

trainData, testData, trainLable, testLable = dataProcessing.getData()
trainLable = np.array(trainLable)
trainData = np.array(trainData)
testLable = np.array(testLable)
testData = np.array(testData)


print(trainLable[0:10])
print(testLable[0:10])
logisticRegresionModel = LogisticRegresionModel(trainData, testData, trainLable, testLable)
logisticRegresionModel.model()
#print(trainData)
