import pandas as pd
import nltk
import tensorflow
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import sys
import os

from SVMModel import SVMModel
from LogisticRegresionModel import LogisticRegresionModel
from DataProcessing import DataProcessing

dirname = os.path.dirname(__file__)
dirname=os.path.abspath(os.path.join(dirname, os.pardir))
trainData = pd.read_csv(os.path.join(dirname,
    'Data Parsing\\labeledData.csv'))
testData = pd.read_csv(os.path.join(dirname,
    'Data Parsing\\TestLabeledData.csv'))

np.random.seed(5)

permutation = list(np.random.permutation(trainData.shape[0]))
reviews = trainData.iloc[permutation, 0]
numberGrades = trainData.iloc[permutation, 1].divide(10)
grades = trainData.iloc[permutation, 2]

permutation = list(np.random.permutation(testData.shape[0]))
testReviews = testData.iloc[permutation, 0]
testNumberGrades = testData.iloc[permutation, 1].divide(10)
testGrades = testData.iloc[permutation, 2]

dataProcessing = DataProcessing(reviews, testReviews, grades, testGrades, True)
dataProcessing.cleanData()
dataProcessing.tokenize()

trainData, testData, trainLable, testLable = dataProcessing.getData()
trainLable = np.array(trainLable)
#trainData = np.hstack((trainData,np.transpose(np.matrix(numberGrades.values))))
testLable = np.array(testLable)
#testData = np.hstack((testData,np.transpose(np.matrix(testNumberGrades.values))))


print(trainLable[0:10])
print(testLable[0:10])
learningModel = SVMModel(trainData, testData, trainLable, testLable)
learningModel.model()
#print(trainData)
