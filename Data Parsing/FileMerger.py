import glob
import sys
import csv
import os

def MergeFeatureFiles(filePath,label, outFile):
    files = glob.glob(filePath)   
    for name in files:
        try:
            with open(name) as inFile: 
                outFile.writerow([inFile.read(),name.split('_')[1].split('.')[0],label])
        except:
            print(name)

dirname = os.path.dirname(__file__)
dirname=os.path.abspath(os.path.join(dirname, os.pardir))
filePath = os.path.join(dirname,"Unprocessed data/aclImdb/train/pos/*") #sys.argv[1]
label = "1" #sys.argv[2]
outFileName = os.path.join(dirname,"Data Parsing/labeledData.csv") #sys.argv[3]

outFile = csv.writer(open(outFileName, "w", newline=''))

MergeFeatureFiles(filePath,label,outFile)


filePath = os.path.join(dirname,"Unprocessed data/aclImdb/train/neg/*") #sys.argv[1]
label = "0" #sys.argv[2]

MergeFeatureFiles(filePath,label,outFile)

outFileName = os.path.join(dirname,"Data Parsing/testLabeledData.csv")
outFile = csv.writer(open(outFileName, "w", newline=''))

filePath = filePath = os.path.join(dirname,"Unprocessed data/aclImdb/test/pos/*") #sys.argv[1]
label = "1" #sys.argv[2]

MergeFeatureFiles(filePath,label,outFile)

filePath = filePath = os.path.join(dirname,"Unprocessed data/aclImdb/test/neg/*") #sys.argv[1]
label = "0" #sys.argv[2]

MergeFeatureFiles(filePath,label,outFile)